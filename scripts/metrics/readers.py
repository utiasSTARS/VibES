from tracemalloc import start
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import h5py
import cv2
import os
from pathlib import Path


def get_hdf5_file_size(filepath, unit="MB"):
    """
    Get the file size of an HDF5 file.

    Args:
        filepath (str): Path to the HDF5 file
        unit (str): Unit for file size ('B', 'KB', 'MB', 'GB'). Default is 'MB'.

    Returns:
        float: File size in the specified unit

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the unit is not supported
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    # Get file size in bytes
    file_size_bytes = os.path.getsize(filepath)

    # Convert to requested unit
    unit_conversions = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

    if unit.upper() not in unit_conversions:
        raise ValueError(
            f"Unsupported unit '{unit}'. Supported units: {list(unit_conversions.keys())}"
        )

    file_size = file_size_bytes / unit_conversions[unit.upper()]

    logging.debug(f"HDF5 file '{filepath}' size: {file_size:.2f} {unit.upper()}")

    return file_size


class BaseEventReader:
    def __init__(self, fp, extract_cam_geom=True, downsample_factor=1):
        self._fp = fp
        self._cam_w = None
        self._cam_h = None
        self._downsample_factor = downsample_factor

        # check file extension
        file_extension = os.path.splitext(self._fp)[1]
        assert file_extension in [".hdf5", ".h5"]

        logging.info(f"Reading HDF5 file: {self._fp} with downsample factor: {downsample_factor}")

        if downsample_factor > 1:
            self._events_df = self._load_downsampled_data()
        else:
            self._events_df = pd.read_hdf(self._fp, key="CD/events")

        self._num_events = len(self._events_df)
        self._min_time = self._events_df["t"].min() if len(self._events_df) > 0 else 0
        self._max_time = self._events_df["t"].max() if len(self._events_df) > 0 else 0
        logging.debug("done reading events")

        logging.info("Total events loaded: %d", self._num_events)
        logging.info(f"Event time range: {self._min_time} to {self._max_time} us")

        if extract_cam_geom:
            self._extract_camera_geometry()

    def _load_downsampled_data(self):
        """Load data with downsampling applied during loading."""
        logging.info(f"Loading with downsample factor {self._downsample_factor}")

        # Method 1: Use h5py to read every nth row directly
        with h5py.File(self._fp, "r") as f:
            dataset = f["CD/events"]
            total_events = dataset.shape[0]

            # Calculate indices for downsampling
            indices = np.arange(0, total_events, self._downsample_factor)
            downsampled_events = dataset[indices]

            # Convert to DataFrame
            events_df = pd.DataFrame(downsampled_events, columns=["x", "y", "p", "t"])

            logging.info(f"Downsampled from {total_events} to {len(events_df)} events")
            return events_df

    def _extract_camera_geometry(self):
        """Extract camera geometry from HDF5 file or estimate from data."""
        try:
            with h5py.File(self._fp, 'r') as f:
                # Try to get geometry from metadata first
                if 'geometry' in f.attrs:
                    self._cam_w = f.attrs['geometry'][0]
                    self._cam_h = f.attrs['geometry'][1]
                elif 'width' in f.attrs and 'height' in f.attrs:
                    self._cam_w = f.attrs['width']
                    self._cam_h = f.attrs['height']
                else:
                    # Fallback: estimate from data
                    dataset = f['CD/events']
                    sample_size = min(10000, dataset.shape[0])
                    sample_data = dataset[:sample_size]
                    self._cam_w = int(np.max(sample_data[:, 0])) + 1  # x column
                    self._cam_h = int(np.max(sample_data[:, 1])) + 1  # y column
        except Exception as e:
            logging.warning(f"Could not extract camera geometry: {e}")
            self._cam_w = 640  # default fallback
            self._cam_h = 480

    def get_geom_width(self):
        return self._cam_w

    def get_geom_height(self):
        return self._cam_h

    @property
    def df(self):
        """For ChunkedBaseEventReader, this should return current chunk or full data."""
        if hasattr(self, '_events_df'):
            return self._events_df
        else:
            return self._current_chunk if self._current_chunk is not None else pd.DataFrame()

class ChunkedBaseEventReader:
    """
    Event reader that loads HDF5 data in chunks with native downsampling support.
    """

    def __init__(self, fp, extract_cam_geom=True, chunk_size=100_000_000, downsample_factor=1):
        self._fp = fp
        self._cam_w = None
        self._cam_h = None
        self._chunk_size = chunk_size
        self._downsample_factor = downsample_factor
        self._current_chunk = None
        self._iterator_exhausted = False
        self._current_index = 0
        self._total_events = None
        self._downsampled_indices = None  # Pre-computed indices for downsampling
        self._events_df = None  # Will store complete dataset for compatibility

        # check file extension
        file_extension = os.path.splitext(self._fp)[1]
        assert file_extension in [".hdf5", ".h5"]

        logging.info(f"Reading HDF5 file: {self._fp} with downsample factor: {downsample_factor}")

        # Initialize h5py-based chunked reading
        self._init_chunk_reader()

        if extract_cam_geom:
            self._extract_camera_geometry()

        # Load complete dataset for compatibility with main script
        self._load_complete_dataset()

    def _extract_camera_geometry(self):
        """Extract camera geometry from HDF5 file or estimate from data."""
        try:
            with h5py.File(self._fp, 'r') as f:
                # Try to get geometry from metadata first
                if 'geometry' in f.attrs:
                    self._cam_w = f.attrs['geometry'][0]
                    self._cam_h = f.attrs['geometry'][1]
                    logging.info(f"Camera geometry from metadata: {self._cam_w}x{self._cam_h}")
                elif 'width' in f.attrs and 'height' in f.attrs:
                    self._cam_w = f.attrs['width']
                    self._cam_h = f.attrs['height']
                    logging.info(f"Camera geometry from attributes: {self._cam_w}x{self._cam_h}")
                else:
                    # Fallback: estimate from data sample
                    dataset = f['CD/events']
                    sample_size = min(10000, dataset.shape[0])
                    sample_data = dataset[:sample_size]
                    self._cam_w = int(np.max(sample_data[:, 0])) + 1  # x column
                    self._cam_h = int(np.max(sample_data[:, 1])) + 1  # y column
                    logging.info(f"Camera geometry estimated from data: {self._cam_w}x{self._cam_h}")
        except Exception as e:
            logging.warning(f"Could not extract camera geometry: {e}")
            self._cam_w = 640  # default fallback
            self._cam_h = 480
            logging.info(f"Using default camera geometry: {self._cam_w}x{self._cam_h}")

    def get_geom_width(self):
        """Get camera width."""
        return self._cam_w

    def get_geom_height(self):
        """Get camera height."""
        return self._cam_h

    def _init_chunk_reader(self):
        """Initialize h5py-based chunked reading with downsampling."""
        try:
            with h5py.File(self._fp, "r") as f:
                dataset = f["CD/events"]
                self._total_events = len(dataset)

                if self._downsample_factor > 1:
                    # Pre-compute downsampled indices
                    self._downsampled_indices = np.arange(0, self._total_events, self._downsample_factor)
                    effective_events = len(self._downsampled_indices)
                    logging.info(f"Pre-computed {effective_events} downsampled indices from {self._total_events} total events")
                else:
                    effective_events = self._total_events

                self._current_index = 0
                logging.info(f"Initialized h5py chunked reader. Effective events: {effective_events}")
        except Exception as e:
            logging.error(f"Failed to initialize h5py chunked reader: {e}")
            raise

    def _load_complete_dataset(self):
        """Load the complete downsampled dataset for compatibility."""
        logging.info("Loading complete downsampled dataset...")

        try:
            with h5py.File(self._fp, "r") as f:
                dataset = f["CD/events"]

                if self._downsample_factor > 1:
                    # Load downsampled data
                    downsampled_events = dataset[self._downsampled_indices]
                    self._events_df = pd.DataFrame(downsampled_events, columns=["x", "y", "p", "t"])
                    logging.info(f"Loaded {len(self._events_df)} downsampled events from {self._total_events} total")
                else:
                    # Load all data
                    all_events = dataset[:]
                    self._events_df = pd.DataFrame(all_events, columns=["x", "y", "p", "t"])
                    logging.info(f"Loaded {len(self._events_df)} events")

        except Exception as e:
            logging.error(f"Error loading complete dataset: {e}")
            self._events_df = pd.DataFrame(columns=["x", "y", "p", "t"])

    @property
    def df(self):
        """Return the complete dataset as DataFrame."""
        return self._events_df

    def get_time_windows(self, window_size_us, overlap_us=0):
        """
        Generator that yields time windows from the dataset.

        Args:
            window_size_us: Size of each time window in microseconds
            overlap_us: Overlap between consecutive windows in microseconds

        Yields:
            Tuple of (start_time, end_time, window_dataframe)
        """
        if len(self._events_df) == 0:
            logging.warning("No events loaded, cannot generate time windows")
            return

        min_time = self._events_df['t'].min()
        max_time = self._events_df['t'].max()

        step_size = window_size_us - overlap_us
        current_start = min_time

        logging.info(f"Generating time windows: {window_size_us}μs windows, {overlap_us}μs overlap")
        logging.info(f"Time range: {min_time} to {max_time} μs")

        while current_start < max_time:
            current_end = current_start + window_size_us

            # Extract events in current time window
            mask = (self._events_df['t'] >= current_start) & (self._events_df['t'] < current_end)
            window_data = self._events_df[mask].copy()

            if len(window_data) > 0:
                yield current_start, current_end, window_data

            current_start += step_size

    def _load_next_chunk(self):
        """Load the next chunk of events with downsampling applied."""
        if self._downsample_factor > 1:
            return self._load_next_chunk_downsampled()
        else:
            return self._load_next_chunk_normal()

    def _load_next_chunk_normal(self):
        """Load the next chunk without downsampling."""
        if self._current_index >= self._total_events:
            self._iterator_exhausted = True
            self._current_chunk = pd.DataFrame(columns=["x", "y", "p", "t"])
            return False

        try:
            with h5py.File(self._fp, "r") as f:
                dataset = f["CD/events"]
                start_idx = self._current_index
                end_idx = min(start_idx + self._chunk_size, self._total_events)

                # Read chunk data
                chunk_data = dataset[start_idx:end_idx]
                self._current_chunk = pd.DataFrame(chunk_data, columns=["x", "y", "p", "t"])
                self._current_index = end_idx

                logging.info(f"Loaded chunk with {len(self._current_chunk)} events (index {start_idx}:{end_idx})")
                return True

        except Exception as e:
            logging.error(f"Error loading chunk: {e}")
            self._iterator_exhausted = True
            self._current_chunk = pd.DataFrame(columns=["x", "y", "p", "t"])
            return False

    def _load_next_chunk_downsampled(self):
        """Load the next chunk with downsampling applied."""
        if self._current_index >= len(self._downsampled_indices):
            self._iterator_exhausted = True
            self._current_chunk = pd.DataFrame(columns=["x", "y", "p", "t"])
            return False

        try:
            with h5py.File(self._fp, "r") as f:
                dataset = f["CD/events"]

                # Calculate which downsampled indices to load in this chunk
                start_ds_idx = self._current_index
                end_ds_idx = min(start_ds_idx + (self._chunk_size // self._downsample_factor),
                                 len(self._downsampled_indices))

                # Get the actual file indices
                indices_to_load = self._downsampled_indices[start_ds_idx:end_ds_idx]

                # Load data at these specific indices
                chunk_data = dataset[indices_to_load]
                self._current_chunk = pd.DataFrame(chunk_data, columns=["x", "y", "p", "t"])

                self._current_index = end_ds_idx

                logging.info(f"Loaded downsampled chunk with {len(self._current_chunk)} events "
                             f"(downsampled indices {start_ds_idx}:{end_ds_idx})")
                return True

        except Exception as e:
            logging.error(f"Error loading downsampled chunk: {e}")
            self._iterator_exhausted = True
            self._current_chunk = pd.DataFrame(columns=["x", "y", "p", "t"])
            return False

    def has_more_chunks(self):
        """Check if there are more chunks to load."""
        if self._downsample_factor > 1:
            return self._current_index < len(self._downsampled_indices)
        else:
            return self._current_index < self._total_events

    def get_chunk_iterator(self):
        """
        Generator that yields data chunks.
        Useful for processing very large datasets without loading everything into memory.
        """
        self._current_index = 0
        self._iterator_exhausted = False

        while not self._iterator_exhausted and self.has_more_chunks():
            if self._load_next_chunk():
                yield self._current_chunk
            else:
                break


def choose_event_reader(filepath, size_threshold_mb=2000, downsample_factor=1, **kwargs):
    """
    Automatically choose between BaseEventReader and ChunkedBaseEventReader based on file size.
    Now supports native downsampling during loading.

    Args:
        filepath (str): Path to the HDF5 file
        size_threshold_mb (float): File size threshold in MB
        downsample_factor (int): Downsample factor to apply during loading
        **kwargs: Additional arguments to pass to the reader constructor

    Returns:
        BaseEventReader or ChunkedBaseEventReader: The appropriate reader instance
    """
    try:
        file_size_mb = get_hdf5_file_size(filepath, unit="MB")

        if file_size_mb > size_threshold_mb:
            logging.info(
                f"File size ({file_size_mb:.2f} MB) exceeds threshold ({size_threshold_mb} MB). "
                f"Using ChunkedBaseEventReader with downsample factor {downsample_factor}."
            )
            return ChunkedBaseEventReader(filepath, downsample_factor=downsample_factor, **kwargs)
        else:
            logging.info(
                f"File size ({file_size_mb:.2f} MB) is below threshold ({size_threshold_mb} MB). "
                f"Using BaseEventReader with downsample factor {downsample_factor}."
            )
            return BaseEventReader(filepath, downsample_factor=downsample_factor, **kwargs)

    except Exception as e:
        logging.warning(
            f"Could not determine file size for '{filepath}': {e}. "
            f"Defaulting to BaseEventReader with downsample factor {downsample_factor}."
        )
        return BaseEventReader(filepath, downsample_factor=downsample_factor, **kwargs)


# Alternative: Random sampling approach for better statistical properties
class RandomSamplingEventReader(BaseEventReader):
    """Event reader that uses random sampling instead of regular downsampling."""

    def __init__(self, fp, extract_cam_geom=True, sample_ratio=1.0, random_seed=None):
        self._sample_ratio = sample_ratio
        self._random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        # Calculate equivalent downsample factor for logging
        downsample_factor = int(1.0 / sample_ratio) if sample_ratio < 1.0 else 1

        super().__init__(fp, extract_cam_geom, downsample_factor=downsample_factor)

    def _load_downsampled_data(self):
        """Load data with random sampling applied during loading."""
        logging.info(f"Loading with random sampling ratio {self._sample_ratio}")

        with h5py.File(self._fp, "r") as f:
            dataset = f["CD/events"]
            total_events = dataset.shape[0]

            # Calculate number of events to sample
            n_sample = int(total_events * self._sample_ratio)

            # Generate random indices
            indices = np.random.choice(total_events, size=n_sample, replace=False)
            indices = np.sort(indices)  # Sort to maintain temporal order

            sampled_events = dataset[indices]
            events_df = pd.DataFrame(sampled_events, columns=["x", "y", "p", "t"])

            logging.info(f"Randomly sampled {len(events_df)} events from {total_events} total events")
            return events_df


# Time-based sampling for better temporal distribution
class TimeBasedSamplingEventReader(BaseEventReader):
    """Event reader that samples events within specified time windows."""

    def __init__(self, fp, extract_cam_geom=True, time_start=None, time_end=None,
                 max_events=None, time_stride=None):
        self._time_start = time_start
        self._time_end = time_end
        self._max_events = max_events
        self._time_stride = time_stride

        super().__init__(fp, extract_cam_geom, downsample_factor=1)

    def _load_downsampled_data(self):
        """Load data with time-based sampling."""
        logging.info(f"Loading with time-based sampling: start={self._time_start}, "
                     f"end={self._time_end}, max_events={self._max_events}")

        with h5py.File(self._fp, "r") as f:
            dataset = f["CD/events"]

            # Load all timestamps first (much smaller than full data)
            timestamps = dataset[:, 3]  # t column

            # Apply time filtering
            if self._time_start is not None and self._time_end is not None:
                time_mask = (timestamps >= self._time_start) & (timestamps < self._time_end)
                valid_indices = np.where(time_mask)[0]
            else:
                valid_indices = np.arange(len(timestamps))

            # Apply stride if specified
            if self._time_stride is not None and self._time_stride > 1:
                valid_indices = valid_indices[::self._time_stride]

            # Apply max events limit
            if self._max_events is not None and len(valid_indices) > self._max_events:
                # Sample uniformly across the time range
                step = len(valid_indices) // self._max_events
                valid_indices = valid_indices[::step][:self._max_events]

            # Load the actual data
            sampled_events = dataset[valid_indices]
            events_df = pd.DataFrame(sampled_events, columns=["x", "y", "p", "t"])

            logging.info(f"Time-based sampling: loaded {len(events_df)} events from {len(timestamps)} total events")
            return events_df