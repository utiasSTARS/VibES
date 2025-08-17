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
                    # Fallback: estimate from data
                    if len(self._events_df) > 0:
                        self._cam_w = int(self._events_df['x'].max()) + 1
                        self._cam_h = int(self._events_df['y'].max()) + 1
                        logging.info(f"Camera geometry estimated from data: {self._cam_w}x{self._cam_h}")
                    else:
                        # No data available, use defaults
                        self._cam_w = 640
                        self._cam_h = 480
                        logging.info(f"No data available, using default geometry: {self._cam_w}x{self._cam_h}")
        except Exception as e:
            logging.warning(f"Could not extract camera geometry: {e}")
            # Estimate from loaded data if available
            if len(self._events_df) > 0:
                self._cam_w = int(self._events_df['x'].max()) + 1
                self._cam_h = int(self._events_df['y'].max()) + 1
                logging.info(f"Camera geometry estimated from loaded data: {self._cam_w}x{self._cam_h}")
            else:
                self._cam_w = 640  # default fallback
                self._cam_h = 480
                logging.info(f"Using default camera geometry: {self._cam_w}x{self._cam_h}")

    def get_geom_width(self):
        """Get camera width."""
        return self._cam_w

    def get_geom_height(self):
        """Get camera height."""
        return self._cam_h

    @property
    def df(self):
        """Return the events dataframe."""
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

        min_time = self._min_time
        max_time = self._max_time

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


class OptimizedChunkedBaseEventReader:
    """
    Optimized event reader that loads HDF5 data in chunks with native downsampling support.
    """

    def __init__(self, fp, extract_cam_geom=True, chunk_size=100_000_000, downsample_factor=1):
        self._fp = fp
        self._cam_w = None
        self._cam_h = None
        self._chunk_size = chunk_size
        self._downsample_factor = downsample_factor
        self._total_events = None
        self._events_df = None

        # Keep file handle open for better performance
        self._h5_file = None
        self._dataset = None

        # check file extension
        file_extension = os.path.splitext(self._fp)[1]
        assert file_extension in [".hdf5", ".h5"]

        logging.info(f"Reading HDF5 file: {self._fp} with downsample factor: {downsample_factor}")

        # Initialize and keep file open
        self._init_file_handle()

        if extract_cam_geom:
            self._extract_camera_geometry()

        # Load complete dataset efficiently
        self._load_complete_dataset_optimized()

    def __del__(self):
        """Cleanup: close file handle when object is destroyed."""
        self._close_file_handle()

    def _init_file_handle(self):
        """Initialize and keep HDF5 file handle open for better performance."""
        try:
            self._h5_file = h5py.File(self._fp, "r")
            self._dataset = self._h5_file["CD/events"]
            self._total_events = self._dataset.shape[0]
            logging.info(f"Opened HDF5 file with {self._total_events} total events")
        except Exception as e:
            logging.error(f"Failed to open HDF5 file: {e}")
            self._close_file_handle()
            raise

    def _close_file_handle(self):
        """Close HDF5 file handle."""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
                self._h5_file = None
                self._dataset = None
            except Exception as e:
                logging.warning(f"Error closing HDF5 file: {e}")

    def _extract_camera_geometry(self):
        """Extract camera geometry from HDF5 file or estimate from data."""
        try:
            # Try to get geometry from metadata first
            if 'geometry' in self._h5_file.attrs:
                self._cam_w = self._h5_file.attrs['geometry'][0]
                self._cam_h = self._h5_file.attrs['geometry'][1]
                logging.info(f"Camera geometry from metadata: {self._cam_w}x{self._cam_h}")
            elif 'width' in self._h5_file.attrs and 'height' in self._h5_file.attrs:
                self._cam_w = self._h5_file.attrs['width']
                self._cam_h = self._h5_file.attrs['height']
                logging.info(f"Camera geometry from attributes: {self._cam_w}x{self._cam_h}")
            else:
                # Fallback: estimate from data sample (use smaller sample for speed)
                sample_size = min(1000, self._total_events)  # Reduced sample size
                sample_data = self._dataset[:sample_size]
                self._cam_w = int(np.max(sample_data[:, 0])) + 1  # x column
                self._cam_h = int(np.max(sample_data[:, 1])) + 1  # y column
                logging.info(f"Camera geometry estimated from {sample_size} samples: {self._cam_w}x{self._cam_h}")
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

    def _load_complete_dataset_optimized(self):
        """Load the complete downsampled dataset with optimizations."""
        logging.info("Loading complete downsampled dataset with optimizations...")

        try:
            if self._downsample_factor > 1:
                # Method 1: Use numpy slicing directly on HDF5 dataset (fastest)
                if self._downsample_factor <= 100:  # For reasonable downsample factors
                    logging.info("Using direct HDF5 slicing for downsampling")
                    downsampled_events = self._dataset[::self._downsample_factor]
                else:
                    # Method 2: Generate indices and use fancy indexing for very high downsample
                    logging.info("Using index-based downsampling for high downsample factor")
                    indices = np.arange(0, self._total_events, self._downsample_factor)
                    # Read in chunks to avoid memory issues
                    chunk_size = 1_000_000
                    chunks = []
                    for i in range(0, len(indices), chunk_size):
                        chunk_indices = indices[i:i+chunk_size]
                        chunks.append(self._dataset[chunk_indices])
                    downsampled_events = np.vstack(chunks)

                # Create DataFrame with explicit dtype specification for speed
                self._events_df = pd.DataFrame(
                    downsampled_events,
                    columns=["x", "y", "p", "t"],
                    dtype={'x': np.uint16, 'y': np.uint16, 'p': np.uint8, 't': np.uint64}
                )
                logging.info(f"Loaded {len(self._events_df)} downsampled events from {self._total_events} total")
            else:
                # Load all data efficiently
                logging.info("Loading complete dataset")
                all_events = self._dataset[:]
                self._events_df = pd.DataFrame(
                    all_events,
                    columns=["x", "y", "p", "t"],
                    dtype={'x': np.uint16, 'y': np.uint16, 'p': np.uint8, 't': np.uint64}
                )
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
        Optimized generator that yields time windows from the dataset.

        Args:
            window_size_us: Size of each time window in microseconds
            overlap_us: Overlap between consecutive windows in microseconds

        Yields:
            Tuple of (start_time, end_time, window_dataframe)
        """
        if len(self._events_df) == 0:
            logging.warning("No events loaded, cannot generate time windows")
            return

        # Use pre-computed min/max for efficiency
        min_time = self._events_df['t'].iloc[0]  # Assuming sorted data
        max_time = self._events_df['t'].iloc[-1]

        step_size = window_size_us - overlap_us
        current_start = min_time

        logging.info(f"Generating time windows: {window_size_us}μs windows, {overlap_us}μs overlap")
        logging.info(f"Time range: {min_time} to {max_time} μs")

        # Pre-compute time values as numpy array for faster comparisons
        time_values = self._events_df['t'].values

        while current_start < max_time:
            current_end = current_start + window_size_us

            # Use numpy operations for faster masking
            mask = (time_values >= current_start) & (time_values < current_end)

            if np.any(mask):
                # Use iloc with boolean indexing for speed
                window_data = self._events_df[mask].copy()
                yield current_start, current_end, window_data

            current_start += step_size

    def get_streaming_time_windows(self, window_size_us, overlap_us=0, max_memory_mb=1000):
        """
        Memory-efficient streaming version that doesn't load full dataset.
        Useful for very large files where even downsampled data is too big.
        """
        step_size = window_size_us - overlap_us

        # Get time range without loading all data
        first_event = self._dataset[0]
        last_event = self._dataset[-1]
        min_time = first_event[3]  # t column
        max_time = last_event[3]

        logging.info(f"Streaming time windows: {window_size_us}μs windows, {overlap_us}μs overlap")
        logging.info(f"Time range: {min_time} to {max_time} μs")

        # Estimate events per time unit for chunk sizing
        total_duration = max_time - min_time
        events_per_us = self._total_events / total_duration if total_duration > 0 else 1
        estimated_events_per_window = int(events_per_us * window_size_us * 1.5)  # 50% buffer

        current_start = min_time
        current_pos = 0

        while current_start < max_time:
            current_end = current_start + window_size_us

            # Find events in current time window
            window_events = []
            search_pos = current_pos

            # Binary search for start position (approximate)
            while search_pos < self._total_events:
                chunk_size = min(100000, self._total_events - search_pos)
                chunk = self._dataset[search_pos:search_pos + chunk_size]

                # Find events in time window
                time_mask = (chunk[:, 3] >= current_start) & (chunk[:, 3] < current_end)
                matching_events = chunk[time_mask]

                if len(matching_events) > 0:
                    window_events.append(matching_events)

                # Stop if we've passed the window
                if chunk[-1, 3] > current_end:
                    break

                search_pos += chunk_size

            if window_events:
                all_window_events = np.vstack(window_events)
                if self._downsample_factor > 1:
                    all_window_events = all_window_events[::self._downsample_factor]

                window_df = pd.DataFrame(
                    all_window_events,
                    columns=["x", "y", "p", "t"],
                    dtype={'x': np.uint16, 'y': np.uint16, 'p': np.uint8, 't': np.uint64}
                )

                yield current_start, current_end, window_df

            current_start += step_size

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

