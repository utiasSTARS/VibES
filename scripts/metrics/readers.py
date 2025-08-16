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


def choose_event_reader(filepath, size_threshold_mb=2000, **kwargs):
    """
    Automatically choose between BaseEventReader and ChunkedBaseEventReader based on file size.

    Args:
        filepath (str): Path to the HDF5 file
        size_threshold_mb (float): File size threshold in MB. Files larger than this will use ChunkedBaseEventReader.
        **kwargs: Additional arguments to pass to the reader constructor

    Returns:
        BaseEventReader or ChunkedBaseEventReader: The appropriate reader instance
    """
    try:
        file_size_mb = get_hdf5_file_size(filepath, unit="MB")

        if file_size_mb > size_threshold_mb:
            logging.info(
                f"File size ({file_size_mb:.2f} MB) exceeds threshold ({size_threshold_mb} MB). Using ChunkedBaseEventReader."
            )
            return ChunkedBaseEventReader(filepath, **kwargs)
        else:
            logging.info(
                f"File size ({file_size_mb:.2f} MB) is below threshold ({size_threshold_mb} MB). Using BaseEventReader."
            )
            return BaseEventReader(filepath, **kwargs)

    except Exception as e:
        logging.warning(
            f"Could not determine file size for '{filepath}': {e}. Defaulting to BaseEventReader."
        )
        return BaseEventReader(filepath, **kwargs)


class ChunkedBaseEventReader:
    """
    Event reader that loads HDF5 data in chunks to handle large files efficiently.
    Provides the same interface as BaseEventReader but loads data incrementally.
    Uses h5py for direct chunked reading of HDF5 files.
    """

    def __init__(self, fp, extract_cam_geom=True, chunk_size=100_000_000):
        self._fp = fp
        self._cam_w = None
        self._cam_h = None
        self._chunk_size = chunk_size
        self._current_chunk = None
        self._iterator_exhausted = False
        self._current_index = 0
        self._total_events = None

        # check file extension
        file_extension = os.path.splitext(self._fp)[1]
        assert file_extension in [".hdf5", ".h5"]

        logging.info(f"Reading HDF5 file: {self._fp}")

        # Initialize h5py-based chunked reading
        self._init_chunk_reader()

        if extract_cam_geom:
            self._extract_camera_geometry()

        # Load first chunk
        self._load_next_chunk()

    def _init_chunk_reader(self):
        """Initialize h5py-based chunked reading."""
        try:
            with h5py.File(self._fp, "r") as f:
                dataset = f["CD/events"]
                self._total_events = len(dataset)
                self._current_index = 0
                logging.info(
                    f"Initialized h5py chunked reader. Total events: {self._total_events}"
                )
        except Exception as e:
            logging.error(f"Failed to initialize h5py chunked reader: {e}")
            raise

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError("Subclasses must implement __next__() method")

    def __del__(self):
        """Clean up resources"""
        pass

    def _load_next_chunk(self):
        """Load the next chunk of events using h5py."""
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

                # Convert to DataFrame
                self._current_chunk = pd.DataFrame(
                    chunk_data, columns=["x", "y", "p", "t"]
                )

                self._current_index = end_idx
                logging.info(
                    f"Loaded chunk with {len(self._current_chunk)} events (index {start_idx}:{end_idx})"
                )
                return True

        except Exception as e:
            logging.error(f"Error loading chunk: {e}")
            self._iterator_exhausted = True
            self._current_chunk = pd.DataFrame(columns=["x", "y", "p", "t"])
            return False

    def df(self):
        """Return the current chunk DataFrame."""
        return self._current_chunk

    def get_geom_width(self):
        return self._cam_w

    def get_geom_height(self):
        return self._cam_h

    def get_min_time(self):
        """Get minimum timestamp from current chunk."""
        if self._current_chunk is None or len(self._current_chunk) == 0:
            return None
        return self._current_chunk["t"].min()

    def get_max_time(self):
        """Get maximum timestamp from current chunk."""
        if self._current_chunk is None or len(self._current_chunk) == 0:
            return None
        return self._current_chunk["t"].max()

    def _parse_hdf5_geom_string(self, geom_str):
        """Parse geometry string from HDF5 attributes."""
        try:
            geom_parts = geom_str.split("x")
            if len(geom_parts) == 2:
                return int(geom_parts[0]), int(geom_parts[1])
            else:
                raise ValueError(f"Invalid geometry format: {geom_str}")
        except Exception as e:
            logging.error(f"Error parsing geometry string '{geom_str}': {e}")
            return None, None

    def _extract_camera_geometry(self):
        """Extract camera width and height from HDF5 attributes or infer from data."""
        logging.debug("Extracting camera geometry")

        try:
            # Use h5py temporarily just for geometry extraction
            with h5py.File(self._fp, "r") as hdf5_file:
                if "geometry" in hdf5_file.attrs:
                    self._cam_w, self._cam_h = self._parse_hdf5_geom_string(
                        hdf5_file.attrs["geometry"]
                    )
                    logging.debug(
                        f"Parsed camera geometry from hdf5 attributes: {self._cam_w}x{self._cam_h}"
                    )
                else:
                    # Try different attribute locations
                    if "width" in hdf5_file.attrs:
                        self._cam_w = hdf5_file.attrs["width"]
                    elif "CD" in hdf5_file and "width" in hdf5_file["CD"].attrs:
                        self._cam_w = hdf5_file["CD"].attrs["width"]
                    elif "sensor_size" in hdf5_file.attrs:
                        self._cam_w = hdf5_file.attrs["sensor_size"][0]
                    else:
                        # Infer from a sample of the data to avoid loading all events
                        hdf5_dataset = hdf5_file["CD/events"]
                        sample_size = min(100000, len(hdf5_dataset))
                        sample_data = hdf5_dataset[:sample_size]
                        self._cam_w = int(np.max(sample_data[:, 0])) + 1
                        logging.debug(
                            f"Inferred camera width from data sample: {self._cam_w}"
                        )

                    if "height" in hdf5_file.attrs:
                        self._cam_h = hdf5_file.attrs["height"]
                    elif "CD" in hdf5_file and "height" in hdf5_file["CD"].attrs:
                        self._cam_h = hdf5_file["CD"].attrs["height"]
                    elif "sensor_size" in hdf5_file.attrs:
                        self._cam_h = hdf5_file.attrs["sensor_size"][1]
                    else:
                        # Infer from a sample of the data
                        hdf5_dataset = hdf5_file["CD/events"]
                        sample_size = min(100000, len(hdf5_dataset))
                        sample_data = hdf5_dataset[:sample_size]
                        self._cam_h = int(np.max(sample_data[:, 1])) + 1
                        logging.debug(
                            f"Inferred camera height from data sample: {self._cam_h}"
                        )
        except Exception as e:
            logging.warning(f"Could not determine camera geometry ({e})")
            self._cam_w, self._cam_h = 640, 480

        logging.info(f"Using camera geometry: {self._cam_w}x{self._cam_h}")

    def _get_time_slice(self, start_time, end_time):
        """Get events within a specific time window from current chunk."""
        if self._current_chunk is None or len(self._current_chunk) == 0:
            return pd.DataFrame(columns=["x", "y", "p", "t"])

        mask = (self._current_chunk["t"] >= start_time) & (
            self._current_chunk["t"] < end_time
        )
        return self._current_chunk[mask]

    def reset(self):
        """Reset the iterator to the beginning."""
        self._iterator_exhausted = False
        self._current_chunk = None
        self._current_index = 0
        self._load_next_chunk()

    def has_more_chunks(self):
        """Check if there are more chunks to load."""
        return self._current_index < self._total_events

    def load_next_chunk(self):
        """Load the next chunk of events. Returns True if successful, False if no more chunks."""
        return self._load_next_chunk()

    def get_current_chunk_info(self):
        """Get information about the current chunk."""
        if self._current_chunk is None:
            return {
                "events": 0,
                "exhausted": self._iterator_exhausted,
                "has_more": self.has_more_chunks(),
            }

        chunk_events = len(self._current_chunk)
        start_idx = max(0, self._current_index - self._chunk_size)
        end_idx = self._current_index

        return {
            "events": chunk_events,
            "exhausted": self._iterator_exhausted,
            "has_more": self.has_more_chunks(),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "progress": (
                f"{end_idx}/{self._total_events} ({100*end_idx/self._total_events:.1f}%)"
                if self._total_events > 0
                else "0%"
            ),
        }

    def get_time_windows(
        self,
        window_size_us,
        overlap_us=0,
        default_event_start_time=10_000_000,
        default_event_end_time=20_000_000,
    ):
        """Generator that yields time windows of specified size across chunks."""
        # Use provided start and end times
        global_start_time = default_event_start_time
        global_end_time = default_event_end_time

        if global_start_time is None or global_end_time is None:
            logging.error(
                "Both default_event_start_time and default_event_end_time must be specified"
            )
            return

        step_size = window_size_us - overlap_us
        current_window_start = global_start_time

        print(f"Generating time windows from {global_start_time} to {global_end_time}")
        print(f"Window size: {window_size_us} us, overlap: {overlap_us} us")

        counter = 0

        while current_window_start < global_end_time:
            current_window_end = min(
                current_window_start + window_size_us, global_end_time
            )

            # Check if we need to load more chunks to cover this time window
            while (
                self._current_chunk is None
                or len(self._current_chunk) == 0
                or (
                    self._current_chunk["t"].max() < current_window_end
                    and self.has_more_chunks()
                )
            ):

                if not self.has_more_chunks():
                    logging.info("No more chunks available")
                    break

                success = self.load_next_chunk()
                if not success:
                    logging.info("Failed to load next chunk")
                    break

                logging.info(
                    f"Loaded chunk with time range: {self._current_chunk['t'].min()} to {self._current_chunk['t'].max()}"
                )

            # If we still don't have data, we've reached the end
            if self._current_chunk is None or len(self._current_chunk) == 0:
                logging.info("No more data available")
                break

            # If current chunk's max time is still less than window start,
            # we need to skip ahead or stop
            chunk_max_time = self._current_chunk["t"].max()
            if chunk_max_time < current_window_start:
                if self.has_more_chunks():
                    continue  # Try loading next chunk
                else:
                    break  # No more data

            # Get events for this time window
            window_data = self._get_time_slice(current_window_start, current_window_end)

            print(
                f"Yielding window {counter}: {current_window_start} to {current_window_end}, events: {len(window_data)}"
            )

            if len(window_data) > 0:  # Only yield non-empty windows
                yield current_window_start, current_window_end, window_data
                counter += 1

            current_window_start += step_size


class BaseEventReader:
    def __init__(self, fp, extract_cam_geom=True):
        self._fp = fp
        self._cam_w = None
        self._cam_h = None

        # check file extension
        file_extension = os.path.splitext(self._fp)[1]
        assert file_extension in [".hdf5", ".h5"]

        # Open HDF5 file to read events and possibly camera geometry
        logging.info(f"Reading HDF5 file: {self._fp}")
        self._events_df = pd.read_hdf(self._fp, key="CD/events")
        self._num_events = len(self._events_df)
        self._min_time = self._events_df["t"].min()
        self._max_time = self._events_df["t"].max()
        logging.debug("done reading events")

        logging.info("Total events loaded: %d", self._num_events)
        logging.info(f"Event time range: {self._min_time} to {self._max_time} us")

        if extract_cam_geom:
            self._extract_camera_geometry()

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError("Subclasses must implement __next__() method")

    def _parse_hdf5_geom_string(self, geom_str):
        """Parse geometry string from HDF5 attributes."""
        try:
            geom_parts = geom_str.split("x")
            if len(geom_parts) == 2:
                return int(geom_parts[0]), int(geom_parts[1])
            else:
                raise ValueError(f"Invalid geometry format: {geom_str}")
        except Exception as e:
            logging.error(f"Error parsing geometry string '{geom_str}': {e}")
            return None, None

    def _extract_camera_geometry(self):
        """Extract camera width and height from HDF5 attributes or infer from data."""

        logging.debug("Extracting camera geometry")
        with h5py.File(self._fp, "r") as f:
            df_dict = self._events_df.to_dict(orient="list")

            try:
                if "geometry" in f.attrs:
                    self._cam_w, self._cam_h = self._parse_hdf5_geom_string(
                        f.attrs["geometry"]
                    )
                    logging.debug(
                        f"Parsed camera geometry from hdf5 attributes: {self._cam_w}x{self._cam_h}"
                    )
                else:
                    # Try different attribute locations
                    if "width" in f.attrs:
                        self._cam_w = f.attrs["width"]
                    elif "CD" in f and "width" in f["CD"].attrs:
                        self._cam_w = f["CD"].attrs["width"]
                    elif "sensor_size" in f.attrs:
                        self._cam_w = f.attrs["sensor_size"][0]
                    else:
                        self._cam_w = int(np.max(df_dict["x"])) + 1
                        logging.debug(f"Inferred camera width from data: {self._cam_w}")

                    if "height" in f.attrs:
                        self._cam_h = f.attrs["height"]
                    elif "CD" in f and "height" in f["CD"].attrs:
                        self._cam_h = f["CD"].attrs["height"]
                    elif "sensor_size" in f.attrs:
                        self._cam_h = f.attrs["sensor_size"][1]
                    else:
                        self._cam_h = int(np.max(df_dict["y"])) + 1
                        logging.debug(
                            f"Inferred camera height from data: {self._cam_h}"
                        )
            except Exception as e:
                logging.warning(f"Could not determine camera geometry ({e})")
                self._cam_w, self._cam_h = 640, 480

        logging.info(f"Using camera geometry: {self._cam_w}x{self._cam_h}")

    def _get_time_slice(self, start_time, end_time):
        """Get events within a specific time window."""
        mask = (self._events_df["t"] >= start_time) & (self._events_df["t"] < end_time)
        return self._events_df[mask]

    def df(self):
        """Return the events DataFrame."""
        return self._events_df

    def get_geom_width(self):
        return self._cam_w

    def get_geom_height(self):
        return self._cam_h

    def get_min_time(self):
        return self._min_time

    def get_max_time(self):
        return self._max_time

    def get_time_windows(
        self,
        window_size_us,
        overlap_us=0,
        default_event_start_time=10_000_000,
        default_event_end_time=20_000_000,
    ):
        """Generator that yields time windows of specified size."""
        start_time = (
            self._min_time
            if default_event_start_time is None
            else default_event_start_time
        )
        end_time = (
            self._max_time if default_event_end_time is None else default_event_end_time
        )

        # Start from 10 seconds into the data
        current_start = start_time
        if current_start > self._max_time:
            current_start = self._min_time
            logging.info("Starting from the beginning of the data")
        if end_time > self._max_time:
            end_time = self._max_time
            logging.info(
                f"Generating time windows from {current_start} to {end_time} with window size {window_size_us} us"
            )
        step_size = window_size_us - overlap_us

        print(f"Generating time windows from {current_start} to {end_time}")
        print(f"Window size: {window_size_us} us, overlap: {overlap_us} us")

        counter = 0
        while current_start < end_time:  # and counter < max_windows:
            current_end = min(current_start + window_size_us, end_time)
            window_data = self._get_time_slice(current_start, current_end)

            print(
                f"Yielding window {counter}: {current_start} to {current_end}, events: {len(window_data)}"
            )
            if len(window_data) > 0:  # Only yield non-empty windows
                yield current_start, current_end, window_data
                counter += 1

            current_start += step_size


class FixedSizeEventReader(BaseEventReader):
    def __init__(self, file_path, window_size, start_index=0):
        super().__init__(file_path)
        self._window_size = window_size
        self._current_index = start_index
        self._total_events = len(self._events_df)

    def __iter__(self):
        return self

    def __next__(self):
        """Yield fixed-size event windows from the events DataFrame."""
        if self._events_df.empty:
            raise StopIteration

        # Iterate over the DataFrame in chunks of _window_size
        start = self._current_index
        end = min(self._current_index + self._window_size, self._total_events)
        window_df = self._events_df.iloc[start:end]

        if window_df.empty:
            raise StopIteration

        self._current_index += self._window_size
        return window_df.values


class TimeWindowEventReader(BaseEventReader):
    def __init__(self, file_path, delta_t, start_time=0):
        super().__init__(file_path)
        self._delta_t = delta_t  # Time window size in microseconds
        self._current_time = start_time

    def __iter__(self):
        return self

    def __next__(self):
        """Yield events within a time window defined by delta_t."""
        if self._events_df.empty:
            raise StopIteration

        # Filter events within the current time window
        window_df = self._get_time_slice(
            self._current_time, self._current_time + self._delta_t
        )

        if window_df.empty:
            raise StopIteration

        # Update current time to the next window start
        self._current_time += self._delta_t
        return window_df.values


class FrameReader:
    def __init__(self, fp):
        self._fp = fp
        self._width = None
        self._height = None
        self._image_files = []
        self._get_geometry()  # Initialize geometry from first image

    def _get_geometry(self):
        """Get image dimensions from the first image in the directory."""
        if not os.path.exists(self._fp):
            raise ValueError(f"Directory does not exist: {self._fp}")

        # Get list of image files (common image extensions)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        self._image_files = []

        for file_path in Path(self._fp).iterdir():
            if file_path.suffix.lower() in image_extensions:
                self._image_files.append(file_path)

        if not self._image_files:
            raise ValueError(f"No image files found in directory: {self._fp}")

        # Sort files by name to ensure consistent ordering
        self._image_files.sort()

        # Read the first image to get dimensions (as grayscale)
        first_image = cv2.imread(str(self._image_files[0]), cv2.IMREAD_GRAYSCALE)
        if first_image is None:
            raise ValueError(f"Could not read image: {self._image_files[0]}")

        self._height, self._width = first_image.shape
        print(f"Image geometry: {self._width}x{self._height}")
        print(f"Total images found: {len(self._image_files)}")

    def get_geom_width(self):
        return self._width

    def get_geom_height(self):
        return self._height

    def get_frame_count(self):
        return len(self._image_files)

    def __iter__(self):
        """Iterator to yield frames one by one without loading all into memory."""
        for img_path in self._image_files:
            # Load image using OpenCV as grayscale
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Warning: Could not load image {img_path}, skipping...")
                continue

            yield img

    def load(self):
        """Load all images from the directory as numpy arrays (for backward compatibility)."""
        print("Loading frames from directory...")
        frames = []

        for frame in tqdm(self, desc="Loading frames", total=len(self._image_files)):
            frames.append(frame)

        print(f"Total frames loaded: {len(frames)}")

        if len(frames) == 0:
            raise ValueError("No frames were successfully loaded.")

        return np.array(frames)


def test_frame_reader():
    """Test the FrameReader class functionality."""
    import tempfile
    import shutil
    from PIL import Image

    # Create a temporary directory for test images
    test_dir = tempfile.mkdtemp()

    try:
        print("\n" + "=" * 50)
        print("Testing FrameReader class...")
        print("=" * 50)

        # Create test images with different content
        test_images = []
        image_width, image_height = 320, 240

        for i in range(5):
            # Create test image with different patterns
            img_array = np.zeros((image_height, image_width, 3), dtype=np.uint8)

            # Create different patterns for each image
            if i == 0:
                # Gradient pattern
                for y in range(image_height):
                    for x in range(image_width):
                        img_array[y, x] = [x % 256, y % 256, (x + y) % 256]
            elif i == 1:
                # Checkerboard pattern
                for y in range(image_height):
                    for x in range(image_width):
                        if (x // 20 + y // 20) % 2 == 0:
                            img_array[y, x] = [255, 255, 255]
                        else:
                            img_array[y, x] = [0, 0, 0]
            elif i == 2:
                # Red channel dominant
                img_array[:, :, 0] = 200  # Red
                img_array[:, :, 1] = 50  # Green
                img_array[:, :, 2] = 50  # Blue
            elif i == 3:
                # Green channel dominant
                img_array[:, :, 0] = 50  # Red
                img_array[:, :, 1] = 200  # Green
                img_array[:, :, 2] = 50  # Blue
            else:
                # Blue channel dominant
                img_array[:, :, 0] = 50  # Red
                img_array[:, :, 1] = 50  # Green
                img_array[:, :, 2] = 200  # Blue

            # Save as PNG (to test different formats)
            img_path = os.path.join(test_dir, f"frame_{i:03d}.png")
            img_pil = Image.fromarray(img_array, "RGB")
            img_pil.save(img_path)
            test_images.append(img_path)

        print(f"Created {len(test_images)} test images in {test_dir}")

        # Test FrameReader initialization
        print("\n1. Testing FrameReader initialization...")
        reader = FrameReader(test_dir)

        # Test API methods
        print("\n2. Testing API methods...")
        width = reader.get_geom_width()
        height = reader.get_geom_height()
        frame_count = reader.get_frame_count()

        print(f"   - Geometry: {width}x{height}")
        print(f"   - Frame count: {frame_count}")

        # Verify geometry
        assert (
            width == image_width
        ), f"Width mismatch: expected {image_width}, got {width}"
        assert (
            height == image_height
        ), f"Height mismatch: expected {image_height}, got {height}"
        assert frame_count == 5, f"Frame count mismatch: expected 5, got {frame_count}"

        # Test iterator functionality
        print("\n3. Testing iterator functionality...")
        frame_iter_count = 0
        frames_for_comparison = []

        for i, frame in enumerate(reader):
            frame_iter_count += 1

            # Verify frame properties
            assert frame is not None, f"Frame {i} is None"
            assert isinstance(frame, np.ndarray), f"Frame {i} is not numpy array"
            assert frame.shape == (
                image_height,
                image_width,
            ), f"Frame {i} shape mismatch: {frame.shape}"
            assert frame.dtype == np.uint8, f"Frame {i} dtype mismatch: {frame.dtype}"

            # Verify frame is not empty (contains non-zero values)
            non_zero_count = np.count_nonzero(frame)
            assert non_zero_count > 0, f"Frame {i} contains only zeros"

            # Check frame content varies (not all frames identical)
            if i > 0:
                prev_frame = frames_for_comparison[i - 1]
                diff = np.sum(
                    np.abs(frame.astype(np.int16) - prev_frame.astype(np.int16))
                )
                assert diff > 0, f"Frame {i} is identical to previous frame"

            print(
                f"   - Frame {i}: shape={frame.shape}, dtype={frame.dtype}, non_zero_pixels={non_zero_count}"
            )

            # Store frames for comparison
            frames_for_comparison.append(frame.copy())

            # Test a few frames in detail
            if i < 3:
                min_val, max_val = frame.min(), frame.max()
                mean_val = frame.mean()
                print(f"     Stats: min={min_val}, max={max_val}, mean={mean_val:.1f}")

        print(f"   - Iterated through {frame_iter_count} frames")
        assert (
            frame_iter_count == frame_count
        ), f"Iterator count mismatch: expected {frame_count}, got {frame_iter_count}"

        # Test load() method
        print("\n4. Testing load() method...")
        frames_array = reader.load()

        assert isinstance(frames_array, np.ndarray), "load() should return numpy array"
        assert frames_array.shape == (
            frame_count,
            image_height,
            image_width,
        ), f"Loaded array shape mismatch: {frames_array.shape}"
        assert (
            frames_array.dtype == np.uint8
        ), f"Loaded array dtype mismatch: {frames_array.dtype}"

        # Verify all frames are loaded correctly
        for i in range(frame_count):
            frame = frames_array[i]
            non_zero_count = np.count_nonzero(frame)
            assert non_zero_count > 0, f"Loaded frame {i} contains only zeros"

        print(f"   - Successfully loaded {frames_array.shape[0]} frames")
        print(f"   - Array shape: {frames_array.shape}")

        # Test error handling
        print("\n5. Testing error handling...")

        # Test with non-existent directory
        try:
            FrameReader("/non/existent/directory")
            assert False, "Should have raised ValueError for non-existent directory"
        except ValueError as e:
            print(f"   - Correctly handled non-existent directory: {e}")

        # Test with empty directory
        empty_dir = tempfile.mkdtemp()
        try:
            FrameReader(empty_dir)
            assert False, "Should have raised ValueError for empty directory"
        except ValueError as e:
            print(f"   - Correctly handled empty directory: {e}")
        finally:
            shutil.rmtree(empty_dir)

        print("\n" + "=" * 50)
        print("✅ FrameReader test completed successfully!")
        print("✅ All API methods work correctly")
        print("✅ Images are loaded properly as grayscale")
        print("✅ Error handling works as expected")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ FrameReader test failed: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # Clean up test directory
        shutil.rmtree(test_dir)
        print(f"Cleaned up test directory: {test_dir}")


def create_test_hdf5(filepath, num_events=100000, time_span=30_000_000):
    """Create a test HDF5 file with event data spanning time_span microseconds."""
    # Generate synthetic event data with realistic timestamps
    x = np.random.randint(0, 640, num_events)
    y = np.random.randint(0, 480, num_events)
    p = np.random.randint(0, 2, num_events)

    # Create timestamps spanning time_span microseconds
    t = np.sort(np.random.randint(0, time_span, num_events))

    # Create DataFrame for return value
    events_df = pd.DataFrame({"x": x, "y": y, "p": p, "t": t})

    # Save to HDF5 in the format expected by ChunkedBaseEventReader
    with h5py.File(filepath, "w") as f:
        # Create the CD group and events dataset
        cd_group = f.create_group("CD")

        # Stack the event data as expected (x, y, p, t columns)
        event_data = np.column_stack((x, y, p, t))
        cd_group.create_dataset("events", data=event_data, dtype=np.int64)

        # Add metadata
        f.attrs["width"] = 640
        f.attrs["height"] = 480
        f.attrs["total_events"] = num_events

    return events_df


def test_cross_chunk_windowing():
    """Test the ChunkedBaseEventReader cross-chunk windowing functionality."""
    test_file = "/tmp/test_cross_chunk_events.hdf5"

    try:
        print("Creating test HDF5 file...")
        original_events = create_test_hdf5(
            test_file, num_events=50000, time_span=30_000_000
        )
        print(f"Created test file with {len(original_events)} events")
        print(
            f"Time range: {original_events['t'].min()} to {original_events['t'].max()} us"
        )

        print("\nTesting ChunkedBaseEventReader with small chunks...")
        # Use small chunk size to force cross-chunk behavior
        reader = ChunkedBaseEventReader(test_file, chunk_size=10000)

        print(f"Camera geometry: {reader.get_geom_width()}x{reader.get_geom_height()}")

        # Test windowing across chunks
        start_time = 5_000_000  # 5 seconds
        end_time = 15_000_000  # 15 seconds
        window_size = 2_000_000  # 2 seconds
        overlap = 500_000  # 0.5 seconds

        print(f"\nTesting time windows from {start_time} to {end_time}")
        print(f"Window size: {window_size} us, overlap: {overlap} us")

        window_count = 0
        total_events_in_windows = 0

        for window_start, window_end, window_data in reader.get_time_windows(
            window_size_us=window_size,
            overlap_us=overlap,
            default_event_start_time=start_time,
            default_event_end_time=end_time,
        ):
            window_count += 1
            events_in_window = len(window_data)
            total_events_in_windows += events_in_window

            print(
                f"Window {window_count}: {window_start} to {window_end}, events: {events_in_window}"
            )

            # Verify window data is within expected time range
            if events_in_window > 0:
                min_t = window_data["t"].min()
                max_t = window_data["t"].max()
                assert (
                    min_t >= window_start
                ), f"Event time {min_t} is before window start {window_start}"
                assert (
                    max_t < window_end
                ), f"Event time {max_t} is after window end {window_end}"

            # Limit output for readability
            if window_count >= 10:
                print("... (stopping after 10 windows for readability)")
                break

        print(f"\nTotal windows generated: {window_count}")
        print(f"Total events in windows: {total_events_in_windows}")

        # Verify we got some windows
        assert window_count > 0, "No windows were generated"

        # Test reset functionality
        print("\nTesting reset...")
        reader.reset()
        first_window_after_reset = next(
            reader.get_time_windows(
                window_size_us=window_size,
                overlap_us=overlap,
                default_event_start_time=start_time,
                default_event_end_time=start_time + window_size,
            ),
            None,
        )

        if first_window_after_reset:
            print(
                f"First window after reset: {first_window_after_reset[0]} to {first_window_after_reset[1]}, events: {len(first_window_after_reset[2])}"
            )

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    ex_metavision_hdf5 = (
        "/mnt/d/Datasets/harmeda/harmeda_dataset/pattern/pattern_ev.hdf5"
    )
    ex_amiev_hdf5 = "/mnt/d/Datasets/harmeda/harmeda_dataset/AMI/amiev.hdf5"

    # Example usage
    reader1 = TimeWindowEventReader(ex_metavision_hdf5, delta_t=10000)

    print(f"Camera 1 geometry: {reader1.get_geom_width()}x{reader1.get_geom_height()}")
    assert reader1.get_geom_width() == 1280
    assert reader1.get_geom_height() == 720

    for events in reader1:
        print(f"Loaded {len(events)} events in current time window.")
        assert len(events) > 0, "No events loaded in time window"
        assert events[-1, 3] - events[0, 3] <= 10000, "Events not within time window"
        break

    reader2 = TimeWindowEventReader(ex_amiev_hdf5, delta_t=10000)

    print(f"Camera 2 geometry: {reader2.get_geom_width()}x{reader2.get_geom_height()}")
    assert reader2.get_geom_width() == 640
    assert reader2.get_geom_height() == 480

    for events in reader2:
        print(f"Loaded {len(events)} events in current time window.")
        assert len(events) > 0, "No events loaded in time window"
        assert events[-1, 3] - events[0, 3] <= 10000, "Events not within time window"
        break

    reader1 = FixedSizeEventReader(ex_metavision_hdf5, window_size=1000)

    for events in reader1:
        print(f"Loaded {len(events)} events in fixed-size window.")
        assert len(events) == 1000
        break

    reader2 = FixedSizeEventReader(ex_amiev_hdf5, window_size=1000)

    for events in reader2:
        print(f"Loaded {len(events)} events in fixed-size window.")
        assert len(events) == 1000
        break

    test_cross_chunk_windowing()

    # Test FrameReader
    test_frame_reader()
