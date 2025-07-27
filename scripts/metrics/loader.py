import numpy as np
import os
import cv2
from pathlib import Path

from metavision_core.event_io import EventsIterator
from tqdm import tqdm


class EventLoader:
    def __init__(self, fp, delta_t=10000):
        self._fp = fp
        self._delta_t = delta_t
        self._ev_iterator = EventsIterator(input_path=self._fp, delta_t=self._delta_t)
        self._width, self._height = self._ev_iterator.get_size()  # Camera Geometry
        self.events = np.empty((0, 4), dtype=np.float32)  # Initialize empty array

    def get_geom_width(self):
        return self._width

    def get_geom_height(self):
        return self._height

    def get_events_iterator(self):
        return self._ev_iterator

    def load(self):
        print("Reading events from iterator...")
        event_list = []  # Collect all events in a list first

        for ev_slice in tqdm(self._ev_iterator, desc="Loading event slices"):
            # ev_slice is a dictionary with keys like 'x', 'y', 'p', 't'
            # Extract the arrays for each component
            x = ev_slice["x"]
            y = ev_slice["y"]
            p = ev_slice["p"]
            t = ev_slice["t"]

            # Stack them into an Nx4 array for this slice
            slice_events = np.column_stack((x, y, p, t))
            event_list.append(slice_events)

        # Concatenate all slices into one big Nx4 array
        if event_list:
            self.events = np.vstack(event_list).astype(np.float32)
        else:
            self.events = np.empty((0, 4), dtype=np.float32)

        print(f"Total events read: {len(self.events)}")

        if len(self.events) == 0:
            raise ValueError("No events read from the iterator.")
        return self.events


class FrameLoader:
    def __init__(self, fp):
        self._fp = fp
        self._frames = []
        self._width = None
        self._height = None
        self._get_geometry()  # Initialize geometry from first image

    def _get_geometry(self):
        """Get image dimensions from the first image in the directory."""
        if not os.path.exists(self._fp):
            raise ValueError(f"Directory does not exist: {self._fp}")

        # Get list of image files (common image extensions)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = []

        for file_path in Path(self._fp).iterdir():
            if file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)

        if not image_files:
            raise ValueError(f"No image files found in directory: {self._fp}")

        # Read the first image to get dimensions (as grayscale)
        first_image = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
        if first_image is None:
            raise ValueError(f"Could not read image: {image_files[0]}")

        self._height, self._width = first_image.shape
        print(f"Image geometry: {self._width}x{self._height}")

    def get_geom_width(self):
        return self._width

    def get_geom_height(self):
        return self._height

    def load(self):
        """Load all images from the directory as numpy arrays."""
        print("Loading frames from directory...")

        if not os.path.exists(self._fp):
            raise ValueError(f"Directory does not exist: {self._fp}")

        # Get list of image files and sort them
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = []

        for file_path in Path(self._fp).iterdir():
            if file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)

        # Sort files by name to ensure consistent ordering
        image_files.sort()

        if not image_files:
            raise ValueError(f"No image files found in directory: {self._fp}")

        self._frames = []

        for img_path in tqdm(image_files, desc="Loading frames"):
            # Load image using OpenCV as grayscale
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Warning: Could not load image {img_path}, skipping...")
                continue

            # Image is already grayscale, no conversion needed
            self._frames.append(img)

        print(f"Total frames loaded: {len(self._frames)}")

        if len(self._frames) == 0:
            raise ValueError("No frames were successfully loaded.")

        return np.array(self._frames)
