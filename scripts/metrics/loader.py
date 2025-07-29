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


