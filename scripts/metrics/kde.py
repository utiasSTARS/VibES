import numpy as np

import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm

from metavision_core.event_io import EventsIterator

from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

from niqe.niqe import niqe


class KDEMetrics:
    def __init__(self, cam_w, cam_h, events_iterator):
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.events_iterator = events_iterator

        self.events = np.empty((0, 4), dtype=np.float32)  # Initialize empty array
        self.get_events()  # Read events from the iterator

    def get_events(self):  # Read events from the iterator
        print("Reading events from iterator...")
        event_list = []  # Collect all events in a list first

        for ev_slice in tqdm(self.events_iterator, desc="Loading event slices"):
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

    def point_distribution(self):
        # Compute point distribution metric

        pts = self.events[:, 1:3].astype(float)  # Extract x, y coordinates
        pts[:, 0] /= self.cam_w  # Normalize x coordinates by camera width
        pts[:, 1] /= self.cam_h  # Normalize y coordinates by camera height

        # Use sklearn's KernelDensity for 2D density estimation
        kde2d = KernelDensity(
            kernel="gaussian", bandwidth=0.05
        )  # tune bandwidth as needed
        kde2d.fit(pts)

        # Evaluate KDE at each event location (returns log densities)
        log_densities = kde2d.score_samples(pts)
        densities = np.exp(log_densities)  # Convert to actual densities

        print("Variance of KDE densities:", np.var(densities))

        plt.figure()
        plt.hist(densities, bins=80, density=True)
        plt.xlabel("KDE density at event locations")
        plt.ylabel("Probability density")
        plt.title("Distribution of event densities (1-D)")
        plt.tight_layout()
        plt.show()

        pass

    def niqe(image, avg_window=None, extend_mode="reflect", C=1.0):

        pass


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 event data.")
    parser.add_argument("file_path", type=str, help="Path to the HDF5 file")
    parser.add_argument(
        "--time_window_us",
        "-t",
        type=float,
        default=-1.0,
        help="Time window in microseconds",
    )
    parser.add_argument(
        "--delta_t",
        type=int,
        default=10000,
        help="Delta time for reading events (default: 1000 us)",
    )
    args = parser.parse_args()

    ev_iterator = EventsIterator(input_path=args.file_path, delta_t=args.delta_t)
    width, height = ev_iterator.get_size()  # Camera Geometry

    print(f"Processing file: {args.file_path}")
    print(f"Camera Geometry: {height}x{width}")

    m = KDEMetrics(cam_w=width, cam_h=height, events_iterator=ev_iterator)

    m.point_distribution()
