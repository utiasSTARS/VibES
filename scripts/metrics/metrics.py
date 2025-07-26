import numpy as np

import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm

from metavision_core.event_io import EventsIterator

from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

from niqe.niqe import niqe


def h5_tree(val, pre="", out=""):
    length = len(val)
    for key, val in val.items():
        length -= 1
        if length == 0:  # the last item
            if type(val) == h5py._hl.group.Group:
                out += pre + "└── " + key + "\n"
                out = h5_tree(val, pre + "    ", out)
            else:
                out += pre + "└── " + key + f" {val.shape}\n"
        else:
            if type(val) == h5py._hl.group.Group:
                out += pre + "├── " + key + "\n"
                out = h5_tree(val, pre + "│   ", out)
            else:
                out += pre + "├── " + key + f" {val.shape}\n"
    return out


def read_events_from_hdf5(file_path, time_window_us):
    import h5py

    initial_time = -1
    events = []
    with h5py.File(file_path, "r") as f:
        dataset = f["CD/events"]
        initial_time = dataset[0][3]  # First timestamp in dataset
        for e in dataset:
            x, y, p, ts = e
            if x < 0 or y < 0 or x >= 1280 or y >= 720:
                continue
            delta = ts - initial_time
            events.append((ts, x, y, p))

            if time_window_us != -1.0 and delta > time_window_us:
                break  # Stop when reaching the time window or max events

    # print info
    print(f"Read {len(events)} events from {file_path}")
    return events


def get_camera_geometry(file_path, delta_t=1000):

    # Initialize RawReader

    return height, width


class Metrics:
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
        "--pd", action="store_true", default=True, help="Metric: Point Distribution"
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

    m = Metrics(cam_w=width, cam_h=height, events_iterator=ev_iterator)

    if args.pd:
        m.point_distribution()

    # Read events from the HDF5 file
    # events = read_events_from_hdf5(file_path, time_window_us)
