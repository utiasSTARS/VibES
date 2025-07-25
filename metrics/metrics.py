import numpy as np
import h5py

from metavision_core.event_io import EventsIterator

import argparse


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


class Metrics:
    def __init__(self, cam_w, cam_h, event_iter):
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.event_iter = event_iter

    def point_distribution(self):
        # Compute point distribution metric

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
        default=1000,
        help="Delta time in microseconds for event iterator",
    )
    parser.add_argument("--pd", action="store_true", help="Metric: Point Distribution")
    args = parser.parse_args()

    # Initialize RawReader
    mv_iterator = EventsIterator(input_path=args.file_path, delta_t=args.delta_t)
    height, width = mv_iterator.get_size()  # Camera Geometry

    print(f"Processing file: {args.file_path}")
    print(f"Camera Geometry: {height}x{width}")

    m = Metrics(cam_w=width, cam_h=height, event_iter=mv_iterator)

    # Read events from the HDF5 file
    # events = read_events_from_hdf5(file_path, time_window_us)
