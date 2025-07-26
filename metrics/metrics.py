import numpy as np

import argparse
import h5py

import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity


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
    from metavision_core.event_io import EventsIterator

    # Initialize RawReader
    my_iterator = EventsIterator(input_path=file_path, delta_t=delta_t)
    width, height = my_iterator.get_size()  # Camera Geometry

    return height, width


class Metrics:
    def __init__(self, cam_w, cam_h, events):
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.events = np.array(events)

    def point_distribution(self):
        # Compute point distribution metric

        kde = KernelDensity(kernel="gaussian", bandwidth=0.2)
        pts = self.events[:, 1:3].astype(float)  # Extract x, y coordinates
        pts[:, 0] /= self.cam_w  # Normalize x coordinates by camera width
        pts[:, 1] /= self.cam_h  # Normalize y coordinates by camera height

        print(pts)

        kde.fit(pts)

        log_density = kde.score_samples(pts)

        print(np.exp(log_density))

        data_range = np.linspace(0, 1, log_density.shape[0])

        fig = plt.figure()
        # plt.plot(x_range, np.exp(log_density), color="gray", linewidth=2)
        plt.fill_between(pts[:, 0], np.exp(log_density), alpha=0.5)
        plt.plot(pts[:, 0], np.full_like(pts[:, 0], -0.01), "|k", markeredgewidth=1)
        # plt.ylim(-0.02, 0.22)

        plt.title("Point Distribution")
        plt.show()

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
    args = parser.parse_args()

    width, height = get_camera_geometry(args.file_path, delta_t=1000)

    print(f"Processing file: {args.file_path}")
    print(f"Camera Geometry: {height}x{width}")
    print("Reading events...")
    events = read_events_from_hdf5(args.file_path, args.time_window_us)
    print("Done.")

    m = Metrics(cam_w=width, cam_h=height, events=events)

    if args.pd:
        m.point_distribution()

    # Read events from the HDF5 file
    # events = read_events_from_hdf5(file_path, time_window_us)
