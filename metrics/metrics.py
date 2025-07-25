import numpy as np
import h5py

import argparse


def read_events_from_hdf5(file_path, time_window_us=-1):
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

            if time_window_us != -1 and delta > time_window_us:
                break  # Stop when reaching the time window or max events

    # print info
    print(f"Read {len(events)} events from {file_path}")
    return events


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 event data.")
    parser.add_argument("file_path", type=str, help="Path to the HDF5 file")
    parser.add_argument(
        "--time_window_us", type=int, default=-1, help="Time window in microseconds"
    )
    parser.add_argument("--pd", action="store_true", help="Metric: Point Distribution")
    args = parser.parse_args()

    # Example file path (adjust as needed)
    file_path = args.file_path
    time_window_us = args.time_window_us
