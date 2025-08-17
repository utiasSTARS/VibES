import argparse
import h5py
import numpy as np
from sklearn.neighbors import KernelDensity
import pickle
import os

def _parse_hdf5_geom_string(geom_str):
    """Parse geometry string from HDF5 attributes."""
    try:
        geom_parts = geom_str.split("x")
        if len(geom_parts) == 2:
            return int(geom_parts[0]), int(geom_parts[1])
        else:
            raise ValueError(f"Invalid geometry format: {geom_str}")
    except Exception as e:
        raise ValueError(f"Error parsing geometry string '{geom_str}': {e}")

def load_events(file_path, downsample_rate=1):
    """Load x,y events from an HDF5 file."""
    cam_w = -1
    cam_h = -1
    with h5py.File(file_path, "r") as f:
        # Adjust dataset name if needed (common ones: 'events', 'CD/events')
        if "events" in f:
            events = f["events"][:]
        elif "CD/events" in f:
            events = f["CD/events"][:]
        else:
            raise KeyError("No 'events' dataset found in HDF5 file")

        if "geometry" in f.attrs:
            cam_w, cam_h = _parse_hdf5_geom_string(f.attrs["geometry"])
            # print(f"Camera geometry: {cam_w}x{cam_h}")
        else:
            raise KeyError("No 'geometry' attribute found in HDF5 file")

    # Assuming events are structured with ['x','y',...] or columns
    if events.dtype.names is not None:
        x = events["x"].astype(float)
        y = events["y"].astype(float)
    else:
        # fallback if stored as NxM array
        x, y = events[:, 0].astype(float), events[:, 1].astype(float)

    # nomalize coordinates to [0, 1]
    x /= cam_w
    y /= cam_h

    # print(f"Loaded {y.shape}")
    if downsample_rate > 1:
        # print(f"Downsampling events by a factor of {downsample_rate}")
        x = x[::downsample_rate]
        y = y[::downsample_rate]

    return np.vstack([x, y]).T


def compute_kde(points, bw=0.05):
    """Compute KDE on 2D points using SciPy."""
    kde2d = KernelDensity(kernel="gaussian", bandwidth=bw)
    # points = points.T  # shape (N, 2)
    kde2d.fit(points)
    log_densities = kde2d.score_samples(points)
    return np.exp(log_densities)


def main():
    parser = argparse.ArgumentParser(description="Compute KDE on event camera HDF5 file")
    parser.add_argument("input_file", help="Path to events_*.hdf5 file")
    parser.add_argument("--output", help="Optional output filename")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample rate")
    args = parser.parse_args()

    print(f"Loading events from {args.input_file}... {args.downsample}")
    pts = load_events(args.input_file, downsample_rate=int(args.downsample) if args.downsample else 1)

    # print number of events
    print(f"Loaded {pts.shape[0]} events with shape {pts.shape}")

    # print("Computing KDE...")
    densities = compute_kde(pts)

    # derive output name if not given
    if args.output:
        out_file = args.output
    else:
        base, _ = os.path.splitext(args.input_file)
        out_file = base + "_densities.pkl"

    # print(f"Saving densities to {out_file}...")
    with open(out_file, "wb") as f:
        pickle.dump(densities, f)

    print("Done ✅")


if __name__ == "__main__":
    main()
