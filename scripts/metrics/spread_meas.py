import argparse
import h5py
import numpy as np
from sklearn.neighbors import KernelDensity
import pickle
import os

import numpy as np
from scipy.ndimage import sobel  # if you prefer pure numpy, use np.gradient

def events_to_count_image(x, y, W, H):
    """x,y in [0,1] floats or pixel ints; returns HxW counts image."""
    if x.max() <= 1.0 and y.max() <= 1.0:
        xi = np.clip((x * W).astype(int), 0, W-1)
        yi = np.clip((y * H).astype(int), 0, H-1)
    else:
        xi = np.clip(x.astype(int), 0, W-1)
        yi = np.clip(y.astype(int), 0, H-1)
    img = np.zeros((H, W), dtype=np.int64)
    np.add.at(img, (yi, xi), 1)
    return img

def gini(x):
    v = x.ravel().astype(float)
    if v.size == 0:
        return 0.0
    v = v[v>=0]
    if v.sum() == 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    cum = np.cumsum(v)
    # Gini via Lorenz curve
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n

def entropy(x, eps=1e-12):
    p = x.ravel().astype(float)
    s = p.sum()
    if s == 0: return 0.0
    p = p / s
    p = p[p>0]
    return -np.sum(p * np.log(p + eps))

def hoyer_sparsity(x):
    v = x.ravel().astype(float)
    n = v.size
    l1 = np.sum(np.abs(v))
    l2 = np.sqrt(np.sum(v**2))
    if l2 == 0: return 0.0
    return (np.sqrt(n) - (l1 / (l2 + 1e-12))) / (np.sqrt(n) - 1 + 1e-12)

def gradient_energy(img):
    gx = sobel(img, axis=1, mode='nearest')
    gy = sobel(img, axis=0, mode='nearest')
    mag = np.hypot(gx, gy)
    return mag.mean()

def border_ratio(img, band_px=8):
    H, W = img.shape
    mask = np.zeros_like(img, dtype=bool)
    mask[:band_px, :] = True
    mask[-band_px:, :] = True
    mask[:, :band_px] = True
    mask[:, -band_px:] = True
    edge = img[mask].sum()
    interior = img[~mask].sum()
    total = edge + interior
    return (edge / total) if total > 0 else 0.0

def summarize_events(x, y, W, H, band_px=8):
    img = events_to_count_image(x, y, W, H)
    return {
        "var_counts": float(np.var(img)),
        "gini": float(gini(img)),
        "entropy": float(entropy(img)),
        "hoyer": float(hoyer_sparsity(img)),
        "grad_energy": float(gradient_energy(img)),
        "border_ratio": float(border_ratio(img, band_px)),
        "total_events": int(img.sum()),
    }


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

    res = summarize_events(x, y, cam_w, cam_h)
    return res



def main():
    parser = argparse.ArgumentParser(description="Compute KDE on event camera HDF5 file")
    parser.add_argument("input_file", help="Path to events_*.hdf5 file")
    parser.add_argument("--output", help="Optional output filename")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample rate")
    args = parser.parse_args()

    print(f"Loading events from {args.input_file}... {args.downsample}")
    res = load_events(args.input_file, downsample_rate=int(args.downsample) if args.downsample else 1)

    # derive output name if not given
    if args.output:
        out_file = args.output
    else:
        base, _ = os.path.splitext(args.input_file)
        out_file = base + "_res.pkl"

    # print(f"Saving densities to {out_file}...")
    with open(out_file, "wb") as f:
        pickle.dump(res, f)

    print("Done ✅")


if __name__ == "__main__":
    main()
