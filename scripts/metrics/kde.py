"""
Kernel Density Estimation (KDE) Analysis for Event Data.

This script computes the spatial density of events from an HDF5 recording.
It is useful for analyzing the distribution of events across the sensor plane,
which can indicate focus quality, texture richness, or the effectiveness of
motion compensation (sharper peaks = better compensation).

The script:
1. Loads event data (x, y coordinates) from HDF5 files.
2. Normalizes coordinates to [0, 1] based on sensor geometry.
3. Computes a Gaussian Kernel Density Estimate (KDE).
4. Saves the resulting density scores to a pickle file for further analysis.

Usage:
    python kde_analysis.py input_events.hdf5 --output densities.pkl --downsample 10

Dependencies:
    - h5py
    - scikit-learn
    - numpy

@author Vincenzo Polizzi - STARS Lab
@date Dec 27 2025
"""

import argparse
import h5py
import numpy as np
from sklearn.neighbors import KernelDensity
import pickle
import os
import sys
from typing import Tuple

def _parse_hdf5_geom_string(geom_str: str) -> Tuple[int, int]:
    """
    Parse geometry string (e.g., "1280x720") from HDF5 attributes.
    
    Args:
        geom_str (str): Geometry string from file metadata.
        
    Returns:
        Tuple[int, int]: (width, height)
        
    Raises:
        ValueError: If the format is incorrect.
    """
    try:
        # Handle cases where geometry might be bytes
        if isinstance(geom_str, bytes):
            geom_str = geom_str.decode('utf-8')

        geom_parts = geom_str.split("x")
        if len(geom_parts) == 2:
            return int(geom_parts[0]), int(geom_parts[1])
        else:
            raise ValueError(f"Invalid geometry format: {geom_str}")
    except Exception as e:
        raise ValueError(f"Error parsing geometry string '{geom_str}': {e}")

def load_events(file_path: str, downsample_rate: int = 1) -> np.ndarray:
    """
    Load and normalize x,y event coordinates from an HDF5 file.
    
    Coordinates are normalized to the [0, 1] range to make KDE bandwidth 
    independent of sensor resolution.
    
    Args:
        file_path (str): Path to the .hdf5 file.
        downsample_rate (int): Factor to reduce data size (e.g., 10 takes every 10th event).
        
    Returns:
        np.ndarray: N x 2 array of normalized (x, y) coordinates.
    """
    cam_w = -1
    cam_h = -1

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        # 1. Locate the Event Dataset
        # Metavision and other formats use different keys
        if "events" in f:
            events = f["events"][:]
        elif "CD/events" in f:
            events = f["CD/events"][:]
        else:
            # List keys to help debugging
            keys = list(f.keys())
            raise KeyError(f"No 'events' or 'CD/events' dataset found in HDF5 file. Available keys: {keys}")

        # 2. Extract Geometry for Normalization
        # Look in root attributes first
        if "geometry" in f.attrs:
            cam_w, cam_h = _parse_hdf5_geom_string(f.attrs["geometry"])
        # Some formats might store geometry in the events dataset attributes
        elif "events" in f and "geometry" in f["events"].attrs:
            cam_w, cam_h = _parse_hdf5_geom_string(f["events"].attrs["geometry"])
        # Fallback hardcoded or raise error (Safer to raise error)
        else:
            raise KeyError("No 'geometry' attribute found in HDF5 file metadata.")

    print(f"  [Loader] Sensor Geometry: {cam_w}x{cam_h}")

    # 3. Extract Coordinates
    # Handle structured arrays (named fields) vs unstructured arrays
    if events.dtype.names is not None:
        if 'x' in events.dtype.names and 'y' in events.dtype.names:
            x = events["x"].astype(float)
            y = events["y"].astype(float)
        else:
            raise ValueError(f"Event dataset is structured but missing 'x' or 'y' fields. Fields: {events.dtype.names}")
    else:
        # Fallback if stored as NxM array (Column 0=x, Column 1=y typically)
        if events.shape[1] >= 2:
            x = events[:, 0].astype(float)
            y = events[:, 1].astype(float)
        else:
            raise ValueError(f"Event dataset has invalid shape: {events.shape}")

    # 4. Normalize Coordinates to [0, 1]
    # This is crucial for KDE bandwidth consistency
    x /= cam_w
    y /= cam_h

    # 5. Downsampling
    if downsample_rate > 1:
        x = x[::downsample_rate]
        y = y[::downsample_rate]

    return np.vstack([x, y]).T


def compute_kde(points: np.ndarray, bw: float = 0.005) -> np.ndarray:
    """
    Compute Gaussian KDE on 2D points.
    
    Args:
        points (np.ndarray): N x 2 array of normalized coordinates.
        bw (float): Bandwidth for the Gaussian kernel. 
                    Since coords are [0,1], 0.005 is roughly 0.5% of the sensor width.
        
    Returns:
        np.ndarray: Array of density scores (exponential of log-likelihood).
    """
    print(f"  [KDE] Fitting model with bandwidth={bw} on {len(points)} points...")

    # Use KDTree for faster queries on low-dimensional data
    kde2d = KernelDensity(kernel="gaussian", bandwidth=bw, algorithm='kd_tree')
    kde2d.fit(points)

    # Score samples returns log-density
    print("  [KDE] Scoring samples...")
    log_densities = kde2d.score_samples(points)

    return np.exp(log_densities)


def main():
    parser = argparse.ArgumentParser(description="Compute KDE density map for event camera data")
    parser.add_argument("input_file", help="Path to events .hdf5 file")
    parser.add_argument("--output", help="Optional output filename for .pkl result")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample rate to speed up computation (default: 1, no downsampling)")
    parser.add_argument("--bandwidth", type=float, default=0.005,
                        help="KDE Bandwidth (default: 0.005 relative to image size)")

    args = parser.parse_args()

    print(f"=== Starting KDE Analysis ===")
    print(f"Input: {args.input_file}")

    try:
        # Load Data
        pts = load_events(args.input_file, downsample_rate=args.downsample)
        print(f"Loaded {pts.shape[0]} events for processing (Downsample: {args.downsample})")

        if pts.shape[0] == 0:
            print("Error: No events loaded.")
            return

        # Compute KDE
        densities = compute_kde(pts, bw=args.bandwidth)

        # Determine Output Filename
        if args.output:
            out_file = args.output
        else:
            base, _ = os.path.splitext(args.input_file)
            out_file = base + "_densities.pkl"

        # Save Results
        print(f"Saving {len(densities)} density scores to {out_file}...")
        with open(out_file, "wb") as f:
            # We save both the densities and the points used to generate them
            # This allows reconstructing the heatmap later
            data = {
                "points": pts,
                "densities": densities,
                "bandwidth": args.bandwidth
            }
            pickle.dump(data, f)

        print(f"Done ✅")

        # Print basic stats
        print(f"Density Stats -> Min: {np.min(densities):.4f}, Max: {np.max(densities):.4f}, Mean: {np.mean(densities):.4f}")

    except Exception as e:
        print(f"Fatal Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()