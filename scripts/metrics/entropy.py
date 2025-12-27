"""
Entropy Analysis Tool for Image Sequences.

This script evaluates the information content (Shannon Entropy) of image sequences
to compare the quality or characteristics of different datasets (e.g., standard Event-based
reconstructions vs. Vibrating/Compensated reconstructions).

It computes:
1. **Per-frame Shannon Entropy:** Measures the amount of information/disorder in the image.
   Higher entropy in event accumulations often correlates with richer texture or less motion blur.
2. **Rolling Statistics:** Computes median, min, and max entropy over a sliding window to visualize trends.
3. **Comparative Visualization:** Generates high-quality LaTeX-formatted plots comparing two datasets.

Usage:
    python entropy_analysis.py /path/to/dataset [bin|gray] [window_us]

    Example:
    python entropy_analysis.py ./data gray 10000

Expected Directory Structure:
    /path/to/dataset/
    ├── ev/
    │   └── img_[type]_[us]/   (e.g., img_gray_10000)
    └── vibes/
        └── img_[type]_[us]/

@author Vincenzo Polizzi - STARS Lab
@date Dec 27 2025
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
import os
import cv2
from pathlib import Path
import re
import pandas as pd
from typing import Tuple, List, Optional, Generator

import skimage.measure


class FrameLoader:
    """
    Efficient iterator-based image loader for processing large sequences.

    This class handles directory scanning, sorting by frame index, and
    lazy loading of images to minimize memory usage.
    """
    def __init__(self, fp: str):
        """
        Initialize the loader.

        Args:
            fp (str): File path to the directory containing images.
        """
        self._fp = fp
        self._width = None
        self._height = None
        self._image_files = []
        self._get_geometry()  # Initialize geometry/metadata from the first available image

    def _get_geometry(self) -> None:
        """
        Scans the directory and determines image dimensions.

        Raises:
            ValueError: If directory does not exist or contains no valid images.
        """
        if not os.path.exists(self._fp):
            raise ValueError(f"Directory does not exist: {self._fp}")

        # Supported extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        self._image_files = []

        for file_path in Path(self._fp).iterdir():
            if file_path.suffix.lower() in image_extensions:
                self._image_files.append(file_path)

        if not self._image_files:
            raise ValueError(f"No image files found in directory: {self._fp}")

        # Natural sort files by the number in their filename (e.g., img_1.png, img_2.png)
        # This prevents ordering issues like img_10.png coming before img_2.png
        self._image_files.sort(key=lambda p: int(re.search(r'\d+', p.stem).group()))

        # Read the first image to establish dimensions
        first_image = cv2.imread(str(self._image_files[0]), cv2.IMREAD_GRAYSCALE)
        if first_image is None:
            raise ValueError(f"Could not read image: {self._image_files[0]}")

        self._height, self._width = first_image.shape
        print(f"[FrameLoader] Image geometry: {self._width}x{self._height}")
        print(f"[FrameLoader] Total images found: {len(self._image_files)}")

    def get_geom_width(self) -> int:
        return self._width

    def get_geom_height(self) -> int:
        return self._height

    def get_frame_count(self) -> int:
        return len(self._image_files)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """
        Yields frames one by one.

        Yields:
            np.ndarray: Grayscale image data.
        """
        for img_path in self._image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Warning: Could not load image {img_path}, skipping...")
                continue

            yield img

    def load(self) -> np.ndarray:
        """
        Load all images into a single numpy array.

        Warning: This may consume significant RAM for large datasets.

        Returns:
            np.ndarray: Array of shape (N, H, W) containing all frames.
        """
        print("Loading all frames into memory...")
        frames = []

        for frame in tqdm(self, desc="Loading frames", total=len(self._image_files)):
            frames.append(frame)

        print(f"Total frames loaded: {len(frames)}")

        if len(frames) == 0:
            raise ValueError("No frames were successfully loaded.")

        return np.array(frames)


def rolling_min_max(data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate rolling min and max values over a sliding window.

    Args:
        data (np.ndarray): Input 1D array.
        window (int): Size of the sliding window.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays containing the rolling min and max respectively.
    """
    rolling_min = []
    rolling_max = []
    half_window = window // 2

    for i in range(len(data)):
        # Define window boundaries, clamping to array limits
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)

        window_data = data[start:end]
        rolling_min.append(np.min(window_data))
        rolling_max.append(np.max(window_data))

    return np.array(rolling_min), np.array(rolling_max)


def compute_entropy_for_folder(folder_path: str, folder_name: str) -> np.ndarray:
    """
    Computes Shannon Entropy for every image in the specified folder.

    Args:
        folder_path (str): Path to the image directory.
        folder_name (str): Label for the dataset (for logging).

    Returns:
        np.ndarray: Array of entropy values for each frame. Returns empty array if processing fails.
    """
    print(f"\n=== Processing {folder_name}: {folder_path} ===")

    try:
        frame_loader = FrameLoader(folder_path)
    except ValueError as e:
        print(f"Error initializing loader: {e}")
        return np.array([])

    entropy = []

    # Process frames individually to keep memory footprint low
    for i, frame in enumerate(
            tqdm(frame_loader, desc=f"Computing entropy for {folder_name}", total=frame_loader.get_frame_count())):

        # Ensure proper type for entropy calculation
        if frame.dtype != np.float64:
            frame = frame.astype(np.float64)

        # Simple masking/normalization if needed (logic preserved from original)
        # Assuming we want entropy of the non-zero structure or full image
        # frame = (frame > 0).astype(np.uint8)

        try:
            # Calculate Shannon Entropy: -sum(p * log2(p))
            entropy_value = skimage.measure.shannon_entropy(frame)
            entropy.append(entropy_value)
        except Exception as e:
            print(f"Error computing entropy for frame {i}: {e}")
            continue

    if entropy:
        entropy_arr = np.array(entropy)

        print(f"\nEntropy Results for {folder_name}:")
        print(f"  Processed: {len(entropy_arr)} frames")
        print(f"  Mean: {np.mean(entropy_arr):.4f}")
        print(f"  Std:  {np.std(entropy_arr):.4f}")
        print(f"  Min:  {np.min(entropy_arr):.4f}")
        print(f"  Max:  {np.max(entropy_arr):.4f}")

        return entropy_arr
    else:
        print(f"No valid entropy computed for {folder_name}!")
        return np.array([])


def save_entropy_to_csv(entropy1, entropy2, entropy1_windowed, entropy2_windowed,
                        min_entropy1, max_entropy1, min_entropy2, max_entropy2,
                        output_path: Path, window_size: int) -> Tuple[Path, Path]:
    """
    Exports raw entropy data and summary statistics to CSV files.
    """
    # Pad shorter arrays with NaN to ensure DataFrame consistency
    max_length = max(len(entropy1), len(entropy2))

    data = {
        'frame_number': list(range(max_length)),
        'ev_entropy': list(entropy1) + [np.nan] * (max_length - len(entropy1)),
        'vibes_entropy': list(entropy2) + [np.nan] * (max_length - len(entropy2)),
        'ev_entropy_windowed': list(entropy1_windowed) + [np.nan] * (max_length - len(entropy1_windowed)),
        'vibes_entropy_windowed': list(entropy2_windowed) + [np.nan] * (max_length - len(entropy2_windowed)),
        'ev_entropy_min': list(min_entropy1) + [np.nan] * (max_length - len(min_entropy1)),
        'ev_entropy_max': list(max_entropy1) + [np.nan] * (max_length - len(max_entropy1)),
        'vibes_entropy_min': list(min_entropy2) + [np.nan] * (max_length - len(min_entropy2)),
        'vibes_entropy_max': list(max_entropy2) + [np.nan] * (max_length - len(max_entropy2))
    }

    # 1. Save Full Time-Series Data
    df = pd.DataFrame(data)
    csv_file = output_path / f"entropy_data_window_{window_size}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Time-series data saved to: {csv_file}")

    # 2. Save Summary Statistics
    summary_data = {
        'dataset': ['EV', 'vibes'],
        'frame_count': [len(entropy1), len(entropy2)],
        'mean_entropy': [np.mean(entropy1), np.mean(entropy2)],
        'std_entropy': [np.std(entropy1), np.std(entropy2)],
        'min_entropy': [np.min(entropy1), np.min(entropy2)],
        'max_entropy': [np.max(entropy1), np.max(entropy2)],
        'mean_entropy_windowed': [np.mean(entropy1_windowed), np.mean(entropy2_windowed)],
        'std_entropy_windowed': [np.std(entropy1_windowed), np.std(entropy2_windowed)]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_csv_file = output_path / f"entropy_summary_window_{window_size}.csv"
    summary_df.to_csv(summary_csv_file, index=False)
    print(f"Summary statistics saved to: {summary_csv_file}")

    return csv_file, summary_csv_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run entropy quality metric on 'ev' (baseline) and 'vibes' (method) image sequences.")
    parser.add_argument(
        "base_path", type=str, help="Base directory containing 'ev' and 'vibes' subfolders."
    )
    parser.add_argument("type", type=str, choices=["bin", "gray"],
                        help="Type of images to process (binary or grayscale).")
    parser.add_argument("us", type=int,
                        help="Accumulation time window in microseconds (e.g., 10000). Used to find subfolders.")

    args = parser.parse_args()

    # --- Path Construction ---
    base_path = Path(args.base_path)
    # Expected structure: base/ev/img_gray_10000 and base/vibes/img_gray_10000
    ev_path = base_path / f"ev/img_{args.type}_{args.us}"
    vibes_path = base_path / f"vibes/img_{args.type}_{args.us}"

    # Validation
    if not ev_path.exists():
        raise ValueError(f"'ev' folder not found at: {ev_path}")
    if not vibes_path.exists():
        raise ValueError(f"'vibes' folder not found at: {vibes_path}")

    print(f"Base path: {base_path}")
    print(f"EV (Baseline) path: {ev_path}")
    print(f"Vibes (Method) path: {vibes_path}")

    # --- 1. Compute Entropy ---
    entropy1 = compute_entropy_for_folder(str(ev_path), "EV")
    entropy2 = compute_entropy_for_folder(str(vibes_path), "vibes")

    if len(entropy1) > 0 and len(entropy2) > 0:
        window_size = 10

        # --- 2. Post-Processing (Filtering) ---
        # Apply median filter to smooth out frame-to-frame noise
        if len(entropy1) >= window_size:
            entropy1_windowed = ndimage.median_filter(entropy1, size=window_size, mode="reflect")
        else:
            entropy1_windowed = entropy1

        if len(entropy2) >= window_size:
            entropy2_windowed = ndimage.median_filter(entropy2, size=window_size, mode="reflect")
        else:
            entropy2_windowed = entropy2

        # Calculate bands (min/max) for visualization
        min_entropy1, max_entropy1 = rolling_min_max(entropy1, window_size)
        min_entropy2, max_entropy2 = rolling_min_max(entropy2, window_size)

        # --- 3. Save Data ---
        save_entropy_to_csv(
            entropy1, entropy2, entropy1_windowed, entropy2_windowed,
            min_entropy1, max_entropy1, min_entropy2, max_entropy2,
            base_path, window_size
        )

        # --- 4. Plotting ---
        # Use LaTeX font rendering for publication-quality plots
        try:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "axes.labelsize": 10,
                "font.size": 10,
                "legend.fontsize": 10,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "text.latex.preamble": r"\usepackage{amsmath}"
            })
        except Exception:
            print("Warning: LaTeX support not available. Using default fonts.")

        # Determine shared x-axis limit
        frame_numbers1 = np.arange(len(entropy1))
        frame_numbers2 = np.arange(len(entropy2))
        min_length = 2500 # Set fixed length for plot consistency, or use min(len1, len2)

        # Plot Setup
        fig_width_in = 3.29  # Standard single-column width
        fig_height_in = 1.5 * fig_width_in
        fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

        # Plot Curves
        ax.plot(frame_numbers1[:min_length], entropy1_windowed[:min_length],
                linewidth=1.0, alpha=0.8,
                color="red", label=fr"EV (Windowed, median={window_size})")
        ax.plot(frame_numbers2[:min_length], entropy2_windowed[:min_length],
                linewidth=1.0, alpha=0.8,
                color="orange", label=fr"Ours (Windowed, median={window_size})")

        # Plot Range Areas (Min/Max bands)
        ax.fill_between(frame_numbers1[:min_length],
                        min_entropy1[:min_length], max_entropy1[:min_length],
                        color="lightblue", alpha=0.3, label="EV Range")
        ax.fill_between(frame_numbers2[:min_length],
                        min_entropy2[:min_length], max_entropy2[:min_length],
                        color="lightgreen", alpha=0.3, label="Ours Range")

        # Styling
        ax.set_xlabel(r"Frame Number", fontsize=10, labelpad=3)
        ax.set_ylabel(r"Entropy", fontsize=10, labelpad=3)
        ax.set_title(r"\textbf{Entropy Comparison}", fontsize=10, pad=5)
        ax.grid(True, alpha=0.3)

        # Legend
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.3),
                  ncol=2, borderaxespad=0, handlelength=1, fontsize=10, markerscale=0.7)

        fig.tight_layout(rect=[-0.2, 0, 1.2, 0.85])

        # Save Plot
        output_plot = base_path / "entropy_comparison_windowed.pdf"
        fig.savefig(output_plot, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to: {output_plot}")

        # --- 5. Final Statistics ---
        print(f"\n=== COMPARATIVE STATISTICS ===")
        print(f"\nEV (Baseline):")
        print(f"  Mean Entropy: {np.mean(entropy1):.4f}")
        print(f"  Std Entropy:  {np.std(entropy1):.4f}")

        print(f"\nVibes (Method):")
        print(f"  Mean Entropy: {np.mean(entropy2):.4f}")
        print(f"  Std Entropy:  {np.std(entropy2):.4f}")

        diff = np.mean(entropy1) - np.mean(entropy2)
        print(f"\nDifference (EV - Vibes): {diff:.4f}")

    else:
        if len(entropy1) == 0:
            print("Error: No valid entropy computed for EV folder.")
        if len(entropy2) == 0:
            print("Error: No valid entropy computed for Vibes folder.")