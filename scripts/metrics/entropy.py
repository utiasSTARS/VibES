import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
import os
import cv2
from pathlib import Path

import skimage.measure


class FrameLoader:
    def __init__(self, fp):
        self._fp = fp
        self._width = None
        self._height = None
        self._image_files = []
        self._get_geometry()  # Initialize geometry from first image

    def _get_geometry(self):
        """Get image dimensions from the first image in the directory."""
        if not os.path.exists(self._fp):
            raise ValueError(f"Directory does not exist: {self._fp}")

        # Get list of image files (common image extensions)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        self._image_files = []

        for file_path in Path(self._fp).iterdir():
            if file_path.suffix.lower() in image_extensions:
                self._image_files.append(file_path)

        if not self._image_files:
            raise ValueError(f"No image files found in directory: {self._fp}")

        # Sort files by name to ensure consistent ordering
        self._image_files.sort()

        # Read the first image to get dimensions (as grayscale)
        first_image = cv2.imread(str(self._image_files[0]), cv2.IMREAD_GRAYSCALE)
        if first_image is None:
            raise ValueError(f"Could not read image: {self._image_files[0]}")

        self._height, self._width = first_image.shape
        print(f"Image geometry: {self._width}x{self._height}")
        print(f"Total images found: {len(self._image_files)}")

    def get_geom_width(self):
        return self._width

    def get_geom_height(self):
        return self._height

    def get_frame_count(self):
        return len(self._image_files)

    def __iter__(self):
        """Iterator to yield frames one by one without loading all into memory."""
        for img_path in self._image_files:
            # Load image using OpenCV as grayscale
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Warning: Could not load image {img_path}, skipping...")
                continue

            yield img

    def load(self):
        """Load all images from the directory as numpy arrays (for backward compatibility)."""
        print("Loading frames from directory...")
        frames = []

        for frame in tqdm(self, desc="Loading frames", total=len(self._image_files)):
            frames.append(frame)

        print(f"Total frames loaded: {len(frames)}")

        if len(frames) == 0:
            raise ValueError("No frames were successfully loaded.")

        return np.array(frames)


def rolling_min_max(data, window):
    """Calculate rolling min and max values over a sliding window."""
    rolling_min = []
    rolling_max = []
    half_window = window // 2

    for i in range(len(data)):
        # Define window boundaries
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)

        # Get min and max for this window
        window_data = data[start:end]
        rolling_min.append(np.min(window_data))
        rolling_max.append(np.max(window_data))

    return np.array(rolling_min), np.array(rolling_max)


def compute_entropy_for_folder(folder_path, folder_name):
    """Compute entropy for all frames in a folder."""
    print(f"\n=== Processing {folder_name}: {folder_path} ===")

    # Load events from the file
    frame_loader = FrameLoader(folder_path)

    print(f"Found {frame_loader.get_frame_count()} frames")
    print(f"Frame dimensions: {frame_loader.get_geom_width()}x{frame_loader.get_geom_height()}")

    entropy = []

    # Use iterator to process frames one by one (memory efficient)
    for i, frame in enumerate(
            tqdm(frame_loader, desc=f"Computing entropy for {folder_name}", total=frame_loader.get_frame_count())):
        # Frame is already grayscale from FrameLoader

        # Ensure the image is in float format (0-255 range is fine for entropy)
        if frame.dtype != np.float64:
            frame = frame.astype(np.float64)

        try:
            entropy_value = skimage.measure.shannon_entropy(frame)
            entropy.append(entropy_value)
        except Exception as e:
            print(f"Error computing entropy for frame {i}: {e}")
            continue

    if entropy:
        entropy = np.array(entropy)

        print(f"\nEntropy Results for {folder_name}:")
        print(f"Processed {len(entropy)} frames")
        print(f"Mean Entropy: {np.mean(entropy):.4f}")
        print(f"Std Entropy: {np.std(entropy):.4f}")
        print(f"Min Entropy: {np.min(entropy):.4f}")
        print(f"Max Entropy: {np.max(entropy):.4f}")

        return entropy
    else:
        print(f"No valid entropy computed for {folder_name}!")
        return np.array([])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run entropy metric on 'ev' and 'harmeda' folders within a given path.")
    parser.add_argument(
        "base_path", type=str, help="Base path containing 'ev' and 'harmeda' subfolders"
    )
    parser.add_argument("type", type=str, choices=["bin", "gray"],
                        help="Type of images to process (bin or gray)")
    args = parser.parse_args()

    # Construct paths to the two expected folders
    base_path = Path(args.base_path)
    ev_path = base_path / f"ev/img_{args.type}"
    harmeda_path = base_path / f"harmeda/img_{args.type}"

    # Check if both folders exist
    if not ev_path.exists():
        raise ValueError(f"'ev' folder not found at: {ev_path}")
    if not harmeda_path.exists():
        raise ValueError(f"'harmeda' folder not found at: {harmeda_path}")

    print(f"Base path: {base_path}")
    print(f"EV folder: {ev_path}")
    print(f"Harmeda folder: {harmeda_path}")

    # Compute entropy for both folders
    entropy1 = compute_entropy_for_folder(str(ev_path), "EV")
    entropy2 = compute_entropy_for_folder(str(harmeda_path), "Harmeda")

    # Get base directory for saving graphs
    parent_dir = base_path

    if len(entropy1) > 0 and len(entropy2) > 0:
        window_size = 10

        # Create windowed versions if we have enough frames
        if len(entropy1) >= window_size:
            entropy1_windowed = ndimage.median_filter(
                entropy1, size=window_size, mode="reflect"
            )
        else:
            entropy1_windowed = entropy1

        if len(entropy2) >= window_size:
            entropy2_windowed = ndimage.median_filter(
                entropy2, size=window_size, mode="reflect"
            )
        else:
            entropy2_windowed = entropy2

        # Create frame numbers for each dataset
        frame_numbers1 = np.arange(len(entropy1))
        frame_numbers2 = np.arange(len(entropy2))

        # Plot 1: Original entropy comparison
        plt.figure(figsize=(15, 8))
        plt.plot(
            frame_numbers1,
            entropy1,
            linewidth=1.5,
            marker="o",
            markersize=3,
            alpha=0.7,
            color="blue",
            label="EV (Original)"
        )
        plt.plot(
            frame_numbers2,
            entropy2,
            linewidth=1.5,
            marker="s",
            markersize=3,
            alpha=0.7,
            color="green",
            label="Harmeda (Original)"
        )
        plt.xlabel("Frame Number")
        plt.ylabel("Entropy")
        plt.title("Entropy Comparison: Original Data")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(parent_dir / "entropy_comparison_original.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # Plot 2: Windowed entropy comparison
        plt.figure(figsize=(15, 8))
        plt.plot(
            frame_numbers1,
            entropy1_windowed,
            linewidth=2,
            marker="o",
            markersize=2,
            alpha=0.8,
            color="red",
            label=f"EV (Windowed, median={window_size})"
        )
        plt.plot(
            frame_numbers2,
            entropy2_windowed,
            linewidth=2,
            marker="s",
            markersize=2,
            alpha=0.8,
            color="orange",
            label=f"Harmeda (Windowed, median={window_size})"
        )

        # Add rolling min/max for both datasets
        min_entropy1, max_entropy1 = rolling_min_max(entropy1, window_size)
        min_entropy2, max_entropy2 = rolling_min_max(entropy2, window_size)

        plt.fill_between(
            frame_numbers1,
            min_entropy1,
            max_entropy1,
            color="lightblue",
            alpha=0.3,
            label="EV Range"
        )
        plt.fill_between(
            frame_numbers2,
            min_entropy2,
            max_entropy2,
            color="lightgreen",
            alpha=0.3,
            label="Harmeda Range"
        )

        plt.xlabel("Frame Number")
        plt.ylabel("Entropy")
        plt.title(f"Entropy Comparison: Windowed Data (Median over {window_size} frames)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(parent_dir / "entropy_comparison_windowed.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # Plot 3: Combined view with both original and windowed
        plt.figure(figsize=(15, 10))

        # Original data (lighter colors)
        plt.plot(
            frame_numbers1,
            entropy1,
            linewidth=1,
            alpha=0.4,
            color="blue",
            label="EV (Original)"
        )
        plt.plot(
            frame_numbers2,
            entropy2,
            linewidth=1,
            alpha=0.4,
            color="green",
            label="Harmeda (Original)"
        )

        # Windowed data (bolder colors)
        plt.plot(
            frame_numbers1,
            entropy1_windowed,
            linewidth=3,
            alpha=0.9,
            color="darkblue",
            label="EV (Windowed)"
        )
        plt.plot(
            frame_numbers2,
            entropy2_windowed,
            linewidth=3,
            alpha=0.9,
            color="darkgreen",
            label="Harmeda (Windowed)"
        )

        plt.xlabel("Frame Number")
        plt.ylabel("Entropy")
        plt.title(f"Complete Entropy Comparison: Original vs Windowed (window size={window_size})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(parent_dir / "entropy_comparison_complete.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # Print comparative statistics
        print(f"\n=== COMPARATIVE STATISTICS ===")
        print(f"\nEV:")
        print(f"  Frames: {len(entropy1)}")
        print(f"  Mean Entropy: {np.mean(entropy1):.4f}")
        print(f"  Std Entropy: {np.std(entropy1):.4f}")
        print(f"  Mean Windowed Entropy: {np.mean(entropy1_windowed):.4f}")
        print(f"  Std Windowed Entropy: {np.std(entropy1_windowed):.4f}")

        print(f"\nHarmeda:")
        print(f"  Frames: {len(entropy2)}")
        print(f"  Mean Entropy: {np.mean(entropy2):.4f}")
        print(f"  Std Entropy: {np.std(entropy2):.4f}")
        print(f"  Mean Windowed Entropy: {np.mean(entropy2_windowed):.4f}")
        print(f"  Std Windowed Entropy: {np.std(entropy2_windowed):.4f}")

        print(f"\nDifferences (EV - Harmeda):")
        print(f"  Mean Entropy Difference: {np.mean(entropy1) - np.mean(entropy2):.4f}")
        print(f"  Mean Windowed Entropy Difference: {np.mean(entropy1_windowed) - np.mean(entropy2_windowed):.4f}")

        print(f"\nGraphs saved to: {parent_dir}")

    else:
        if len(entropy1) == 0:
            print("No valid entropy computed for EV folder!")
        if len(entropy2) == 0:
            print("No valid entropy computed for Harmeda folder!")
