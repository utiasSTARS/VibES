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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run entropy metric on event data.")
    parser.add_argument(
        "file_path", type=str, help="Path to the accumulated event frames"
    )
    args = parser.parse_args()

    # Load events from the file
    frame_loader = FrameLoader(args.file_path)

    # Get parent directory for saving graphs
    parent_dir = Path(args.file_path).parent

    print(f"Found {frame_loader.get_frame_count()} frames")
    print(f"Frame dimensions: {frame_loader.get_geom_width()}x{frame_loader.get_geom_height()}")

    entropy = []

    # Use iterator to process frames one by one (memory efficient)
    for i, frame in enumerate(tqdm(frame_loader, desc="Computing entropy", total=frame_loader.get_frame_count())):
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

        print(f"\nEntropy Results:")
        print(f"Processed {len(entropy)} frames")
        print(f"Mean Entropy: {np.mean(entropy):.4f}")
        print(f"Std Entropy: {np.std(entropy):.4f}")
        print(f"Min Entropy: {np.min(entropy):.4f}")
        print(f"Max Entropy: {np.max(entropy):.4f}")

        # Create windowed version using median filter over 10 frames
        window_size = 10
        if len(entropy) >= window_size:
            # Use scipy's median filter for windowed median
            entropy_windowed = ndimage.median_filter(
                entropy, size=window_size, mode="reflect"
            )

            print(f"\nWindowed Entropy (median over {window_size} frames):")
            print(f"Mean Windowed Entropy: {np.mean(entropy_windowed):.4f}")
            print(f"Std Windowed Entropy: {np.std(entropy_windowed):.4f}")
            print(f"Min Windowed Entropy: {np.min(entropy_windowed):.4f}")
            print(f"Max Windowed Entropy: {np.max(entropy_windowed):.4f}")
        else:
            print(
                f"Warning: Not enough frames ({len(entropy)}) for windowing (need >= {window_size})"
            )
            entropy_windowed = entropy

        frame_numbers = np.arange(len(entropy))

        # Plot 1: Original entropy only (as before)
        plt.figure(figsize=(12, 6))
        plt.plot(
            frame_numbers,
            entropy,
            linewidth=1.5,
            marker="o",
            markersize=3,
            alpha=0.7,
        )
        plt.xlabel("Frame Number")
        plt.ylabel("Entropy")
        plt.title("Entropy vs Frame Number")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(parent_dir / "entropy_vs_frame_number.png")
        plt.show()
        plt.close()

        min_entropy, max_entropy = rolling_min_max(entropy, window_size)

        # Plot 2: Comparison of original and windowed entropy
        plt.figure(figsize=(15, 8))
        plt.plot(
            frame_numbers,
            entropy_windowed,
            linewidth=2,
            marker="o",
            markersize=2,
            alpha=0.8,
            color="red",
            label=f"Windowed Entropy (median, window={window_size})",
        )
        plt.fill_between(
            frame_numbers,
            min_entropy,
            max_entropy,
            color="lightgray",
            alpha=0.5,
            label="Entropy Range",
        )
        plt.xlabel("Frame Number")
        plt.ylabel("Entropy")
        plt.title(
            f"Windowed Entropy vs Frame Number (Median over {window_size} frames)"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(parent_dir / "windowed_entropy_vs_frame_number.png")
        plt.show()
        plt.close()

        print(f"\nGraphs saved to: {parent_dir}")

    else:
        print("No valid entropy computed!")