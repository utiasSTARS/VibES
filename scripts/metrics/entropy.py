import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage

import skimage.measure

from loader import FrameLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run entropy metric on event data.")
    parser.add_argument(
        "file_path", type=str, help="Path to the accumulated event frames"
    )
    args = parser.parse_args()

    # Load events from the file
    frame_loader = FrameLoader(args.file_path)
    frames = frame_loader.load()

    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}")

    entropy = []

    # Iterate over all frames and compute NIQE scores
    for i, frame in enumerate(tqdm(frames, desc="Computing entropy")):
        # Frame is already grayscale from FrameLoader

        # Ensure the image is in float format (0-255 range is fine for NIQE)
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
        plt.show()
        plt.savefig("entropy_vs_frame_number.png")
        plt.close()

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
        plt.xlabel("Frame Number")
        plt.ylabel("Entropy")
        plt.title(
            f"Windowed Entropy vs Frame Number (Median over {window_size} frames)"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig("windowed_entropy_vs_frame_number.png")
        plt.close()

    else:
        print("No valid entropy computed!")
