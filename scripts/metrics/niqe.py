import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from loader import FrameLoader

from niqe.niqe import niqe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NIQE metric on event data.")
    parser.add_argument(
        "file_path", type=str, help="Path to the accumulated event frames"
    )
    args = parser.parse_args()

    # Load events from the file
    frame_loader = FrameLoader(args.file_path)
    frames = frame_loader.load()

    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}")

    niqe_scores = []

    # Iterate over all frames and compute NIQE scores
    for i, frame in enumerate(tqdm(frames, desc="Computing NIQE scores")):
        # Frame is already grayscale from FrameLoader

        # Check if frame is large enough for NIQE (requires > 192x192)
        if frame.shape[0] <= 193 or frame.shape[1] <= 193:
            print(f"Warning: Frame {i} is too small ({frame.shape}), skipping...")
            continue

        # Ensure the image is in float format (0-255 range is fine for NIQE)
        if frame.dtype != np.float64:
            frame = frame.astype(np.float64)

        try:
            score = niqe(frame)
            niqe_scores.append(score)
        except Exception as e:
            print(f"Error computing NIQE for frame {i}: {e}")
            continue

    if niqe_scores:
        niqe_scores = np.array(niqe_scores)

        print(f"\nNIQE Results:")
        print(f"Processed {len(niqe_scores)} frames")
        print(f"Mean NIQE score: {np.mean(niqe_scores):.4f}")
        print(f"Std NIQE score: {np.std(niqe_scores):.4f}")
        print(f"Min NIQE score: {np.min(niqe_scores):.4f}")
        print(f"Max NIQE score: {np.max(niqe_scores):.4f}")

        # Plot line chart of NIQE scores vs frame number
        plt.figure(figsize=(12, 6))
        frame_numbers = np.arange(len(niqe_scores))
        plt.plot(
            frame_numbers,
            niqe_scores,
            linewidth=1.5,
            marker="o",
            markersize=3,
            alpha=0.7,
        )
        plt.xlabel("Frame Number")
        plt.ylabel("NIQE Score")
        plt.title("NIQE Score vs Frame Number")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid NIQE scores computed!")
