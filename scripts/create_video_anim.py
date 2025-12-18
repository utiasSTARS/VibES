import cv2
import numpy as np

def process_videos(path1, path2, output_path, skip_seconds=30):
    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)

    # Get video properties from the first video
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate how many frames to skip (30 seconds * FPS)
    frames_to_skip = int(skip_seconds * fps)

    # Set start positions
    cap1.set(cv2.CAP_PROP_POS_FRAMES, frames_to_skip)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frames_to_skip)

    # Define codec and create VideoWriter for stacked output (Height is doubled)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height * 2))

    print(f"Processing... skipping first {skip_seconds}s ({frames_to_skip} frames).")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Stack frames vertically
        stacked_frame = np.vstack((frame1, frame2))

        # Write to output
        out.write(stacked_frame)

    cap1.release()
    cap2.release()
    out.release()
    print(f"Done! Saved to {output_path}")
