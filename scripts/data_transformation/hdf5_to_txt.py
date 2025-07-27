#!/usr/bin/env python3
"""
Convert HDF5 event files to TXT format for ETAP dataset loader.

This script converts HDF5 event files containing event data (x, y, t, p)
into a single TXT file with format: t x y p

Usage:
    python hdf5_to_txt.py --input-file events.h5 --output-dir ./output
"""

import argparse
import os

import cv2
import numpy as np
import h5py
import hdf5plugin
from pathlib import Path
from tqdm import tqdm


def read_events_from_hdf5(file_path, time_window_us):
    initial_time = -1
    xy_s = []
    p_s = []
    t_s = []
    x_min = 0 # 320
    y_min = 0 # 120
    # x_max = 960
    # y_max = 600

    with h5py.File(file_path, 'r') as f:
        dataset = f['CD/events']
        initial_time = dataset[0][3]  # First timestamp in dataset
        for e in tqdm(dataset):
            x, y, p, ts = e
            # if x < 0 or y < 0 or x >= 1280 or y >= 720:
            # if x < x_min or y < y_min or x >= x_max or y >= y_max:
            #     continue
            delta = ts - initial_time
            xy_s.append([x - x_min, y - y_min])
            p_s.append(p)
            t_s.append(ts / 1e6)  # Convert to seconds

            if delta > time_window_us:
                break  # Stop when reaching the time window or max events

    # print info
    print(f"Read {len(xy_s)} events from {file_path}")

    xy_s = np.array(xy_s, dtype=np.uint16)
    p_s = np.array(p_s, dtype=np.uint8)
    t_s = np.array(t_s, dtype=np.float32)
    return xy_s, p_s, t_s


def convert_hdf5_to_txt(input_file, output_dir):
    """
    Convert HDF5 event file to TXT format.

    Args:
        input_file (str): Path to input HDF5 file
        output_dir (str): Output directory for TXT file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting {input_file} to TXT format...")

    xy, p, t = read_events_from_hdf5(input_file, time_window_us=5 * 1e6)  # 2 second time window

    # accumulate events into an image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Assuming a frame size of 480x640
    for i in range(len(xy)):
        x, y = xy[i]
        p_val = p[i]
        if p_val == 1:
            frame[y, x, 0] = 255  # Positive polarity
        else:
            frame[y, x, 2] = 255  # Negative polarity

    # save the frame as an image
    output_image_path = os.path.join(output_dir, 'events_frame.png')
    cv2.imwrite(output_image_path, frame)

    # Save events to TXT file with format: t x y p
    txt_file = os.path.join(output_dir, 'events.txt')
    print(f"Saving events to {txt_file}...")

    with open(txt_file, 'w') as f:
        for i in range(len(xy)):
            x, y = xy[i]
            timestamp = t[i]
            polarity = p[i]
            f.write(f"{timestamp} {x} {y} {polarity}\n")

    print(f"Saved {len(xy)} events to {txt_file}")


def batch_convert(input_dir, output_dir, pattern="*.hdf5"):
    """
    Batch convert multiple HDF5 files to TXT format.

    Args:
        input_dir (str): Input directory containing HDF5 files
        output_dir (str): Output directory for TXT files
        pattern (str): File pattern to match
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all HDF5 files
    h5_files = list(input_path.glob(pattern))

    if not h5_files:
        print(f"No HDF5 files found in {input_dir} matching pattern '{pattern}'")
        return

    print(f"Found {len(h5_files)} HDF5 files to convert")

    for h5_file in tqdm(h5_files, desc="Converting files"):
        # Create output subdirectory for this file
        file_output_dir = output_path / h5_file.stem
        convert_hdf5_to_txt(str(h5_file), str(file_output_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 event files to TXT format for ETAP dataset loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python hdf5_to_txt.py --input-file events.h5 --output-dir ./output
  
  # Batch convert all HDF5 files in a directory
  python hdf5_to_txt.py --input-dir ./data --output-dir ./converted --batch
  
  # Batch convert with custom pattern
  python hdf5_to_txt.py --input-dir ./data --output-dir ./converted --batch --pattern "events_*.h5"
        """
    )

    # Input/output arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-file', '-i',
        help='Input HDF5 file path'
    )

    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Output directory for TXT file'
    )

    args = parser.parse_args()

    try:
        convert_hdf5_to_txt(args.input_file, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())