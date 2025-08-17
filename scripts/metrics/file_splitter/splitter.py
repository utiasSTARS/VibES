import os
import h5py
import numpy as np
from pathlib import Path
from .loaders import get_optimal_duration_event_reader

def split_hdf5_into_chunks(input_file, output_dir, duration_ms=10.0):
    """
    Splits a large HDF5 event file into smaller HDF5 files, each containing events of `duration_ms`.

    Args:
        input_file (str): Path to the input HDF5 file.
        output_dir (str): Directory where output chunks will be stored.
        duration_ms (float): Duration of each split in milliseconds.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Use your adaptive loader (handles big/small files efficiently)
    reader = get_optimal_duration_event_reader(
        path_to_event_file=input_file,
        duration_ms=duration_ms
    )

    file_index = 0
    for window in reader:
        if len(window) == 0:
            continue

        out_path = os.path.join(output_dir, f"chunk_{file_index:06d}.hdf5")
        with h5py.File(out_path, "w") as f:
            # Create dataset in the same structure as input
            dset = f.create_dataset(
                "CD/events",
                data=window,
                compression="gzip",
                compression_opts=4
            )
        file_index += 1

    print(f"✅ Done! Wrote {file_index} files of {duration_ms} ms each to {output_dir}")
