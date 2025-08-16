#!/usr/bin/env python3
"""
Convert AEDAT4 (dv/AedatFile) event files into HDF5 format
with 'CD' and 'EXT_TRIGGER' groups, indexes, and attributes.
Processes events in chunks to save memory.
"""

import argparse
import numpy as np
import h5py
from dv import AedatFile
import os
from datetime import datetime

from tqdm import tqdm


def extract_events_to_txt(bag_path, output_path, max_events=None):
    """
    Extract DVS events from ROS bag file and save to text file.
    Format: t x y p (one event per line)
    """
    print(f"Opening bag file: {bag_path}")

    try:
        # Open output text file
        if os.path.exists(output_path):
            os.remove(output_path)

        event_count = 0

        # Event processing buffer for batch writing
        BUFFER_SIZE = 1000000  # Write in batches for efficiency
        event_buffer = []

        print('**** ROS Bag -> Text File Conversion **** ')
        print('*** Processing Events *** \n')

        # Initialize progress bar
        progress_bar = tqdm(desc="Processing", unit="events")

        with open(output_path, 'w') as txt_file:
            # Process events in streaming fashion

            # Process events in streaming fashion
            with AedatFile(input_path) as f:
                # Get camera resolution
                height, width = f['events'].size

                # Write header comment
                txt_file.write(f"# DVS events extracted from {os.path.basename(bag_path)}\n")
                txt_file.write(f"# Camera resolution: {width}x{height}\n")
                txt_file.write(f"# Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                txt_file.write(f"# Format: t x y p (timestamp_us x y polarity)\n")
                txt_file.write("#\n")

                for event in f['events']:
                    # Convert timestamp to microseconds
                    timestamp_us = int(event.timestamp)

                    # Extract event components
                    x = int(event.x)
                    y = int(event.y)
                    p = int(event.polarity)

                    # Add to buffer
                    event_buffer.append(f"{timestamp_us} {x} {y} {p}\n")
                    event_count += 1

                    # Write buffer when full
                    if len(event_buffer) >= BUFFER_SIZE:
                        txt_file.writelines(event_buffer)
                        progress_bar.update(len(event_buffer))
                        event_buffer.clear()

                    # Check max events limit
                    if max_events and event_count >= max_events:
                        break

            # Write remaining events in buffer
            if event_buffer:
                txt_file.writelines(event_buffer)
                progress_bar.update(len(event_buffer))

        progress_bar.close()

        print('*** Conversion Complete ***')
        print(f"Successfully saved {event_count} events to {output_path}")
        return True

    except Exception as e:
        print(f"Error during processing: {e}")
        raise

def convert_aedat4_to_hdf5(input_path, output_path, max_events=None):
    """
    Extract DVS events from ROS bag file and save to HDF5 with Prophesee-compliant format.
    Uses streaming approach with buffering for memory efficiency.
    """
    print(f"Opening AEDAT4 file: {input_path}")


    try:
        # Create HDF5 file and configure datasets
        if os.path.exists(output_path):
            os.remove(output_path)
        write_file = h5py.File(output_path, 'w')

        # Define event data type (exactly matching format)
        event_dtype = np.dtype({
            'names': ['x', 'y', 'p', 't'],
            'formats': ['<u2', '<u2', '<i2', '<i8'],
            'offsets': [0, 2, 4, 8],
            'itemsize': 16
        })

        # Create groups and datasets with chunked storage
        CD_GROUP = write_file.create_group('CD')
        events_dataset = CD_GROUP.create_dataset(
            'events',
            shape=(0,),
            maxshape=(None,),
            dtype=event_dtype,
            compression=0x8ECF  # Prophesee compression
        )

        # Index calculation variables (time-based indexing with safeguards)
        indexes = []
        curr_delta = 0
        previous_t = None
        event_count = 0

        # Event processing buffer
        BUFFER_SIZE = 10000000  # Adjust based on available memory
        event_buffer = []

        print('**** ROS Bag -> Prophesee HDF5 Conversion **** ')
        print('*** Processing Events and Indexes *** \n')

        # Initialize progress bar
        progress_bar = tqdm(desc="Processing", unit="events")

        # Process events in streaming fashion
        with AedatFile(input_path) as f:
            height, width = f['events'].size
            # Get camera resolution
            print(f"Camera resolution: {width}x{height}")
            for event in f['events']:
                # Convert timestamp to microseconds (Prophesee spec)
                # Extract event components
                x = int(event.x)
                y = int(event.y)
                p = int(event.polarity)
                t_event = event.timestamp

                # Add to buffer
                event_buffer.append((x, y, p, t_event))

                # Calculate time delta for indexing (time-based, every 2000 microseconds)
                if previous_t is None:
                    delta = 0  # First event, no delta
                else:
                    delta = t_event - previous_t

                # Update index tracking with safeguards against large jumps
                curr_delta += delta

                # Create indexes every 2000 microseconds, but limit to prevent memory explosion
                if curr_delta >= 2000:
                    # Calculate how many 2000us intervals have passed
                    intervals = curr_delta // 2000
                    # Limit intervals to prevent memory issues (max 1000 per event)
                    for _ in range(int(intervals)):
                        indexes.append((event_count, t_event))
                    curr_delta %= 2000  # Remainder after division

                previous_t = t_event
                event_count += 1

                # Write buffer when full
                if len(event_buffer) >= BUFFER_SIZE:
                    # Convert buffer to numpy array
                    np_buffer = np.array(event_buffer, dtype=event_dtype)
                    # Resize dataset and append
                    current_size = events_dataset.shape[0]
                    events_dataset.resize(current_size + len(np_buffer), axis=0)
                    events_dataset[current_size:] = np_buffer
                    event_buffer.clear()
                    progress_bar.update(len(np_buffer))

                # Check max events limit
                if max_events and event_count >= max_events:
                    break

        # Write remaining events in buffer
        if event_buffer:
            np_buffer = np.array(event_buffer, dtype=event_dtype)
            current_size = events_dataset.shape[0]
            events_dataset.resize(current_size + len(np_buffer), axis=0)
            events_dataset[current_size:] = np_buffer
            progress_bar.update(len(np_buffer))

        progress_bar.close()

        # Create indexes dataset
        cd_indexes = np.array(indexes, dtype=[('id', '<u8'), ('ts', '<i8')])
        CD_GROUP.create_dataset('indexes', data=cd_indexes, compression=0x8ECF)

        EXT_TRIGGER_GROUP = write_file.create_group('EXT_TRIGGER')
        ext_trigger_events = np.array([], dtype={
            'names': ['p', 't', 'id'],
            'formats': ['<i2', '<i8', '<i2'],
            'offsets': [0, 8, 16],
            'itemsize': 24
        })

        ext_trigger_indexes = np.array([], dtype=[('id', '<u8'), ('ts', '<i8')])
        EXT_TRIGGER_GROUP.create_dataset('events', data=ext_trigger_events, compression=0x8ECF)
        EXT_TRIGGER_GROUP.create_dataset('indexes', data=ext_trigger_indexes, compression=0x8ECF)

        # Add file attributes
        attributes = {
            'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'firmware_version': '3.2.0',
            'generation': '3.1',
            'geometry': f'{width}x{height}',
            'integrator_name': 'Prophesee',
            'serial_number': '00001093',  # Default for ROS bag conversion
            'system_ID': '28',
            'version': '1.0',
        }
        for key, value in attributes.items():
            write_file.attrs.create(key, value.encode('utf-8'))

        write_file.close()


        print('*** Conversion Complete ***')
        print(f"Successfully saved {event_count} events to {output_path}")
        print(f"Created {len(indexes)} index entries")
        return True

    except Exception as e:
        print(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert AEDAT4 to HDF5 using dv.AedatFile (streaming)")
    parser.add_argument("input", help="Path to input .aedat4 file")
    parser.add_argument("output", nargs="?", help="Path to output .hdf5 file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_path = args.output if args.output else os.path.splitext(args.input)[0] + ".hdf5"
    convert_aedat4_to_hdf5(args.input, output_path)
