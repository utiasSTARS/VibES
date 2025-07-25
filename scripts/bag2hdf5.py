#!/usr/bin/env python3
"""
ROS bag to Prophesee-compliant HDF5 converter
Converts DVS events from ROS bag files to HDF5 format matching Prophesee specification
"""

import sys
import rosbag
import h5py
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime
import tempfile
import os


def get_camera_info(bag_path, camera_info_topic='/camera/infra1/camera_info'):
    """Extract camera resolution from camera_info messages."""
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[camera_info_topic]):
                return msg.width, msg.height
    except:
        print(f"Warning: Could not read camera info from {camera_info_topic}, using default 640x480")
        return 640, 480


def extract_events_streaming(bag_path, output_path, topic_name='/dvs/events_left', 
                           camera_info_topic='/camera/infra1/camera_info', max_events=None):
    """
    Extract DVS events from ROS bag file and save to HDF5 with Prophesee-compliant format.
    Uses streaming approach with buffering for memory efficiency.
    """
    print(f"Opening bag file: {bag_path}")
    
    # Get camera resolution
    width, height = get_camera_info(bag_path, camera_info_topic)
    print(f"Camera resolution: {width}x{height}")
    
    # Get total number of messages for progress bar
    with rosbag.Bag(bag_path, 'r') as bag:
        bag_info = bag.get_type_and_topic_info()
        if topic_name not in bag_info.topics:
            print(f"Error: Topic {topic_name} not found in bag file!")
            available_topics = list(bag_info.topics.keys())
            print(f"Available topics: {available_topics}")
            return False
        
        total_messages = bag_info.topics[topic_name].message_count
        bag_start_time = bag.get_start_time()
    
    print(f"Total messages to process: {total_messages}")
    
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

        first_event_time = -1  # Track first event time for indexing
        
        # Process events in streaming fashion
        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                for event in msg.events:
                    # Convert timestamp to microseconds (Prophesee spec)
                    timestamp_ns = int(event.ts.to_nsec())
                    if first_event_time == -1:
                        first_event_time = timestamp_ns
                    
                    # Extract event components
                    x = int(event.x)
                    y = int(event.y)
                    p = int(event.polarity)
                    t_event = timestamp_ns-first_event_time
                    
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
            'Date': datetime.fromtimestamp(bag_start_time).strftime('%Y-%m-%d %H:%M:%S'),
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


def verify_hdf5_file(hdf5_path, num_samples=5):
    """Verify the HDF5 file matches Prophesee format."""
    print(f"\nVerifying HDF5 file: {hdf5_path}")
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Check structure
            print("File structure:")
            print(f"  Groups: {list(f.keys())}")
            
            # Print attributes
            print("\nFile attributes:")
            for key in f.attrs.keys():
                value = f.attrs[key]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                print(f"  {key}: {value}")
            
            # Check CD group
            if 'CD' in f:
                cd_group = f['CD']
                print(f"\nCD group datasets: {list(cd_group.keys())}")
                
                if 'events' in cd_group:
                    events = cd_group['events']
                    print(f"  Events shape: {events.shape}")
                    print(f"  Events dtype: {events.dtype}")
                    
                    if len(events) > 0:
                        print("\nSample events:")
                        for i in range(min(num_samples, len(events))):
                            e = events[i]
                            print(f"    Event {i}: x={e['x']}, y={e['y']}, p={e['p']}, t={e['t']} us")
                
                if 'indexes' in cd_group:
                    indexes = cd_group['indexes']
                    print(f"\nIndexes shape: {indexes.shape}")
                    print(f"Indexes dtype: {indexes.dtype}")
                    if len(indexes) > 0:
                        print(f"First index: id={indexes[0]['id']}, ts={indexes[0]['ts']}")
                        print(f"Last index: id={indexes[-1]['id']}, ts={indexes[-1]['ts']}")
            
            # Check EXT_TRIGGER group
            if 'EXT_TRIGGER' in f:
                ext_group = f['EXT_TRIGGER']
                print(f"\nEXT_TRIGGER group datasets: {list(ext_group.keys())}")
                if 'events' in ext_group:
                    print(f"  EXT_TRIGGER events shape: {ext_group['events'].shape}")
                if 'indexes' in ext_group:
                    print(f"  EXT_TRIGGER indexes shape: {ext_group['indexes'].shape}")
            
            print("\nHDF5 file verification successful!")
                
    except Exception as e:
        print(f"Error verifying HDF5 file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Extract DVS events from ROS bag to Prophesee-compliant HDF5')
    parser.add_argument('bag_file', help='Input ROS bag file path')
    parser.add_argument('output_file', help='Output HDF5 file path')
    parser.add_argument('--topic', default='/dvs/events_left', 
                       help='DVS events topic name (default: /dvs/events_left)')
    parser.add_argument('--camera-info', default='/camera/infra1/camera_info',
                       help='Camera info topic (default: /camera/infra1/camera_info)')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maximum number of events to process (default: all)')
    parser.add_argument('--verify', action='store_true', 
                       help='Verify the output HDF5 file after creation')
    
    args = parser.parse_args()
    
    try:
        success = extract_events_streaming(args.bag_file, args.output_file, args.topic, 
                                         args.camera_info, args.max_events)
        
        if success and args.verify:
            verify_hdf5_file(args.output_file)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())