#!/usr/bin/env python3
"""
ROS bag to text file converter
Converts DVS events from ROS bag files to simple text format: t x y p
"""

import sys
import rosbag
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime
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


def extract_events_to_txt(bag_path, output_path, topic_name='/dvs/events_left', 
                         camera_info_topic='/camera/infra1/camera_info', max_events=None):
    """
    Extract DVS events from ROS bag file and save to text file.
    Format: t x y p (one event per line)
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
            # Write header comment
            txt_file.write(f"# DVS events extracted from {os.path.basename(bag_path)}\n")
            txt_file.write(f"# Camera resolution: {width}x{height}\n")
            txt_file.write(f"# Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            txt_file.write(f"# Format: t x y p (timestamp_us x y polarity)\n")
            txt_file.write("#\n")
            
            # Process events in streaming fashion
            with rosbag.Bag(bag_path, 'r') as bag:
                for topic, msg, t in bag.read_messages(topics=[topic_name]):
                    for event in msg.events:
                        # Convert timestamp to microseconds
                        timestamp_us = int(event.ts.to_nsec() / 1000)
                        
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


def verify_txt_file(txt_path, num_samples=5):
    """Verify the text file and show sample events."""
    print(f"\nVerifying text file: {txt_path}")
    
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        # Count header lines (starting with #)
        header_lines = 0
        for line in lines:
            if line.startswith('#'):
                header_lines += 1
            else:
                break
        
        data_lines = len(lines) - header_lines
        print(f"Total lines: {len(lines)}")
        print(f"Header lines: {header_lines}")
        print(f"Data lines: {data_lines}")
        
        # Show header
        print("\nHeader:")
        for i in range(min(header_lines, 10)):
            print(f"  {lines[i].strip()}")
        
        # Show sample data
        if data_lines > 0:
            print(f"\nSample events (first {min(num_samples, data_lines)}):")
            for i in range(header_lines, min(header_lines + num_samples, len(lines))):
                parts = lines[i].strip().split()
                if len(parts) == 4:
                    t, x, y, p = parts
                    print(f"  Event {i-header_lines+1}: t={t} us, x={x}, y={y}, p={p}")
                else:
                    print(f"  Line {i+1}: {lines[i].strip()} (unexpected format)")
            
            # Show statistics
            print(f"\nFile statistics:")
            timestamps = []
            x_coords = []
            y_coords = []
            polarities = []
            
            # Sample first 10000 lines for statistics
            sample_size = min(10000, data_lines)
            for i in range(header_lines, header_lines + sample_size):
                parts = lines[i].strip().split()
                if len(parts) == 4:
                    timestamps.append(int(parts[0]))
                    x_coords.append(int(parts[1]))
                    y_coords.append(int(parts[2]))
                    polarities.append(int(parts[3]))
            
            if timestamps:
                print(f"  Time range: {min(timestamps)} - {max(timestamps)} us")
                print(f"  Duration: {(max(timestamps) - min(timestamps)) / 1e6:.3f} seconds")
                print(f"  X range: {min(x_coords)} - {max(x_coords)}")
                print(f"  Y range: {min(y_coords)} - {max(y_coords)}")
                print(f"  Polarities: {set(polarities)}")
        
        print("\nText file verification successful!")
                
    except Exception as e:
        print(f"Error verifying text file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Extract DVS events from ROS bag to text file')
    parser.add_argument('bag_file', help='Input ROS bag file path')
    parser.add_argument('output_file', help='Output text file path')
    parser.add_argument('--topic', default='/dvs/events_left', 
                       help='DVS events topic name (default: /dvs/events_left)')
    parser.add_argument('--camera-info', default='/camera/infra1/camera_info',
                       help='Camera info topic (default: /camera/infra1/camera_info)')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maximum number of events to process (default: all)')
    parser.add_argument('--verify', action='store_true', 
                       help='Verify the output text file after creation')
    
    args = parser.parse_args()
    
    try:
        success = extract_events_to_txt(args.bag_file, args.output_file, args.topic, 
                                       args.camera_info, args.max_events)
        
        if success and args.verify:
            verify_txt_file(args.output_file)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())