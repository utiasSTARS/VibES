#!/bin/bash

# List of datasets to process
datasets=("amiev" "pattern" "logo" "checkerpattern")

# Base paths
base_input_path="results"
executable="./build/variance_grad_metrics"

# Time window parameter
time_window="10000"

# Function to run variance_grad_metrics
run_variance_grad() {
    local dataset=$1
    local input_file=$2
    local output_dir=$3

    echo "Processing: $dataset - $input_file -> $output_dir"

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Run the command
    $executable -i "$input_file" -o "$output_dir" --time-window-us $time_window

    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed: $dataset - $output_dir"
    else
        echo "✗ Failed to process: $dataset - $output_dir"
    fi
    echo ""
}

# Process each dataset
for dataset in "${datasets[@]}"; do
    echo "=== Processing dataset: $dataset ==="

    # 1. EV undistorted
    input_file="$base_input_path/$dataset/ev/compensated_events/undistorted.hdf5"
    output_dir="$base_input_path/$dataset/ev/und/"
    run_variance_grad "$dataset" "$input_file" "$output_dir"

    # 2. VibES no compensation
    input_file="$base_input_path/$dataset/VibES/compensated_events/VibES_nocomp.hdf5"
    output_dir="$base_input_path/$dataset/VibES/nocomp/"
    run_variance_grad "$dataset" "$input_file" "$output_dir"

    # 3. VibES compensated
    input_file="$base_input_path/$dataset/VibES/compensated_events/VibES_compensated.hdf5"
    output_dir="$base_input_path/$dataset/VibES/comp/"
    run_variance_grad "$dataset" "$input_file" "$output_dir"

    echo "=== Completed dataset: $dataset ==="
    echo ""
done

echo "All datasets processed!"