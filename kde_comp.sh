#!/bin/bash

# Path to your folder containing events_*.hdf5
DATASET_DIR=${1:-"/path/to/your/dataset"}
SCRIPT_PATH=${2:-"/path/to/your/kde_script.py"}
DOWNSAMPLE_RATE=${3:-60}  # Default downsample rate

# Batch size
BATCH_SIZE=50
count=0
pids=()

for file in "$DATASET_DIR"/events_*.hdf5; do
    echo "Launching KDE for $file ..."
    python3 "$SCRIPT_PATH" "$file" --output "${file%.hdf5}_kde.pkl" --downsample "$DOWNSAMPLE_RATE" &

    # Track background job
    pids+=($!)
    ((count++))

    # Wait after every BATCH_SIZE files
    if (( count % BATCH_SIZE == 0 )); then
        echo "Waiting for batch of $BATCH_SIZE jobs to finish... (count: $count)"
        wait "${pids[@]}"
        pids=()
    fi
done

# Wait for any remaining jobs
if (( ${#pids[@]} > 0 )); then
    echo "Waiting for final batch..."
    wait "${pids[@]}"
fi

echo "All KDE computations finished ✅"