#!/bin/bash

# prefix
prefix="results/"
echo "Running edges eval..."

# Function to run and log
run_with_log () {
    path="$1"
    shift
    log_file="${path%/}/run_connectivity.log"
    python3 scripts/metrics/edge_metrics.py --path "$path" "$@" > "$log_file" 2>&1
}

# Runs
run_with_log "$prefix/logo/"          --analysis continuity --resize .5 --gaussian_kernel 7 --block_size 3 --start_idx 0 --end_idx -1
run_with_log "$prefix/amiev/"         --analysis continuity --resize 1. --gaussian_kernel 3 --block_size 5 --start_idx 0 --end_idx -1
run_with_log "$prefix/checkerpattern/" --analysis continuity --resize .5 --gaussian_kernel 7 --block_size 3 --start_idx 0 --end_idx -1
run_with_log "$prefix/pattern/"       --analysis continuity --resize .5 --gaussian_kernel 7 --block_size 3 --start_idx 0 --end_idx -1
run_with_log "$prefix/sim/texture/"   --analysis continuity --resize 1. --gaussian_kernel 5 --block_size 3 --start_idx 0 --end_idx -1

#python entropy.py ../$prefix/texture/ bin 10000

echo "Done"
