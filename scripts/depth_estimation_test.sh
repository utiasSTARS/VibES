#!/bin/sh

prefix="dataset"
mkdir -p "$prefix/results/depth_estimation"

echo "Log file: $log_file"

log_file="$prefix/results/depth_estimation/run_depth_estimation_21.log"
echo "Log file: $log_file"
echo "Running depth estimation on stagger_30hz_21..." > "$log_file"
./cmake-build-release/depth_estimation_iter -i "$prefix/stagger_30hz_21.hdf5" --trackers-x 105 230 455 383 --trackers-y 190 160 213 247 --front-trackers 2 > "$log_file" 2>&1

log_file="$prefix/results/depth_estimation/run_depth_estimation_31.log"
echo "Log file: $log_file"
echo "Running depth estimation on stagger_30hz_31..." > "$log_file"
./cmake-build-release/depth_estimation_iter -i "$prefix/stagger_30hz_31.hdf5" --trackers-x 105 230 455 383 --trackers-y 190 160 213 247 --front-trackers 2 > "$log_file" 2>&1

log_file="$prefix/results/depth_estimation/run_depth_estimation_32.log"
echo "Log file: $log_file"
echo "Running depth estimation on stagger_30hz_32...\n" > "$log_file"
./cmake-build-release/depth_estimation_iter -i "$prefix/stagger_30hz_32.hdf5" --trackers-x 105 230 419 517 --trackers-y 190 160 232 239 --front-trackers 2 > "$log_file" 2>&1

echo "Done!"