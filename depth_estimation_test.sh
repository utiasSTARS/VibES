#!/bin/sh

prefix="/home/viciopoli/datasets/event_harmeda/harmeda_dataset"
mkdir -p "$prefix/results/depth_estimation"

echo "Log file: $log_file"

log_file="$prefix/results/depth_estimation/run_depth_estimation_31.log"
echo "Log file: $log_file"
echo "Running depth estimation on harmeda_30hz_sm_stagger_31..." > "$log_file"
./cmake-build-release/depth_estimation_iter -i "$prefix/harmeda_30hz_sm_stagger_31.hdf5" --trackers-x 105 230 455 383 --trackers-y 190 160 213 247 --front-trackers 2 > "$log_file" 2>&1

log_file="$prefix/results/depth_estimation/run_depth_estimation_21.log"
echo "Log file: $log_file"
echo "Running depth estimation on harmeda_40hz_sm_stagger...\n" > "$log_file"
./cmake-build-release/depth_estimation_iter -i "$prefix/harmeda_40hz_sm_stagger.hdf5" --trackers-x 110 235 419 517 --trackers-y 190 163 232 239 --front-trackers 2  > "$log_file" 2>&1

log_file="$prefix/results/depth_estimation/run_depth_estimation_32.log"
echo "Log file: $log_file"
echo "Running depth estimation on harmeda_30hz_sm_stagger_32...\n" > "$log_file"
./cmake-build-release/depth_estimation_iter -i "$prefix/harmeda_30hz_sm_stagger_32.hdf5" --trackers-x 105 230 419 517 --trackers-y 190 160 232 239 --front-trackers 2 > "$log_file" 2>&1

# 109 175 229 409 497 420 --trackers-y 185 240 161 208 219 271
echo "Done!"