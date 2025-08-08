#!/bin/sh

echo "Running depth estimation on harmeda_40hz_sm_stagger...\n"
./cmake-build-release/depth_estimation_iter -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/harmeda_40hz_sm_stagger.hdf5 --trackers-x 110 235 419 517 --trackers-y 190 163 232 239 --front-trackers 2

./cmake-build-release/depth_estimation_iter -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/harmeda_30hz_sm_stagger_32.hdf5 --trackers-x 105 230 419 517 --trackers-y 190 160 232 239 --front-trackers 2

./cmake-build-release/depth_estimation_iter -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/harmeda_30hz_sm_stagger_31.hdf5 --trackers-x 105 230 455 383 --trackers-y 190 160 213 247 --front-trackers 2
# 109 175 229 409 497 420 --trackers-y 185 240 161 208 219 271
echo