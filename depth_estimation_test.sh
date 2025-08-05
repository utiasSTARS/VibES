#!/bin/sh

echo "Running depth estimation on harmeda_40hz_sm_stagger...\n"
./cmake-build-release/depth_estimation_iter -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/harmeda_40hz_sm_stagger.hdf5 --trackers-x 114 440 --trackers-y 194 215
./cmake-build-release/depth_estimation_iter -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/harmeda_30hz_sm_stagger_32.hdf5
--trackers-x
100
434
--trackers-y
197
214
# 109 175 229 409 497 420 --trackers-y 185 240 161 208 219 271
echo