#!/bin/sh

# prefix
prefix="/home/viciopoli/datasets/event_harmeda/harmeda_dataset/results"
echo "Running image extraction at 100 Hz (10 ms)"
#
./cmake-build-release/compensation_store_imgs --time-window-us 33333 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/amiev.hdf5 -o "$prefix/amiev/ev/" --nocompensation
./cmake-build-release/compensation_store_imgs --time-window-us 33333 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/amiev_r.hdf5 -o "$prefix/amiev/harmeda/"  --tracker-x 537 --tracker-y 186
#
#python entropy.py ../../results/amiev/ bin 10000
#
./cmake-build-release/compensation_store_imgs --time-window-us 33333 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/logo_baseline.hdf5 -c /home/viciopoli/datasets/event_harmeda/intrinsics.json -o "$prefix/logo/ev/" --nocompensation
./cmake-build-release/compensation_store_imgs --time-window-us 33333 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/logo_harmeda.hdf5 -c /home/viciopoli/datasets/event_harmeda/intrinsics.json -o "$prefix/logo/harmeda/" --tracker-x 557 --tracker-y 242
#
#python entropy.py ../$prefix/logo/ bin 10000
#
./cmake-build-release/compensation_store_imgs --time-window-us 33333 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/checkerpattern_baseline.hdf5 -c /home/viciopoli/datasets/event_harmeda/intrinsics.json -o "$prefix/checkerpattern/ev/" --nocompensation
./cmake-build-release/compensation_store_imgs --time-window-us 33333 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/checkerpattern_harmeda.hdf5 -c /home/viciopoli/datasets/event_harmeda/intrinsics.json -o "$prefix/checkerpattern/harmeda/" --tracker-x 335 --tracker-y 193
#
./cmake-build-release/compensation_store_imgs --time-window-us 33333 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/pattern_baseline.hdf5 -c /home/viciopoli/datasets/event_harmeda/intrinsics.json -o "$prefix/pattern/ev/" --nocompensation
./cmake-build-release/compensation_store_imgs --time-window-us 33333 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/pattern_harmeda.hdf5 -c /home/viciopoli/datasets/event_harmeda/intrinsics.json -o "$prefix/pattern/harmeda/" --tracker-x 412 --tracker-y 133
#
#python entropy.py ../$prefix/texture/ bin 10000
#
./cmake-build-release/compensation_store_imgs --time-window-us 33333 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/texture_baseline.hdf5 -o "$prefix/sim/texture/ev/" --nocompensation
./cmake-build-release/compensation_store_imgs --time-window-us 33333 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/harmeda_10hz_sm_texture.hdf5 -o "$prefix/sim/texture/harmeda/" --tracker-x 300 --tracker-y 195
#
#python entropy.py ../$prefix/texture/ bin 10000
#
echo "Done"