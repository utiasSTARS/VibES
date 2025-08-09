#!/bin/sh

echo "Running image extraction at 100 Hz (10 ms)"
#
./cmake-build-release/compensation_store_imgs --time-window-us 10000 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/amiev.hdf5 -o ../results/amiev/ev/ --nocompensation
./cmake-build-release/compensation_store_imgs --time-window-us 10000 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/amiev_r.hdf5 -o ../results/amiev/harmeda/  --tracker-x 537 --tracker-y 186
#
python entropy.py ../../results/amiev/ bin 10000
#
./cmake-build-release/compensation_store_imgs --time-window-us 10000 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/logo_baseline.hdf5 -c /home/viciopoli/datasets/event_harmeda/intrinsics.json -o ../results/logo/ev/ --nocompensation
./cmake-build-release/compensation_store_imgs --time-window-us 10000 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/logo_harmeda.hdf5 -c /home/viciopoli/datasets/event_harmeda/intrinsics.json -o ../results/logo/harmeda/ --tracker-x 730 --tracker-y 187
#
python entropy.py ../../results/logo/ bin 10000
#
./cmake-build-release/compensation_store_imgs --time-window-us 10000 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/texture_baseline.hdf5 -o ../results/sim/texture/ev/ --nocompensation
./cmake-build-release/compensation_store_imgs --time-window-us 10000 -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/harmeda_10hz_sm_texture.hdf5 -o ../results/sim/texture/harmeda/ --tracker-x 300 --tracker-y 195
#
python entropy.py ../../results/texture/ bin 10000
#
echo "Done"