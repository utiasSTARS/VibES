#! bin/bash

echo "Installing dv-processing... This contains OpenCV deps"
add-apt-repository ppa:inivation-ppa/inivation
apt-get update
apt-get install -y dv-processing

echo "Install fftw3"
apt-get install -y libfftw3-dev

echo "Install Open3D"
apt-get install -y libopen3d-dev
