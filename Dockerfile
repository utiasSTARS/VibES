FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Toronto
ENV LANGUAGE=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    vim \
    apt-utils \
    libopencv-dev libboost-all-dev libusb-1.0-0-dev libprotobuf-dev protobuf-compiler \
    libhdf5-dev hdf5-tools libglew-dev libglfw3-dev libcanberra-gtk-module ffmpeg \
    software-properties-common libfftw3-dev libeigen3-dev libgflags2.2 libgflags-dev libgoogle-glog-dev

# Create working directory
RUN mkdir -p /VibES
WORKDIR /VibES

# Install dependencies
RUN git clone https://github.com/prophesee-ai/openeb.git --branch 5.1.1
RUN cd openeb &&  \
    mkdir build && cd build &&  \
    cmake .. -DCOMPILE_PYTHON3_BINDINGS=OFF &&  \
    cmake --build . --config Release -- -j 4 && \
    cmake --build . --target install

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV HDF5_PLUGIN_PATH="/usr/local/hdf5/lib/plugin:${HDF5_PLUGIN_PATH}"

# Install VibES
COPY cmake /VibES/cmake
COPY include /VibES/include
COPY demo /VibES/demo
COPY thirdparty /VibES/thirdparty
COPY CMakeLists.txt /VibES/CMakeLists.txt

# Build VibES
RUN mkdir -p /VibES/build
WORKDIR /VibES/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4

# Set entrypoint
ENTRYPOINT ["./compensation"]
CMD [ "-c", "/datasets/intrinsics.json", "-i", "/datasets/data/logo_vib.hdf5", "-o", "../results/logo/harmeda/", "--tracker-x", "557", "--tracker-y", "242"]
