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
    software-properties-common

# Create working directory
RUN mkdir -p /VibES
WORKDIR /VibES

# Install dependencies
RUN apt-get install -y libfftw3-dev

# Install VibES
COPY cmake /VibES/cmake
COPY include /VibES/include
COPY demo_old /VibES/demo
COPY CMakeLists.txt /VibES/CMakeLists.txt
COPY camera /VibES/camera
COPY tests /VibES/tests

# Build VibES
RUN mkdir -p /VibES/build
WORKDIR /VibES/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4

# Run demo_old
CMD ["./helix_vis_thread"]