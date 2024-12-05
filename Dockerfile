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
RUN mkdir -p /HARMEDA
WORKDIR /HARMEDA

# Install dependencies
COPY scripts/install_deps.sh /tmp/install_deps.sh
RUN chmod +x /tmp/install_deps.sh
RUN bash /tmp/install_deps.sh

# Install HARMEDA
COPY cmake /HARMEDA/cmake
COPY include /HARMEDA/include
COPY demo /HARMEDA/demo
COPY CMakeLists.txt /HARMEDA/CMakeLists.txt
COPY camera /HARMEDA/camera
COPY tests /HARMEDA/tests

# Build HARMEDA
RUN mkdir -p /HARMEDA/build
WORKDIR /HARMEDA/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4

# Run demo
CMD ["./helix_vis_thread"]