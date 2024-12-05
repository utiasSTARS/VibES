# HARMEDA

## Install Locally

### Dependencies

Run the following command to install the dependencies:

```bash
./scripts/install_deps.sh
```

### Build

Run the following command to build the project:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

### Run

To run the project, run the following command:

```bash
./build/helix_vis_thread
```

See [Run Arguments](#run-arguments) for more options.

## Run using Docker

Make sure to have [Docker](https://docs.docker.com/engine/install/) installed on your machine.

To build the Docker image, run the following command in the root directory of the project:

```bash
docker build -t harmeda .
```

To run the Docker container, run the following command:

```bash
docker run --net=host --rm --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it harmeda ./helix_vis_thread_vis
```

See [Run Arguments](#run-arguments) for more options.

# Run Arguments

To run different scripts in the project, run the following commands:

- `./helix_vis_thread_vis`
- `./helix_vis_thread`

To use the camera or a `aedat4` file as input, run the following commands:

- `./helix_vis_thread_vis camera`
- `./helix_vis_thread /path/to/aedat4/file.aedat4`

