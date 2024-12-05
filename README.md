# HARMEDA

## Run using Docker

Make sure to have [Docker](https://docs.docker.com/engine/install/) installed on your machine.

To build the Docker image, run the following command in the root directory of the project:

```bash
docker build -t harmeda .
```

To run the Docker container, run the following command:

```bash
docker run --net=host --rm --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it harmeda
```

To run different scripts pass the following arguments to the `docker run` command:

- ./helix_vis_thread_vis
- ./helix_vis_thread

To use the camera or a `aedat4` file as input, pass the following arguments to the `docker run` command:

- ./helix_vis_thread_vis -camera
- ./helix_vis_thread /path/to/aedat4/file.aedat4

