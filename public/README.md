<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache 2.0 License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
<h2 align="center">VibES: Induced Vibration for Persistent Event-based Sensing</h2>

  <p align="center">
C++ library for applying motion compensation to induced vibrational motion.
    <br />
    <a href="https://papers.starslab.ca/vibes">Webpage</a>
    ·
    <a href="https://github.com/utiasSTARS/VibES/issues">Report Bug</a>
    ·
    <a href="https://github.com/utiasSTARS/VibES/issues">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
    </li>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li><a href="#structure">Structure</a></li>
    <li><a href="#usage">Usage</a></li>
        <ul><a href="#requirements">Requirements</a></ul>
        <ul><a href="#installation">Installation</a></ul>
        <ul><a href="#docker">Docker</a></ul>
        <ul><a href="#interface">Interface</a></ul>
        <ul><a href="#demo">Demo</a></ul>
    <li><a href="datasets">Datasets</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About

<div align="center">
  <a href="https://github.com/jpl-x/x_multi_agent">
    <img src="images/ctio.gif" alt="demo" >
  </a>
</div>

This is the code for the paper **VibES: Induced Vibration for Persistent Event-based Sensing**
([PDF](https://)) by [Vincenzo Polizzi](https://github.com/viciopoli01/),
[Stephen Yang](https://),
[Quentin Clark](https://),
[Jonathan Kelly](https://starslab.ca/people/prof-jonathan-kelly/),
[Igor Gilitschenski](https://)
and [David Lindell](http://).
For an overview of our method, check out our [webpage](https://papers.starslab.ca/vibes).

If you use any of this code, please cite the following publication:

```bibtex
@ARTICLE{Polizzi25arxiv,
  
}
```

## Abstract

<p align="center">
Event cameras are a bio-inspired class of sensors that asynchronously measure per-pixel intensity changes. Under fixed illumination conditions in static or low-motion scenes, rigidly mounted event cameras are unable to generate any events, becoming unsuitable for most computer vision tasks.
To address this limitation, recent work has investigated motion-induced event stimulation that often requires complex hardware or additional optical components.
In contrast, we introduce a lightweight approach to sustain persistent event generation by employing a simple rotating unbalanced mass to induce periodic vibrational motion. This is combined with a motion-compensation pipeline that removes the injected motion and yields clean, motion-corrected events for downstream perception tasks.
We demonstrate our approach with a hardware prototype and evaluate it on real-world captured datasets.
Our method reliably recovers motion parameters and improves both image reconstruction and edge detection over event-based sensing without motion induction.
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Structure

- `include`: contains all the header files for the library
- `third_party`: contains Haste
- `demos`: contains the files to run the demos

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

This code was tested on `Ubuntu 22.04` using the `Metavision SDK 4.6.2`.

### Requirements

The following libraries are needed for installing the VibES library:

- [Metavision SDK 4.6.2](https://) if not available [OpenEB](https://) can be used
- [Boost 1.74](https://)
- [Eigen]()
- [gflags]()
- [glog]()

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

To install the library you can download the .deb package for your architecture here, or build it by yourself by doing:

```bash
$ git clone git@github.com:jpl-x/x_multi_agent.git
$ mkdir build && cd build
$ cmake ..
$ make package
$ sudo dpkg -i x_1.2.3_$(dpkg --print-architecture).deb
```

To enable/disable some features, set to ON/OFF the CMake options (_Note that visualization affect thte runtime
operations_):

- `VISUALIZATION`: if `ON`(by default) allows the visualization of accumulated images
- `FANCY_VISUALIZATION`: if `ON` allows to see the difference between the raw incoming events and the compensated ones
  in the accumulated images
- `STORE`: if `ON` stores accumulated frames

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Docker

Make sure to have installed Docker, you don't need any other dependency here!
The Docker container has been built with `FANCY_VISUALIZATION` flag `ON`, it contains the executable to run the
visualizatin demo, from file or from the real camera.

Execute the following command to run the image:

```bash
docker run --net=host --rm --privileged -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -v $(pwd):/data --name vibes -it viciopoli/vibes:latest  -c PATH_TO_INTRINSICS -i PATH_TO_FILE
```

In the case `PATH_TO_INTRINSICS` is not specified the incoming events are not distorted.
In the case `PATH_TO_FILE` is not specified the code looks for available cameras.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Interface

The x Library accepts as input various sensors' measurements, that are then fused together in the IEKF.
To use the library in your project add in your `CMakeLists.txt`:

```Cmake
find_package(x 1.2.3 REQUIRED)

# ...

include_directories(
        OTHER_INCLUDES
        ${x_INCLUDE_DIRS}
)

# ...

target_link_libraries(${PROJECT_NAME}
        OTHER_LIBRARIES
        ${x_LIBRARIES}
)
```

Usage example:

- Initialization

```c++
#include <x/vio.h>
#include <ctime>

VIO vio;

const auto params = vio.loadParamsFromYaml("PATH_TO_A_YAML_FILE");

vio.setUp(params);

time_t now = time(0);
vio_.initAtTime((double)now);
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Demo

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Datasets

The datasets we collected are available [here](https://).

The scenes with the induced motion are named with `_vib` the others have no motion induced.
Simulated data are labeled with `sim_`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

This code can be run with proprietary libraries from Prophesee as well as with their opensource library OpenEB.
All the experiments have been conducted with the proprietary one.

Distributed under the Apache 2.0 License. See [LICENSE](LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

A particular thanks goes also to [Cedric Le Gentil](https://www.linkedin.com/in/) for brainstorming.

Readme template layout from [Best-README-Template](https://github.com/othneildrew/Best-README-Template).
<p align="right">(<a href="#readme-top">back to top</a>)</p>


[contributors-shield]: https://img.shields.io/github/contributors/jpl-x/x_multi_agent.svg?style=for-the-badge

[contributors-url]: https://github.com/jpl-x/x_multi_agent/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/jpl-x/x_multi_agent.svg?style=for-the-badge

[forks-url]: https://github.com/jpl-x/x_multi_agent/network/members

[stars-shield]: https://img.shields.io/github/stars/jpl-x/x_multi_agent.svg?style=for-the-badge

[stars-url]: https://github.com/jpl-x/x_multi_agent/stargazers

[issues-shield]: https://img.shields.io/github/issues/jpl-x/x_multi_agent.svg?style=for-the-badge

[issues-url]: https://github.com/jpl-x/x_multi_agent/issues

[license-shield]: https://img.shields.io/github/license/jpl-x/x_multi_agent.svg?style=for-the-badge

[license-url]: https://github.com/viciopoli01/jpl-x/tree/build_and_play/LICENSE

[product-screenshot]: images/demo.gif
