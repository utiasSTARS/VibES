# VibES: Induced Vibration for Persistent Event-Based Sensing (3DV 2025)

[![arXiv](https://img.shields.io/badge/arXiv-2508.19094-b31b1b.svg)](https://arxiv.org/abs/2508.19094)
[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://papers.starslab.ca/vibes/)

VibES is a lightweight framework that enables **event cameras to sense persistently in static or low-motion scenes** by inducing controlled mechanical vibrations. Using a simple rotating unbalanced mass, VibES generates continuous event streams and compensates for the induced motion **online and in real time**, producing clean, motion-corrected data suitable for downstream perception tasks.

> **Why VibES?** Event cameras are inherently motion-driven. VibES introduces a minimal, energy-efficient source of motion—paired with principled signal estimation—to unlock their use in otherwise static environments.

---

## 🚀 Key Features

* **Persistent Event Generation**
  Enables sensing in static scenes without mirrors, pan–tilt units, or complex optical setups.

* **Ultra-Low Power**
  Requires only **0.282 W** using a small DC motor—orders of magnitude lower than robotic actuators.

* **Online Motion Compensation**
  Estimates and removes induced vibrations in real time **without prior calibration** or knowledge of physical parameters.

* **High-Frequency Signal Estimation**
  Robust frequency detection over **30–500 rad/s** using **Non-Uniform FFT (NUFFT)** and tracking with **Iterative EKFs (IEKF)**.

* **Multi-Harmonic Support**
  Accurately models complex vibration profiles beyond a single sinusoid.

* **Proven Performance Gains**
  Improves image reconstruction (NIQE ↓ **41%**) and edge detection (contour length ↑ **87%**).

* **Broad Applicability**
  Benefits edge detection, E2VID image reconstruction, feature tracking, and relative depth estimation via vibration-induced parallax.

---

## 🧩 System Overview

1. **Mechanical Excitation**: A rotating unbalanced mass induces small-amplitude, periodic motion of the event camera.
2. **Event Stream Analysis**: The induced motion produces a persistent stream of events, even in static scenes.
3. **Frequency Estimation**: NUFFT identifies dominant vibration frequencies from non-uniformly sampled events.
4. **State Tracking**: IEKFs track phase, amplitude, and offsets online.
5. **Motion Compensation**: Estimated motion is removed to yield stabilized, task-ready event data.

---

## 🛠️ Installation

### Local Setup

#### 1. Clone the Repository

```bash
git clone git@github.com:utiasSTARS/VibES.git --branch main
cd VibES
git submodule update --init --recursive
```


#### 2. System Dependencies

  Ensure FFTW3 and the Metavision SDK or [OpenEB](https://github.com/prophesee-ai/openeb/tree/5.1.1) are installed

```bash
sudo apt-get update
sudo apt-get install -y libfftw3-dev libeigen3-dev libgflags2 libgflags-dev
```

#### 3. Build (C++20)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Docker (Recommended)

Run with X11 forwarding for visualization, mount the dataset directory:

```bash
docker docker run --net=host --rm -v /PATH_TO/vibes_dataset/:/datasets --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it viciopoli/vibes:latest 
```

If you `/bin/bash` into the container different executables can be run.
Example:

```bash
./compensation -c /datasets/intrinsics.json -i /datasets/data/logo_vib.hdf5 -o ../results/logo/ --tracker-x 557 --tracker-y 242
```

Press o to enable the overlay visualization.

---

## 💻 Usage

### Core Executables
* **`compensation`**: Real-time vibration estimation and motion rectification.
* **`multi_tracker`**: Multi-feature tracking using the HASTE backend.
* **`frequency_estimation`**: Offline vibration frequency analysis.

### Running the Compensation Tool
The primary tool is `compensation`. It requires an initial feature location (x, y) to begin tracking the vibration.

**Basic Syntax:**
```bash
./compensation -c <intrinsics.json> -i <file.hdf5> --tracker-x <x> --tracker-y <y>
```

If `-i` is not provided, the program will attempt to connect to a live camera.

---

## 📂 Dataset Download
To reproduce the results or run the examples, download the dataset from [Hugging Face](https://huggingface.co/datasets/viciopoli/VibES).

### Using Git (Recommended) 

Ensure you have git-lfs installed.

```bash
# Install Git LFS if needed
sudo apt-get install git-lfs
git lfs install

# Clone the dataset
git clone https://huggingface.co/datasets/viciopoli/VibES vibes_dataset
```

Files can also be downloaded manually from the [Hugging Face](https://huggingface.co/datasets/viciopoli/VibES).

---

## 📚 Citation

If you use VibES in your research, please cite:

```bibtex
@inproceedings{polizzi_2025_vibes,
  title     = {VibES: Induced Vibration for Persistent Event-Based Sensing},
  author    = {Polizzi, Vincenzo and Yang, Stephen and Clark, Quentin and
               Kelly, Jonathan and Gilitschenski, Igor and Lindell, David B.},
  booktitle = {International Conference on 3D Vision (3DV)},
  year      = {2025}
}
```

---

## 🤝 Acknowledgments

* **FINUFFT** – Non-uniform FFT computations
* **HASTE** – Event-based feature tracking
* **Metavision SDK** – Prophesee event camera interfacing

---

## 📎 Links

* 📄 Paper: [https://arxiv.org/abs/2508.19094](https://arxiv.org/abs/2508.19094)
* 🌐 Project Page: [https://papers.starslab.ca/vibes/](https://papers.starslab.ca/vibes/)
