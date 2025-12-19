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

### 1. System Dependencies

Ensure FFTW3 and the Metavision SDK are installed.

```bash
sudo apt-get update
sudo apt-get install -y libfftw3-dev
```

### 2. Local Build (C++20)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 3. Docker (Recommended)

Build and run with X11 forwarding for visualization:

```bash
docker build -t vibes .
docker run --net=host --rm --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -it vibes ./helix_vis_thread_vis
```

---

## 💻 Usage

### Core Executables

* `compensation` – Real-time vibration estimation and motion rectification
* `multi_tracker` – Multi-feature tracking using the HASTE backend
* `depth_estimation` – Relative depth from vibration-induced parallax
* `frequency_estimation` – Offline vibration frequency analysis

### Examples

**Live camera with visualization**

```bash
./build/helix_vis_thread_vis camera
```

**Process a recording**

```bash
./build/helix_vis_thread /path/to/recording.aedat4
```

---

## 📊 Evaluation & Metrics

To enable quantitative evaluation (Entropy, NIQE, edge metrics):

```bash
cmake .. -DBUILD_METRICS=ON
make -j
```

This builds:

* `edges_metrics`
* `variance_grad_metrics`

for systematic performance analysis.

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
