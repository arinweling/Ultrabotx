# Genesis Ultrasound Simulation

Teleoperable Franka Panda robot arm in [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world) with a live ultrasound feed from [raysim](https://github.com/isaac-for-healthcare/i4h-sensor-simulation) — an NVIDIA OptiX-based GPU raytracing ultrasound simulator — scanning an i4h ABDPhantom abdominal body model.



https://github.com/user-attachments/assets/798cf530-abfe-49bd-858c-fa269113867a


https://github.com/user-attachments/assets/56157d4c-f88d-4a47-b10c-1672e9d224d1


## What it does

- Loads the **i4h ABDPhantom** (full abdominal body with 14 organ meshes) as a fixed rigid body in a Genesis physics scene
- Attaches a **Clarius ultrasound probe** (box proxy) rigidly to the Franka end-effector
- Renders a live **B-mode ultrasound image** via raysim every 10 physics steps — the image updates as you move the probe over the phantom
- Full **keyboard teleoperation** of the robot arm using Genesis's IK controller

## Repository structure

```
genesis_testing/
├── src/
│   ├── test.py                  # Main scene: Franka + phantom + live ultrasound
│   ├── download_phantom.py      # One-time download of i4h ABDPhantom assets
│   └── organ_phantom_scene.py   # Standalone scene with individual organ meshes
├── docs/
│   └── superpowers/             # Design docs and implementation plans
├── genesis-world/               # Submodule: Genesis physics engine
├── i4h-asset-catalog/           # Submodule: i4h asset download helpers
└── i4h-sensor-simulation/       # Submodule: raysim ultrasound simulator
```

## Requirements

- Linux (tested on Ubuntu 22.04)
- NVIDIA GPU with driver ≥ 555 (tested: RTX 3060, driver 595)
- CUDA ≥ 12.6 toolkit installed (tested: 13.2)
- CMake ≥ 3.24 (`conda install -c conda-forge cmake`)
- Conda

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/<your-username>/genesis_testing.git
cd genesis_testing
```

### 2. Download assets

The robot and phantom assets are hosted on Hugging Face. Download and extract them into the `assets/` folder at the repo root:

```bash
# Using the Hugging Face CLI (pip install huggingface_hub if needed)
huggingface-cli download yunkao/SonoGym_assets_models \
  --repo-type dataset \
  --local-dir assets

# Or download manually from:
# https://huggingface.co/datasets/yunkao/SonoGym_assets_models/tree/main
# and place the contents into the assets/ directory
```

The `assets/` directory is gitignored — its contents are not committed to the repo.

### 3. Run the setup script

```bash
bash setup.sh
```

The script handles everything automatically:
- Creates the `genesis` conda environment (Python 3.11)
- Installs `nvcc` and `ninja` via conda
- Installs Genesis, i4h asset helper, PyMUST, and OpenCV
- Builds raysim from source using CMake + CUDA
- Downloads the ABDPhantom assets (~500 MB) to `~/.cache/i4h-assets/`

> **Note:** The raysim build step takes 3–5 minutes on first run. Subsequent runs of `setup.sh` are fast — all steps are skipped if already complete.

<details>
<summary>Manual setup (if you prefer step-by-step)</summary>

```bash
# Conda env
conda create -n genesis python=3.11 libstdcxx-ng -c conda-forge -y

# Build tools
conda install -n genesis -c nvidia cuda-nvcc=13.2.78 -y
conda install -n genesis -c conda-forge ninja -y

# Python packages
conda run -n genesis pip install -e genesis-world
conda run -n genesis pip install -e i4h-asset-catalog
conda run -n genesis pip install pymust opencv-python

# Build raysim
cd i4h-sensor-simulation/ultrasound-raytracing
conda run -n genesis cmake \
  -DASSIMP_WARNINGS_AS_ERRORS=OFF \
  -DPYTHON_EXECUTABLE=$(conda run -n genesis which python) \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_EXAMPLES=OFF \
  -B build-release
conda run -n genesis cmake --build build-release -j$(nproc)
cd ../..

# Download assets
conda run -n genesis python src/download_phantom.py
```

</details>

## Running

```bash
cd genesis-world
conda run -n genesis python ../src/test.py
```

Two windows open:
- **Genesis viewer** — 3D scene with the Franka arm, phantom body, and probe
- **Ultrasound (raysim)** — live B-mode image from the probe's perspective

### Keyboard controls

| Key | Action |
|-----|--------|
| `↑` / `↓` | Sweep probe along its face |
| `←` / `→` | Slide probe across its face |
| `m` / `n` | Press / release probe normal to the face |
| `j` / `k` | Rotate probe footprint |
| `q` / `e` | Rock probe side-to-side |
| `t` / `g` | Tilt probe heel-toe |
| `space` (hold) | Close gripper |
| `\` | Reset robot pose |
| `esc` | Quit |

### Getting ultrasound signal

The robot starts at `z = 0.40 m` above the phantom. Press `m` repeatedly to lower the probe toward the body. The organ meshes are roughly in the range `z = 0.0–0.15 m` world. The B-mode window updates when the probe face enters the body volume.

## How it works

### Coordinate synchronization

raysim loads organ OBJ meshes at startup in millimetre coordinates (the phantom's local frame). The phantom is spawned **fixed** in Genesis at world position `(0.5, 0.0, 0.0)`. Each simulation step, the probe face position is computed from the robot's hand link and converted to the phantom's mm frame:

```python
pos_mm = (probe_face_world_m - phantom_origin_world_m) * 1000.0
```

This keeps raysim's static mesh world in sync with Genesis without rebuilding the OptiX scene each frame.

### GPU memory

Genesis (Taichi) and raysim (OptiX) share the same GPU. Taichi is capped at 2 GB before Genesis initialises so raysim's BVH has room:

```python
import taichi as ti
ti.init(arch=ti.cuda, device_memory_GB=2)
gs.init(backend=gs.gpu)
```

## Submodules

| Submodule | Purpose | Upstream |
|-----------|---------|----------|
| `genesis-world` | Physics simulation engine | [Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) |
| `i4h-asset-catalog` | Asset download helpers | [isaac-for-healthcare/i4h-asset-catalog](https://github.com/isaac-for-healthcare/i4h-asset-catalog) |
| `i4h-sensor-simulation` | raysim OptiX ultrasound simulator | [isaac-for-healthcare/i4h-sensor-simulation](https://github.com/isaac-for-healthcare/i4h-sensor-simulation) |
| `i4h-workflows` | i4H reference workflows | [isaac-for-healthcare/i4h-workflows](https://github.com/isaac-for-healthcare/i4h-workflows) |

## Known issues

- The raysim probe orientation may need calibration depending on the robot's end-effector pose — adjust the Euler angle mapping in `_quat_wxyz_to_euler_xyz` if the B-mode image appears rotated
- `Lungs.obj`, `Skin.obj`, and `Spleen.obj` are missing from the v0.2.0 ABDPhantom asset release; the simulator runs without them
