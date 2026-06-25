# CLAUDE.md — ef/ workspace

## Workspace layout

```
ef/
├── Ultrabotx/          ← MAIN PROJECT (this is the active repo)
├── EchoWorld/          ← experimental reference (CVPR 2025 echocardiography world model)
├── SonoGym/            ← experimental reference (NVIDIA IsaacLab robotic US platform)
├── NV-Segment-CTMR/    ← experimental reference (NVIDIA medical segmentation models)
├── model-zoo/          ← experimental reference (MONAI model zoo)
├── dataset/            ← local dataset cache (Isaac robot US guidance)
└── BTP2_Research_Summary.md  ← WaveX BTP-2 research notes (separate project)
```

**Only `Ultrabotx/` is the active project.** The sibling repos are cloned for reference/inspiration only — do not modify them unless explicitly asked.

---

## Ultrabotx — project overview

A teleoperable **Franka Panda robot arm** in [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world) with a live **B-mode ultrasound feed** from [raysim](https://github.com/isaac-for-healthcare/i4h-sensor-simulation) (NVIDIA OptiX GPU raytracing), scanning an i4h ABDPhantom abdominal body model.

**Goal:** robotic ultrasound simulation — a robot arm autonomously (or via teleoperation) sweeps an ultrasound probe over a phantom body and receives real-time B-mode feedback.

### Key concepts

- **Genesis** — physics simulation engine (Taichi-based, GPU). Handles robot kinematics, IK, and scene rendering.
- **raysim** — NVIDIA OptiX raytracing ultrasound simulator. Compiled as a CUDA `.so` and called from Python. Takes probe pose → returns B-mode image.
- **i4h ABDPhantom** — 14-organ abdominal phantom mesh, downloaded via i4h-asset-catalog to `~/.cache/i4h-assets/`.
- **Clarius ultrasound probe** — box-proxy geometry rigidly attached to Franka end-effector.
- **Coordinate sync** — raysim uses mm in phantom-local frame; Genesis uses metres in world frame. Conversion: `pos_mm = (probe_face_world_m - phantom_origin_world_m) * 1000.0`
- **GPU memory split** — Taichi capped at 2 GB (`ti.init(arch=ti.cuda, device_memory_GB=2)`) before Genesis init so OptiX BVH has headroom.

---

## Repository structure

```
Ultrabotx/
├── src/
│   ├── test_custom.py           # Main scene: Franka + phantom + live US (primary entry point)
│   ├── test_isaac_assets.py     # Scene using Isaac/i4h assets directly
│   ├── test_isaac_assets_cube.py
│   ├── test_custom_cube.py
│   ├── organ_phantom_scene.py   # Standalone scene with individual organ meshes
│   ├── probe_test.py            # Probe-only test
│   └── download_phantom.py      # One-time ABDPhantom asset download
├── scripts/
│   ├── segment_ct_image.py      # NVIDIA VISTA-3D API call for CT segmentation
│   └── extract_probe_mesh.py    # Probe mesh extraction utility
├── config/
│   ├── raysim_custom.toml       # raysim probe + sim + world parameters (primary config)
│   └── raysim_isaac.toml        # Alternative config for Isaac asset layout
├── assets/
│   ├── data/                    # SonoGym assets (robots, human models, surgical tools) — gitignored
│   │   ├── Robots/Franka/
│   │   ├── HumanModels/
│   │   └── SurgicalTools/
│   └── Robots/                  # Additional robot asset scripts
├── xml/franka_emika_panda/      # Custom MuJoCo MJCF with symlinked Genesis panda assets
├── output/
│   ├── nii/                     # NIfTI output files
│   └── obj/                     # OBJ mesh outputs
├── genesis-world/               # Submodule: Genesis physics engine
├── i4h-asset-catalog/           # Submodule: asset download helpers
├── i4h-sensor-simulation/       # Submodule: raysim (ultrasound-raytracing + fluoro-simulator)
├── i4h-workflows/               # Submodule: i4h reference workflows
└── setup.sh                     # One-shot environment setup
```

---

## Environment

- **Conda env:** `genesis` (Python 3.11)
- **CUDA:** 13.2 (nvcc via conda), driver ≥ 555
- **GPU:** NVIDIA (tested RTX 3060)
- **OS:** Ubuntu 22.04 / WSL2

### Running the simulation

```bash
cd Ultrabotx/genesis-world
conda run -n genesis python ../src/test_custom.py
```

Two windows open: Genesis 3D viewer + raysim B-mode display.

### Setup (first time)

```bash
cd Ultrabotx
bash setup.sh
```

Setup is idempotent — safe to re-run.

---

## Submodules

| Submodule | Purpose |
|-----------|---------|
| `genesis-world` | Genesis physics engine |
| `i4h-asset-catalog` | Asset download helpers (`get_i4h_local_asset_path`) |
| `i4h-sensor-simulation` | raysim OptiX US simulator + fluoro-simulator |
| `i4h-workflows` | i4h reference workflows |

raysim is built from source (`i4h-sensor-simulation/ultrasound-raytracing/build-release/`). The `.so` is added to `sys.path` at runtime — do not move the build dir.

---

## raysim config (`config/raysim_custom.toml`)

Key parameters to tune:

| Parameter | Location | Effect |
|-----------|----------|--------|
| `probe.frequency` | `[probe]` | Higher = finer axial res, less penetration |
| `sim.t_far` | `[sim]` | Max imaging depth in mm |
| `world.phantom_pos` | `[world]` | Genesis world position of phantom origin (metres) |
| `world.organ_euler_deg` | `[world]` | Rotation applied to organ OBJs in both raysim and Genesis |
| `sensor.tcp_offset_m` | `[sensor]` | Offset from probe link to transducer face (metres) |

---

## Key scripts

### `src/test_custom.py` (main entry point)
- Loads ABDPhantom (i4h assets) as fixed rigid body
- Attaches Clarius probe to Franka end-effector
- Calls raysim every 10 physics steps for B-mode image
- Full keyboard teleoperation (IK-based)
- Config loaded from `config/raysim_custom.toml` via `tomllib`

### `scripts/segment_ct_image.py`
- Calls NVIDIA VISTA-3D API (`health.api.nvidia.com`) for CT segmentation
- Requires `NVIDIA_API_KEY` env var
- Input: NIfTI file at `output/image.nii.gz`
- Output: segmentation `.nrrd` file

---

## Assets

- **i4h ABDPhantom + ClariusUltrasoundProbe + Franka:** downloaded to `~/.cache/i4h-assets/` by setup.sh
- **SonoGym assets** (fr3_US.usd, HumanModels, SurgicalTools): downloaded to `assets/data/` by setup.sh — gitignored
- **Missing meshes:** `Lungs.obj`, `Skin.obj`, `Spleen.obj` absent from v0.2.0 ABDPhantom release — simulation runs without them

---

## Experimental repos (reference only)

| Repo | What it is | Why cloned |
|------|------------|-----------|
| `SonoGym/` | NVIDIA IsaacLab robotic US simulation platform | Architecture/task reference for robotic US |
| `EchoWorld/` | CVPR 2025 motion-aware world model for echocardiography probe guidance | Model architecture reference |
| `NV-Segment-CTMR/` | NVIDIA MONAI segmentation models (CT+MRI, 345+ classes) | Segmentation pipeline reference |
| `model-zoo/` | MONAI model zoo | Model bundle reference |

---

## Known issues

- raysim probe orientation may need calibration — adjust Euler angle mapping in `_quat_wxyz_to_euler_xyz` if B-mode appears rotated
- Taichi must be capped at 2 GB before Genesis init or OptiX BVH runs out of VRAM
- The `fixture_nomtl.usda` (Clarius probe without MDL materials) is generated by setup.sh — Genesis cannot load MDL shaders
