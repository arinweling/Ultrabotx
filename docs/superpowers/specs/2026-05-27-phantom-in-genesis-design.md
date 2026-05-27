# Design: Load i4h ABDPhantom into Genesis

**Date:** 2026-05-27
**Status:** Approved

## Overview

Load the i4h ABDPhantom asset (abdominal phantom with organ meshes) into a Genesis scene as rigid bodies for future robot interaction work. Two scripts: one for one-time asset download, one for the Genesis scene.

## File Layout

```
genesis_testing/
└── src/
    ├── download_phantom.py    # one-time download via i4h_asset_helper
    └── phantom_scene.py       # Genesis scene with phantom loaded
```

## `download_phantom.py`

- Calls `retrieve_asset(sub_path="Props/ABDPhantom")` from `i4h_asset_helper.assets`
- Downloads the full ABDPhantom folder (phantom.usda + organ OBJs + textures) to the default cache dir `~/.cache/i4h-assets`, or a custom path via the `I4H_DOWNLOAD_DIR` env var
- Prints the local path on success
- No Genesis dependency — runs standalone

## `phantom_scene.py`

- Reads the phantom path from `~/.cache/i4h-assets` (or `I4H_DOWNLOAD_DIR` if set) using `get_i4h_local_asset_path()` from `i4h_asset_helper.assets`
- Constructs the path to `phantom.usda` within the downloaded folder
- Initialises Genesis with the rasterizer renderer (`gs.renderers.Rasterizer`)
- Builds a scene:
  - Ground plane as a static rigid body
  - Phantom loaded via `scene.add_stage(gs.morphs.USD(file=<phantom.usda path>))`, which parses the USD and returns one rigid entity per prim
- Adds a camera at a sensible default position looking at the origin
- Calls `scene.build()` and opens the interactive viewer

## Out of Scope (for now)

- Robot arm or gripper
- Soft-body simulation of organs
- Individual organ loading (OBJ meshes separately)
- Custom collision geometry tuning

## Success Criteria

- `download_phantom.py` completes without error and prints the local asset path
- `phantom_scene.py` opens the Genesis viewer showing the phantom on a ground plane
