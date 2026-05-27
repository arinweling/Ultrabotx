# ABDPhantom in Genesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create two scripts — one to download the i4h ABDPhantom asset, one to load it into a Genesis rigid-body scene.

**Architecture:** `download_phantom.py` uses `i4h_asset_helper.retrieve_asset` to pull `Props/ABDPhantom` from S3 into `~/.cache/i4h-assets`. `phantom_scene.py` reads that cached path, builds a Genesis scene with a ground plane, and loads `phantom.usda` via `scene.add_stage()`. Both scripts run under the `genesis` conda environment.

**Tech Stack:** Python 3.10+, Genesis (`genesis-world`), `i4h_asset_helper`, conda env `genesis`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/download_phantom.py` | Create | Download ABDPhantom assets via i4h_asset_helper |
| `src/phantom_scene.py` | Create | Load phantom.usda into Genesis scene |

---

## Task 1: Install i4h_asset_helper into the genesis conda env

**Files:**
- No code files — environment setup only

- [ ] **Step 1: Install i4h_asset_helper from the local repo**

```bash
conda run -n genesis pip install -e /home/arin/genesis_testing/i4h-asset-catalog
```

Expected output ends with: `Successfully installed i4h-asset-helper-0.5.0`

- [ ] **Step 2: Verify the install**

```bash
conda run -n genesis python -c "from i4h_asset_helper.assets import retrieve_asset, get_i4h_local_asset_path; print('ok')"
```

Expected: `ok`

---

## Task 2: Create `src/download_phantom.py`

**Files:**
- Create: `src/download_phantom.py`

- [ ] **Step 1: Create the src directory**

```bash
mkdir -p /home/arin/genesis_testing/src
```

- [ ] **Step 2: Write `download_phantom.py`**

Create `/home/arin/genesis_testing/src/download_phantom.py`:

```python
import os
from i4h_asset_helper.assets import retrieve_asset, get_i4h_local_asset_path

def main():
    print("Downloading ABDPhantom assets...")
    local_dir = retrieve_asset(sub_path="Props/ABDPhantom", verbose=True)
    phantom_path = os.path.join(local_dir, "Props", "ABDPhantom", "phantom.usda")
    print(f"\nDownload complete.")
    print(f"Asset root: {local_dir}")
    print(f"Phantom USD: {phantom_path}")
    if not os.path.exists(phantom_path):
        print("WARNING: phantom.usda not found at expected path — check the asset layout.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the download script**

```bash
conda run -n genesis python /home/arin/genesis_testing/src/download_phantom.py
```

Expected: progress bars from tqdm, then prints the local asset root and phantom.usda path. No WARNING line.

- [ ] **Step 4: Verify phantom.usda exists on disk**

The script will print the path. Confirm the file exists:

```bash
ls "$(conda run -n genesis python -c "
from i4h_asset_helper.assets import get_i4h_local_asset_path
print(get_i4h_local_asset_path())
")/Props/ABDPhantom/phantom.usda"
```

Expected: prints the file path (not "No such file").

---

## Task 3: Create `src/phantom_scene.py`

**Files:**
- Create: `src/phantom_scene.py`

- [ ] **Step 1: Write `phantom_scene.py`**

Create `/home/arin/genesis_testing/src/phantom_scene.py`:

```python
import argparse
import os
import genesis as gs
from i4h_asset_helper.assets import get_i4h_local_asset_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True,
                        help="Open interactive viewer (default: True)")
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    local_dir = get_i4h_local_asset_path()
    phantom_path = os.path.join(local_dir, "Props", "ABDPhantom", "phantom.usda")

    if not os.path.exists(phantom_path):
        raise FileNotFoundError(
            f"phantom.usda not found at {phantom_path}. "
            "Run download_phantom.py first."
        )

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(gravity=(0, 0, -9.8)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, 0.5, 0.5),
            camera_lookat=(0, 0, 0),
            camera_up=(0, 0, 1),
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    entities = scene.add_stage(gs.morphs.USD(file=phantom_path))
    print(f"Loaded {len(entities)} rigid entities from phantom.usda")

    scene.build()

    for _ in range(1000):
        scene.step()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the scene script with the viewer**

```bash
conda run -n genesis python /home/arin/genesis_testing/src/phantom_scene.py --vis
```

Expected: Genesis initialises, prints `Loaded N rigid entities from phantom.usda`, viewer window opens showing the phantom on a ground plane.

- [ ] **Step 3: Verify headless run works (no viewer)**

```bash
conda run -n genesis python /home/arin/genesis_testing/src/phantom_scene.py --no-vis 2>&1 | tail -5
```

Wait — `--no-vis` isn't wired up. Use `--vis` flag absence instead. Rewrite the argparse default to `False` and pass `--vis` explicitly when you want the window:

Update line in `phantom_scene.py`:
```python
    parser.add_argument("-v", "--vis", action="store_true", default=False,
```

Then headless test:
```bash
conda run -n genesis python /home/arin/genesis_testing/src/phantom_scene.py
```

Expected: runs 1000 steps silently, exits cleanly (no viewer window).
