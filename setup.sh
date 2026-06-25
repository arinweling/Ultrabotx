#!/usr/bin/env bash
# setup.sh — one-shot environment setup for genesis_testing
# Usage: bash setup.sh
# Requires: conda, NVIDIA GPU, CUDA driver >= 555

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="genesis"
CUDA_VERSION="13.2.78"

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
section() { echo -e "\n${BOLD}━━━ $* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
die()     { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── preflight checks ─────────────────────────────────────────────────────────
section "Preflight checks"

command -v conda >/dev/null 2>&1 || die "conda not found. Install Miniconda first."
command -v cmake >/dev/null 2>&1 || die "cmake not found. Run: conda install -c conda-forge cmake"
nvidia-smi >/dev/null 2>&1      || die "nvidia-smi failed — is an NVIDIA GPU present?"

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
info "NVIDIA driver: $DRIVER_VERSION"
info "Repo root:     $REPO_DIR"

# ── WSL2: OptiX driver fix ────────────────────────────────────────────────────
if grep -qiE "microsoft|WSL" /proc/version 2>/dev/null; then
    section "WSL2 OptiX driver check"
    WSL_LIB="/usr/lib/wsl/lib"
    OPTIX_DST="$WSL_LIB/libnvoptix.so.1"

    # Search common locations for driver-dist folder
    DRIVER_DIST=""
    for candidate in \
        "/mnt/wslg/distro/home/${USER}/driver-dist" \
        "/mnt/c/Users/${USER}/driver-dist" \
        "$HOME/driver-dist"; do
        if [ -f "${candidate}/libnvoptix.so.1" ]; then
            DRIVER_DIST="$candidate"
            break
        fi
    done

    if [ -z "$DRIVER_DIST" ]; then
        warn "WSL2 detected but driver-dist folder not found."
        warn "OptiX (raysim) requires an up-to-date libnvoptix.so.1 in $WSL_LIB."
        warn "To fix OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND, do the following:"
        warn "  1. Locate your NVIDIA driver package and extract libnvoptix.so.1"
        warn "     (commonly found at C:\\Windows\\System32\\libnvoptix.so.1 or"
        warn "      in the driver installer folder)"
        warn "  2. Copy it into WSL:"
        warn "     sudo cp /mnt/c/path/to/libnvoptix.so.1 $OPTIX_DST"
        warn "     sudo ldconfig"
        warn "  Or re-run setup.sh after placing the file in ~/driver-dist/"
    else
        OPTIX_SRC="$DRIVER_DIST/libnvoptix.so.1"
        if [ ! -f "$OPTIX_DST" ] || ! cmp -s "$OPTIX_SRC" "$OPTIX_DST"; then
            info "Updating $OPTIX_DST from $DRIVER_DIST ..."
            sudo cp "$OPTIX_SRC" "$OPTIX_DST"
            sudo ldconfig
            info "libnvoptix.so.1 updated"
        else
            info "libnvoptix.so.1 already up to date"
        fi
    fi
fi

# # ── submodules ────────────────────────────────────────────────────────────────
# section "Initialising submodules"
# git -C "$REPO_DIR" submodule update --init --recursive
# info "Submodules ready"

# ── conda environment ─────────────────────────────────────────────────────────
section "Conda environment: $ENV_NAME"
if conda env list | grep -q "^${ENV_NAME} "; then
    info "Environment '$ENV_NAME' already exists — skipping creation"
else
    conda create -n "$ENV_NAME" python=3.11 pip libstdcxx-ng -c conda-forge -y
    info "Environment '$ENV_NAME' created"
fi

ENV_PIP="$(conda info --base)/envs/${ENV_NAME}/bin/pip"
[ -x "$ENV_PIP" ] || die "Cannot find pip in conda env '${ENV_NAME}': $ENV_PIP"
info "Using pip: $ENV_PIP"

# ── nvcc (needed to build raysim) ─────────────────────────────────────────────
section "CUDA compiler (nvcc)"
if conda run -n "$ENV_NAME" nvcc --version >/dev/null 2>&1; then
    info "nvcc already installed"
else
    info "Installing cuda-nvcc ${CUDA_VERSION}..."
    conda install -n "$ENV_NAME" -c nvidia "cuda-nvcc=${CUDA_VERSION}" -y
fi

# ── ninja (needed by CMake) ───────────────────────────────────────────────────
section "Build tools (ninja)"
if conda run -n "$ENV_NAME" ninja --version >/dev/null 2>&1; then
    info "ninja already installed"
else
    conda install -n "$ENV_NAME" -c conda-forge ninja -y
fi

# ── Taichi ───────────────────────────────────────────────────────────────────
section "Taichi"
if conda run -n "$ENV_NAME" python -c "import taichi" >/dev/null 2>&1; then
    info "Taichi already installed"
else
    info "Installing Taichi..."
    "$ENV_PIP" install taichi
fi

# ── Genesis ───────────────────────────────────────────────────────────────────
section "Genesis physics engine"
GENESIS_DIR="$REPO_DIR/genesis-world"
# Remove any non-editable stub that would shadow the editable install
GENESIS_STUB="$(conda run -n "$ENV_NAME" python -c "import sys; print(next((p for p in sys.path if p.endswith('site-packages')), ''))")/genesis"
if [ -d "$GENESIS_STUB" ] && [ ! -f "$GENESIS_STUB/__init__.py" ]; then
    warn "Removing non-editable Genesis stub at $GENESIS_STUB"
    rm -rf "$GENESIS_STUB"
fi
if conda run -n "$ENV_NAME" python -c "import genesis; assert genesis.__file__" >/dev/null 2>&1; then
    info "Genesis already installed (editable)"
else
    info "Installing Genesis in editable mode from $GENESIS_DIR ..."
    "$ENV_PIP" install -e "$GENESIS_DIR"[usd]
fi

# ── i4h asset helper ──────────────────────────────────────────────────────────
section "i4h asset helper"
CATALOG_DIR="$REPO_DIR/i4h-asset-catalog"
if conda run -n "$ENV_NAME" python -c "from i4h_asset_helper.assets import get_i4h_local_asset_path" >/dev/null 2>&1; then
    info "i4h_asset_helper already installed"
else
    info "Installing i4h_asset_helper..."
    "$ENV_PIP" install -e "$CATALOG_DIR"
fi

# ── raysim (OptiX ultrasound simulator) ───────────────────────────────────────
section "raysim (NVIDIA OptiX ultrasound simulator)"
RAYSIM_DIR="$REPO_DIR/i4h-sensor-simulation/ultrasound-raytracing"
SO_FILE=$(find "$RAYSIM_DIR/raysim" -name "ray_sim_python*.so" 2>/dev/null | head -1)

if [ -n "$SO_FILE" ]; then
    info "raysim already built: $SO_FILE"
else
    MATERIAL_HPP="$RAYSIM_DIR/include/raysim/core/material.hpp"
    if ! grep -q '#include <cstdint>' "$MATERIAL_HPP"; then
        info "Patching material.hpp: adding #include <cstdint>"
        sed -i 's/#include <memory>/#include <cstdint>\n#include <memory>/' "$MATERIAL_HPP"
    fi

    info "Configuring raysim with CMake..."
    PYTHON_BIN=$(conda run -n "$ENV_NAME" which python)
    conda run -n "$ENV_NAME" cmake \
        -DASSIMP_WARNINGS_AS_ERRORS=OFF \
        -DPYTHON_EXECUTABLE="$PYTHON_BIN" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DBUILD_EXAMPLES=OFF \
        -B "$RAYSIM_DIR/build-release" \
        -S "$RAYSIM_DIR"

    info "Building raysim (this takes a few minutes)..."
    conda run -n "$ENV_NAME" cmake --build "$RAYSIM_DIR/build-release" -j"$(nproc)"
    info "raysim built successfully"
fi

# ── omniverse-kit (USD baking for Genesis) ────────────────────────────────────
section "omniverse-kit (USD baking)"
if conda run -n "$ENV_NAME" python -c "import omni" >/dev/null 2>&1; then
    info "omniverse-kit already installed"
else
    info "Installing omniverse-kit..."
    "$ENV_PIP" install --extra-index-url https://pypi.nvidia.com omniverse-kit
fi

for RC in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$RC" ] && ! grep -q 'OMNI_KIT_ACCEPT_EULA' "$RC"; then
        echo 'export OMNI_KIT_ACCEPT_EULA=yes' >> "$RC"
        info "Added OMNI_KIT_ACCEPT_EULA=yes to $RC"
    fi
done
export OMNI_KIT_ACCEPT_EULA=yes

# ── OpenCV ────────────────────────────────────────────────────────────────────
section "OpenCV"
if conda run -n "$ENV_NAME" python -c "import cv2" >/dev/null 2>&1; then
    info "OpenCV already installed"
else
    "$ENV_PIP" install opencv-python
fi

# ── nibabel (NIfTI I/O for seg_to_obj.py) ────────────────────────────────────
section "nibabel"
if conda run -n "$ENV_NAME" python -c "import nibabel" >/dev/null 2>&1; then
    info "nibabel already installed"
else
    "$ENV_PIP" install nibabel
fi

# ── Symlink panda MJCF assets so custom XML can use relative meshdir ─────────
section "Panda MJCF asset symlink"
GENESIS_PANDA_ASSETS=$(conda run -n "$ENV_NAME" python -c \
    "import genesis, os; print(os.path.join(os.path.dirname(genesis.__file__), 'assets', 'xml', 'franka_emika_panda', 'assets'))")
ln -sfn "$GENESIS_PANDA_ASSETS" "$REPO_DIR/xml/franka_emika_panda/assets"
info "assets → $GENESIS_PANDA_ASSETS"

# ── Download required i4h assets ──────────────────────────────────────────────
section "i4h assets (ABDPhantom + ClariusUltrasoundProbe)"
conda run -n "$ENV_NAME" python - << 'PYEOF'
from i4h_asset_helper.assets import (
    _get_s3_client, _get_asset_env, _S3_BUCKETS,
    get_i4h_local_asset_path, get_i4h_asset_hash, get_i4h_asset_version,
)
import os

bucket  = _S3_BUCKETS[_get_asset_env()]
s3      = _get_s3_client()
local   = get_i4h_local_asset_path()
ver     = get_i4h_asset_version()
h       = get_i4h_asset_hash(version=ver)
prefix  = f"Assets/Isaac/Healthcare/{ver}/{h}/"

needed = [
    "Props/ABDPhantom/",
    "Props/ClariusUltrasoundProbe/fixture.usda",
    "Robots/Franka/",
]

paginator = s3.get_paginator("list_objects_v2")
for np_ in needed:
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + np_):
        for obj in page.get("Contents", []):
            rel  = obj["Key"][len(prefix):]
            dest = os.path.join(local, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest):
                print(f"Downloading {rel}")
                s3.download_file(bucket, obj["Key"], dest)
print("i4h assets ready")
PYEOF

# ── Download SonoGym assets ───────────────────────────────────────────────────
section "SonoGym assets"
SONOGYM_ASSET_URL="https://huggingface.co/datasets/yunkao/SonoGym_assets_models/resolve/main/assets.tar.gz"
SONOGYM_ASSET_MARKER="$REPO_DIR/assets/data/Robots/Franka/fr3_US.usd"
if [ -f "$SONOGYM_ASSET_MARKER" ]; then
    info "SonoGym assets already present"
else
    info "Downloading SonoGym assets from Hugging Face..."
    TMP_ASSET_TAR=$(mktemp)
    curl -L "$SONOGYM_ASSET_URL" -o "$TMP_ASSET_TAR"
    tar -xzf "$TMP_ASSET_TAR" -C "$REPO_DIR"
    rm -f "$TMP_ASSET_TAR"
    info "SonoGym assets extracted"
fi

# Regenerate fixture_nomtl.usda (MDL materials stripped, needed by Genesis)
NOMTL_CHECK=$(conda run -n "$ENV_NAME" python -c "
from i4h_asset_helper.assets import get_i4h_local_asset_path
import os
p = os.path.join(get_i4h_local_asset_path(), 'Props', 'ClariusUltrasoundProbe', 'fixture_nomtl.usda')
print('ok' if os.path.exists(p) else 'missing')
" 2>/dev/null || echo "missing")

if [ "$NOMTL_CHECK" = "ok" ]; then
    info "fixture_nomtl.usda already exists"
else
    info "Generating fixture_nomtl.usda (stripping MDL materials)..."
    conda run -n "$ENV_NAME" python - << 'PYEOF'
from i4h_asset_helper.assets import get_i4h_local_asset_path
from pxr import Usd
import os
local = get_i4h_local_asset_path()
src = os.path.join(local, "Props", "ClariusUltrasoundProbe", "fixture.usda")
dst = os.path.join(local, "Props", "ClariusUltrasoundProbe", "fixture_nomtl.usda")
stage = Usd.Stage.Open(src)
to_remove = [p.GetPath() for p in stage.Traverse() if p.GetTypeName() == "Material"]
for path in to_remove:
    stage.RemovePrim(path)
stage.Export(dst)
print(f"fixture_nomtl.usda written to {dst}")
PYEOF
fi

# ── done ──────────────────────────────────────────────────────────────────────
section "Setup complete"
echo -e "${GREEN}${BOLD}"
echo "  Everything is ready. To run the simulation:"
echo ""
echo "    cd $REPO_DIR/genesis-world"
echo "    conda run -n $ENV_NAME python $REPO_DIR/src/test.py"
echo -e "${NC}"
