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

# ── submodules ────────────────────────────────────────────────────────────────
section "Initialising submodules"
git -C "$REPO_DIR" submodule update --init --recursive
info "Submodules ready"

# ── conda environment ─────────────────────────────────────────────────────────
section "Conda environment: $ENV_NAME"
if conda env list | grep -q "^${ENV_NAME} "; then
    info "Environment '$ENV_NAME' already exists — skipping creation"
else
    conda create -n "$ENV_NAME" python=3.11 libstdcxx-ng -c conda-forge -y
    info "Environment '$ENV_NAME' created"
fi

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

# ── Genesis ───────────────────────────────────────────────────────────────────
section "Genesis physics engine"
GENESIS_DIR="$REPO_DIR/genesis-world"
if conda run -n "$ENV_NAME" python -c "import genesis" >/dev/null 2>&1; then
    info "Genesis already installed"
else
    info "Installing Genesis from $GENESIS_DIR ..."
    conda run -n "$ENV_NAME" pip install -e "$GENESIS_DIR"
fi

# ── i4h asset helper ──────────────────────────────────────────────────────────
section "i4h asset helper"
CATALOG_DIR="$REPO_DIR/i4h-asset-catalog"
if conda run -n "$ENV_NAME" python -c "from i4h_asset_helper.assets import get_i4h_local_asset_path" >/dev/null 2>&1; then
    info "i4h_asset_helper already installed"
else
    info "Installing i4h_asset_helper..."
    conda run -n "$ENV_NAME" pip install -e "$CATALOG_DIR"
fi

# ── raysim (OptiX ultrasound simulator) ───────────────────────────────────────
section "raysim (NVIDIA OptiX ultrasound simulator)"
RAYSIM_DIR="$REPO_DIR/i4h-sensor-simulation/ultrasound-raytracing"
SO_FILE=$(find "$RAYSIM_DIR/raysim" -name "ray_sim_python*.so" 2>/dev/null | head -1)

if [ -n "$SO_FILE" ]; then
    info "raysim already built: $SO_FILE"
else
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

# ── PyMUST (fallback CPU ultrasound simulator) ────────────────────────────────
section "PyMUST"
if conda run -n "$ENV_NAME" python -c "import pymust" >/dev/null 2>&1; then
    info "PyMUST already installed"
else
    conda run -n "$ENV_NAME" pip install pymust
fi

# ── OpenCV ────────────────────────────────────────────────────────────────────
section "OpenCV"
if conda run -n "$ENV_NAME" python -c "import cv2" >/dev/null 2>&1; then
    info "OpenCV already installed"
else
    conda run -n "$ENV_NAME" pip install opencv-python
fi

# ── Download ABDPhantom assets ────────────────────────────────────────────────
section "ABDPhantom assets"
PHANTOM_CHECK=$(conda run -n "$ENV_NAME" python -c "
from i4h_asset_helper.assets import get_i4h_local_asset_path
import os
p = os.path.join(get_i4h_local_asset_path(), 'Props', 'ABDPhantom', 'phantom.usda')
print('ok' if os.path.exists(p) else 'missing')
" 2>/dev/null || echo "missing")

if [ "$PHANTOM_CHECK" = "ok" ]; then
    info "ABDPhantom assets already downloaded"
else
    info "Downloading ABDPhantom assets (~500 MB)..."
    conda run -n "$ENV_NAME" python "$REPO_DIR/src/download_phantom.py"
fi

# ── done ──────────────────────────────────────────────────────────────────────
section "Setup complete"
echo -e "${GREEN}${BOLD}"
echo "  Everything is ready. To run the simulation:"
echo ""
echo "    cd $REPO_DIR/genesis-world"
echo "    conda run -n $ENV_NAME python $REPO_DIR/src/test.py"
echo -e "${NC}"
