#!/bin/bash
# scripts/setup_segger_env.sh

# Exit on error
set -e

echo "=== Setting up Segger Environment (PIP Only) ==="

# 1. Activate Conda Env (Isolation Only)
source $(conda info --base)/etc/profile.d/conda.sh
# Create a clean environment with just Python 3.10 and pip
conda create -n segger_env python=3.10 pip -y || echo "Env exists, proceeding..."
conda activate segger_env

echo "Active Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

# 2. Install PyTorch Stable (2.5.1 for CUDA 12.4)
# We install this FIRST to ensure subsequent packages respect this version
echo "Installing PyTorch 2.5.1 (CUDA 12.4)..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Fix for "undefined symbol: __nvJitLinkComplete_12_4"
# Ensure nvjitlink is up to date (Torch 2.5.1/CUDA 12.4 requirement)
echo "Fixing nvidia-nvjitlink version..."
pip install "nvidia-nvjitlink-cu12>=12.4.127"

# 3. Install RAPIDS (CUDA 12) via PIP
# Using NVIDIA PyPI index
echo "Installing RAPIDS (cudf, cuml, cugraph, cuspatial) for CUDA 12..."
pip install \
    --extra-index-url https://pypi.nvidia.com \
    "cudf-cu12==24.8.*" \
    "cuml-cu12==24.8.*" \
    "cugraph-cu12==24.8.*" \
    "cuspatial-cu12==24.8.*" \
    "cupy-cuda12x"

# 4. Install PyTorch Geometric Dependencies
# Must match Torch 2.5.1 + CUDA 12.4 exactly
echo "Installing PyTorch Geometric Dependencies..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install torch-geometric

# 5. Install Standard Science Stack & Utilities
echo "Installing Science Stack..."
# Note: dask-cuda is needed for RAPIDS scaling
pip install \
    "numpy>=1.21.0" \
    "pandas>=1.3.0" \
    "scipy>=1.7.0" \
    "scanpy>=1.9.3" \
    "squidpy>=1.2.0" \
    "geopandas>=0.9.0" \
    "shapely>=1.7.0" \
    "dask-geopandas>=0.4.0" \
    "dask" \
    "lightning>=1.9.0" \
    "torchmetrics>=0.5.0" \
    "adjustText>=0.8" \
    "dask-cuda>=23.10.0"

# 6. Install Segger (Editable)
echo "Installing Segger in editable mode..."
# Since the script is now in stereosegger/scripts/, the root is one level up
REPO_ROOT="$(dirname "$0")/.."
pip install -e "${REPO_ROOT}"

echo "=== Setup Complete! ==="
echo "To use: conda activate segger_env"