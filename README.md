# StereoSegger: Fast and Accurate Cell Segmentation for Spatial Omics

> **Note:** This project is heavily inspired by the original **Segger** implementation by Elyas Heidari. You can find the original repository at [EliHei2/segger_dev](https://github.com/EliHei2/segger_dev).

## Installation

StereoSegger requires **CUDA 12** (specifically configured for **CUDA 12.4** compatibility) for GPU acceleration.

### Option 1: Automated Setup (Recommended)

We provide a setup script that handles the complex dependency chain (PyTorch 2.5.1, RAPIDS 24.08, CUDA 12.4) automatically. This is the **most reliable method** to ensure GPU acceleration works.

```bash
# Clone the repository
git clone https://github.com/nrclaudio/stereosegger.git
cd stereosegger

# Run the setup script (requires Conda installed)
bash scripts/setup_segger_env.sh

# Activate the environment
conda activate segger_env
```

### Option 2: Pip Install (Advanced)

If you are managing your own CUDA environment, you can install via pip. Note that you **must** include the NVIDIA and PyTorch indices to get the correct GPU-accelerated wheels.

```bash
pip install stereosegger \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

---

## Inputs & Outputs

Understanding the data flow is critical for using StereoSegger effectively.

### 1. Inputs

StereoSegger primarily operates on **Parquet** files derived from standard spatial formats.

#### A. Raw Input (SAW Output)
- **Format:** `h5ad` (AnnData)
- **Source:** Output from the SAW pipeline (Stereo-seq Analysis Workflow).
- **Requirements:**
    - `.X`: Sparse matrix of gene counts.
    - `.obsm['spatial']`: (x, y) coordinates of the bins.
    - `.var`: Index must contain unique gene names.

#### B. Processed Input (StereoSegger Native)
If you are skipping the conversion step, you must provide a directory containing:

1.  **`transcripts.parquet`** (The Graph Nodes):
    - **Schema:**
      - `transcript_id` (int64): Unique ID for each gene-location occurrence.
      - `gene_id` (int32): Index mapping to `genes.parquet`.
      - `x`, `y` (float): Spatial coordinates.
      - `count` (int32): UMI count (intensity).
      - `bx`, `by` (int32): Grid coordinates (for bin1 data).
      - `overlaps_nucleus` (int/bool, optional): Ground truth label (1=nucleus, 0=cytoplasm).
      - `cell_id` (int/str, optional): Ground truth cell assignment.

2.  **`genes.parquet`** (The Dictionary):
    - **Schema:**
      - `gene_id` (int32): Primary key.
      - `gene_name` (string): Human-readable gene symbol.

3.  **`boundaries.parquet`** (Optional - Context):
    - **Schema:**
      - `boundary_id` (int/str): Unique ID.
      - `geometry` (binary): WKB-encoded polygon (e.g., nuclei masks).

### 2. Outputs

#### A. Intermediate Dataset
The `create_dataset_fast` command produces a directory of **PyTorch Geometric Data objects (`.pt`)**.
- These are tiled crops of the tissue graph, ready for high-performance training.
- **Location:** Your specified `--data_dir`.

#### B. Segmentation Results
The `predict` command produces the final cell segmentation.
- **Format:** `h5ad` and/or `csv`.
- **Content:**
    - **Cell Labels:** Each transcript is assigned a `seg_label` (predicted cell ID).
    - **Cell X/Y:** Centroids of the predicted cells.
    - **Confidence:** Probability scores for the assignments.
- **Location:** Your specified `--benchmarks_dir`.

---

## Quickstart: Stereo-seq SAW bin1

### 1. Convert Data & Create Dataset

Convert a SAW bin1 `h5ad` to StereoSegger parquet and build the graph dataset.

```bash
# 1. Convert H5AD to Parquet
python -m stereosegger.cli.convert_saw_h5ad_to_segger_parquet \
  --h5ad C04895D5_tissue.h5ad \
  --out_dir ./raw_data \
  --bin_pitch 1.0 \
  --min_count 1

# 2. Build Graph Dataset
python -m stereosegger.cli.create_dataset_fast \
  --base_dir ./raw_data \
  --data_dir ./processed_dataset \
  --sample_type saw_bin1 \
  --tx_graph_mode grid_bins \
  --grid_connectivity 8 \
  --within_bin_edges star
```

### 2. Train Model

Train the model on your processed dataset.

```bash
python -m stereosegger.cli.train_model \
  --dataset_dir ./processed_dataset \
  --models_dir ./models \
  --sample_tag my_sample \
  --max_epochs 200 \
  --accelerator cuda \
  --devices 1
```

### 3. Run Segmentation (Predict)

Apply the trained model to segment new or validation data.

```bash
python -m stereosegger.cli.predict \
  --segger_data_dir ./processed_dataset \
  --models_dir ./models \
  --benchmarks_dir ./results \
  --transcripts_file ./raw_data/transcripts.parquet \
  --model_version 0
```

---

## Architecture & Design Choices

### Why Grid Graphs for SAW bin1?
- **Efficiency:** SAW bin1 data is naturally gridded. Using grid adjacency (neighbors are pixels up/down/left/right) is `O(1)` compared to `O(N log N)` for k-Nearest Neighbors.
- **Context:** The `star` topology connects all genes within a bin to a central hub, preserving local co-expression while maintaining spatial continuity across the tissue.

### Feature Engineering
- **Genes:** Learned embeddings for each gene identity.
- **Counts:** `log1p(count)` is used to scale embeddings, allowing the model to distinguish between low-expression noise and high-expression signal without exploding the graph size.
- **Boundaries:** If nuclei masks are available, the model computes geometric features (Area, Convexity, Circularity) to guide segmentation. Area is log-transformed to handle the large variance in cell sizes.