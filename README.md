# StereoSegger: Fast and Accurate Cell Segmentation for Spatial Omics

> **Note:** This project is heavily inspired by the original **Segger** implementation by Elyas Heidari. You can find the original repository at [EliHei2/segger_dev](https://github.com/EliHei2/segger_dev).

## Installation

StereoSegger requires **CUDA 12** (specifically **CUDA 12.4** compatibility) for GPU acceleration.

### Quick Install (One-Liner)

To install StereoSegger with full GPU acceleration (PyTorch + RAPIDS), use the following command:

```bash
pip install stereosegger --extra-index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.nvidia.com
```

### Option 1: Automated Setup (Recommended for HPC/Conda)

We provide a setup script that handles the complex dependency chain (PyTorch 2.5.1, RAPIDS 24.08, CUDA 12.4) automatically inside a clean Conda environment.

```bash
# Clone the repository
git clone https://github.com/nrclaudio/stereosegger.git
cd stereosegger

# Run the setup script (requires Conda)
bash scripts/setup_segger_env.sh

# Activate the environment
conda activate segger_env
```

### Why the extra index URLs?
*   **`https://download.pytorch.org/whl/cu124`**: Ensures you get the version of PyTorch compiled for CUDA 12.4. Without this, `pip` may download a CPU-only version or an incompatible CUDA build.
*   **`https://pypi.nvidia.com`**: Provides the RAPIDS (cuDF, cuML, etc.) wheels. While some RAPIDS components are moving to standard PyPI, this index ensures you get the most stable, CUDA-linked binaries.

---

## Inputs & Outputs

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
If you are skipping the conversion step, provide a directory containing:
- **`transcripts.parquet`**: Long-form table of gene-location occurrences (`transcript_id`, `gene_id`, `x`, `y`, `count`, `bx`, `by`).
- **`genes.parquet`**: Mapping of `gene_id` to `gene_name`.
- **`boundaries.parquet`** (Optional): WKB-encoded polygons (e.g., nuclei masks).

---

## Quickstart: Stereo-seq SAW bin1

### 1. Convert Data & Create Dataset

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

For large datasets (like SAW bin1) using the grid optimizations, use the **fast** prediction script:

```bash
python -m stereosegger.cli.predict_fast \
  --segger_data_dir ./processed_dataset \
  --models_dir ./models \
  --benchmarks_dir ./results \
  --transcripts_file ./raw_data/transcripts.parquet \
  --model_version 0 \
  --tx_graph_mode grid_bins \
  --grid_connectivity 8
```

---

## Command Reference

### 1. `convert_saw_h5ad_to_segger_parquet`
Converts Stereo-seq SAW pipeline output (H5AD) into the Parquet format required by StereoSegger.

**Options:**
- `--h5ad PATH`: Path to SAW bin1 h5ad file. **(Required)**
- `--out_dir PATH`: Output directory for Segger parquet files. **(Required)**
- `--bin_pitch FLOAT`: Bin pitch for rounding to grid coordinates. Default: `1.0`.
- `--min_count INT`: Minimum count to keep a bin-gene entry. Default: `1`.
- `--labels_tif PATH`: Optional label TIFF for boundary polygons.
- `--tissue_mask_tif PATH`: Optional tissue mask TIFF.
- `--bbox FLOAT FLOAT FLOAT FLOAT`: Bounding box `xmin xmax ymin ymax`.
- `--gene_name_source TEXT`: Column in `adata.var` for gene names. Default: `"real_gene_name"`.
- `--top_genes INT`: Keep only top K genes by total counts.

### 2. `create_dataset` (Fast)
Creates the graph-based dataset used for training and inference.

**Options:**
- `--base_dir PATH`: Directory containing raw parquet files. **(Required)**
- `--data_dir PATH`: Directory to save the processed dataset. **(Required)**
- `--sample_type TEXT`: e.g., `"xenium"`, `"merscope"`, `"saw_bin1"`.
- `--tx_graph_mode [kdtree|grid_bins]`: Strategy for transcript edges. Default: `"grid_bins"`.
- `--grid_connectivity INT`: Grid connectivity (4 or 8). Default: `8`.
- `--within_bin_edges [none|star]`: Within-bin edge strategy. Default: `"star"`.
- `--tile_size INT`: Size of the spatial tiles.
- `--n_workers INT`: Number of parallel workers. Default: `0` (serial).

### 3. `train_model`
Trains the Segger segmentation model.

**Options:**
- `--dataset_dir PATH`: Directory containing the processed dataset. **(Required)**
- `--models_dir PATH`: Directory to save the model and logs. **(Required)**
- `--sample_tag TEXT`: Unique tag for the sample. **(Required)**
- `--batch_size INT`: Training batch size. Default: `1`.
- `--max_epochs INT`: Number of training epochs. Default: `300`.
- `--accelerator TEXT`: `"cuda"` or `"cpu"`. Default: `"cuda"`.
- `--devices INT`: Number of GPUs to use. Default: `4`.
- `--learning_rate FLOAT`: Learning rate. Default: `1e-4`.

### 4. `predict` / `predict_fast`
Runs the segmentation inference. `predict_fast` is optimized for large grid-based datasets.

**Options:**
- `--segger_data_dir PATH`: Processed dataset directory. **(Required)**
- `--models_dir PATH`: Trained models directory. **(Required)**
- `--benchmarks_dir PATH`: Output results directory. **(Required)**
- `--transcripts_file PATH`: Original transcripts parquet file. **(Required)**
- `--model_version INT`: Version of the model to load. Default: `0`.
- `--tx_graph_mode [kdtree|grid_bins]`: Strategy for transcript edges. Default: `"grid_bins"`.
- `--grid_connectivity INT`: Grid connectivity (4 or 8). Default: `8`.
- `--within_bin_edges [none|star]`: Within-bin edge strategy. Default: `"star"`.
- `--use_cc BOOL`: Use connected components for unassigned transcripts. Default: `False`.
- `--file_format TEXT`: Output format (`"anndata"`, `"parquet"`, `"csv"`). Default: `"anndata"`.
- `--k_bd` / `--dist_bd`: Boundary neighborhood parameters.
- `--k_tx` / `--dist_tx`: Transcript neighborhood parameters.

---

## Technical Details

### Stereo-seq SAW bin1 Methodology
StereoSegger implements specific logic to handle SAW bin1 data efficiently:
- **Regular Grid**: SAW bin1 data is already a regular grid. We leverage this by using grid adjacency (neighbors are pixels up/down/left/right) which is `O(1)` compared to `O(N log N)` for distance-based kNN on pseudo-points.
- **Consistency**: Grid adjacency keeps local structure consistent with the chip layout and avoids sensitivity to sparsity or count magnitude.

#### Graph Modes & Definitions
1.  **Pseudo-transcript (Gene-Bin Node)**: A node created from a nonzero (bin, gene) entry. Connects all genes in a bin to a central "hub" gene, and connects hubs across adjacent bins. **Recommended for SAW**.
2.  **Aggregated Bin Node**: A node representing an entire spatial bin, aggregating all transcripts within it. Features are `[log(total_count), log(n_genes)]`.
3.  **Grid Adjacency**: Two bins are neighbors if their integer grid coordinates differ by one step. `grid_connectivity=8` includes diagonals.

### Architecture
StereoSegger employs a Heterogeneous Graph Attention Network (GATv2) to segment transcripts based on their spatial neighborhood and identity.

#### 1. Nodes (The Graph Components)
- **Transcript Nodes (`tx`)**: Represents a specific gene at a spatial location. Gene embeddings are scaled by `(1 + log(count))` to represent signal intensity without exploding graph size.
- **Boundary Nodes (`bd`)**: Represents polygon boundaries (e.g., nuclei). Features like Area are log-transformed for numerical stability.

#### 2. Edges (The Connections)
- **`tx` $\leftrightarrow$ `tx` (Transcript-Transcript)**: Star topology (within bin) + Grid adjacency (across bins).
- **`tx` $\rightarrow$ `bd` (Transcript-Boundary Neighbors)**: Connects transcripts to nearby candidate cells.
- **`tx` $\rightarrow$ `bd` (Supervision)**: Connects a transcript to the _correct_ ground-truth boundary during training.
