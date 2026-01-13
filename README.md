# StereoSegger: Fast and Accurate Cell Segmentation for Spatial Omics

## Stereo-seq SAW bin1 Quickstart

Convert a SAW bin1 `h5ad` to StereoSegger parquet and build a dataset with grid-based transcript graphs:

```bash
python -m segger.cli.convert_saw_h5ad_to_segger_parquet \
  --h5ad C04895D5_tissue.h5ad \
  --out_dir /path/to/base_dir \
  --bin_pitch 1.0 \
  --min_count 1

python -m segger.cli.create_dataset_fast \
  --base_dir /path/to/base_dir \
  --data_dir /path/to/stereosegger_dataset \
  --sample_type saw_bin1 \
  --tx_graph_mode grid_same_gene \
  --grid_connectivity 8 \
  --within_bin_edges star
```

Recommended defaults: `grid_connectivity=8`, `within_bin_edges=star` (optional), `bin_pitch=1.0` when coords are in bin units.

---

## End-to-End Workflow

After generating the dataset (see [Stereo-seq SAW bin1 Quickstart](#stereo-seq-saw-bin1-quickstart)), you can train a model and run segmentation.

### 1. Train Model

Train the StereoSegger model on your processed dataset.

```bash
python -m segger.cli.train_model \
  --dataset_dir /path/to/stereosegger_dataset \
  --models_dir /path/to/models_dir \
  --sample_tag my_sample \
  --max_epochs 200 \
  --accelerator cuda \
  --devices 1
```

*   `--dataset_dir`: Path to the processed StereoSegger dataset (output of `create_dataset`).
*   `--models_dir`: Directory where checkpoints and logs will be saved.
*   `--sample_tag`: Tag used during dataset creation (e.g., sample name).

### 2. Run Segmentation (Predict)

Apply the trained model to segment transcripts.

```bash
python -m segger.cli.predict \
  --segger_data_dir /path/to/stereosegger_dataset \
  --models_dir /path/to/models_dir \
  --benchmarks_dir /path/to/output_dir \
  --transcripts_file /path/to/base_dir/transcripts.parquet \
  --model_version 0
```

*   `--segger_data_dir`: Path to the processed StereoSegger dataset.
*   `--models_dir`: Directory containing the trained model (same as used in training).
*   `--benchmarks_dir`: Directory to save segmentation results (`.h5ad`, `.csv`, etc.).
*   `--transcripts_file`: Path to the `transcripts.parquet` file (generated during `convert_saw_h5ad_to_segger_parquet` or dataset creation).
*   `--model_version`: Version number of the training run (e.g., `0` for `version_0`).

---

## Inputs & Outputs

### Input Format
StereoSegger relies on a standard H5AD file (AnnData) for Stereo-seq data, which is then converted into intermediate Parquet files.

1.  **Source H5AD:**
    *   **Expression Matrix (`.X`):** Sparse matrix of gene counts.
    *   **Coordinates (`.obsm['spatial']`):** (x, y) coordinates of the bins.
    *   **Variables (`.var`):** Must contain gene names (index or a specific column).

2.  **Intermediate Parquet:**
    The conversion tool (`convert_saw_h5ad_to_segger_parquet`) generates a directory with:
    *   `transcripts.parquet`: Contains pseudo-transcripts derived from non-zero bins. Columns: `x`, `y`, `gene_id`, `count`, `bx` (bin x), `by` (bin y).
    *   `genes.parquet`: Mapping of `gene_id` to `gene_name`.
    *   `boundaries.parquet` (Optional): Polygon geometries from existing segmentation masks (if provided).

### Output
1.  **StereoSegger Dataset:**
    *   A directory of processed PyTorch Geometric graphs (`.pt` files) representing tiled regions of the tissue.
    *   Nodes: Pseudo-transcripts (with count features) and boundaries (if available).
    *   Edges: Spatial grid connections (same gene across neighbors) and local co-expression (within-bin).

    *   **Trained Model:**
    *   PyTorch Lightning checkpoints (`.ckpt`) saved in the `models_dir`.

### Data Requirements
If bypassing the H5AD conversion tools, ensure your input Parquet files adhere to these schemas:

1.  **`transcripts.parquet`** (Required):
    *   One row per detected gene-location (long format).
    *   **Columns:**
        *   `transcript_id` (int64): Unique identifier for the transcript node.
        *   `x`, `y` (float): Spatial coordinates.
        *   `gene_id` (int32): Integer index corresponding to `genes.parquet`.
        *   `count` (int32, optional): UMI count for this gene at this location (used for scaling embeddings).
        *   `bx`, `by` (int32, optional): Grid indices (required if using grid-based graph construction).
        *   `overlaps_nucleus` (int/bool, optional): 1 if inside a nucleus, 0 otherwise (for supervision).
        *   `cell_id` (int/str, optional): Ground truth cell assignment (for supervision).

2.  **`genes.parquet`** (Required):
    *   Mapping between integer IDs and gene names.
    *   **Columns:**
        *   `gene_id` (int32): Matching `transcripts.parquet`.
        *   `gene_name` (string): Human-readable gene symbol.

3.  **`boundaries.parquet`** (Optional):
    *   Polygon geometries for cell/nuclei boundaries.
    *   **Columns:**
        *   `boundary_id` (int/str): Unique identifier.
        *   `geometry` (bytes/binary): WKB (Well-Known Binary) encoded polygon.

---

## Installation (Linux with NVIDIA GPU)

StereoSegger requires **CUDA 11** or **CUDA 12** for GPU acceleration. The recommended way to install is using `pip` inside a clean environment.

### Step 1: Create a clean environment
```bash
# Using conda for isolation
conda create -n stereosegger python=3.10 pip -y
conda activate stereosegger
```

### Step 2: Install Core Dependencies (PyTorch, RAPIDS, CuPy)
StereoSegger relies on GPU acceleration. We recommend installing the core stack first to ensure compatibility.

**Using Pip (Recommended for Linux/WSL with CUDA 12):**
```bash
# 1. Install PyTorch (match your CUDA version, e.g., 12.4)
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Install RAPIDS (cuDF, cuML, cuGraph) and CuPy
pip install cudf-cu12==24.8.* cuml-cu12==24.8.* cugraph-cu12==24.8.* cuspatial-cu12==24.8.* cupy-cuda12x \
    --extra-index-url https://pypi.nvidia.com

# 3. Install PyTorch Geometric and optimized kernels
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

**Using Conda:**
```bash
# Install compatible versions of PyTorch, RAPIDS, and CuPy
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge \
    pytorch=2.5.1 torchvision pytorch-cuda=12.4 \
    rapids=24.8 python=3.10 cuda-version=12.4 \
    cupy
pip install torch_geometric
```

### Step 3: Install StereoSegger
```bash
pip install -e .
```


---

## Stereo-seq SAW bin1 Notes and Design Choices

Why SAW bin1 is handled differently:

- SAW bin1 is already a regular grid with counts per (bin, gene), not per-molecule coordinates. Treating each nonzero entry as a pseudo-transcript preserves gene identity while keeping bin-level counts.
- Grid adjacency is a more faithful neighborhood for bins than a distance-based kNN on pseudo-points. It keeps local structure consistent with the chip layout and avoids sensitivity to sparsity or count magnitude.
- The added `log1p(count)` feature lets the model see expression strength without exploding the number of nodes (no count expansion).

Definitions:

- Pseudo-transcript: a node created from a nonzero (bin, gene) entry. It carries the gene identity, uses the bin’s (x, y) coordinates, and stores `log1p(count)` as an extra scalar feature. It is not a single molecule; it is a bin-level aggregate.
- Grid adjacency: two bins are neighbors if their integer grid coordinates differ by one step. With 4-connectivity this is up/down/left/right; with 8-connectivity it also includes diagonals.

Graph mode guidance:

- `grid_same_gene`: connect same gene across neighboring bins. This preserves gene-specific spatial continuity and is the default for SAW.
- `within_bin_edges=star`: optionally connect co-expressed genes within a bin to share signal locally.
    - **Optimization:** For dense data, set to `none` to avoid performance bottlenecks.
- `grid_bins`: optional ablation/debug mode that collapses nodes to unique bins with aggregate features.
- `kdtree`: original behavior for Xenium/MERSCOPE; still supported for SAW if you want distance-based adjacency.

Modeling choices and impact:

- Token-based gene embeddings (default): each transcript node carries a gene ID token, and the model learns an embedding per gene. For SAW bin1 this is the standard path because transcripts are indexed by `gene_id`.
- scRNAseq cell-type abundance embeddings: if you pass `--scrnaseq_file` and `--celltype_column`, StereoSegger uses gene-by-cell-type abundance vectors as fixed features. This injects biological priors but requires gene-name alignment; missing genes are filtered.
- Count feature (`log1p(count)`): when a `count` column exists, StereoSegger adds expression strength without expanding nodes. For token-based embeddings it scales the gene embedding by `(1 + log1p(count))`; for scRNAseq embeddings it appends `log1p(count)` as an extra feature.

Token-based embedding example:

The gene name is treated as a categorical ID, not as text. StereoSegger maps each gene to an integer and looks up a trainable vector:

```
gene_name -> gene_id (0..N-1)
embedding = E[gene_id]  # E is a learned matrix (num_genes x emb_dim)
```

The embedding matrix `E` is updated during training to make gene identities useful for the segmentation task.

scRNAseq embedding example:

If you pass `--scrnaseq_file`, each gene is represented by a fixed vector of cell-type abundances (fraction of cells of each type that express the gene). Those vectors are used directly as node features and then linearly projected inside the model. This injects biological prior knowledge but only works if gene names match between scRNAseq and the spatial data.

Boundaries:

- SAW bin1 conversion can optionally write `boundaries.parquet` from label TIFFs.
- If boundaries are missing, dataset creation and prediction can still run in a prediction-only mode (no training labels).

---

## Architecture

StereoSegger employs a Heterogeneous Graph Attention Network (GATv2) to segment transcripts based on their spatial neighborhood and identity.

### 1. Nodes (The Graph Components)
The graph consists of two distinct node types:

*   **Transcript Nodes (`tx`):**
    *   **Identity:** Represents a specific gene detected at a specific spatial location.
    *   **Resolution:** For Stereo-seq (SAW bin1), each node corresponds to a `(bin, gene)` tuple. If a single bin contains 10 distinct genes, it generates 10 distinct nodes.
    *   **Features:**
        *   **Gene Embedding:** A learnable vector representing the gene identity (e.g., "Ins2" gets a unique vector, "Nphs1" gets another).
        *   **UMI Count:** The embedding vector is scaled by `(1 + log(count))` to represent the signal intensity without creating duplicate nodes.

*   **Boundary Nodes (`bd`):**
    *   **Identity:** Represents a polygon boundary (e.g., a cell or nucleus segmentation).
    *   **Features:** Geometric properties like Area, Convexity, Elongation, and Circularity.

### 2. Edges (The Connections)
Information flows between nodes via three types of directed edges:

*   **`tx` $\leftrightarrow$ `tx` (Transcript-Transcript):**
    *   Connects transcripts of the **same gene** in adjacent spatial bins (Grid Adjacency).
    *   Allows the model to recognize contiguous "blobs" of expression belonging to the same gene/transcript.
    *   Optionally connects different genes within the same bin ("star" topology) to model local co-expression.

*   **`tx` $\rightarrow$ `bd` (Transcript-Boundary Neighbors):**
    *   Connects a transcript to nearby boundaries (potential candidate cells).
    *   The model uses this edge to decide if the transcript belongs to that boundary.

*   **`tx` $\rightarrow$ `bd` (Transcript-Boundary Assignment - Supervision):**
    *   Connects a transcript to the *correct* boundary (Ground Truth) during training.

### 3. The Neural Network (GATv2)
The model processes the graph in layers:

1.  **Input Layer:**
    *   Initializes learnable embeddings for `tx` nodes.
    *   Initializes geometric feature encodings for `bd` nodes.

2.  **Message Passing Layers (SkipGAT):**
    *   Uses **GATv2Conv** (Graph Attention Version 2) to aggregate information.
    *   **Mechanism:** Each node updates its representation by attending to its neighbors. A transcript "asks" its neighbors: *"Are you part of the same cell? What gene are you?"*
    *   **Skip Connections:** Residual connections are added to preserve signal depth.

3.  **Decoding (Prediction):**
    *   For every `tx` $\rightarrow$ `bd` edge, the model computes a similarity score (dot product) between the final transcript embedding and the boundary embedding.
    *   **Output:** A probability score (0 to 1) indicating confidence that the transcript belongs to that cell.

---
