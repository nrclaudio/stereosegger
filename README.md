# 🍳 Welcome to segger!

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/EliHei2/segger_dev/main.svg)](https://results.pre-commit.ci/latest/github/EliHei2/segger_dev/main)

**Important note (Dec 2024)**: As segger is currently undergoing constant development, we highly recommend installing it directly via GitHub.

**segger** is a cutting-edge tool for **cell segmentation** in **single-molecule spatial omics** datasets. By leveraging **graph neural networks (GNNs)** and heterogeneous graphs, segger offers unmatched accuracy and scalability.

# How segger Works

![Segger Model](docs/images/Segger_model_08_2024.png)

---

# Quick Links

- 💾 **[Installation Guide](https://elihei2.github.io/segger_dev/installation/)**  
  Get started with installing segger on your machine.

- 📖 **[User Guide](https://elihei2.github.io/segger_dev/user_guide/)**  
  Learn how to use segger for cell segmentation tasks.

- 💻 **[Command-Line Interface (CLI)](https://elihei2.github.io/segger_dev/cli/)**  
  Explore the CLI options for working with segger.

- 📚 **[API Reference](https://elihei2.github.io/segger_dev/api/)**  
  Dive into the detailed API documentation for advanced usage.

- 📝 **[Sample Workflow](https://elihei2.github.io/segger_dev/notebooks/segger_tutorial/)**  
  Check out our tutorial showcasing a sample workflow with segger.

---

# Stereo-seq SAW bin1 Quickstart

Convert a SAW bin1 `h5ad` to Segger parquet and build a dataset with grid-based transcript graphs:

```bash
python -m segger.cli.convert_saw_h5ad_to_segger_parquet \
  --h5ad C04895D5_tissue.h5ad \
  --out_dir /path/to/base_dir \
  --bin_pitch 1.0 \
  --min_count 1

python -m segger.cli.create_dataset_fast \
  --base_dir /path/to/base_dir \
  --data_dir /path/to/segger_dataset \
  --sample_type saw_bin1 \
  --tx_graph_mode grid_same_gene \
  --grid_connectivity 8 \
  --within_bin_edges star
```

Recommended defaults: `grid_connectivity=8`, `within_bin_edges=star` (optional), `bin_pitch=1.0` when coords are in bin units.

---

# Stereo-seq SAW bin1 Notes and Design Choices

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
- `grid_bins`: optional ablation/debug mode that collapses nodes to unique bins with aggregate features.
- `kdtree`: original behavior for Xenium/MERSCOPE; still supported for SAW if you want distance-based adjacency.

Modeling choices and impact:

- Token-based gene embeddings (default): each transcript node carries a gene ID token, and the model learns an embedding per gene. For SAW bin1 this is the standard path because transcripts are indexed by `gene_id`.
- scRNAseq cell-type abundance embeddings: if you pass `--scrnaseq_file` and `--celltype_column`, Segger uses gene-by-cell-type abundance vectors as fixed features. This injects biological priors but requires gene-name alignment; missing genes are filtered.
- Count feature (`log1p(count)`): when a `count` column exists, Segger adds expression strength without expanding nodes. For token-based embeddings it scales the gene embedding by `(1 + log1p(count))`; for scRNAseq embeddings it appends `log1p(count)` as an extra feature.

Boundaries:

- SAW bin1 conversion can optionally write `boundaries.parquet` from label TIFFs.
- If boundaries are missing, dataset creation and prediction can still run in a prediction-only mode (no training labels).

---

# Why segger?

- **Highly parallelizable** – Optimized for multi-GPU environments
- **Fast and efficient** – Trains in a fraction of the time compared to alternatives
- **Transfer learning** – Easily adaptable to new datasets and technologies

### Challenges in Segmentation

Spatial omics segmentation faces issues like:

- **Over/Under-segmentation**
- **Transcript contamination**
- **Scalability limitations**

segger tackles these with a **graph-based approach**, achieving superior segmentation accuracy.

---

## Installation

**Important note (Dec 2024)**: As segger is currently undergoing constant development, we highly recommend installing it directly via GitHub.

### GitHub Installation

For a straightforward local installation from GitHub, clone the repository and install the package using `pip`:

```bash
git clone https://github.com/EliHei2/segger_dev.git
cd segger_dev
pip install -e ".[cuda12]"
```

segger requires **CUDA 11** or **CUDA 12** for GPU acceleration.
You can find more detailed information in our **[Installation Guide](https://elihei2.github.io/segger_dev/installation/)**.
To avoid dependency conflicts, we recommend installing segger in a virtual environment or using a containerized environment.

### Docker Installation

We provide an easy-to-use Docker container for those who prefer a containerized environment. To pull and run the Docker image:

```bash
docker pull danielunyi42/segger_dev:cuda121
docker run --gpus all -it danielunyi42/segger_dev:cuda121
```

The official Docker image comes with all dependencies pre-installed, including the CUDA toolkit, PyTorch, and CuPy.
The current images support **CUDA 11.8** and **CUDA 12.1**, which can be specified in the image tag.

---

# Powered by

- **PyTorch Lightning & PyTorch Geometric**: Enables fast, efficient graph neural network (GNN) implementation for heterogeneous graphs.
- **Dask**: Scalable parallel processing and distributed task scheduling, ideal for handling large transcriptomic datasets.
- **Shapely & Geopandas**: Utilized for spatial operations such as polygon creation, scaling, and spatial relationship computations.
- **RAPIDS**: Provides GPU-accelerated computation for tasks like k-nearest neighbors (KNN) graph construction.
- **AnnData & Scanpy**: Efficient processing for single-cell datasets.
- **SciPy**: Facilitates spatial graph construction, including distance metrics and convex hull calculations for transcript clustering.

---

# Contributions

segger is **open-source** and welcomes contributions. Join us in advancing spatial omics segmentation!

- **Source Code**  
  [GitHub](https://github.com/EliHei2/segger_dev)

- **Bug Tracker**  
  [Report Issues](https://github.com/EliHei2/segger_dev/issues)

- **Full Documentation**  
  [API Reference](https://elihei2.github.io/segger_dev/api/)
