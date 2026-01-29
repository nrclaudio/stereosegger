# StereoSegger: Fast and Accurate Cell Segmentation for Stereo-seq

> **Note:** This project is heavily inspired by the original **Segger** implementation by Elyas Heidari. You can find the original repository at [EliHei2/segger_dev](https://github.com/EliHei2/segger_dev). This version is specifically optimized and refactored for **Stereo-seq** (SAW bin1) workflows.

## Installation

StereoSegger requires **CUDA 12** (specifically **CUDA 12.4** compatibility) for GPU acceleration.

### Quick Install (One-Liner)

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

---

## Inputs & Outputs

StereoSegger operates on **Parquet** files. We provide a built-in command to convert raw Stereo-seq H5AD files into this format.

### 1. Raw Input (SAW Output)
- **Format:** `h5ad` (AnnData)
- **Source:** Output from the SAW pipeline (Stereo-seq Analysis Workflow).
- **Conversion:** Use `stereosegger convert_saw` to prepare this for the pipeline.

### 2. Processed Input (Parquet)
The core pipeline expects a directory containing:
- **`transcripts.parquet`**: Long-form table of gene-location occurrences.
- **`genes.parquet`**: Mapping of `gene_id` to `gene_name`.
- **`boundaries.parquet`**: Polygons (required for training, optional for prediction).

---

### 1. Prepare Data

#### Path A: For Training (Kidneys)
Training requires ground-truth labels. You **must** provide a label TIFF (e.g., `ssdna_mask`).

```bash
# 1. Convert with labels
stereosegger convert_saw \
  --h5ad kidney_sample.h5ad \
  --labels_tif ssdna_mask.tif \
  --out_dir ./raw_data_labeled

# 2. Build Dataset
stereosegger create_dataset \
  --base_dir ./raw_data_labeled \
  --data_dir ./dataset_labeled
```

#### Path B: For Prediction (Whole Chip)
Prediction on new data uses a pre-trained model and does not require a mask.

```bash
# 1. Convert without labels
stereosegger convert_saw \
  --h5ad whole_chip.h5ad \
  --out_dir ./raw_data_unlabeled

# 2. Build Dataset
stereosegger create_dataset \
  --base_dir ./raw_data_unlabeled \
  --data_dir ./dataset_unlabeled
```

### 2. Train Model (Requires Labeled Data)
Training requires that you provided a `--labels_tif` during the `convert_saw` step.

```bash
stereosegger train_model \
  --dataset_dir ./processed_dataset \
  --models_dir ./models \
  --sample_tag my_sample \
  --max_epochs 300 \
  --devices 1
```

### 3. Run Segmentation (Predict)

```bash
stereosegger predict_fast \
  --segger_data_dir ./processed_dataset \
  --models_dir ./models \
  --benchmarks_dir ./results \
  --transcripts_file ./raw_data/transcripts.parquet \
  --model_version 0
```

---

## Command Reference

### 1. `convert_saw`
Converts Stereo-seq SAW pipeline output (H5AD) into Parquet format.

**Options:**
- `--h5ad PATH`: Path to SAW bin1 h5ad file.
- `--out_dir PATH`: Output directory.
- `--labels_tif PATH`: (Optional) Label TIFF for boundary polygons (Required if you intend to train).
- `--bin_pitch FLOAT`: Bin pitch for rounding. Default: `1.0`.

### 2. `create_dataset`
Creates the graph-based dataset used for training and inference.

**Options:**
- `--base_dir PATH`: Directory containing raw parquet files.
- `--data_dir PATH`: Directory to save the processed dataset.
- `--tx_graph_mode [kdtree|grid_bins]`: Transcript edge strategy. Default: `"grid_bins"`.
- `--grid_connectivity INT`: Grid connectivity (4 or 8). Default: `8`.
- `--within_bin_edges [none|star]`: Within-bin edge strategy. Default: `"star"`.

### 3. `train_model`
Trains the Segger model. **Will stop if the dataset is unlabeled.**

**Options:**
- `--dataset_dir PATH`: Processed dataset directory.
- `--models_dir PATH`: Directory to save the model.
- `--sample_tag TEXT`: Unique tag for the sample.
- `--max_epochs INT`: Number of training epochs. Default: `300`.

### 4. `predict_fast`
Runs fast segmentation inference for large grid-based datasets.

**Options:**
- `--segger_data_dir PATH`: Processed dataset directory.
- `--models_dir PATH`: Trained models directory.
- `--benchmarks_dir PATH`: Output results directory.
- `--transcripts_file PATH`: Original transcripts parquet file.
- `--model_version INT`: Version of the model to load. Default: `0`.

---

## Technical Details

### Architecture
StereoSegger employs a Heterogeneous Graph Attention Network (GATv2) to segment transcripts based on their spatial neighborhood and identity.

- **Transcript Nodes (`tx`)**: Represents a specific gene at a spatial location.
- **Boundary Nodes (`bd`)**: Represents polygon boundaries (e.g., nuclei).
- **Supervision**: During training, the model learns to predict "belongs" edges between transcripts and ground-truth boundaries.