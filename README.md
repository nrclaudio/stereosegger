# 🍳 Welcome to segger!


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

Token-based embedding example:

The gene name is treated as a categorical ID, not as text. Segger maps each gene to an integer and looks up a trainable vector:

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

# Why segger?

- **Highly parallelizable** – Optimized for multi-GPU environments
- **Fast and efficient** – Trains in a fraction of the time compared to alternatives
- **Transfer learning** – Easily adaptable to new datasets and technologies


  [API Reference](https://elihei2.github.io/segger_dev/api/)
