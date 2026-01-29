import os
import scanpy as sc
from stereosegger.data.utils import calculate_gene_celltype_abundance_embedding
from stereosegger.cli.utils import CustomFormatter
from pathlib import Path
import logging
import argparse
from stereosegger.data.parquet.sample import STSampleParquet
import time

# CLI command to create a Segger dataset
help_msg = "Create Segger dataset from spatial transcriptomics data (Stereo-seq)."


def create_dataset(args):
    # Setup logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[ch])

    # If scRNAseq file is provided, calculate gene-celltype embeddings
    gene_celltype_abundance_embedding = None
    if args.scrnaseq_file:
        logging.info("Calculating gene and celltype embeddings...")
        scRNAseq = sc.read(args.scrnaseq_file)
        sc.pp.subsample(scRNAseq, 0.1)
        gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(scRNAseq, args.celltype_column)

    # Initialize the sample class
    logging.info("Initializing sample...")
    sample = STSampleParquet(
        base_dir=args.base_dir,
        n_workers=args.n_workers,
        weights=gene_celltype_abundance_embedding,
    )

    # Save Segger dataset
    logging.info("Saving dataset for Segger...")
    start_time = time.time()
    sample.save(
        data_dir=args.data_dir,
        k_bd=args.k_bd,
        dist_bd=args.dist_bd,
        k_tx=args.k_tx,
        dist_tx=args.dist_tx,
        tx_graph_mode=args.tx_graph_mode,
        grid_connectivity=args.grid_connectivity,
        within_bin_edges=args.within_bin_edges,
        bin_pitch=args.bin_pitch,
        tile_size=args.tile_size,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        neg_sampling_ratio=args.neg_sampling_ratio,
        frac=args.frac,
        val_prob=args.val_prob,
        test_prob=args.test_prob,
    )
    end_time = time.time()
    logging.info(f"Time to save dataset: {end_time - start_time} seconds")
    logging.info("Dataset saved successfully.")


def main():
    parser = argparse.ArgumentParser(description=help_msg)
    parser.add_argument("--base_dir", type=Path, required=True, help="Directory containing the raw dataset.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory to save the processed Segger dataset.")
    parser.add_argument("--scrnaseq_file", type=Path, default=None, help="Path to the scRNAseq file.")
    parser.add_argument(
        "--celltype_column", type=str, default=None, help="Column name for cell type annotations in the scRNAseq file."
    )
    parser.add_argument("--k_bd", type=int, default=3, help="Number of nearest neighbors for boundary nodes.")
    parser.add_argument("--dist_bd", type=float, default=15.0, help="Maximum distance for boundary neighbors.")
    parser.add_argument("--k_tx", type=int, default=3, help="Number of nearest neighbors for transcript nodes.")
    parser.add_argument("--dist_tx", type=float, default=5.0, help="Maximum distance for transcript neighbors.")
    parser.add_argument(
        "--tile_size",
        type=int,
        default=None,
        help="If provided, specifies the size of the tile. Overrides `tile_width` and `tile_height`.",
    )
    parser.add_argument(
        "--tile_width", type=int, default=None, help="Width of the tiles in pixels. Ignored if `tile_size` is provided."
    )
    parser.add_argument(
        "--tile_height", type=int, default=None, help="Height of the tiles in pixels. Ignored if `tile_size` is provided."
    )
    parser.add_argument("--neg_sampling_ratio", type=float, default=5.0, help="Ratio of negative samples.")
    parser.add_argument("--frac", type=float, default=1.0, help="Fraction of the dataset to process.")
    parser.add_argument("--val_prob", type=float, default=0.1, help="Proportion of data for use for validation split.")
    parser.add_argument("--test_prob", type=float, default=0.2, help="Proportion of data for use for test split.")
    parser.add_argument("--n_workers", type=int, default=0, help="Number of workers for parallel processing.")
    parser.add_argument(
        "--tx_graph_mode",
        choices=["kdtree", "grid_bins"],
        default="grid_bins",
        help="Strategy for transcript-transcript edges.",
    )
    parser.add_argument(
        "--grid_connectivity",
        type=int,
        default=8,
        help="Grid connectivity (4 or 8) for grid-based transcript graphs.",
    )
    parser.add_argument(
        "--within_bin_edges",
        choices=["none", "star"],
        default="star",
        help="Within-bin edge strategy for grid graphs.",
    )
    parser.add_argument("--bin_pitch", type=float, default=1.0, help="Bin pitch for grid graph fallbacks.")

    args = parser.parse_args()
    create_dataset(args)


if __name__ == "__main__":
    main()
