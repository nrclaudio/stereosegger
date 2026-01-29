import argparse
from stereosegger.training.segger_data_module import SeggerDataModule
from stereosegger.prediction.predict_parquet import segment, load_model
from stereosegger.cli.utils import CustomFormatter
from pathlib import Path
import logging

help_msg = "Run the Segger segmentation model (fast version)."


def run_segmentation(args):
    # Setup logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[ch])
    logger = logging.getLogger(__name__)

    logger.info("Initializing Segger data module...")
    dm = SeggerDataModule(
        data_dir=args.segger_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dm.setup()

    logger.info("Loading the model...")
    model_path = Path(args.models_dir) / "lightning_logs" / f"version_{args.model_version}"
    model = load_model(model_path / "checkpoints")

    logger.info("Running segmentation...")
    segment(
        model,
        dm,
        save_dir=args.benchmarks_dir,
        seg_tag=args.save_tag,
        transcript_file=args.transcripts_file,
        file_format=args.file_format,
        receptive_field={"k_bd": args.k_bd, "dist_bd": args.dist_bd, "k_tx": args.k_tx, "dist_tx": args.dist_tx},
        min_transcripts=args.min_transcripts,
        cell_id_col=args.cell_id_col,
        use_cc=args.use_cc,
        knn_method=args.knn_method,
        tx_graph_mode=args.tx_graph_mode,
        grid_connectivity=args.grid_connectivity,
        within_bin_edges=args.within_bin_edges,
        bin_pitch=args.bin_pitch,
        verbose=True,
    )

    logger.info("Segmentation completed.")


def main():
    parser = argparse.ArgumentParser(description=help_msg)
    parser.add_argument("--segger_data_dir", type=Path, required=True, help="Directory containing processed dataset.")
    parser.add_argument("--models_dir", type=Path, required=True, help="Directory containing trained models.")
    parser.add_argument("--benchmarks_dir", type=Path, required=True, help="Directory to save results.")
    parser.add_argument("--transcripts_file", type=str, required=True, help="Path to transcripts file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")
    parser.add_argument("--model_version", type=int, default=0, help="Model version.")
    parser.add_argument("--save_tag", type=str, default="segger_segmentation", help="Tag for saving results.")
    parser.add_argument("--min_transcripts", type=int, default=5, help="Min transcripts per cell.")
    parser.add_argument("--cell_id_col", type=str, default="segger_cell_id", help="Column for cell IDs.")
    parser.add_argument("--use_cc", action="store_true", default=False, help="Use connected components.")
    parser.add_argument("--knn_method", type=str, default="kd_tree", help="KNN method.")
    parser.add_argument("--file_format", type=str, default="anndata", help="Output format.")
    parser.add_argument("--k_bd", type=int, default=3, help="K for boundary.")
    parser.add_argument("--dist_bd", type=float, default=15.0, help="Dist for boundary.")
    parser.add_argument("--k_tx", type=int, default=3, help="K for transcript.")
    parser.add_argument("--dist_tx", type=float, default=5.0, help="Dist for transcript.")
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
    run_segmentation(args)


if __name__ == "__main__":
    main()