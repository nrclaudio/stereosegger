import sys
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import warnings

# === SETUP ===
# Add local stereosegger source to path if not installed via pip
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent  # Assumes script is in root/scripts/
segger_src = repo_root / "src"

if str(segger_src) not in sys.path:
    sys.path.insert(0, str(segger_src))

try:
    import stereosegger
    from stereosegger.cli.convert_saw_h5ad_to_segger_parquet import convert_saw_h5ad_to_parquet
    from stereosegger.data.parquet.sample import STSampleParquet
    from stereosegger.training.segger_data_module import SeggerDataModule
    from stereosegger.training.train import LitSegger
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import CSVLogger

    # Try importing prediction modules (requires CuPy/CUDA)
    try:
        from stereosegger.prediction.predict import predict_batch

        HAS_PREDICTION_SUPPORT = True
    except ImportError:
        HAS_PREDICTION_SUPPORT = False
        print("Warning: CuPy not found. Inference step will be skipped.")

    print(f"StereoSegger imported from: {stereosegger.__file__}")
except ImportError as e:
    print(f"Error importing stereosegger: {e}")
    sys.exit(1)


def main():
    # === CONFIGURATION ===
    # Input H5AD file (Update this path to your file)
    # Example placeholder path
    input_h5ad = repo_root / "data" / "example.h5ad"

    # Create a dummy H5AD if it doesn't exist for tutorial purposes?
    # For now, we'll check if it exists and warn user.
    if not input_h5ad.exists():
        print(f"\n[!] Input file {input_h5ad} not found.")
        print("Please edit the 'input_h5ad' variable in this script to point to your .h5ad file.")
        print("Exiting...")
        return

    # Output Base Directory for this experiment
    experiment_name = "tutorial_experiment"
    base_out_dir = repo_root / "tutorial_output" / experiment_name

    # Subdirectories
    inputs_dir = base_out_dir / "segger_inputs"
    dataset_dir = base_out_dir / "segger_dataset"
    model_dir = base_out_dir / "segger_model"

    # Cleanup previous run (Optional)
    if base_out_dir.exists():
        print(f"Cleaning up old directory: {base_out_dir}")
        shutil.rmtree(base_out_dir)

    base_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {base_out_dir}")

    # === 1. CONVERSION (H5AD -> Parquet) ===
    print(f"\n=== 1. Converting {input_h5ad.name} ===")

    convert_saw_h5ad_to_parquet(
        h5ad_path=input_h5ad,
        out_dir=inputs_dir,
        bin_pitch=1.0,
        min_count=1,
        labels_tif=None,  # Optional: Path to label image for supervision
        tissue_mask_tif=None,  # Optional: Path to tissue mask
        bbox=None,  # Optional: (xmin, xmax, ymin, ymax) to crop
        gene_name_source="gene_name",  # Adjust based on your h5ad.var columns
        top_genes=None,  # Optional: Limit to top K genes
    )

    print("Conversion complete. Files generated:")
    for f in inputs_dir.glob("*"):
        print(f" - {f.name}")

    # === 2. DATASET CREATION ===
    print("\n=== 2. Creating Segger Dataset ===")

    # Initialize Sample Loader
    sample = STSampleParquet(
        base_dir=inputs_dir,
        n_workers=4,  # Adjust based on your CPU cores
        sample_type="saw_bin1",
        weights=None,  # None = Use learnable embeddings
    )

    # Save (Create) the dataset tiles
    # Using 'grid_bins' with 'star' topology as recommended for Stereo-seq
    sample.save(
        data_dir=dataset_dir,
        k_bd=3,
        dist_bd=15.0,
        k_tx=3,
        dist_tx=5.0,
        tx_graph_mode="grid_bins",
        grid_connectivity=8,
        within_bin_edges="star",
    )

    print(f"Dataset created at {dataset_dir}")

    # === 3. TRAINING ===
    print("\n=== 3. Training Model ===")

    # Data Module
    dm = SeggerDataModule(
        data_dir=dataset_dir, batch_size=1, num_workers=0  # 0 for safety in notebooks/mac to avoid forking issues
    )
    dm.setup()

    if len(dm.train) == 0:
        print("Error: No training data found.")
        return

    # Determine input feature size
    sample_data = dm.train[0]

    # Logic to determine if token-based
    genes_df = pd.read_parquet(inputs_dir / "genes.parquet")
    num_tx_tokens = len(genes_df) + 10

    is_token_based = False
    num_tx_features = 0

    # Flexible check for feature dimensions
    if hasattr(sample_data["tx"], "token_based") and sample_data["tx"].token_based:
        is_token_based = True
        num_tx_features = num_tx_tokens
    elif "tx" in sample_data.x_dict and sample_data.x_dict["tx"].ndim == 1:
        is_token_based = True
        num_tx_features = num_tx_tokens
    else:
        # Fallback/Check for (N, 2) case manually if needed
        # Commonly for SAW bin1: [Index, Count] -> token based logic inside model
        if sample_data.x_dict["tx"].ndim == 2 and sample_data.x_dict["tx"].shape[1] == 2:
            is_token_based = True
            num_tx_features = num_tx_tokens
        else:
            is_token_based = False
            num_tx_features = sample_data.x_dict["tx"].shape[1]

    print(f"Model Config: is_token_based={is_token_based}, num_features={num_tx_features}")

    # Initialize Model
    model = LitSegger(
        is_token_based=is_token_based,
        num_node_features={"tx": num_tx_features, "bd": 0},  # bd=0 if no boundaries
        init_emb=8,
        hidden_channels=32,
        out_channels=8,
        heads=2,
        num_mid_layers=2,
        aggr="sum",
        learning_rate=1e-3,
    )

    # Trainer
    trainer = Trainer(
        accelerator="auto",  # 'mps' on Mac, 'cuda' on Linux
        devices=1,
        max_epochs=2,  # Short run for tutorial
        default_root_dir=model_dir,
        logger=CSVLogger(model_dir),
        enable_checkpointing=True,
    )

    print("Starting Training...")
    trainer.fit(model=model, datamodule=dm)
    print("Training Complete.")

    # === 4. INFERENCE ===
    print("\n=== 4. Running Inference ===")

    if not HAS_PREDICTION_SUPPORT or not torch.cuda.is_available():
        print("[!] Skipping Inference: Requires NVIDIA GPU and CuPy.")
        print("To run inference, execute this script on a machine with a CUDA-enabled GPU.")
        return

    # If we are here, we have CUDA and CuPy
    print("GPU detected. Running inference on test set...")
    model.eval()
    model.to("cuda")

    all_assignments = []
    # Using the test dataloader for demonstration
    loader = dm.test_dataloader()

    receptive_field = {"k_bd": 3, "dist_bd": 15.0, "k_tx": 3, "dist_tx": 5.0}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            try:
                # predict_batch returns a DataFrame with assignments
                df = predict_batch(
                    model, batch, score_cut=0.5, receptive_field=receptive_field, use_cc=True, knn_method="kd_tree"
                )
                all_assignments.append(df)
            except Exception as e:
                print(f"Batch prediction failed: {e}")
                pass

    if all_assignments:
        full_df = pd.concat(all_assignments, ignore_index=True)
        print(f"Generated assignments for {len(full_df)} transcripts.")
        print(full_df.head())

        # Save results
        out_file = base_out_dir / "segmentation.parquet"
        full_df.to_parquet(out_file)
        print(f"Saved to {out_file}")
    else:
        print("No assignments generated.")


if __name__ == "__main__":
    main()
