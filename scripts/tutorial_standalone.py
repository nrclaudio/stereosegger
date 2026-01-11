import sys
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

# === SETUP ===
# Add local segger source to path if not installed via pip
# Adjust this path to point to your 'tools/segger/src' directory if needed
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent.parent # Assuming tools/segger/scripts/
segger_src = repo_root / "tools" / "segger" / "src"

if str(segger_src) not in sys.path:
    sys.path.insert(0, str(segger_src))

try:
    import segger
    from segger.cli.convert_saw_h5ad_to_segger_parquet import convert_saw_h5ad_to_parquet
    from segger.data.parquet.sample import STSampleParquet
    from segger.training.segger_data_module import SeggerDataModule
    from segger.training.train import LitSegger
    from segger.prediction import torch_predict
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import CSVLogger
    print(f"Segger imported from: {segger.__file__}")
except ImportError as e:
    print(f"Error importing segger: {e}")
    sys.exit(1)

def main():
    # === CONFIGURATION ===
    # Input H5AD file (Update this path to your file)
    # Example path based on your workspace
    input_h5ad = repo_root / "data" / "raw" / "realigned" / "realigned_C04895D5" / "C04895D5_tissue.h5ad"

    # Output Base Directory for this experiment
    experiment_name = "tutorial_C04895D5"
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
    
    if not input_h5ad.exists():
        print(f"Error: Input file {input_h5ad} not found.")
        return

    convert_saw_h5ad_to_parquet(
        h5ad_path=input_h5ad,
        out_dir=inputs_dir,
        bin_pitch=1.0,
        min_count=1,
        labels_tif=None,       # Optional: Path to label image for supervision
        tissue_mask_tif=None,  # Optional: Path to tissue mask
        bbox=None,             # Optional: (xmin, xmax, ymin, ymax) to crop
        gene_name_source="gene_name", # or 'real_gene_name' depending on your h5ad
        top_genes=None         # Optional: Limit to top K genes
    )

    print("Conversion complete. Files generated:")
    for f in inputs_dir.glob("*"):
        print(f" - {f.name}")

    # === 2. DATASET CREATION ===
    print("\n=== 2. Creating Segger Dataset ===")

    # Initialize Sample Loader
    sample = STSampleParquet(
        base_dir=inputs_dir,
        n_workers=4,           # Adjust based on your CPU cores
        sample_type="saw_bin1",
        weights=None           # None = Use learnable embeddings
    )

    # Save (Create) the dataset tiles
    sample.save(
        data_dir=dataset_dir,
        k_bd=3, dist_bd=15.0,  # Boundary connectivity (irrelevant if no labels)
        k_tx=3, dist_tx=5.0,   # Transcript connectivity (used if kdtree mode)
        tx_graph_mode="grid_same_gene",
        grid_connectivity=8,
        within_bin_edges="star",
        bin_pitch=1.0,
        allow_missing_boundaries=True, # Important for inference-only runs
        tile_width=200,
        tile_height=200,
        val_prob=0.1,
        test_prob=0.1,
        neg_sampling_ratio=5.0,
        frac=1.0               # Use 100% of the data
    )

    print(f"Dataset created at {dataset_dir}")

    # === 3. TRAINING ===
    print("\n=== 3. Training Model ===")

    # Data Module
    dm = SeggerDataModule(
        data_dir=dataset_dir,
        batch_size=1,
        num_workers=0  # 0 for safety in notebooks/mac
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

    if hasattr(sample_data["tx"], "token_based") and sample_data["tx"].token_based:
        is_token_based = True
        num_tx_features = num_tx_tokens
    elif "tx" in sample_data.x_dict and sample_data.x_dict["tx"].ndim == 1:
        is_token_based = True
        num_tx_features = num_tx_tokens
    else:
        # Fallback/Check for (N, 2) case manually if needed
        if sample_data.x_dict["tx"].ndim == 2 and sample_data.x_dict["tx"].shape[1] == 2:
            # This handles the specific case of [Index, Count] inputs
            is_token_based = True
            num_tx_features = num_tx_tokens
        else:
            is_token_based = False
            num_tx_features = sample_data.x_dict["tx"].shape[1]

    print(f"Model Config: is_token_based={is_token_based}, num_features={num_tx_features}")

    # Initialize Model
    model = LitSegger(
        is_token_based=is_token_based,
        num_node_features={"tx": num_tx_features, "bd": 0}, # bd=0 if no boundaries
        init_emb=8,
        hidden_channels=32,
        out_channels=8,
        heads=2,
        num_mid_layers=2,
        aggr="sum",
        learning_rate=1e-3
    )

    # Trainer
    trainer = Trainer(
        accelerator="auto", # 'mps' on Mac, 'cuda' on Linux
        devices=1,
        max_epochs=5,       # Short run for tutorial
        default_root_dir=model_dir,
        logger=CSVLogger(model_dir),
    )

    print("Starting Training...")
    trainer.fit(model=model, datamodule=dm)

    # === 4. INFERENCE ===
    print("\n=== 4. Running Inference ===")
    
    model.eval()
    model.to("cpu") # Run on CPU for simple inference

    all_assignments = []
    loaders = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]

    for loader in loaders:
        for batch in tqdm(loader, leave=False):
            try:
                # use_cc=True refines predictions using connected components
                df = torch_predict.predict_batch(model.model, batch, score_cut=0.5, use_cc=True)
                all_assignments.append(df)
            except Exception:
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
        print("No assignments generated (Likely due to missing boundaries in training data for this tutorial).")

if __name__ == "__main__":
    main()
