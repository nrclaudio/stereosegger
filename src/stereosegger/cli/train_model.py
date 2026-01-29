import argparse
from stereosegger.cli.utils import CustomFormatter
from pathlib import Path
import logging

help_msg = "Train the Segger segmentation model."


def train_model(args):
    # Setup logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[ch])

    # Import packages
    logging.info("Importing packages...")
    import torch
    import json
    from stereosegger.training.train import LitSegger
    from stereosegger.training.segger_data_module import SeggerDataModule
    from stereosegger.prediction.predict_parquet import load_model
    from lightning.pytorch.loggers import CSVLogger
    from pytorch_lightning import Trainer

    logging.info("Done.")

    # Load datasets
    logging.info("Loading Stereo-seq datasets...")
    dm = SeggerDataModule(
        data_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dm.setup()
    logging.info("Done.")

    # Check for unlabeled data (Stop training if no labels found)
    if dm.train is None or len(dm.train) == 0:
        raise ValueError(f"No training tiles found in {args.dataset_dir}. Did you run create_dataset on unlabeled data?")
    
    # Check first tile for labels
    first_tile = dm.train[0]
    if not hasattr(first_tile["tx", "belongs", "bd"], "edge_label_index"):
        raise ValueError(
            f"The dataset at {args.dataset_dir} does not contain training labels (belongs edges). "
            "Training requires labeled data. Please ensure boundaries.parquet and supervision columns "
            "were provided during dataset creation."
        )

    # Determine num_tx_tokens
    num_tx_tokens = args.num_tx_tokens
    metadata_path = args.dataset_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            
            # BLOCK TRAINING ON INFERENCE-ONLY DATA
            if metadata.get("dataset_mode") == "inference":
                raise ValueError(
                    f"The dataset at {args.dataset_dir} was created in INFERENCE mode (no labels). "
                    "You cannot train on this dataset. Please recreate the dataset with labeled data."
                )

            if "num_tx_tokens" in metadata:
                if args.num_tx_tokens == 500:
                    num_tx_tokens = metadata["num_tx_tokens"]
                    logging.info(f"Using num_tx_tokens={num_tx_tokens} from metadata.json")

    # Final safety check: scan first few batches if token based
    logging.info("Checking dataset for token-based features...")
    token_flag = getattr(dm.train[0]["tx"], "token_based", None)
    if token_flag is not None:
        if torch.is_tensor(token_flag):
            is_token_based = bool(token_flag.item())
        else:
            is_token_based = bool(token_flag)
    else:
        is_token_based = dm.train[0].x_dict["tx"].ndim == 1

    if is_token_based:
        max_found_token = 0
        for i in range(min(10, len(dm.train))):
            tx_x = dm.train[i].x_dict["tx"]
            if tx_x.ndim == 2 and tx_x.shape[1] == 2:
                current_max = int(tx_x[:, 0].max().item())
            else:
                current_max = int(tx_x.max().item())
            max_found_token = max(max_found_token, current_max)

        if max_found_token >= num_tx_tokens:
            num_tx_tokens = max_found_token + 100
            logging.warning(
                f"Increased num_tx_tokens to {num_tx_tokens} to accommodate found tokens (max={max_found_token})"
            )

    # Initialize model
    if args.pretrained_model_dir is not None:
        logging.info("Loading pretrained model...")
        ls = load_model(args.pretrained_model_dir / "lightning_logs" / f"version_{args.pretrained_model_version}" / "checkpoints")
    else:
        logging.info("Creating new model...")
        if is_token_based:
            tx_x = dm.train[0].x_dict["tx"]
            num_tx_features = num_tx_tokens
            print("Using token-based embeddings as node features, number of tokens: ", num_tx_features)
        else:
            num_tx_features = dm.train[0].x_dict["tx"].shape[1]
            print("Using scRNAseq embeddings as node features, number of features: ", num_tx_features)
        num_bd_features = dm.train[0].x_dict["bd"].shape[1]
        print("Number of boundary node features: ", num_bd_features)
        ls = LitSegger(
            is_token_based=is_token_based,
            num_node_features={"tx": num_tx_features, "bd": num_bd_features},
            init_emb=args.init_emb,
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            heads=args.heads,
            num_mid_layers=args.num_mid_layers,
            aggr="sum",
            learning_rate=args.learning_rate,
        )
    logging.info("Done.")

    # Initialize the Lightning trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        precision=args.precision,
        devices=args.devices,
        max_epochs=args.max_epochs,
        default_root_dir=args.models_dir,
        logger=CSVLogger(args.models_dir),
    )

    logging.info("Done.")

    # Train model
    logging.info("Training model...")
    trainer.fit(model=ls, datamodule=dm)
    logging.info("Done.")


def main():
    parser = argparse.ArgumentParser(description=help_msg)
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Directory containing the processed Segger dataset.")
    parser.add_argument("--models_dir", type=Path, required=True, help="Directory to save the trained model.")
    parser.add_argument("--sample_tag", type=str, required=True, help="Sample tag for the dataset.")
    parser.add_argument("--init_emb", type=int, default=8, help="Size of the embedding layer.")
    parser.add_argument("--hidden_channels", type=int, default=32, help="Size of hidden channels in the model.")
    parser.add_argument("--num_tx_tokens", type=int, default=500, help="Number of transcript tokens.")
    parser.add_argument("--out_channels", type=int, default=8, help="Number of output channels.")
    parser.add_argument("--heads", type=int, default=2, help="Number of attention heads.")
    parser.add_argument("--num_mid_layers", type=int, default=2, help="Number of mid layers in the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--accelerator", type=str, default="cuda", help='Device type ("cuda", "cpu").')
    parser.add_argument("--max_epochs", type=int, default=300, help="Number of epochs for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--pretrained_model_dir", type=Path, default=None, help="Directory containing pretrained model.")
    parser.add_argument("--pretrained_model_version", type=int, default=None, help="Version of pretrained model.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices (GPUs) to use.")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy.")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Precision for training.")

    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()