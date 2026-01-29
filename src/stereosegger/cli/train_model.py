import click
from stereosegger.cli.utils import add_options, CustomFormatter
from pathlib import Path
import logging
from argparse import Namespace

# Path to default YAML configuration file
train_yml = Path(__file__).parent / "configs" / "train" / "default.yaml"

help_msg = "Train the Segger segmentation model."


@click.command(name="train_model", help=help_msg)
@add_options(config_path=train_yml)
@click.option("--dataset_dir", type=Path, required=True, help="Directory containing the processed Segger dataset.")
@click.option(
    "--models_dir", type=Path, required=True, help="Directory to save the trained model and the training logs."
)
@click.option("--sample_tag", type=str, required=True, help="Sample tag for the dataset.")
@click.option("--init_emb", type=int, default=8, help="Size of the embedding layer.")
@click.option("--hidden_channels", type=int, default=32, help="Size of hidden channels in the model.")
@click.option("--num_tx_tokens", type=int, default=500, help="Number of transcript tokens.")
@click.option("--out_channels", type=int, default=8, help="Number of output channels.")
@click.option("--heads", type=int, default=2, help="Number of attention heads.")
@click.option("--num_mid_layers", type=int, default=2, help="Number of mid layers in the model.")
@click.option("--batch_size", type=int, default=1, help="Batch size for training.")
@click.option("--num_workers", type=int, default=0, help="Number of workers for data loading.")
@click.option(
    "--accelerator", type=str, default="cuda", help='Device type to use for training (e.g., "cuda", "cpu").'
)  # Ask for accelerator
@click.option("--max_epochs", type=int, default=300, help="Number of epochs for training.")
@click.option("--save_best_model", type=bool, default=True, help="Whether to save the best model.")  # unused for now
@click.option("--learning_rate", type=float, default=1e-4, help="Learning rate for training.")
@click.option(
    "--pretrained_model_dir",
    type=Path,
    default=None,
    help="Directory containing the pretrained modelDirectory containing the pretrained model to use (if any).",
)
@click.option("--pretrained_model_version", type=int, default=None, help="Version of pretrained model.")
@click.option("--devices", type=int, default=4, help="Number of devices (GPUs) to use.")
@click.option("--strategy", type=str, default="auto", help="Training strategy for the trainer.")
@click.option("--precision", type=str, default="16-mixed", help="Precision for training.")
def train_model(args: Namespace):

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
    logging.info("Loading Xenium datasets...")
    dm = SeggerDataModule(
        data_dir=args.dataset_dir,
        batch_size=args.batch_size,  # Hard-coded batch size
        num_workers=args.num_workers,  # Hard-coded number of workers
    )

    dm.setup()
    logging.info("Done.")

    # Determine num_tx_tokens cleverly
    num_tx_tokens = args.num_tx_tokens
    metadata_path = args.dataset_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            if "num_tx_tokens" in metadata:
                # If user didn't override the default 500 significantly, or if they didn't provide it
                # We assume metadata value is better.
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
        # Check if we need to increase num_tx_tokens based on actual data
        max_found_token = 0
        for i in range(min(10, len(dm.train))):
            tx_x = dm.train[i].x_dict["tx"]
            if tx_x.ndim == 2 and tx_x.shape[1] == 2:
                # [token_id, count]
                current_max = int(tx_x[:, 0].max().item())
            else:
                current_max = int(tx_x.max().item())
            max_found_token = max(max_found_token, current_max)
        
        if max_found_token >= num_tx_tokens:
            num_tx_tokens = max_found_token + 100
            logging.warning(f"Increased num_tx_tokens to {num_tx_tokens} to accommodate found tokens (max={max_found_token})")

    # Initialize model
    if args.pretrained_model_dir is not None:
        logging.info("Loading pretrained model...")
        ls = load_model(args.pretrained_model_dir / "lightning_logs" / f"version_{args.model_version}" / "checkpoints")
    else:
        logging.info("Creating new model...")
        if is_token_based:
            # if the model is token-based, the input is a 1D tensor of token indices
            tx_x = dm.train[0].x_dict["tx"]
            assert tx_x.ndim in (1, 2)
            if tx_x.ndim == 1:
                assert tx_x.dtype == torch.long
            else:
                assert tx_x.shape[1] == 2
            num_tx_features = num_tx_tokens
            print("Using token-based embeddings as node features, number of tokens: ", num_tx_features)
        else:
            # if the model is not token-based, the input is a 2D tensor of scRNAseq embeddings
            assert dm.train[0].x_dict["tx"].ndim == 2
            assert dm.train[0].x_dict["tx"].dtype == torch.float32
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
            aggr="sum",  # Hard-coded value
            learning_rate=args.learning_rate,
        )
    logging.info("Done.")

    # Initialize the Lightning trainer
    trainer = Trainer(
        accelerator=args.accelerator,  # Directly use the specified accelerator
        strategy=args.strategy,  # Hard-coded value
        precision=args.precision,  # Hard-coded value
        devices=args.devices,  # Hard-coded value
        max_epochs=args.max_epochs,  # Hard-coded value
        default_root_dir=args.models_dir,
        logger=CSVLogger(args.models_dir),
    )

    logging.info("Done.")

    # Train model
    logging.info("Training model...")
    trainer.fit(model=ls, datamodule=dm)
    logging.info("Done.")


@click.command(name="slurm", help="Train on Slurm cluster")
@add_options(config_path=train_yml)
def train_slurm(args):
    train_model(args)


@click.group(help="Train the Segger model")
def train():
    pass


train.add_command(train_slurm)

if __name__ == "__main__":
    train_model()
