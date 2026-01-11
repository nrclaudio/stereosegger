from stereosegger.cli.create_dataset_fast import create_dataset
from stereosegger.cli.train_model import train
from stereosegger.cli.predict import predict
import click


# Setup main CLI command
@click.group(help="Command line interface for the StereoSegger segmentation package")
def stereosegger():
    pass


# Add sub-commands to main CLI commands
stereosegger.add_command(create_dataset)
stereosegger.add_command(train)
stereosegger.add_command(predict)
