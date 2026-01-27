from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader
import os
import sys
from pathlib import Path
from stereosegger.data.parquet.pyg_dataset import STPyGDataset


# TODO: Add documentation
class SeggerDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: os.PathLike,
        batch_size: int = 4,
        num_workers: int = 1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = None
        self.test = None
        self.val = None

    # TODO: Add documentation
    def setup(self, stage=None):
        if self.train is None:
            self.train = STPyGDataset(root=self.data_dir / "train_tiles")
        if self.test is None:
            self.test = STPyGDataset(root=self.data_dir / "test_tiles")
        if self.val is None:
            self.val = STPyGDataset(root=self.data_dir / "val_tiles")

        self.loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # TODO: Add documentation
    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, **self.loader_kwargs)

    # TODO: Add documentation
    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, **self.loader_kwargs)

    # TODO: Add documentation
    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, **self.loader_kwargs)
