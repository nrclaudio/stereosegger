from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader
import os
from pathlib import Path
from stereosegger.data.parquet.pyg_dataset import STPyGDataset


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

    def setup(self, stage=None):
        if self.train is None:
            train_dir = self.data_dir / "train_tiles"
            if train_dir.exists():
                self.train = STPyGDataset(root=train_dir)
        if self.test is None:
            test_dir = self.data_dir / "test_tiles"
            if test_dir.exists():
                self.test = STPyGDataset(root=test_dir)
        if self.val is None:
            val_dir = self.data_dir / "val_tiles"
            if val_dir.exists():
                self.val = STPyGDataset(root=val_dir)

        self.loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        if self.train is not None and len(self.train) > 0:
            return DataLoader(self.train, shuffle=True, **self.loader_kwargs)
        return None

    def test_dataloader(self):
        if self.test is not None and len(self.test) > 0:
            return DataLoader(self.test, shuffle=False, **self.loader_kwargs)
        return None

    def val_dataloader(self):
        if self.val is not None and len(self.val) > 0:
            return DataLoader(self.val, shuffle=False, **self.loader_kwargs)
        return None