from typing import List, Optional, Callable
from torch_geometric.data import Dataset, Data
import glob
import os
from pathlib import Path
import torch


class STPyGDataset(Dataset):
    """
    A dataset class for handling training using spatial
    transcriptomics data, loading from disk.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        print(f"Indexing dataset at {self.processed_dir}...")
        
        if not os.path.exists(self.processed_dir):
            print(f"Warning: Processed directory {self.processed_dir} does not exist.")
            self._processed_file_paths = []
        else:
            # Faster than glob.glob for large directories
            paths = [
                f.path
                for f in os.scandir(self.processed_dir)
                if f.is_file() and f.name.startswith("tiles_") and f.name.endswith(".pt")
            ]
            self._processed_file_paths = sorted(paths)
        print(f"Found {len(self._processed_file_paths)} tiles.")

    @property
    def raw_file_names(self) -> List[str]:
        """
        Return a list of raw file names in the raw directory.

        Returns:
            List[str]: List of raw file names.
        """
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self) -> List[str]:
        """
        Return a list of processed file names in the processed directory.

        Returns:
            List[str]: List of processed file names.
        """
        return [os.path.basename(p) for p in self._processed_file_paths]

    def len(self) -> int:
        """
        Return the number of processed files.

        Returns:
            int: Number of processed files.
        """
        return len(self._processed_file_paths)

    def get(self, idx: int) -> Data:
        """
        Get a processed data object.

        Args:
            idx (int): Index of the data object to retrieve.

        Returns:
            Data: The processed data object.
        """
        data = torch.load(self._processed_file_paths[idx], weights_only=False)
        # this is an issue in PyG's RandomLinkSplit, dimensions are not consistent if there is only one edge in the graph
        if hasattr(data["tx", "belongs", "bd"], "edge_label_index"):
            if data["tx", "belongs", "bd"].edge_label_index.dim() == 1:
                data["tx", "belongs", "bd"].edge_label_index = data["tx", "belongs", "bd"].edge_label_index.unsqueeze(1)
                data["tx", "belongs", "bd"].edge_label = data["tx", "belongs", "bd"].edge_label.unsqueeze(0)
            assert data["tx", "belongs", "bd"].edge_label_index.dim() == 2
        return data
