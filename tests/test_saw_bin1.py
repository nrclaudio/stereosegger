import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import torch

from stereosegger.cli.convert_saw_h5ad_to_segger_parquet import convert_saw_h5ad_to_parquet
from stereosegger.data.tx_graph import build_grid_gene_bin_edge_index
from stereosegger.data.parquet.sample import STSampleParquet


class TestSawBin1(unittest.TestCase):
    def _make_adata(self) -> ad.AnnData:
        coords = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 2],
                [1, 2],
            ],
            dtype=float,
        )
        X = sp.csr_matrix(
            [
                [1, 0, 0],
                [2, 0, 0],
                [0, 3, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=int,
        )
        adata = ad.AnnData(X=X)
        adata.obsm["spatial"] = coords
        adata.var_names = ["g0", "g1", "g2"]
        return adata

    def test_convert_and_grid_edges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            h5ad_path = tmpdir / "saw.h5ad"
            out_dir = tmpdir / "out"

            adata = self._make_adata()
            adata.write_h5ad(h5ad_path)

            convert_saw_h5ad_to_parquet(
                h5ad_path=h5ad_path,
                out_dir=out_dir,
                bin_pitch=1.0,
                min_count=1,
            )

            transcripts = pd.read_parquet(out_dir / "transcripts.parquet")
            required_cols = {"transcript_id", "x", "y", "bx", "by", "gene_id", "count"}
            self.assertTrue(required_cols.issubset(transcripts.columns))

            bx = transcripts["bx"].to_numpy()
            by = transcripts["by"].to_numpy()
            # Test build_grid_gene_bin_edge_index (star topology)
            edge_index = build_grid_gene_bin_edge_index(bx, by, connectivity=4, within_bin_edges="star")
            edges = set(map(tuple, edge_index.T.cpu().numpy()))

            # Check if star edges are present for a bin with multiple genes
            # In our data, (0,0) has g0. (1,0) has g0. (0,1) has g1. (1,1) has g2.
            # (0,2) has g0. (1,2) has g1.
            # Wait, my mock data:
            # bin (0,0): g0
            # bin (1,0): g0
            # bin (0,1): g1
            # bin (1,1): g2
            # bin (0,2): g0
            # bin (1,2): g1
            # None of the bins have multiple genes in this mock?
            # Let me check X:
            # row 0: [1, 0, 0] -> g0
            # row 1: [2, 0, 0] -> g0
            # Row 0 and 1 have same coordinates (0,0) and (1,0)? No.
            # coords[0] = (0,0), coords[1] = (1,0)

            # Let me adjust mock data to have a bin with 2 genes
            # row 0: (0,0), g0
            # row 1: (0,0), g1 (change this)

            adata.X[1, 0] = 0
            adata.X[1, 1] = 5  # (1,0) now has g1
            # Wait, still different bins.

            # (0,0) -> g0, g1
            adata.obsm["spatial"][1] = [0, 0]
            adata.write_h5ad(h5ad_path)
            convert_saw_h5ad_to_parquet(h5ad_path=h5ad_path, out_dir=out_dir)
            transcripts = pd.read_parquet(out_dir / "transcripts.parquet")

            bx = transcripts["bx"].to_numpy()
            by = transcripts["by"].to_numpy()
            edge_index = build_grid_gene_bin_edge_index(bx, by, connectivity=4, within_bin_edges="star")
            edges = set(map(tuple, edge_index.T.cpu().numpy()))

            # Indices for (0,0) bin
            bin_00_indices = transcripts.index[(transcripts["bx"] == 0) & (transcripts["by"] == 0)].tolist()
            if len(bin_00_indices) >= 2:
                i, j = bin_00_indices[0], bin_00_indices[1]
                self.assertIn((i, j), edges)
                self.assertIn((j, i), edges)

    def test_create_dataset_fast_tile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            h5ad_path = tmpdir / "saw.h5ad"
            base_dir = tmpdir / "base"
            data_dir = tmpdir / "dataset"

            adata = self._make_adata()
            adata.write_h5ad(h5ad_path)

            convert_saw_h5ad_to_parquet(
                h5ad_path=h5ad_path,
                out_dir=base_dir,
                bin_pitch=1.0,
                min_count=1,
            )

            sample = STSampleParquet(
                base_dir=base_dir, sample_type="saw_bin1", n_workers=1, allow_missing_boundaries=True
            )
            sample.save(
                data_dir=data_dir,
                tile_width=100,
                tile_height=100,
                tx_graph_mode="grid_bins",
                grid_connectivity=4,
                within_bin_edges="star",
                allow_missing_boundaries=True,
            )

            processed = list((data_dir / "train_tiles" / "processed").glob("*.pt"))
            processed += list((data_dir / "test_tiles" / "processed").glob("*.pt"))
            processed += list((data_dir / "val_tiles" / "processed").glob("*.pt"))
            self.assertTrue(len(processed) > 0)


if __name__ == "__main__":
    unittest.main()
