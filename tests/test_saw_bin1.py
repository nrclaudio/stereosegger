import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

from segger.cli.convert_saw_h5ad_to_segger_parquet import convert_saw_h5ad_to_parquet
from segger.data.tx_graph import build_grid_same_gene_edge_index
from segger.data.parquet.sample import STSampleParquet


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

            gene_ids = transcripts["gene_id"].to_numpy()
            bx = transcripts["bx"].to_numpy()
            by = transcripts["by"].to_numpy()
            edge_index = build_grid_same_gene_edge_index(gene_ids, bx, by, connectivity=4, within_bin_edges="none")
            edges = set(map(tuple, edge_index.T.cpu().numpy()))

            idx_a = transcripts.index[
                (transcripts["gene_id"] == 0) & (transcripts["bx"] == 0) & (transcripts["by"] == 0)
            ][0]
            idx_b = transcripts.index[
                (transcripts["gene_id"] == 0) & (transcripts["bx"] == 1) & (transcripts["by"] == 0)
            ][0]
            idx_c = transcripts.index[
                (transcripts["gene_id"] == 0) & (transcripts["bx"] == 0) & (transcripts["by"] == 2)
            ][0]

            self.assertIn((idx_a, idx_b), edges)
            self.assertIn((idx_b, idx_a), edges)
            self.assertFalse(any(edge[0] == idx_c or edge[1] == idx_c for edge in edges))
            self.assertTrue(np.all(gene_ids[edge_index[0]] == gene_ids[edge_index[1]]))

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

            sample = STSampleParquet(base_dir=base_dir, sample_type="saw_bin1", n_workers=1)
            sample.save(
                data_dir=data_dir,
                tile_size=3,
                tx_graph_mode="grid_same_gene",
                grid_connectivity=4,
                within_bin_edges="none",
                allow_missing_boundaries=True,
            )

            processed = list((data_dir / "train_tiles" / "processed").glob("*.pt"))
            processed += list((data_dir / "test_tiles" / "processed").glob("*.pt"))
            processed += list((data_dir / "val_tiles" / "processed").glob("*.pt"))
            self.assertTrue(len(processed) > 0)


if __name__ == "__main__":
    unittest.main()
