import os
import shapely
from pyarrow import parquet as pq, compute as pc
from pyarrow import types as pa_types
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
from stereosegger.data.parquet import _utils as utils
from scipy.spatial import cKDTree as KDTree
from stereosegger.data.parquet._ndtree import NDTree
from functools import cached_property
from typing import List, Optional
import logging
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
import torch
from pqdm.processes import pqdm
import random
from stereosegger.data.parquet.transcript_embedding import TranscriptEmbedding
from stereosegger.data.tx_graph import build_grid_gene_bin_edge_index
from types import SimpleNamespace

# Default settings for Stereo-seq (saw_bin1)
DEFAULT_SETTINGS = SimpleNamespace(
    transcripts=SimpleNamespace(
        filename="transcripts.parquet",
        x="x",
        y="y",
        id="transcript_id",
        label="gene_id",
        gene_id="gene_id",
        count="count",
        bx="bx",
        by="by",
        nuclear="overlaps_nucleus",
        cell_id="cell_id",
        filter_substrings=[],
        xy=["x", "y"],
        xyz=["x", "y"],
        columns=["transcript_id", "x", "y", "bx", "by", "gene_id", "count"],
    ),
    boundaries=SimpleNamespace(
        filename="boundaries.parquet",
        id="boundary_id",
        label="boundary_id",
        geometry="geometry",
        columns=["boundary_id", "geometry"],
    ),
)


class STSampleParquet:
    def __init__(self, base_dir, n_workers=1, weights=None):
        self._base_dir = Path(base_dir)
        self.settings = DEFAULT_SETTINGS
        self._transcripts_filepath = self._base_dir / self.settings.transcripts.filename
        self._boundaries_filepath = self._base_dir / self.settings.boundaries.filename
        
        if not self._transcripts_filepath.exists():
            raise FileNotFoundError(f"Transcripts file not found at {self._transcripts_filepath}")
        
        # MODE LOGGING: Explicitly state if this is for Training or Inference
        tx_meta = pq.read_metadata(self._transcripts_filepath)
        cols = tx_meta.schema.names
        has_labels = "overlaps_nucleus" in cols and "cell_id" in cols and self._boundaries_filepath.exists()
        
        if has_labels:
            print(">>> MODE: TRAINING. Labeled data found. Dataset will include ground-truth edges.")
        else:
            print(">>> MODE: INFERENCE. No labels/boundaries found. Dataset will be inference-only (no ground-truth).")

        self.n_workers = n_workers
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.Logger(f"STSample@{base_dir}")
        self._extents = None
        self._transcripts_metadata = None
        self._boundaries_metadata = None
        self._emb_genes = weights.index.to_list() if weights is not None else None
        classes = self.transcripts_metadata["feature_names"]
        self._transcript_embedding = TranscriptEmbedding(np.array(classes), weights)

    @classmethod
    def _get_parquet_metadata(cls, filepath, columns=None):
        size_map = {"BOOLEAN": 1, "INT32": 4, "FLOAT": 4, "INT64": 8, "DOUBLE": 8, "BYTE_ARRAY": 8, "INT96": 12}
        metadata = pq.read_metadata(filepath)
        if columns is None: columns = metadata.schema.names
        # Check if requested columns exist
        missing = set(columns) - set(metadata.schema.names)
        if len(missing) > 0:
            # We allow supervision columns to be missing for prediction
            supervision = {"overlaps_nucleus", "cell_id"}
            if not missing.issubset(supervision):
                raise KeyError(f"Mandatory columns {', '.join(missing - supervision)} not found in {filepath}.")
            # Return metadata for found columns only
            columns = [c for c in columns if c in metadata.schema.names]

        summary = {"n_rows": metadata.num_rows, "n_columns": len(columns), "column_sizes": {}}
        for c in columns:
            i = metadata.schema.names.index(c)
            dtype = metadata.schema[i].physical_type
            summary["column_sizes"][c] = size_map[dtype]
        return summary

    @cached_property
    def transcripts_metadata(self):
        if self._transcripts_metadata is None:
            # We try to load supervision columns if they exist
            check_columns = list(self.settings.transcripts.columns)
            available = pq.read_metadata(self._transcripts_filepath).schema.names
            if "overlaps_nucleus" in available: check_columns.append("overlaps_nucleus")
            if "cell_id" in available: check_columns.append("cell_id")
            
            metadata = STSampleParquet._get_parquet_metadata(self._transcripts_filepath, check_columns)
            table = pq.read_table(self._transcripts_filepath)
            names = pc.unique(table[self.settings.transcripts.label])
            filter_substrings = getattr(self.settings.transcripts, "filter_substrings", [])
            if self._emb_genes is not None:
                names_str = [x.decode("utf-8") if isinstance(x, bytes) else x for x in names.to_pylist()]
                missing_genes = list(set(names_str) - set(self._emb_genes))
                filter_substrings.extend(missing_genes)
            if filter_substrings and (pa_types.is_string(names.type) or pa_types.is_binary(names.type)):
                pattern = "|".join(filter_substrings)
                mask = pc.invert(pc.match_substring_regex(names, pattern))
                filtered_names = pc.filter(names, mask).to_pylist()
            else:
                filtered_names = names.to_pylist()
            metadata["feature_names"] = [x.decode("utf-8") if isinstance(x, bytes) else x for x in filtered_names]
            self._transcripts_metadata = metadata
        return self._transcripts_metadata

    @cached_property
    def boundaries_metadata(self):
        if self._boundaries_metadata is None:
            if not self._boundaries_filepath.exists():
                return {"n_rows": 0, "n_columns": 0, "column_sizes": {}}
            self._boundaries_metadata = STSampleParquet._get_parquet_metadata(self._boundaries_filepath, self.settings.boundaries.columns)
        return self._boundaries_metadata

    @property
    def n_transcripts(self): return self.transcripts_metadata["n_rows"]

    @cached_property
    def extents(self):
        if self._extents is None:
            tx_extents = utils.get_xy_extents(self._transcripts_filepath, *self.settings.transcripts.xy)
            bd_extents = None
            if self._boundaries_filepath.exists():
                boundaries = gpd.read_parquet(self._boundaries_filepath, columns=[self.settings.boundaries.geometry])
                if len(boundaries) > 0:
                    bd_extents = shapely.box(*boundaries.geometry.total_bounds)
            extents = tx_extents if bd_extents is None else tx_extents.union(bd_extents)
            self._extents = shapely.box(*extents.bounds)
        return self._extents

    def _get_balanced_regions(self):
        if self.n_workers == 1: return [self.extents]
        data = pd.read_parquet(self._transcripts_filepath, columns=self.settings.transcripts.xy).values
        return NDTree(data, self.n_workers).boxes

    @staticmethod
    def _setup_directory(data_dir):
        for split in ["train_tiles", "test_tiles", "val_tiles"]:
            (Path(data_dir) / split / "processed").mkdir(parents=True, exist_ok=True)

    def save(self, data_dir, k_bd=3, dist_bd=15, k_tx=3, dist_tx=5, tx_graph_mode="kdtree", **kwargs):
        data_dir = Path(data_dir)
        STSampleParquet._setup_directory(data_dir)
        # Save metadata
        import json
        tx_meta = pq.read_metadata(self._transcripts_filepath)
        has_labels = "overlaps_nucleus" in tx_meta.schema.names and "cell_id" in tx_meta.schema.names and self._boundaries_filepath.exists()
        mode = "training" if has_labels else "inference"
        
        metadata = {
            "num_tx_tokens": len(self.transcripts_metadata["feature_names"]),
            "feature_names": self.transcripts_metadata["feature_names"],
            "k_bd": k_bd,
            "dist_bd": dist_bd,
            "k_tx": k_tx,
            "dist_tx": dist_tx,
            "tx_graph_mode": tx_graph_mode,
            "dataset_mode": mode
        }
        with open(data_dir / "metadata.json", "w") as f: json.dump(metadata, f, indent=4)

        def func(region):
            xm = STInMemoryDataset(sample=self, extents=region)
            tiles = xm._tile(kwargs.get("tile_width"), kwargs.get("tile_height"), kwargs.get("tile_size"))
            if kwargs.get("frac", 1.0) < 1: tiles = random.sample(tiles, int(len(tiles) * kwargs.get("frac")))
            for tile in tiles:
                # If no boundaries exist, everything goes to test_tiles
                if not self._boundaries_filepath.exists():
                    data_type = "test_tiles"
                else:
                    data_type = np.random.choice(["train_tiles", "test_tiles", "val_tiles"], p=[1-(kwargs.get("test_prob",0.2)+kwargs.get("val_prob",0.1)), kwargs.get("test_prob",0.2), kwargs.get("val_prob",0.1)])
                
                xt = STTile(dataset=xm, extents=tile)
                pyg_data = xt.to_pyg_dataset(k_bd=k_bd, dist_bd=dist_bd, k_tx=k_tx, dist_tx=dist_tx, tx_graph_mode=tx_graph_mode, neg_sampling_ratio=kwargs.get("neg_sampling_ratio", 5))
                if pyg_data is not None: torch.save(pyg_data, data_dir / data_type / "processed" / f"{xt.uid}.pt")

        pqdm(self._get_balanced_regions(), func, n_jobs=self.n_workers)


class STInMemoryDataset:
    def __init__(self, sample, extents, margin=10):
        self.sample, self.extents, self.margin, self.settings = sample, extents, margin, sample.settings
        
        available = pq.read_metadata(sample._transcripts_filepath).schema.names
        load_tx_cols = list(self.settings.transcripts.columns)
        if "overlaps_nucleus" in available: load_tx_cols.append("overlaps_nucleus")
        if "cell_id" in available: load_tx_cols.append("cell_id")

        self.transcripts = utils.read_parquet_region(sample._transcripts_filepath, self.settings.transcripts.x, self.settings.transcripts.y, extents.buffer(margin), extra_columns=load_tx_cols)
        
        if sample._boundaries_filepath.exists():
            self.boundaries = utils.read_parquet_region(sample._boundaries_filepath, None, None, extents.buffer(margin), extra_columns=self.settings.boundaries.columns)
        else:
            self.boundaries = pd.DataFrame(columns=self.settings.boundaries.columns)
            
        xy = self.transcripts[[self.settings.transcripts.x, self.settings.transcripts.y]].values
        self.kdtree_tx = KDTree(xy) if len(xy) > 0 else None

    def _tile(self, width=None, height=None, max_size=None):
        if max_size: width = height = np.sqrt(max_size)
        x_min, y_min, x_max, y_max = self.extents.bounds
        tiles = []
        for x in np.arange(x_min, x_max, width):
            for y in np.arange(y_min, y_max, height):
                tile_box = shapely.box(x, y, x + width, y + height)
                if tile_box.intersects(self.extents): tiles.append(tile_box)
        return tiles


class STTile:
    def __init__(self, dataset, extents):
        self.dataset, self.extents, self.margin, self.settings = dataset, extents, dataset.margin, dataset.settings

    @property
    def uid(self):
        x_min, y_min, x_max, y_max = map(int, self.extents.bounds)
        return f"tiles_x={x_min}_y={y_min}_w={x_max-x_min}_h={y_max-y_min}"

    def to_pyg_dataset(self, k_bd=3, dist_bd=15, k_tx=3, dist_tx=5, tx_graph_mode="kdtree", neg_sampling_ratio=5) -> HeteroData:
        pyg_data = HeteroData()
        outset = self.extents.buffer(self.margin)
        tx = self.dataset.transcripts[self.dataset.transcripts[self.settings.transcripts.x].between(outset.bounds[0], outset.bounds[2]) & self.dataset.transcripts[self.settings.transcripts.y].between(outset.bounds[1], outset.bounds[3])]
        if len(tx) == 0: return None
        
        # Nodes
        if tx_graph_mode == "grid_bins":
            grouped = tx.groupby([self.settings.transcripts.bx, self.settings.transcripts.by, self.settings.transcripts.label], sort=False)
            keys = grouped.size().index.to_frame(index=False)
            bx_vals, by_vals, gene_vals = keys[self.settings.transcripts.bx].values, keys[self.settings.transcripts.by].values, keys[self.settings.transcripts.label].values
            tx_positions = np.stack([grouped[self.settings.transcripts.x].mean().values, grouped[self.settings.transcripts.y].mean().values], axis=1)
            pyg_data["tx"].x = torch.tensor(np.stack([gene_vals.astype(np.float32), np.log1p(grouped.size().values).astype(np.float32)], axis=1))
            nbrs_edge_idx = build_grid_gene_bin_edge_index(bx_vals, by_vals, connectivity=8, within_bin_edges="star")
        else:
            tx_positions = tx[self.settings.transcripts.xyz].values
            pyg_data["tx"].x = self.dataset.sample._transcript_embedding.embed(tx[self.settings.transcripts.label])
            tree = KDTree(tx_positions); dist, idx = tree.query(tx_positions, k_tx, distance_upper_bound=dist_tx)
            e = np.argwhere(dist != np.inf).T; e[1] = idx[dist != np.inf]
            nbrs_edge_idx = torch.tensor(e, dtype=torch.long).contiguous()

        if nbrs_edge_idx.shape[1] == 0: return None
        pyg_data["tx"].pos = torch.tensor(tx_positions, dtype=torch.float32)
        pyg_data["tx", "neighbors", "tx"].edge_index = nbrs_edge_idx

        # Boundaries (Allow empty for prediction)
        bd = self.dataset.boundaries[self.dataset.boundaries.geometry.intersects(outset)]
        if len(bd) == 0:
            # For inference, empty bd is okay
            pyg_data["bd"].pos = torch.empty((0, 2), dtype=torch.float32)
            pyg_data["bd"].x = torch.empty((0, 2), dtype=torch.float32)
            pyg_data["tx", "neighbors", "bd"].edge_index = torch.empty((2, 0), dtype=torch.long)
            return pyg_data
        
        polygons = gpd.GeoSeries(bd[self.settings.boundaries.geometry])
        polygons.index = bd[self.settings.boundaries.id].values
        centroids = polygons.centroid.get_coordinates()
        pyg_data["bd"].pos = torch.tensor(centroids.values, dtype=torch.float32)
        props = pd.DataFrame(index=polygons.index); props["area"] = np.log1p(polygons.area)
        props["convexity"] = polygons.convex_hull.area / (polygons.area + 1e-6)
        pyg_data["bd"].x = torch.as_tensor(props.values).float()

        # Neighbors Edges
        tree_bd = KDTree(centroids.values); dist, idx = tree_bd.query(tx_positions[:, :2], k_bd, distance_upper_bound=dist_bd)
        e = np.argwhere(dist != np.inf).T; e[1] = idx[dist != np.inf]
        pyg_data["tx", "neighbors", "bd"].edge_index = torch.tensor(e, dtype=torch.long).contiguous()
        
        # Ground Truth "Belongs" Edges (Only if columns exist)
        if "overlaps_nucleus" in tx.columns and "cell_id" in tx.columns:
            cell_ids_map = {idx: i for (i, idx) in enumerate(polygons.index)}
            is_nuclear = tx["overlaps_nucleus"].astype(bool) & tx["cell_id"].isin(polygons.index)
            row_idx = np.where(is_nuclear)[0]
            col_idx = tx["cell_id"].iloc[row_idx].map(cell_ids_map).values
            blng_edge_idx = torch.tensor(np.stack([row_idx, col_idx])).long()
            
            if blng_edge_idx.numel() > 0:
                pyg_data["tx", "belongs", "bd"].edge_index = blng_edge_idx
                pyg_data, _, _ = RandomLinkSplit(num_val=0, num_test=0, is_undirected=True, edge_types=[("tx", "belongs", "bd")], neg_sampling_ratio=neg_sampling_ratio)(pyg_data)
        
        return pyg_data
