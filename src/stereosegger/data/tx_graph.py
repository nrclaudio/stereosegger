from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Tuple

import numpy as np
import torch


def _grid_offsets(connectivity: int) -> Tuple[Tuple[int, int], ...]:
    if connectivity == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        return (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
    msg = f"Unsupported grid connectivity: {connectivity}. Expected 4 or 8."
    raise ValueError(msg)


def build_grid_same_gene_edge_index(
    gene_ids: Iterable[int] | np.ndarray | torch.Tensor,
    bx: Iterable[int] | np.ndarray | torch.Tensor,
    by: Iterable[int] | np.ndarray | torch.Tensor,
    connectivity: int = 8,
    within_bin_edges: str = "none",
) -> torch.Tensor:
    gene_ids = np.asarray(gene_ids)
    bx = np.asarray(bx)
    by = np.asarray(by)

    n_nodes = gene_ids.shape[0]
    if n_nodes == 0:
        return torch.empty((2, 0), dtype=torch.long)

    offsets = _grid_offsets(connectivity)
    key_to_idx = {(int(g), int(x), int(y)): i for i, (g, x, y) in enumerate(zip(gene_ids, bx, by))}

    src = []
    dst = []
    for i, (g, x, y) in enumerate(zip(gene_ids, bx, by)):
        g = int(g)
        x = int(x)
        y = int(y)
        for dx, dy in offsets:
            j = key_to_idx.get((g, x + dx, y + dy))
            if j is not None and j != i:
                src.append(i)
                dst.append(j)

    if within_bin_edges not in {"none", "star"}:
        msg = f"Unsupported within_bin_edges mode: {within_bin_edges}. Expected 'none' or 'star'."
        raise ValueError(msg)
    if within_bin_edges == "star":
        bins = defaultdict(list)
        for i, (x, y) in enumerate(zip(bx, by)):
            bins[(int(x), int(y))].append(i)
        for nodes in bins.values():
            if len(nodes) < 2:
                continue
            anchor = nodes[0]
            for other in nodes[1:]:
                src.extend([anchor, other])
                dst.extend([other, anchor])

    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor([src, dst], dtype=torch.long).contiguous()
    return edge_index


def build_grid_bin_edge_index(
    bx: Iterable[int] | np.ndarray | torch.Tensor,
    by: Iterable[int] | np.ndarray | torch.Tensor,
    connectivity: int = 8,
) -> Tuple[torch.Tensor, np.ndarray]:
    bx = np.asarray(bx)
    by = np.asarray(by)
    if bx.size == 0:
        return torch.empty((2, 0), dtype=torch.long), np.empty((0, 2), dtype=int)

    bins = np.unique(np.stack([bx, by], axis=1), axis=0)
    bin_to_idx = {(int(x), int(y)): i for i, (x, y) in enumerate(bins)}

    offsets = _grid_offsets(connectivity)
    src = []
    dst = []
    for i, (x, y) in enumerate(bins):
        x = int(x)
        y = int(y)
        for dx, dy in offsets:
            j = bin_to_idx.get((x + dx, y + dy))
            if j is not None and j != i:
                src.append(i)
                dst.append(j)

    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long), bins

    edge_index = torch.tensor([src, dst], dtype=torch.long).contiguous()
    return edge_index, bins
