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


def build_grid_gene_bin_edge_index(
    bx: Iterable[int] | np.ndarray | torch.Tensor,
    by: Iterable[int] | np.ndarray | torch.Tensor,
    connectivity: int = 8,
    within_bin_edges: str = "none",
) -> torch.Tensor:
    """
    Builds edge index for grid-based gene-bin nodes.
    Each node corresponds to a specific (gene_id, bx, by).

    Nodes within the same bin are connected via 'within_bin_edges' strategy.
    Nodes in adjacent bins are connected via 'connectivity' strategy (connecting hubs).
    """
    bx = np.asarray(bx)
    by = np.asarray(by)
    n_nodes = bx.shape[0]

    if n_nodes == 0:
        return torch.empty((2, 0), dtype=torch.long)

    # Group nodes by bin
    bin_nodes = defaultdict(list)
    for i, (x, y) in enumerate(zip(bx, by)):
        bin_nodes[(int(x), int(y))].append(i)

    src = []
    dst = []

    # 1. Within-bin edges (Star topology)
    # The first node in the list is the "hub".
    hubs = {}  # Map (x, y) -> hub_node_idx

    if within_bin_edges == "star":
        for (x, y), nodes in bin_nodes.items():
            if not nodes:
                continue
            hub = nodes[0]
            hubs[(x, y)] = hub

            # Connect all other nodes to the hub (undirected)
            for other in nodes[1:]:
                src.extend([hub, other])
                dst.extend([other, hub])
    elif within_bin_edges == "none":
        # Even if "none", we need to identify a hub to connect to neighbors
        for (x, y), nodes in bin_nodes.items():
            if nodes:
                hubs[(x, y)] = nodes[0]
    else:
        raise ValueError(f"Unsupported within_bin_edges: {within_bin_edges}")

    # 2. Between-bin edges (Grid topology between hubs)
    offsets = _grid_offsets(connectivity)

    for (x, y), hub_idx in hubs.items():
        for dx, dy in offsets:
            neighbor_hub = hubs.get((x + dx, y + dy))
            if neighbor_hub is not None:
                src.append(hub_idx)
                dst.append(neighbor_hub)

    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor([src, dst], dtype=torch.long).contiguous()
    return edge_index
