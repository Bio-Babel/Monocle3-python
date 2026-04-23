"""order_cells — port of R/order_cells.R.

The R shiny root-picker (``select_trajectory_roots``) is intentionally
not ported — see essential-suggestions §5. The function requires either
``root_cells`` or ``root_pr_nodes``.
"""

from __future__ import annotations

from typing import Any, Sequence

import anndata as ad
import igraph as ig
import numpy as np
import pandas as pd

from ._utils import ensure_monocle_uns, get_monocle_uns

__all__ = ["order_cells", "pseudotime", "principal_graph"]


_PSEUDOTIME_COL = "monocle3_pseudotime"


def _find_nearest_vertex(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Closest *target* (column) for each *query* (column). Returns 1-based idx."""
    a_norm = np.sum(query ** 2, axis=0)[:, None]
    b_norm = np.sum(targets ** 2, axis=0)[None, :]
    d = a_norm + b_norm - 2.0 * query.T @ targets
    return np.argmin(d, axis=1) + 1


def order_cells(
    adata: ad.AnnData,
    reduction_method: str = "UMAP",
    root_pr_nodes: Sequence[str] | None = None,
    root_cells: Sequence[str] | None = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Assign pseudotime from a principal-graph-based shortest-path distance.

    Parameters
    ----------
    adata : anndata.AnnData
        Must have ``uns["monocle3"]["principal_graph"]["UMAP"]`` and
        ``principal_graph_aux["UMAP"]["pr_graph_cell_proj_closest_vertex"]``
        populated by :func:`learn_graph`.
    reduction_method : str, default "UMAP"
        Only ``"UMAP"`` is supported (matches R).
    root_pr_nodes : sequence of str, optional
        Principal graph node names (e.g. ``"Y_12"``) where pseudotime is 0.
    root_cells : sequence of str, optional
        Cell names whose closest principal nodes become the root set.
        Exactly one of ``root_pr_nodes`` / ``root_cells`` must be provided.
    verbose : bool, default False
        Unused.

    Returns
    -------
    anndata.AnnData
        With ``obs["monocle3_pseudotime"]`` populated.
    """
    del verbose
    if reduction_method != "UMAP":
        raise ValueError("Only 'UMAP' is supported for order_cells")

    g: ig.Graph | None = get_monocle_uns(
        adata, "principal_graph", reduction_method
    )
    aux = get_monocle_uns(adata, "principal_graph_aux", reduction_method, default={})

    if g is None or g.vcount() == 0:
        raise ValueError("No principal graph. Run learn_graph first.")

    if root_pr_nodes is None and root_cells is None:
        raise ValueError("Provide root_pr_nodes or root_cells.")
    if root_pr_nodes is not None and root_cells is not None:
        raise ValueError("Provide only one of root_pr_nodes / root_cells.")

    vertex_names = list(g.vs["name"])
    if root_cells is not None:
        closest_vertex_df: pd.DataFrame = aux["pr_graph_cell_proj_closest_vertex"]
        missing = [c for c in root_cells if c not in closest_vertex_df.index]
        if missing:
            raise KeyError(
                f"root_cells missing from AnnData: {missing[:5]}..."
            )
        closest = closest_vertex_df.loc[list(root_cells), "V1"].to_numpy()
        root_pr_nodes = list({f"Y_{int(v)}" for v in closest})

    assert root_pr_nodes is not None
    missing = [n for n in root_pr_nodes if n not in vertex_names]
    if missing:
        raise ValueError(
            f"root_pr_nodes not in principal graph: {missing[:5]}..."
        )

    # Store for downstream introspection (branch_nodes / leaf_nodes).
    aux["root_pr_nodes"] = list(root_pr_nodes)
    ensure_monocle_uns(adata)["principal_graph_aux"][reduction_method] = aux

    # Pseudotime = shortest-path distance from any root node to each cell, where
    # the cell-wise graph assigns each cell to its closest principal vertex.
    closest_vertex_df: pd.DataFrame = aux["pr_graph_cell_proj_closest_vertex"]
    cell_to_vertex = closest_vertex_df["V1"].astype(int).to_numpy()  # 1-based

    # Compute shortest-path distances on the principal graph from the root set.
    root_ids = [vertex_names.index(n) for n in root_pr_nodes]
    dist_matrix = np.asarray(g.distances(source=root_ids))
    dist_matrix = np.where(np.isinf(dist_matrix), np.inf, dist_matrix)
    # For each vertex, minimum distance to any root.
    per_vertex = dist_matrix.min(axis=0)

    pseudo = np.array(
        [per_vertex[v - 1] for v in cell_to_vertex], dtype=float
    )

    adata.obs[_PSEUDOTIME_COL] = pd.Series(pseudo, index=adata.obs_names)
    return adata


def pseudotime(adata: ad.AnnData, reduction_method: str = "UMAP") -> pd.Series:
    """Return ``adata.obs["monocle3_pseudotime"]`` as a Series."""
    if _PSEUDOTIME_COL not in adata.obs.columns:
        raise KeyError(f"{_PSEUDOTIME_COL} not found. Run order_cells first.")
    _ = reduction_method
    return adata.obs[_PSEUDOTIME_COL]


def principal_graph(adata: ad.AnnData) -> dict:
    """Return the ``uns["monocle3"]["principal_graph"]`` mapping."""
    val = adata.uns.get("monocle3", {}).get("principal_graph", {})
    if not val:
        raise KeyError(
            "Principal graph not found. Run learn_graph first."
        )
    return dict(val)
