"""order_cells: assign pseudotime from a principal-graph-based shortest
path distance.

The interactive shiny root-picker from upstream monocle3 is not ported.
The function requires either ``root_cells`` or ``root_pr_nodes``.
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


def order_cells(
    adata: ad.AnnData,
    reduction_method: str = "UMAP",
    root_pr_nodes: Sequence[str] | None = None,
    root_cells: Sequence[str] | None = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Assign pseudotime from a principal-graph-based shortest-path distance.

    For each ``root_pr_node`` the nearest *cell* (in UMAP space) is selected,
    and pseudotime is the shortest-path distance from that cell set to every
    other vertex in the cell-augmented principal graph
    (``pr_graph_cell_proj_tree``).

    Parameters
    ----------
    adata : anndata.AnnData
        Must have ``uns["monocle3"]["principal_graph"]["UMAP"]`` and
        ``principal_graph_aux["UMAP"]["pr_graph_cell_proj_closest_vertex"]``
        populated by :func:`learn_graph`.
    reduction_method : str, default "UMAP"
        Only ``"UMAP"`` is supported.
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
    missing_root = [n for n in root_pr_nodes if n not in vertex_names]
    if missing_root:
        raise ValueError(
            f"root_pr_nodes not in principal graph: {missing_root[:5]}..."
        )

    # Store for downstream introspection (branch_nodes / leaf_nodes).
    aux["root_pr_nodes"] = list(root_pr_nodes)
    ensure_monocle_uns(adata)["principal_graph_aux"][reduction_method] = aux

    # Pseudotime on the cell-augmented tree:
    # 1. for each root Y_# node, find the nearest cell in Z-space;
    # 2. use those cells as shortest-path sources in `pr_graph_cell_proj_tree`;
    # 3. per vertex, take the minimum distance to any root cell.
    proj_tree: ig.Graph | None = aux.get("pr_graph_cell_proj_tree")
    Y = aux.get("dp_mst")
    if proj_tree is None or Y is None:
        raise KeyError(
            "principal_graph_aux['UMAP'] is missing pr_graph_cell_proj_tree "
            "and/or dp_mst. Re-run learn_graph after upgrading the port."
        )

    Z = np.asarray(adata.obsm["X_umap"], dtype=np.float64).T  # D × N
    root_col_idx = np.array([vertex_names.index(n) for n in root_pr_nodes], dtype=np.int64)
    Y_root = Y[:, root_col_idx]

    a_norm = np.sum(Y_root ** 2, axis=0)[:, None]
    b_norm = np.sum(Z ** 2, axis=0)[None, :]
    d_root_to_cells = a_norm + b_norm - 2.0 * Y_root.T @ Z  # (n_roots, n_cells)
    closest_cell_idx = np.argmin(d_root_to_cells, axis=1)
    closest_cell_names = [adata.obs_names[i] for i in closest_cell_idx]

    aug_names = list(proj_tree.vs["name"])
    name_to_aug = {n: i for i, n in enumerate(aug_names)}
    try:
        source_ids = [name_to_aug[c] for c in closest_cell_names]
    except KeyError as e:  # pragma: no cover - only triggered by corrupt input
        raise KeyError(
            f"Root-nearest cell {e.args[0]!r} not in pr_graph_cell_proj_tree."
        )

    weights = proj_tree.es["weight"] if "weight" in proj_tree.es.attributes() else None
    dmat = np.asarray(
        proj_tree.distances(source=source_ids, weights=weights), dtype=np.float64
    )
    if dmat.shape[0] > 1:
        pseudo_per_node = dmat.min(axis=0)
    else:
        pseudo_per_node = dmat[0]

    pseudo = np.full(adata.n_obs, np.nan, dtype=np.float64)
    for i, c in enumerate(adata.obs_names):
        j = name_to_aug.get(c)
        if j is not None:
            pseudo[i] = pseudo_per_node[j]

    adata.obs[_PSEUDOTIME_COL] = pd.Series(pseudo, index=adata.obs_names)
    # Also expose pseudotime inside principal_graph_aux as a named series
    # keyed by cell, for downstream diagnostic / branching code.
    aux["pseudotime"] = pd.Series(pseudo, index=adata.obs_names)
    ensure_monocle_uns(adata)["principal_graph_aux"][reduction_method] = aux
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


def _principal_graph(adata: ad.AnnData, reduction_method: str = "UMAP") -> ig.Graph:
    g = get_monocle_uns(adata, "principal_graph", reduction_method)
    if g is None:
        raise KeyError(
            f"No principal graph for {reduction_method}. Run learn_graph first."
        )
    return g


def _root_nodes(adata: ad.AnnData, reduction_method: str = "UMAP") -> pd.Series:
    """Named positions of vertices listed in
    ``principal_graph_aux[[reduction_method]]$root_pr_nodes``.
    """
    g = _principal_graph(adata, reduction_method)
    aux = get_monocle_uns(
        adata, "principal_graph_aux", reduction_method, default={}
    )
    root_names = list(aux.get("root_pr_nodes", []))
    names = list(g.vs["name"])
    idx = {n: i + 1 for i, n in enumerate(names)}
    return pd.Series(
        {n: idx[n] for n in root_names if n in idx}, dtype=int
    )


def _branch_nodes(adata: ad.AnnData, reduction_method: str = "UMAP") -> pd.Series:
    """Vertices with ``degree > 2`` minus roots."""
    g = _principal_graph(adata, reduction_method)
    roots = set(_root_nodes(adata, reduction_method).index)
    names = list(g.vs["name"])
    degs = g.degree()
    return pd.Series(
        {n: i + 1 for i, (n, d) in enumerate(zip(names, degs))
         if d > 2 and n not in roots},
        dtype=int,
    )


def _leaf_nodes(adata: ad.AnnData, reduction_method: str = "UMAP") -> pd.Series:
    """Vertices with ``degree == 1`` minus roots."""
    g = _principal_graph(adata, reduction_method)
    roots = set(_root_nodes(adata, reduction_method).index)
    names = list(g.vs["name"])
    degs = g.degree()
    return pd.Series(
        {n: i + 1 for i, (n, d) in enumerate(zip(names, degs))
         if d == 1 and n not in roots},
        dtype=int,
    )
