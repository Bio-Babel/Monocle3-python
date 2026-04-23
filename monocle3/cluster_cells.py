"""cluster_cells — port of R/cluster_cells.R.

kNN (any of nn2 / annoy / hnsw) → Jaccard-weighted edges → igraph →
Leiden (via leidenalg; leidenbase is itself a port of leidenalg) or
Louvain (via igraph.community_multilevel). Partitions come from the
significance-testing stage in ``compute_partitions`` which reuses the
``pnorm_over_mat`` helper from ``_clustering_cpp``.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy import sparse as sp

from statsmodels.stats.multitest import multipletests

from ._clustering_cpp import jaccard_coeff, pnorm_over_mat
from ._utils import ensure_monocle_uns, get_monocle_uns
from .nearest_neighbors import make_nn_index, search_nn_index, search_nn_matrix

__all__ = ["cluster_cells", "clusters", "partitions"]


_DEFAULT_NN_EUCLIDEAN = {"method": "annoy", "metric": "euclidean"}
_DEFAULT_NN_COSINE = {"method": "annoy", "metric": "cosine"}

_CLUSTER_COL = "monocle3_clusters"
_PARTITION_COL = "monocle3_partitions"


def _make_knn_graph(
    coords: np.ndarray,
    k: int,
    nn_control: dict,
    cell_names: list[str],
    weight: bool,
) -> tuple[ig.Graph, np.ndarray, np.ndarray]:
    """Build the Jaccard-weighted kNN graph used for clustering.

    Returns
    -------
    (g, neighbor_matrix, dist_matrix)
        *g* is the undirected igraph graph on ``cell_names``;
        *neighbor_matrix* / *dist_matrix* drop the self index (column 0).
    """
    n = coords.shape[0]
    k_plus = int(k) + 1
    if k_plus > n:
        k_plus = n  # graceful degrade; R warns but continues.

    method = nn_control.get("method", "annoy")
    if method == "nn2":
        res = search_nn_matrix(coords, coords, k=k_plus, nn_control=nn_control)
    else:
        idx = make_nn_index(coords, nn_control=nn_control)
        res = search_nn_index(coords, nn_index=idx, k=k_plus, nn_control=nn_control)

    nn_idx = res["nn.idx"]
    nn_dist = res["nn.dists"]

    # Swap the self row to the front so column 0 is 'self' — matches R
    # ``swap_nn_row_index_point``.
    row_idx = np.arange(1, n + 1).reshape(-1, 1)
    equal_self = nn_idx == row_idx
    # If self is present somewhere but not at column 0, swap.
    for i in range(n):
        if nn_idx[i, 0] == i + 1:
            continue
        self_col = np.where(equal_self[i])[0]
        if self_col.size:
            j = int(self_col[0])
            nn_idx[i, [0, j]] = nn_idx[i, [j, 0]]
            nn_dist[i, [0, j]] = nn_dist[i, [j, 0]]
        else:
            # Self missing — shift right by one, insert self at column 0.
            nn_idx[i, 1:] = nn_idx[i, :-1]
            nn_idx[i, 0] = i + 1
            nn_dist[i, 1:] = nn_dist[i, :-1]
            nn_dist[i, 0] = 0.0

    # Drop the self column.
    neighbor_matrix = nn_idx[:, 1:]
    dist_matrix = nn_dist[:, 1:]

    # Jaccard coefficient between kNN sets.
    links = jaccard_coeff(neighbor_matrix, bool(weight))
    # Keep only rows with valid source index (defensive).
    links = links[links[:, 0] > 0]
    # Convert to 0-based for igraph.
    edges = [(int(i - 1), int(j - 1)) for i, j, _ in links]
    weights = [float(w) for *_, w in links]

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    g.vs["name"] = list(cell_names)
    return g, neighbor_matrix, dist_matrix


def _run_leiden(
    g: ig.Graph,
    resolution: float | None,
    num_iter: int,
    random_seed: int | None,
    partition_type: str = "CPMVertexPartition",
) -> tuple[np.ndarray, float]:
    """Leiden partition via leidenalg. Returns ``(membership, modularity)``."""
    if partition_type in {
        "ModularityVertexPartition",
        "SignificanceVertexPartition",
        "SurpriseVertexPartition",
    }:
        resolution = None
    else:
        if resolution is None:
            resolution = 0.0001

    cls_map = {
        "CPMVertexPartition": leidenalg.CPMVertexPartition,
        "RBConfigurationVertexPartition": leidenalg.RBConfigurationVertexPartition,
        "ModularityVertexPartition": leidenalg.ModularityVertexPartition,
        "SignificanceVertexPartition": leidenalg.SignificanceVertexPartition,
        "SurpriseVertexPartition": leidenalg.SurpriseVertexPartition,
    }
    partition_cls = cls_map.get(partition_type, leidenalg.CPMVertexPartition)

    kwargs: dict[str, Any] = {}
    if resolution is not None:
        kwargs["resolution_parameter"] = float(resolution)
    if random_seed is not None:
        kwargs["seed"] = int(random_seed)
    kwargs["n_iterations"] = int(num_iter)

    part = leidenalg.find_partition(g, partition_cls, **kwargs)
    membership = np.asarray(part.membership, dtype=np.int64)
    modularity = float(g.modularity(part.membership))
    return membership, modularity


def _run_louvain(
    g: ig.Graph,
    weights: np.ndarray | None,
    louvain_iter: int,
    random_seed: int | None,
) -> tuple[np.ndarray, float]:
    """Louvain via igraph.community_multilevel — take the best of *louvain_iter*.

    Returns ``(membership, modularity)``.
    """
    import random as _random

    louvain_iter = max(1, int(louvain_iter))
    # R cluster_cells.R:373-375: "if(louvain_iter >= 2) random_seed <- NULL".
    # A fixed seed across multiple iterations would collapse them to the same
    # partition, defeating the "best of N" search.
    if louvain_iter >= 2:
        random_seed = None

    best_membership = None
    best_modularity = -np.inf
    for _ in range(louvain_iter):
        if random_seed is not None:
            ig.set_random_number_generator(_random.Random(int(random_seed)))
        part = g.community_multilevel(weights=weights)
        modularity = g.modularity(part.membership, weights=weights)
        if modularity > best_modularity:
            best_modularity = modularity
            best_membership = np.asarray(part.membership, dtype=np.int64)
    assert best_membership is not None
    return best_membership, float(best_modularity)


def _compute_partitions(
    g: ig.Graph,
    membership: np.ndarray,
    qval_thresh: float,
) -> np.ndarray:
    """Partition-assignment helper. Returns a 1-indexed integer array
    matching R's ``cluster_cells.R::compute_partitions`` factor output."""
    membership = np.asarray(membership)
    unique_clusters = np.unique(membership)
    if unique_clusters.size < 2:
        return np.ones_like(membership, dtype=np.int64)

    # 0/1 membership indicator matrix M: (n_cells, n_clusters).
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    rows = np.arange(membership.size)
    cols = np.array([cluster_to_idx[c] for c in membership], dtype=np.int64)
    M = sp.csr_matrix(
        (np.ones(membership.size), (rows, cols)),
        shape=(membership.size, unique_clusters.size),
    )
    A = g.get_adjacency_sparse(attribute=None)  # 0/1 adjacency
    num_links = (M.T @ A @ M).toarray().astype(float)
    np.fill_diagonal(num_links, 0.0)

    edges_per_module = num_links.sum(axis=1)
    total_edges = num_links.sum()

    # When clusters share no inter-cluster edges (total_edges == 0), the
    # theta / num_links / sig_links divisions produce NaN. We let them flow
    # through; the downstream NaN→0 sanitisation collapses sig_links to the
    # zero matrix and the connected-component split then returns one
    # partition per cluster — matching R's behaviour in the same regime.
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_row = (edges_per_module / total_edges).reshape(-1, 1)
        theta_col = (edges_per_module / total_edges).reshape(1, -1)
        theta = theta_row @ theta_col
        var_null = theta * (1.0 - theta) / total_edges

        num_links_ij = num_links / total_edges - theta
        cluster_mat = pnorm_over_mat(num_links_ij, var_null)
        sig_links = num_links / total_edges
    # NaN → 0 for both the link-magnitude matrix and the p-value matrix
    # (matches `compute_partitions` in cluster_cells.R: lines 628-629).
    sig_links = np.where(np.isnan(sig_links), 0.0, sig_links)
    cluster_mat = np.where(np.isnan(cluster_mat), 0.0, cluster_mat)

    # Holm step-down correction — R `stats::p.adjust()` defaults to "holm".
    flat_q = multipletests(cluster_mat.ravel(), method="holm")[1]
    cluster_qmat = flat_q.reshape(cluster_mat.shape)

    sig_links[cluster_qmat > qval_thresh] = 0.0
    np.fill_diagonal(sig_links, 0.0)

    # Preserve weights on the partition graph (R passes `weighted = TRUE` to
    # `graph_from_adjacency_matrix`).
    cluster_g = ig.Graph.Weighted_Adjacency(
        sig_links.tolist(), mode="undirected"
    )
    comp = cluster_g.connected_components().membership
    comp = np.asarray(comp, dtype=np.int64) + 1  # 1-indexed for R parity

    cell_partition = comp[cols]
    return cell_partition.astype(np.int64)


def cluster_cells(
    adata: ad.AnnData,
    reduction_method: str = "UMAP",
    k: int = 20,
    cluster_method: str = "leiden",
    num_iter: int = 2,
    partition_qval: float = 0.05,
    weight: bool = False,
    resolution: float | None = None,
    random_seed: int | None = 42,
    verbose: bool = False,
    nn_control: dict | None = None,
    **kwargs: Any,
) -> ad.AnnData:
    """Cluster cells with Leiden or Louvain on a kNN graph.

    Parameters
    ----------
    adata : anndata.AnnData
    reduction_method : {"UMAP", "tSNE", "PCA", "LSI", "Aligned"}, default "UMAP"
    k : int, default 20
    cluster_method : {"leiden", "louvain"}, default "leiden"
    num_iter : int, default 2
    partition_qval : float, default 0.05
    weight : bool, default False
    resolution : float, optional
        For Leiden with ``CPMVertexPartition`` only. Default 0.0001.
    random_seed : int, default 42
    verbose : bool, default False
        Unused here.
    nn_control : dict, optional
    **kwargs
        Passed through to the Leiden partition constructor.

    Returns
    -------
    anndata.AnnData
        The same object, with ``obs["monocle3_clusters"]`` and
        ``obs["monocle3_partitions"]`` populated.
    """
    del verbose
    if reduction_method not in {"UMAP", "tSNE", "PCA", "LSI", "Aligned"}:
        raise ValueError(
            "reduction_method must be one of 'UMAP', 'tSNE', 'PCA', 'LSI', 'Aligned'"
        )
    if cluster_method not in {"leiden", "louvain"}:
        raise ValueError("cluster_method must be 'leiden' or 'louvain'")
    if resolution is not None and cluster_method == "louvain":
        cluster_method = "leiden"

    key = f"X_{reduction_method.lower()}"
    if key not in adata.obsm:
        raise KeyError(
            f"Dimensionality reduction for {reduction_method} not found"
        )
    coords = np.asarray(adata.obsm[key], dtype=np.float64)

    default_nn = (
        _DEFAULT_NN_EUCLIDEAN
        if reduction_method in {"tSNE", "UMAP"}
        else _DEFAULT_NN_COSINE
    )
    nn_ctrl = dict(default_nn)
    if nn_control:
        nn_ctrl.update(nn_control)

    cell_names = list(adata.obs_names)
    g, _neighbors, _dists = _make_knn_graph(
        coords, k=int(k), nn_control=nn_ctrl, cell_names=cell_names, weight=bool(weight)
    )

    if cluster_method == "leiden":
        membership, modularity = _run_leiden(
            g,
            resolution=resolution,
            num_iter=int(num_iter),
            random_seed=random_seed,
            partition_type=str(kwargs.pop("partition_type", "CPMVertexPartition")),
        )
    else:
        weights = np.asarray(g.es["weight"], dtype=float) if weight else None
        membership, modularity = _run_louvain(
            g, weights=weights, louvain_iter=int(num_iter), random_seed=random_seed
        )

    # R's factor-level indexing is 1-based; align uns storage + obs labels.
    membership = np.asarray(membership, dtype=np.int64) + 1

    cluster_series = pd.Categorical(
        [str(m) for m in membership],
        categories=[str(x) for x in sorted(set(int(v) for v in membership))],
    )

    if len(set(membership)) > 1:
        partition_arr = _compute_partitions(g, membership, qval_thresh=float(partition_qval))
    else:
        partition_arr = np.ones_like(membership, dtype=np.int64)

    partition_series = pd.Categorical(
        [str(p) for p in partition_arr],
        categories=[str(x) for x in sorted(set(int(v) for v in partition_arr))],
    )

    adata.obs[_CLUSTER_COL] = pd.Series(cluster_series, index=adata.obs_names)
    adata.obs[_PARTITION_COL] = pd.Series(partition_series, index=adata.obs_names)

    uns = ensure_monocle_uns(adata)
    uns.setdefault("clusters", {})
    uns["clusters"][reduction_method] = {
        "k": int(k),
        "cluster_method": cluster_method,
        "partition_qval": float(partition_qval),
        "weight": bool(weight),
        "resolution": None if resolution is None else float(resolution),
        "membership": membership.astype(np.int64),
        "partitions": partition_arr.astype(np.int64),
        "modularity": float(modularity),
    }
    return adata


def _clusters_from_uns(
    adata: ad.AnnData, reduction_method: str, kind: str,
) -> pd.Series:
    """Return a categorical Series for ``kind`` ∈ {"membership", "partitions"}.

    R's accessors dispatch per reduction (``methods-cell_data_set.R``), so the
    Python ones read from ``uns["monocle3"]["clusters"][reduction_method]``
    (populated by :func:`cluster_cells`) rather than a single obs column.
    """
    slot = adata.uns.get("monocle3", {}).get("clusters", {}).get(reduction_method)
    if slot is None or kind not in slot:
        raise KeyError(
            f"No {kind} for reduction_method={reduction_method!r}. "
            f"Run cluster_cells(reduction_method={reduction_method!r}) first."
        )
    arr = np.asarray(slot[kind])
    labels = [str(x) for x in arr]
    cats = [str(x) for x in sorted({int(v) for v in arr})]
    return pd.Series(
        pd.Categorical(labels, categories=cats),
        index=adata.obs_names,
        name=kind,
    )


def clusters(adata: ad.AnnData, reduction_method: str = "UMAP") -> pd.Series:
    """Return cluster assignments as a ``pandas.Series`` (categorical)."""
    return _clusters_from_uns(adata, reduction_method, "membership")


def partitions(adata: ad.AnnData, reduction_method: str = "UMAP") -> pd.Series:
    """Return partition assignments as a ``pandas.Series`` (categorical)."""
    return _clusters_from_uns(adata, reduction_method, "partitions")
