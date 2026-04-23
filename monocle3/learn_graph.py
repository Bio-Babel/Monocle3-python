"""learn_graph — port of R/learn_graph.R (SimplePPT-style reversed graph embedding).

This is the largest single piece of monocle3's numerical orchestration
(~1.3 kLOC in R). The port preserves the R algorithm step-for-step:

1. Per partition, pick ``ncenter`` initial centres via evenly-spaced
   subsampling of the UMAP coords → k-means → density-based medoid
   selection over a kNN graph.
2. Run ``calc_principal_graph`` (SimplePPT): alternate MST over pairwise
   centroid distances with a soft-assignment step and a closed-form
   centroid update until the objective stops improving.
3. Optionally close loops between graph tips whose PAGA q-value passes
   ``qval_thresh`` **and** whose geodesic / euclidean ratios meet the
   caller-configured thresholds.
4. Optionally prune branches shorter than ``minimal_branch_len``.
5. Merge per-partition graphs; project every cell onto the nearest
   centroid and record the mapping in ``uns["monocle3"]["principal_graph_aux"]``.

All numeric defaults match R verbatim per essential-suggestions §2.2.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import igraph as ig
import numpy as np
import pandas as pd
from scipy import sparse as sp

from ._utils import ensure_monocle_uns
from .nearest_neighbors import search_nn_matrix

__all__ = ["learn_graph"]


# ---------------------------------------------------------------------------
# SimplePPT kernel
# ---------------------------------------------------------------------------


def _pairwise_sq_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared pairwise distances, shape (A.shape[1], B.shape[1])."""
    # A, B are (D, n) and (D, m) per R convention.
    a_norm = np.sum(A ** 2, axis=0)[:, None]
    b_norm = np.sum(B ** 2, axis=0)[None, :]
    return a_norm + b_norm - 2.0 * A.T @ B


def _soft_assignment(X: np.ndarray, C: np.ndarray, sigma: float):
    """Match R's ``soft_assignment``.

    Returns ``(P, obj)`` where ``P`` is a cells × centroids membership
    matrix and ``obj`` the per-cell entropy-like cost summed across cells.
    """
    dist_XC = _pairwise_sq_dist(X, C)  # N × K
    min_dist = dist_XC.min(axis=1, keepdims=True)
    shifted = dist_XC - min_dist
    Phi = np.exp(-shifted / sigma)
    rowsums = Phi.sum(axis=1, keepdims=True)
    P = Phi / rowsums
    obj = -sigma * float(np.sum(np.log(rowsums.ravel()) - min_dist.ravel() / sigma))
    return P, obj


def _generate_centers(
    X: np.ndarray, W: np.ndarray, P: np.ndarray, gamma: float
) -> np.ndarray:
    """Closed-form centroid update — matches R ``generate_centers`` (eq. 22)."""
    colsum_W = W.sum(axis=0)
    colsum_P = P.sum(axis=0)
    Q = 2.0 * (np.diag(colsum_W) - W) + gamma * np.diag(colsum_P)
    B = gamma * X @ P  # D × K
    C = np.linalg.solve(Q.T, B.T).T  # C @ Q = B  →  C = B @ Q^-1
    return C


def _calc_principal_graph(
    X: np.ndarray,
    C0: np.ndarray,
    maxiter: int,
    eps: float,
    L1_gamma: float,
    L1_sigma: float,
) -> dict:
    """SimplePPT core — translated from R ``calc_principal_graph``."""
    C = np.array(C0, dtype=np.float64, copy=True)
    K = C.shape[1]
    objs: list[float] = []
    W = np.zeros((K, K), dtype=np.float64)
    P = np.zeros((X.shape[1], K), dtype=np.float64)
    stree_out = np.zeros((K, K), dtype=np.float64)

    for iteration in range(1, int(maxiter) + 1):
        # Pairwise sq distances between centroids.
        Phi = _pairwise_sq_dist(C, C)
        # Symmetrise to avoid float asymmetry tripping igraph's check.
        Phi = 0.5 * (Phi + Phi.T)
        # MST over the centroid graph.
        g = ig.Graph.Weighted_Adjacency(
            Phi.tolist(), mode="undirected", attr="weight", loops=False
        )
        mst = g.spanning_tree(weights=g.es["weight"])
        stree = np.zeros((K, K), dtype=np.float64)
        for e in mst.es:
            i, j = e.tuple
            w = float(e["weight"])
            stree[i, j] = w
            stree[j, i] = w
        stree_out = stree
        W = (stree != 0).astype(np.float64)
        obj_W = float(stree.sum())

        P, obj_P = _soft_assignment(X, C, L1_sigma)
        obj = obj_W + L1_gamma * obj_P
        objs.append(obj)

        if iteration > 1:
            rel_diff = abs(objs[iteration - 2] - obj) / max(abs(objs[iteration - 2]), 1e-12)
            if rel_diff < eps:
                break

        C = _generate_centers(X, W, P, L1_gamma)

    return {"X": X, "C": C, "W": W, "P": P, "objs": np.asarray(objs),
            "Y": C, "R": P, "stree": stree_out}


# ---------------------------------------------------------------------------
# Helpers — projection, k-means, nearest vertex
# ---------------------------------------------------------------------------


def _find_nearest_vertex(data: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """R ``find_nearest_vertex`` — returns 1-based nearest target index per
    data column.

    Both ``data`` and ``targets`` are (D, n) and (D, m).
    """
    d = _pairwise_sq_dist(data, targets)
    return np.argmin(d, axis=1) + 1


def _cal_ncenter(n_clusters: int, n_cells: int, nodes_per_log10: int = 15) -> int:
    return int(round(n_clusters * nodes_per_log10 * np.log10(max(n_cells, 1))))


def _kmeans_with_init(
    points: np.ndarray, n_clusters: int, seed: int = 2016
) -> tuple[np.ndarray, np.ndarray]:
    """R's ``kmeans`` with evenly-spaced initial centers + tiny noise."""
    from sklearn.cluster import KMeans

    n = points.shape[0]
    idx = np.linspace(0, n - 1, num=n_clusters).astype(int)
    centers_init = points[idx].astype(float)
    rng = np.random.default_rng(seed)
    centers_init = centers_init + rng.normal(scale=1e-10, size=centers_init.shape)

    km = KMeans(
        n_clusters=n_clusters,
        init=centers_init,
        n_init=1,
        max_iter=100,
        random_state=seed,
    ).fit(points)
    return km.labels_, km.cluster_centers_


def _project_point_to_line_segment(p: np.ndarray, AB: np.ndarray) -> np.ndarray:
    """R ``project_point_to_line_segment`` — closest point on segment [A,B] to *p*."""
    A = AB[:, 0]
    B = AB[:, 1]
    v = B - A
    denom = float(np.dot(v, v))
    if denom == 0:
        return A.copy()
    t = float(np.dot(p - A, v) / denom)
    if t < 0:
        return A.copy()
    if t > 1:
        return B.copy()
    return A + t * v


# ---------------------------------------------------------------------------
# Loop closure + pruning
# ---------------------------------------------------------------------------


def _partition_q_matrix(g: ig.Graph, membership: np.ndarray) -> np.ndarray:
    """Compute the PAGA-like q-value matrix used by ``connect_tips``."""
    from ._clustering_cpp import pnorm_over_mat
    from scipy.stats import false_discovery_control

    unique = np.unique(membership)
    k = unique.size
    row = np.arange(membership.size)
    col = np.array(
        [np.where(unique == m)[0][0] for m in membership], dtype=np.int64
    )
    M = sp.csr_matrix((np.ones(membership.size), (row, col)),
                      shape=(membership.size, k))
    A = g.get_adjacency_sparse(attribute=None)
    num_links = (M.T @ A @ M).toarray().astype(float)
    np.fill_diagonal(num_links, 0.0)
    edges_per = num_links.sum(axis=1)
    total = num_links.sum()
    if total <= 0:
        return np.ones((k, k))
    theta = (edges_per / total)[:, None] @ (edges_per / total)[None, :]
    var_null = theta * (1.0 - theta) / total
    num_links_ij = num_links / total - theta
    pmat = pnorm_over_mat(num_links_ij, var_null)
    pmat = np.where(np.isnan(pmat), 1.0, pmat)
    qmat = false_discovery_control(pmat.ravel(), method="bh").reshape(k, k)
    return qmat


def _connect_tips(
    stree: np.ndarray,
    Y: np.ndarray,
    reduced_coords: np.ndarray,
    membership: np.ndarray,
    euclidean_distance_ratio: float,
    geodesic_distance_ratio: float,
    qval_thresh: float = 0.05,
    k: int = 25,
) -> np.ndarray:
    """Add tip-to-tip edges that pass geodesic + euclidean + q-value tests."""
    stree01 = (stree != 0).astype(np.float64)
    g_old = ig.Graph.Adjacency(stree01.tolist(), mode="undirected")
    degrees = np.asarray(g_old.degree())
    tip_points = np.where(degrees == 1)[0]

    if tip_points.size < 2:
        return stree

    # Build a kNN graph on the cells to estimate q-values.
    k_used = max(5, min(int(k), reduced_coords.shape[0] - 1))
    nn = search_nn_matrix(
        reduced_coords,
        reduced_coords,
        k=k_used + 1,
        nn_control={"method": "nn2", "metric": "euclidean"},
    )
    idx = nn["nn.idx"] - 1
    edges = []
    for i in range(idx.shape[0]):
        for j in idx[i, 1:]:
            edges.append((int(i), int(j)))
    cell_g = ig.Graph(n=reduced_coords.shape[0], edges=edges, directed=False)

    qmat = _partition_q_matrix(cell_g, np.asarray(membership) - 1)

    diameter = g_old.diameter()
    # Max inter-centroid distance along the MST.
    if stree.size == 0:
        return stree
    mst_edges = g_old.get_edgelist()
    dist_YY = _pairwise_sq_dist(Y, Y)
    if mst_edges:
        edge_weights = [np.sqrt(dist_YY[a, b]) for a, b in mst_edges]
        max_node_dist = float(max(edge_weights))
    else:
        max_node_dist = float(np.sqrt(dist_YY.max()))

    new_stree = stree.copy()
    for i in tip_points:
        for j in tip_points:
            if i >= j:
                continue
            if i >= qmat.shape[0] or j >= qmat.shape[1]:
                continue
            if qmat[i, j] >= qval_thresh:
                continue
            geodesic = g_old.distances(i, j)[0][0]
            euclidean = float(np.sqrt(dist_YY[i, j]))
            if (
                geodesic >= geodesic_distance_ratio * diameter
                and euclidean_distance_ratio * max_node_dist > euclidean
            ):
                new_stree[i, j] = 1.0
                new_stree[j, i] = 1.0
    return new_stree


def _prune_tree(
    stree_ori: np.ndarray,
    stree_loop: np.ndarray,
    minimal_branch_len: int,
) -> np.ndarray:
    """Remove branches shorter than *minimal_branch_len* — R ``prune_tree``."""
    if stree_ori.shape[1] < minimal_branch_len:
        return stree_loop

    g_ori = ig.Graph.Adjacency((stree_ori != 0).astype(int).tolist(), mode="undirected")
    g_loop = ig.Graph.Adjacency((stree_loop != 0).astype(int).tolist(), mode="undirected")

    n = stree_ori.shape[0]
    # Find small branches by BFS from a leaf.
    vertex_to_delete: set[int] = set()
    degrees = np.asarray(g_ori.degree())
    # Start BFS from the first vertex with degree 2 (or 0 if none).
    candidates = np.where(degrees == 2)[0]
    root = int(candidates[0]) if candidates.size else 0

    order = g_ori.bfs(root, mode="all")[0]
    parents = {root: None}
    for v in order:
        if v == root:
            continue
        nbr = [n for n in g_ori.neighbors(v) if n in parents]
        parents[v] = nbr[0] if nbr else None

    for v, parent in parents.items():
        if parent is None:
            continue
        if g_ori.degree(parent) <= 2:
            continue
        # Consider removing the branch rooted at v.
        temp = g_ori.copy()
        temp.delete_edges([(parent, v)])
        comps = temp.components()
        v_comp = [c for c in comps if v in c][0]
        if len(v_comp) < minimal_branch_len:
            vertex_to_delete.update(v_comp)

    keep_mask = np.ones(n, dtype=bool)
    for v in vertex_to_delete:
        keep_mask[v] = False
    # Build pruned adjacency.
    kept = np.where(keep_mask)[0]
    pruned = stree_loop[np.ix_(kept, kept)].copy()
    return pruned, kept


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def learn_graph(
    adata: ad.AnnData,
    use_partition: bool = True,
    close_loop: bool = True,
    learn_graph_control: dict | None = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Learn a principal graph over ``adata.obsm["X_umap"]`` (SimplePPT).

    Parameters
    ----------
    adata : anndata.AnnData
    use_partition : bool, default True
    close_loop : bool, default True
    learn_graph_control : dict, optional
        Valid keys (R-parity defaults): ``euclidean_distance_ratio`` (1),
        ``geodesic_distance_ratio`` (1/3), ``minimal_branch_len`` (10),
        ``orthogonal_proj_tip`` (False), ``prune_graph`` (True),
        ``scale`` (False), ``ncenter`` (auto), ``nn.k`` (25),
        ``maxiter`` (10), ``eps`` (1e-5), ``L1.gamma`` (0.5),
        ``L1.sigma`` (0.01), ``nn.method``, ``nn.metric``.
    verbose : bool, default False

    Returns
    -------
    anndata.AnnData
        With ``uns["monocle3"]["principal_graph"]["UMAP"]`` (igraph.Graph),
        ``uns["monocle3"]["principal_graph_aux"]["UMAP"]["dp_mst"]`` (centroid
        coords, ``numpy.ndarray`` shape ``(n_dim, K)``),
        ``uns["monocle3"]["principal_graph_aux"]["UMAP"]["pr_graph_cell_proj_closest_vertex"]``
        (``pandas.DataFrame`` indexed by cell with 1-based centroid id).
    """
    ctrl = {} if learn_graph_control is None else dict(learn_graph_control)
    if "rann.k" in ctrl and "nn.k" not in ctrl:
        ctrl["nn.k"] = ctrl["rann.k"]

    euclidean_distance_ratio = float(ctrl.get("euclidean_distance_ratio", 1.0))
    geodesic_distance_ratio = float(ctrl.get("geodesic_distance_ratio", 1.0 / 3.0))
    minimal_branch_len = int(ctrl.get("minimal_branch_len", 10))
    orthogonal_proj_tip = bool(ctrl.get("orthogonal_proj_tip", False))
    prune_graph = bool(ctrl.get("prune_graph", True))
    ncenter = ctrl.get("ncenter")
    scale = bool(ctrl.get("scale", False))
    nn_k = int(ctrl.get("nn.k", 25))
    maxiter = int(ctrl.get("maxiter", 10))
    eps = float(ctrl.get("eps", 1e-5))
    L1_gamma = float(ctrl.get("L1.gamma", 0.5))
    L1_sigma = float(ctrl.get("L1.sigma", 0.01))
    nn_control = {
        "method": ctrl.get("nn.method", "annoy"),
        "metric": ctrl.get("nn.metric", "euclidean"),
    }

    if "X_umap" not in adata.obsm:
        raise KeyError(
            "No UMAP reduction. Run reduce_dimension(method='UMAP') first."
        )

    if use_partition:
        if "monocle3_partitions" not in adata.obs.columns:
            raise KeyError(
                "use_partition=True requires cluster_cells to be run first."
            )
        partitions_full = np.asarray(adata.obs["monocle3_partitions"].astype(str))
    else:
        partitions_full = np.asarray(["1"] * adata.n_obs)

    umap_coords = np.asarray(adata.obsm["X_umap"], dtype=np.float64)
    # R uses X = t(reduced) → D × N convention; align with that for kernel calls.
    X_full = umap_coords.T

    merged_Y: np.ndarray | None = None
    merged_closest_vertex: list[int] = []
    merged_cell_names: list[str] = []
    graph_parts: list[ig.Graph] = []
    centroid_index_offset = 0

    cell_names = list(adata.obs_names)

    for partition in sorted(pd.unique(partitions_full)):
        mask = partitions_full == partition
        if not mask.any():
            continue
        X_subset = X_full[:, mask]
        subset_cells = [cell_names[i] for i in np.where(mask)[0]]

        if scale:
            X_subset = (X_subset - X_subset.mean(axis=1, keepdims=True)) / (
                X_subset.std(axis=1, keepdims=True) + 1e-12
            )

        n_cells_in_partition = X_subset.shape[1]
        if ncenter is None:
            if "monocle3_clusters" in adata.obs.columns:
                cluster_sub = adata.obs.loc[subset_cells, "monocle3_clusters"]
                n_clusters = len(pd.unique(cluster_sub))
            else:
                n_clusters = 1
            cur_ncenter = _cal_ncenter(n_clusters, n_cells_in_partition)
            if not cur_ncenter or cur_ncenter >= n_cells_in_partition:
                cur_ncenter = max(1, n_cells_in_partition - 1)
        else:
            cur_ncenter = min(n_cells_in_partition - 1, int(ncenter))

        if cur_ncenter < 2:
            continue

        points = X_subset.T  # N × D
        labels, _km_centers = _kmeans_with_init(points, cur_ncenter)

        # kNN on the full partition points.
        k_knn = max(2, min(nn_k, n_cells_in_partition - 1))
        nn_res = search_nn_matrix(
            points, points, k=k_knn, nn_control=nn_control
        )
        nn_dists = nn_res["nn.dists"][:, 1:]  # drop self
        rho = np.exp(-nn_dists.mean(axis=1))

        # Pick the highest-density point per kmeans cluster → initial medoids.
        medoid_idx_per_cluster: list[int] = []
        for c in np.unique(labels):
            cell_idx = np.where(labels == c)[0]
            pick = cell_idx[np.argmax(rho[cell_idx])]
            medoid_idx_per_cluster.append(int(pick))
        medoids = X_subset[:, medoid_idx_per_cluster]

        # SimplePPT.
        rge = _calc_principal_graph(
            X=X_subset, C0=medoids, maxiter=maxiter, eps=eps,
            L1_gamma=L1_gamma, L1_sigma=L1_sigma,
        )

        stree = rge["stree"]  # K × K symmetric
        Y = rge["Y"]  # D × K

        if close_loop:
            stree = _connect_tips(
                stree=stree,
                Y=Y,
                reduced_coords=X_subset.T,
                membership=labels + 1,
                euclidean_distance_ratio=euclidean_distance_ratio,
                geodesic_distance_ratio=geodesic_distance_ratio,
                k=25,
            )

        stree_ori = (rge["stree"] != 0).astype(int)
        if prune_graph:
            pruned_res = _prune_tree(
                stree_ori=stree_ori.astype(float),
                stree_loop=stree.astype(float),
                minimal_branch_len=minimal_branch_len,
            )
            if isinstance(pruned_res, tuple):
                pruned_stree, keep_idx = pruned_res
                Y = Y[:, keep_idx]
                R = rge["R"][:, keep_idx]
                stree = pruned_stree
            else:
                R = rge["R"]
        else:
            R = rge["R"]

        K = Y.shape[1]
        part_g = ig.Graph.Adjacency(
            (stree != 0).astype(int).tolist(), mode="undirected"
        )
        part_g.vs["name"] = [f"Y_{centroid_index_offset + i + 1}" for i in range(K)]

        closest = np.argmax(R, axis=1).astype(np.int64) + 1 + centroid_index_offset
        merged_closest_vertex.extend(closest.tolist())
        merged_cell_names.extend(subset_cells)

        if merged_Y is None:
            merged_Y = Y.copy()
        else:
            merged_Y = np.concatenate([merged_Y, Y], axis=1)
        graph_parts.append(part_g)
        centroid_index_offset += K

    if merged_Y is None or centroid_index_offset == 0:
        # Edge case: no usable partitions.
        uns = ensure_monocle_uns(adata)
        uns.setdefault("principal_graph", {})
        uns.setdefault("principal_graph_aux", {})
        uns["principal_graph"]["UMAP"] = ig.Graph()
        uns["principal_graph_aux"]["UMAP"] = {}
        return adata

    # Union of per-partition graphs.
    big_g = ig.Graph()
    for sub in graph_parts:
        big_g = big_g.disjoint_union(sub) if big_g.vcount() > 0 else sub
    # Restore names (disjoint_union may drop them).
    all_names = [f"Y_{i + 1}" for i in range(merged_Y.shape[1])]
    big_g.vs["name"] = all_names[: big_g.vcount()]

    # Reorder closest_vertex to match adata.obs_names.
    order_series = pd.Series(merged_closest_vertex, index=merged_cell_names)
    order_series = order_series.reindex(cell_names)
    closest_vertex_df = pd.DataFrame(
        {"V1": order_series.to_numpy()},
        index=cell_names,
    )

    uns = ensure_monocle_uns(adata)
    uns.setdefault("principal_graph", {})
    uns.setdefault("principal_graph_aux", {})
    uns["principal_graph"]["UMAP"] = big_g
    uns["principal_graph_aux"]["UMAP"] = {
        "dp_mst": merged_Y,  # D × K
        "pr_graph_cell_proj_closest_vertex": closest_vertex_df,
    }

    return adata
