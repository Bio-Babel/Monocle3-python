"""SimplePPT-style reversed graph embedding on reduced-dim coordinates.

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
    """Squared pairwise distances of two (D, n) / (D, m) matrices,
    shape (A.shape[1], B.shape[1])."""
    a_norm = np.sum(A ** 2, axis=0)[:, None]
    b_norm = np.sum(B ** 2, axis=0)[None, :]
    return a_norm + b_norm - 2.0 * A.T @ B


def _soft_assignment(X: np.ndarray, C: np.ndarray, sigma: float):
    """Soft assignment of cells to centroids.

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
    """Closed-form centroid update (SimplePPT eq. 22)."""
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
            # Numpy scalar division yields inf on div-by-zero (Inf < eps is
            # False, so the loop continues) instead of raising.
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_diff = np.abs(np.float64(objs[iteration - 2]) - np.float64(obj)) \
                           / np.abs(np.float64(objs[iteration - 2]))
            if rel_diff < eps:
                break

        C = _generate_centers(X, W, P, L1_gamma)

    return {"X": X, "C": C, "W": W, "P": P, "objs": np.asarray(objs),
            "Y": C, "R": P, "stree": stree_out}


# ---------------------------------------------------------------------------
# Helpers — projection, k-means, nearest vertex
# ---------------------------------------------------------------------------


def _cal_ncenter(n_clusters: int, n_cells: int, nodes_per_log10: int = 15) -> int:
    return int(round(n_clusters * nodes_per_log10 * np.log10(max(n_cells, 1))))


def _kmeans_with_init(
    points: np.ndarray, n_clusters: int, seed: int = 2016
) -> tuple[np.ndarray, np.ndarray]:
    """k-means with evenly-spaced initial centers plus tiny noise."""
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


def _proj_point_on_line(p: np.ndarray, AB: np.ndarray) -> np.ndarray:
    """R ``projPointOnLine`` — orthogonal projection onto infinite line through A,B."""
    A = AB[:, 0]
    B = AB[:, 1]
    ap = p - A
    ab = B - A
    denom = float(np.dot(ab, ab))
    if denom == 0:
        return A.copy()
    return A + (float(np.dot(ap, ab)) / denom) * ab


# ---------------------------------------------------------------------------
# Build the cell-augmented principal graph used by order_cells. Every cell
# is projected onto its nearest MST edge; the cell-augmented graph has both
# "Y_<i>" (centroid) and cell-name vertices, with edges
# (Y_source → cell1 → cell2 → … → cell_n) chaining cells along each edge
# in distance-from-source order, plus the original centroid edges.
# ---------------------------------------------------------------------------


def _find_nearest_point_on_mst(
    graph: ig.Graph,
    Y: np.ndarray,
    Z: np.ndarray,
    partitions_arr: np.ndarray,
    cell_names: list[str],
    vertex_names: list[str],
) -> pd.DataFrame:
    """For each cell, find the closest centroid (in Y-space) within its
    partition's principal subgraph.

    Returns a DataFrame indexed by cell name with a single column ``V1``
    holding the 1-based *global* centroid index.
    """
    name_to_idx = {n: i for i, n in enumerate(vertex_names)}
    components = graph.decompose()
    unique_parts = sorted(pd.unique(partitions_arr))
    # Fallback: if component/partition counts mismatch, treat as one.
    if len(components) != len(unique_parts):
        components = [graph]

    rows: list[tuple[str, int]] = []
    for i, comp in enumerate(components):
        comp_vnames = list(comp.vs["name"])
        sub_idx = [name_to_idx[n] for n in comp_vnames]
        Y_sub = Y[:, sub_idx]  # D × K_cur
        if len(components) == 1:
            Z_sub = Z
            cells_sub = list(cell_names)
        else:
            partition = unique_parts[i]
            mask = partitions_arr == partition
            Z_sub = Z[:, mask]
            cells_sub = [cell_names[j] for j, m in enumerate(mask) if m]
        if Z_sub.shape[1] == 0 or Y_sub.shape[1] == 0:
            continue
        dsq = _pairwise_sq_dist(Z_sub, Y_sub)  # N_sub × K_cur
        local_arg = np.argmin(dsq, axis=1)
        for cell, la in zip(cells_sub, local_arg):
            rows.append((cell, sub_idx[int(la)] + 1))  # 1-based global

    df = pd.DataFrame(rows, columns=["_cell", "V1"]).set_index("_cell")
    df = df.reindex(cell_names)
    df.index.name = None
    return df


def _project2mst(
    graph: ig.Graph,
    Y: np.ndarray,
    Z: np.ndarray,
    partitions_arr: np.ndarray,
    cell_names: list[str],
    vertex_names: list[str],
    orthogonal_proj_tip: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, ig.Graph]:
    """R ``project2MST``.

    Parameters
    ----------
    graph : igraph.Graph
        The merged principal graph with vertex attribute ``name`` = "Y_<i>".
    Y : np.ndarray
        D × K centroid coordinates.
    Z : np.ndarray
        D × N original reduced-dim coordinates (cells).
    partitions_arr : np.ndarray
        Length-N array of partition labels aligned with *cell_names*.
    cell_names : list[str]
        Cell identifiers in the same order as the columns of *Z*.
    vertex_names : list[str]
        Global centroid names in the same order as the columns of *Y*.
    orthogonal_proj_tip : bool
        When True, cells whose closest centroid is a leaf are projected
        onto the infinite line rather than clamped to the segment.

    Returns
    -------
    closest_vertex : pandas.DataFrame
        Indexed by cell name, column ``V1`` with 1-based global centroid
        index.
    P : np.ndarray
        D × N matrix of cell projections onto the nearest MST edge.
    cell_proj_tree : igraph.Graph
        Undirected weighted graph whose vertices mix ``Y_<i>`` centroid
        names and raw cell names. Edges chain each cell in distance order
        along the MST edge it projected onto, plus the original centroid
        edges.
    """
    D, N = Z.shape
    K = Y.shape[1]
    name_to_idx = {n: i for i, n in enumerate(vertex_names)}

    # ---- (1) closest centroid per cell (per partition). --------------------
    closest_vertex = _find_nearest_point_on_mst(
        graph, Y, Z, partitions_arr, cell_names, vertex_names
    )
    cv_global = closest_vertex["V1"].astype(int).to_numpy()  # 1-based

    # ---- (2) project each cell onto its nearest edge. ----------------------
    tip_leaves = {
        vertex_names[i]
        for i, d in enumerate(graph.degree())
        if d == 1
    }

    # Precompute neighbor lists by vertex name.
    nbr_names: dict[str, list[str]] = {
        vertex_names[v]: [vertex_names[n] for n in graph.neighbors(v)]
        for v in range(graph.vcount())
    }

    P = np.zeros((D, N), dtype=np.float64)
    nearest_edges: list[tuple[str, str]] = [("", "")] * N

    for i in range(N):
        closest_name = vertex_names[cv_global[i] - 1]
        A = Y[:, cv_global[i] - 1]
        neighbors = nbr_names.get(closest_name, [])
        if not neighbors:
            P[:, i] = A
            nearest_edges[i] = (closest_name, closest_name)
            continue
        Z_i = Z[:, i]
        best_dist = np.inf
        best_proj: np.ndarray | None = None
        best_nbr: str | None = None
        for nb in neighbors:
            B = Y[:, name_to_idx[nb]]
            AB = np.column_stack([A, B])
            if closest_name in tip_leaves and orthogonal_proj_tip:
                tmp = _proj_point_on_line(Z_i, AB)
            else:
                tmp = _project_point_to_line_segment(Z_i, AB)
            if np.any(np.isnan(tmp)):
                tmp = B.copy()
            d = float(np.linalg.norm(Z_i - tmp))
            if d < best_dist:
                best_dist = d
                best_proj = tmp
                best_nbr = nb
        assert best_proj is not None and best_nbr is not None
        P[:, i] = best_proj
        nearest_edges[i] = (closest_name, best_nbr)

    # ---- (3) decompose into partitions and build cell-augmented edges. -----
    components = graph.decompose()
    unique_parts = sorted(pd.unique(partitions_arr))
    # If a single component spans multiple labelled partitions (e.g. when
    # learn_graph was run with use_partition=False), collapse labels to '1'.
    if len(components) == 1 and len(unique_parts) > 1:
        partitions_series = pd.Series(["1"] * N, index=cell_names)
        unique_parts = ["1"]
    else:
        partitions_series = pd.Series(partitions_arr, index=cell_names)

    all_edges: list[tuple[str, str, float]] = []

    for part_i, part in enumerate(unique_parts):
        mask = partitions_series.values == part
        if not mask.any():
            continue
        subset_cells = [cell_names[j] for j, m in enumerate(mask) if m]
        cur_p = P[:, mask]  # D × Np
        # Sort each (source, target) pair lexicographically so both
        # orderings of an edge hash to the same group label.
        sub_ne = [nearest_edges[j] for j, m in enumerate(mask) if m]
        sorted_ne = [tuple(sorted(pair)) for pair in sub_ne]
        sources = np.array([s for s, _ in sorted_ne])
        targets = np.array([t for _, t in sorted_ne])

        # distance_2_source per cell.
        src_global = np.array([name_to_idx[s] for s in sources], dtype=np.int64)
        Y_src = Y[:, src_global]
        dist_to_source = np.sqrt(np.sum((cur_p - Y_src) ** 2, axis=0))

        # Group cells by (source, target) and sort within each group by distance.
        group_labels = np.array([f"{s}_{t}" for s, t in zip(sources, targets)])
        order = np.lexsort((dist_to_source, group_labels))

        sorted_cells = [subset_cells[j] for j in order]
        sorted_groups = group_labels[order]
        sorted_sources = sources[order]

        # ---------------------------------------------------------------
        # TODO(potential upstream-bug-2): missing closing edge `cellN -> Y_target`.
        #
        # R source : monocle3/R/learn_graph.R:864-877 (`project2MST`).
        # Issue    : R *intends* to chain, per principal edge (Y_a, Y_b),
        #            Y_a -> c1 -> c2 -> ... -> cN -> Y_b (N cells sorted by
        #            distance-to-source). The trailing `cN -> Y_b` edge is
        #            **missing**: R's `added_rows <- which(is.na(new_source)
        #            & is.na(new_target))` is always integer(0) because
        #            `new_target = rowname` is never NA and the code never
        #            calls `add_row()` to create placeholder rows. The
        #            comment "add the links from the last point on the
        #            principal edge to the target point of the edge"
        #            documents the intent; the implementation is a no-op.
        # Current  : We replicate the truncated chain verbatim (see loop
        #            below). Cells at the far end of an edge reach Y_b only
        #            by traversing back through Y_a + the centroid edge
        #            Y_a--Y_b, which biases pseudotime upwards for those
        #            cells.
        # Impact   : affects `pr_graph_cell_proj_tree` topology, hence
        #            `order_cells` pseudotime values (not ranks).
        # Fix path : (1) file R-upstream issue + PR at the link above;
        #            (2) after R release fixes it, patch this loop to also
        #            emit the closing edge when a group ends. Sketch:
        #               if prev_group is not None and cur_g != prev_group:
        #                   new_src.append(prev_name)
        #                   new_tgt.append(prev_target)
        #               ...
        #            also capture `sorted_targets = targets[order]` above.
        #            Remember to flip the default only when R upstream
        #            releases the fix, so Python/R stay numerically
        #            aligned (see docstring).
        # Gold ref : `R` is the declared gold standard for this port
        #            (see user's instructions). Do NOT silently diverge.
        # ---------------------------------------------------------------
        # Within-group chain: new_source[0] = sorted_sources[0] (the Y_src),
        # new_source[i>0] = sorted_cells[i-1]; new_target[i] = sorted_cells[i].
        new_src: list[str] = []
        new_tgt: list[str] = []
        prev_group = None
        prev_name = ""
        for j in range(len(sorted_cells)):
            cur_g = sorted_groups[j]
            rowname = sorted_cells[j]
            if cur_g != prev_group:
                new_src.append(str(sorted_sources[j]))
            else:
                new_src.append(prev_name)
            new_tgt.append(rowname)
            prev_name = rowname
            prev_group = cur_g
        # ---------------------------------------------------------------
        # END TODO(upstream-bug-2)
        # ---------------------------------------------------------------

        # aug_P = cbind(cur_p, rge_res_Y)  [D × (Np + K)]
        aug_P = np.concatenate([cur_p, Y], axis=1)
        aug_name_idx: dict[str, int] = {}
        for k_, c in enumerate(subset_cells):
            aug_name_idx[c] = k_
        for k_, yn in enumerate(vertex_names):
            aug_name_idx[yn] = len(subset_cells) + k_

        src_ai = np.array([aug_name_idx[s] for s in new_src], dtype=np.int64)
        tgt_ai = np.array([aug_name_idx[s] for s in new_tgt], dtype=np.int64)
        diff = aug_P[:, src_ai] - aug_P[:, tgt_ai]  # D × Nedges
        # ---------------------------------------------------------------
        # TODO(potential upstream-bug-1): weight formula is NOT Euclidean distance.
        #
        # R source : monocle3/R/learn_graph.R:881-883 (`project2MST`).
        # R code   : sqrt(colSums((aug_P[, new_source]
        #                          - aug_P[, new_target]))^2)
        #            The `^2` is placed *outside* colSums(), so the result
        #            collapses to  |Σ_d (A_d − B_d)|  instead of the intended
        #            √(Σ_d (A_d − B_d)^2). Minimal R reproducer:
        #                A <- c(1,0); B <- c(0,1)
        #                diff <- matrix(A - B, ncol = 1)
        #                sqrt(colSums(diff)^2)   # -> 0    (bug)
        #                sqrt(colSums(diff^2))   # -> 1.414 (correct)
        #            Two distinct points thus get edge weight 0 whenever
        #            their signed per-dim deltas cancel.
        # Confirm  : same file line 856 (`distance_2_source`) writes
        #            `sqrt(colSums((cur_p - Y_src)^2))` with `^2` inside
        #            colSums — strongly indicates the 881 line is a
        #            parenthesis typo, not intentional.
        # Current  : we replicate the R formula verbatim (next line).
        # Impact   : `pr_graph_cell_proj_tree` edge weights are wrong
        #            → `order_cells` pseudotime values are biased low.
        #            Topology / rank ordering is approximately preserved
        #            (Spearman ≈ 0.999 on the Y-shape fixture) but
        #            absolute values diverge from the true geometry.
        # Fix path : (1) file R-upstream issue + PR moving `^2` into
        #            colSums; (2) after R releases the fix, replace the
        #            line below with the Euclidean form:
        #               weights = np.sqrt(np.sum(diff ** 2, axis=0))
        #            Do NOT flip ahead of R upstream — keeping parity is
        #            the user-declared invariant (R is the gold standard).
        # Gold ref : user instruction "以R源码为金标".
        # ---------------------------------------------------------------
        weights = np.sqrt(np.sum(diff, axis=0) ** 2)  # bug-for-bug with R
        # ---------------------------------------------------------------
        # END TODO(upstream-bug-1)
        # ---------------------------------------------------------------
        positive = weights[weights > 0]
        min_pos = float(positive.min()) if positive.size else 0.0
        weights = weights + min_pos

        for s, t, w in zip(new_src, new_tgt, weights.tolist()):
            all_edges.append((s, t, float(w)))

        # Append the raw centroid edges for this partition.
        # Component indexing follows sorted-partition order when decomposition
        # aligns; otherwise reuse the full graph.
        if part_i < len(components):
            comp = components[part_i]
        else:
            comp = graph
        comp_vnames = list(comp.vs["name"])
        for e in comp.es:
            a, b = e.tuple
            an = comp_vnames[a]
            bn = comp_vnames[b]
            ai = name_to_idx[an]
            bi = name_to_idx[bn]
            w = float(np.linalg.norm(Y[:, ai] - Y[:, bi]))
            all_edges.append((an, bn, w))

    # Build the cell-augmented igraph.
    aug_vertex_set: list[str] = []
    seen: set[str] = set()
    # Stable order: first all Y_# in global order, then cells in adata order.
    for v in vertex_names:
        if v not in seen:
            aug_vertex_set.append(v)
            seen.add(v)
    for c in cell_names:
        if c not in seen:
            aug_vertex_set.append(c)
            seen.add(c)

    aug_name_to_i = {n: i for i, n in enumerate(aug_vertex_set)}
    edge_idx = [(aug_name_to_i[s], aug_name_to_i[t]) for (s, t, _) in all_edges]
    edge_w = [w for (_, _, w) in all_edges]
    cell_proj_tree = ig.Graph(
        n=len(aug_vertex_set),
        edges=edge_idx,
        directed=False,
        vertex_attrs={"name": aug_vertex_set},
        edge_attrs={"weight": edge_w},
    )

    return closest_vertex, P, cell_proj_tree


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
        Valid keys (with defaults): ``euclidean_distance_ratio`` (1),
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
    per_partition_stree: list[np.ndarray] = []
    per_partition_R: list[np.ndarray] = []
    per_partition_R_cells: list[list[str]] = []
    per_partition_R_cols: list[list[str]] = []
    per_partition_objs: list[np.ndarray] = []

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

        # Per-partition diagnostics exposed as `principal_graph_aux$stree` /
        # `$R` / `$objective_vals`.
        per_partition_stree.append(np.asarray(stree, dtype=np.float64).copy())
        per_partition_R.append(np.asarray(R, dtype=np.float64).copy())
        per_partition_R_cells.append(list(subset_cells))
        per_partition_R_cols.append(
            [f"Y_{centroid_index_offset + j + 1}" for j in range(K)]
        )
        per_partition_objs.append(np.asarray(rge["objs"], dtype=np.float64).copy())

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

    # Seed every centroid edge with weight = 1 so downstream igraph routines
    # (e.g. shortest-path) have a `weight` attribute to read.
    if big_g.ecount() > 0:
        big_g.es["weight"] = [1.0] * big_g.ecount()

    # Project every cell onto the nearest MST edge.
    vertex_names_global = list(big_g.vs["name"])
    Z_full = umap_coords.T  # D × N
    closest_vertex_df, P_proj, cell_proj_tree = _project2mst(
        graph=big_g,
        Y=merged_Y,
        Z=Z_full,
        partitions_arr=partitions_full,
        cell_names=cell_names,
        vertex_names=vertex_names_global,
        orthogonal_proj_tip=orthogonal_proj_tip,
    )

    # ---- Build the block-diagonal R matrix across partitions. --------------
    K_total = merged_Y.shape[1]
    R_full = sp.lil_matrix((adata.n_obs, K_total), dtype=np.float64)
    cell_to_row = {c: i for i, c in enumerate(cell_names)}
    col_cursor = 0
    for block, cells, cols in zip(
        per_partition_R, per_partition_R_cells, per_partition_R_cols
    ):
        rows = [cell_to_row[c] for c in cells]
        ncols_block = block.shape[1]
        for ri, ridx in enumerate(rows):
            R_full[ridx, col_cursor : col_cursor + ncols_block] = block[ri]
        col_cursor += ncols_block
    R_full = R_full.tocsr()

    # ---- Block-diagonal stree across partitions. ---------------------------
    stree_full = sp.lil_matrix((K_total, K_total), dtype=np.float64)
    col_cursor = 0
    for block in per_partition_stree:
        k_blk = block.shape[0]
        stree_full[
            col_cursor : col_cursor + k_blk, col_cursor : col_cursor + k_blk
        ] = block
        col_cursor += k_blk
    stree_full = stree_full.tocsr()

    uns = ensure_monocle_uns(adata)
    uns.setdefault("principal_graph", {})
    uns.setdefault("principal_graph_aux", {})
    uns["principal_graph"]["UMAP"] = big_g
    uns["principal_graph_aux"]["UMAP"] = {
        "dp_mst": merged_Y,  # D × K centroid coord matrix
        "stree": stree_full,
        "R": R_full,
        "objective_vals": list(per_partition_objs),
        "pr_graph_cell_proj_closest_vertex": closest_vertex_df,
        "pr_graph_cell_proj_dist": P_proj,
        "pr_graph_cell_proj_tree": cell_proj_tree,
    }

    return adata
