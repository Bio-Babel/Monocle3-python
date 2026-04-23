"""graph_test — port of R/graph_test.R.

Per-gene spatial autocorrelation via Moran's I on either a kNN graph in
the reduced space or the principal-graph-projected neighbour list. The
test statistic itself is delegated to :mod:`esda`; kNN construction uses
:mod:`pynndescent` or :mod:`sklearn.neighbors` through
:mod:`monocle3.nearest_neighbors`.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.stats import false_discovery_control

from ._clustering_cpp import jaccard_coeff
from ._utils import get_monocle_uns, sparse_row_sums
from .nearest_neighbors import search_nn_matrix

__all__ = ["graph_test"]


def _knn_list_from_indices(nn_idx: np.ndarray) -> list[list[int]]:
    """Drop self (column 0) and convert to list-of-0-based neighbours."""
    return [list((row[1:] - 1).astype(int)) for row in nn_idx]


def _build_W(
    knn_list: list[list[int]], n: int
) -> sp.csr_matrix:
    """Row-standardized 0/1 neighbour matrix (spdep ``nb2listw`` equivalent)."""
    rows, cols = [], []
    for i, nbrs in enumerate(knn_list):
        for j in nbrs:
            rows.append(i)
            cols.append(j)
    W01 = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n)
    )
    row_sums = np.asarray(W01.sum(axis=1)).ravel()
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    D = sp.diags(1.0 / row_sums)
    return (D @ W01).tocsr()


def _moran_per_row(
    values: np.ndarray,
    W: sp.csr_matrix,
    alternative: str,
) -> tuple[float, float, float]:
    """Vanilla Moran's I for a 0/1 row-standardised W.

    Returns
    -------
    (moran_I, z_statistic, p_value)
    """
    n = values.shape[0]
    x = values - values.mean()
    sum_x2 = float(x @ x)
    if sum_x2 == 0:
        return 0.0, 0.0, 1.0
    lag = W @ x
    I = float(x @ lag) / sum_x2  # n·S0 cancels for row-stdz weights.
    EI = -1.0 / (n - 1)
    S0 = float(W.sum())
    S1 = 0.5 * float((W + W.T).multiply(W + W.T).sum())
    col_sum = np.asarray(W.sum(axis=0)).ravel()
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    S2 = float(((col_sum + row_sum) ** 2).sum())
    K = (n * np.sum(x ** 4)) / (sum_x2 ** 2)
    S02 = S0 * S0
    n1, n2, n3 = n - 1, n - 2, n - 3
    # CAREFUL: spdep defines ``wc$nn = n * n`` despite the suggestive name
    # (not ``n * (n - 1)``). Empirically verified — do not "fix" this.
    nn = n * n
    VI_rand_numer = n * (S1 * (nn - 3 * n + 3) - n * S2 + 3 * S02) - K * (
        S1 * (nn - n) - 2 * n * S2 + 6 * S02
    )
    VI_rand = VI_rand_numer / (n1 * n2 * n3 * S02) - EI ** 2
    if VI_rand <= 0:
        VI_rand = (nn * S1 - n * S2 + 3 * S02) / (S02 * (nn - 1)) - EI ** 2
    if VI_rand <= 0:
        return I, 0.0, 1.0
    Z = (I - EI) / np.sqrt(VI_rand)
    from scipy.stats import norm as _norm
    if alternative == "two.sided":
        p = 2.0 * _norm.sf(abs(Z))
    elif alternative == "greater":
        p = _norm.sf(Z)
    else:  # less
        p = _norm.cdf(Z)
    return I, Z, float(p)


def _geary_per_row(
    values: np.ndarray,
    W: sp.csr_matrix,
    alternative: str,
) -> tuple[float, float, float]:
    """Geary's C for a row-standardised W. Returns ``(C, z_stat, p_value)``."""
    n = values.shape[0]
    mean = float(values.mean())
    xc = values - mean
    sum_x2 = float(xc @ xc)
    if sum_x2 == 0:
        return 1.0, 0.0, 1.0

    row_sum = np.asarray(W.sum(axis=1)).ravel()
    col_sum = np.asarray(W.sum(axis=0)).ravel()
    S0 = float(W.sum())
    if S0 == 0:
        return 1.0, 0.0, 1.0

    # sum_ij w_ij (x_i - x_j)^2 =
    #   sum_i x_i^2 * row_i  +  sum_j x_j^2 * col_j  -  2 x^T W x
    term1 = float(np.sum(xc ** 2 * row_sum))
    term2 = float(np.sum(xc ** 2 * col_sum))
    term3 = float(xc @ (W @ xc))
    numer = term1 + term2 - 2.0 * term3
    C = ((n - 1) / (2.0 * S0)) * numer / sum_x2
    EC = 1.0

    # See the `nn = n * n` note in `_moran_per_row`; same convention here.
    nn = n * n
    n1, n2, n3 = n - 1, n - 2, n - 3
    S1 = 0.5 * float((W + W.T).multiply(W + W.T).sum())
    S2 = float(((col_sum + row_sum) ** 2).sum())
    S02 = S0 * S0
    K = (n * np.sum(xc ** 4)) / (sum_x2 ** 2)

    VC = n1 * S1 * (nn - 3 * n + 3 - K * n1)
    VC = VC - (0.25 * n1 * S2 * (nn + 3 * n - 6 - K * (nn - n + 2)))
    VC = VC + S02 * (nn - 3 - K * n1 * n1)
    VC = VC / (n * n2 * n3 * S02)

    if VC <= 0:
        return C, 0.0, 1.0

    # Sign convention: "greater" tests stronger positive spatial
    # autocorrelation, which corresponds to C < 1, hence ZC > 0.
    ZC = (EC - C) / np.sqrt(VC)
    from scipy.stats import norm as _norm
    if alternative == "two.sided":
        p = 2.0 * _norm.sf(abs(ZC))
    elif alternative == "greater":
        p = _norm.sf(ZC)
    else:  # less
        p = _norm.cdf(ZC)
    return C, ZC, float(p)


def graph_test(
    adata: ad.AnnData,
    neighbor_graph: str = "knn",
    reduction_method: str = "UMAP",
    k: int = 25,
    method: str = "Moran_I",
    alternative: str = "greater",
    expression_family: str = "quasipoisson",
    cores: int = 1,
    verbose: bool = False,
    nn_control: dict | None = None,
) -> pd.DataFrame:
    """Per-gene spatial autocorrelation against a kNN or principal-graph
    neighbour list.

    Supports ``method="Moran_I"`` (default) and ``method="Geary_C"``.
    Returns a ``DataFrame`` with columns ``status``, ``p_value``,
    either ``morans_test_statistic`` + ``morans_I`` or
    ``geary_test_statistic`` + ``geary_C``, ``q_value``, plus every
    existing column of ``adata.var``.
    """
    del cores, verbose
    if method not in {"Moran_I", "Geary_C"}:
        raise ValueError(
            "method must be 'Moran_I' or 'Geary_C'"
        )
    if neighbor_graph not in {"knn", "principal_graph"}:
        raise ValueError("neighbor_graph must be 'knn' or 'principal_graph'")
    if alternative not in {"greater", "less", "two.sided"}:
        raise ValueError(
            "alternative must be 'greater', 'less', or 'two.sided'"
        )

    if reduction_method != "UMAP":
        raise NotImplementedError("Only reduction_method='UMAP' is supported.")

    key = f"X_{reduction_method.lower()}"
    if key not in adata.obsm:
        raise KeyError(
            f"No {reduction_method} reduction. Run reduce_dimension first."
        )
    coords = np.asarray(adata.obsm[key], dtype=np.float64)

    nn_ctrl = {"method": "annoy", "metric": "euclidean"}
    if nn_control:
        nn_ctrl.update(nn_control)

    nn_res = search_nn_matrix(
        coords, coords, k=min(int(k) + 1, coords.shape[0]), nn_control=nn_ctrl
    )
    knn_idx = nn_res["nn.idx"]
    knn_list = _knn_list_from_indices(knn_idx)

    if neighbor_graph == "principal_graph":
        # (1) drop kNN edges whose jaccard shared-neighbour count is 0.
        # (2) build a cell→centroid indicator M.
        # (3) feasibility = M @ (principal_g + I) @ M.T — a pair passes only
        #     if the two cells' closest vertices are the same or directly
        #     connected in the principal graph.
        # (4) element-wise multiply jaccard_adj with feasibility.
        # (5) binarise and emit the neighbour index list.
        aux = get_monocle_uns(adata, "principal_graph_aux", reduction_method,
                              default={})
        if "pr_graph_cell_proj_closest_vertex" not in aux:
            raise KeyError(
                "No principal graph projection. Run learn_graph first."
            )
        closest = aux["pr_graph_cell_proj_closest_vertex"]["V1"].to_numpy()
        pg = get_monocle_uns(adata, "principal_graph", reduction_method)
        pg_adj = pg.get_adjacency_sparse().astype(np.float64)
        pg_adj = pg_adj + sp.eye(pg_adj.shape[0], format="csr")

        # (1) jaccard-weighted kNN edges (jaccard_coeff wants 1-based indices).
        nn_idx_1based = knn_idx[:, 1:]  # drop self column
        links = jaccard_coeff(nn_idx_1based, weight=False)
        keep = links[:, 2] > 0
        links = links[keep]
        rows_e = (links[:, 0].astype(np.int64) - 1)
        cols_e = (links[:, 1].astype(np.int64) - 1)
        vals_e = links[:, 2]
        N = adata.n_obs
        jaccard_adj = sp.csr_matrix(
            (vals_e, (rows_e, cols_e)), shape=(N, N)
        )

        # (2) cell-membership indicator M (N × K).
        closest_0based = (closest.astype(np.int64) - 1)
        M = sp.csr_matrix(
            (np.ones(N), (np.arange(N), closest_0based)),
            shape=(N, pg_adj.shape[0]),
        )
        # (3) feasibility mask.
        feasible = M @ pg_adj @ M.T  # N × N, sparse
        feasible = (feasible > 0).astype(np.float64)

        # (4)(5) element-wise jaccard × feasible → binarised index list.
        masked = jaccard_adj.multiply(feasible)
        masked = masked.tocsr()
        knn_list = [list(masked.getrow(i).indices) for i in range(N)]

    W = _build_W(knn_list, n=adata.n_obs)

    # Per-gene transform: log10(x / sf + 0.1) except for family-passthrough.
    X = adata.X
    sf = adata.obs["Size_Factor"].to_numpy(dtype=float)
    results: list[dict[str, Any]] = []
    gene_names = list(adata.var_names)

    if method == "Moran_I":
        test_fn = _moran_per_row
        stat_key = "morans_test_statistic"
        estimate_key = "morans_I"
    else:  # Geary_C
        test_fn = _geary_per_row
        stat_key = "geary_test_statistic"
        estimate_key = "geary_C"

    for g_idx, gene in enumerate(gene_names):
        col = X[:, g_idx]
        if sp.issparse(col):
            col = np.asarray(col.todense()).ravel()
        else:
            col = np.asarray(col, dtype=float).ravel()
        if expression_family in {"uninormal", "binomialff"}:
            values = col
        else:
            values = np.log10(col / sf + 0.1)
        try:
            estimate, Z, p = test_fn(values, W, alternative)
            results.append(
                {
                    "status": "OK",
                    "p_value": p,
                    stat_key: Z,
                    estimate_key: estimate,
                }
            )
        except Exception:
            results.append(
                {
                    "status": "FAIL",
                    "p_value": np.nan,
                    stat_key: np.nan,
                    estimate_key: np.nan,
                }
            )

    df = pd.DataFrame(results, index=gene_names)
    df["q_value"] = 1.0
    ok_mask = df["status"] == "OK"
    if ok_mask.any():
        df.loc[ok_mask, "q_value"] = false_discovery_control(
            df.loc[ok_mask, "p_value"].to_numpy(dtype=float), method="bh"
        )

    # Join var metadata.
    merged = pd.concat([df, adata.var], axis=1)
    # Drop duplicated columns if var had any of our output names.
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged
