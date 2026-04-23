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
    # Randomisation-based expectation/variance (spdep default).
    EI = -1.0 / (n - 1)
    # Approximate variance for row-standardised W; good enough for
    # ranking genes, matches R closely for this use case.
    W2 = W.multiply(W)
    S0 = float(W.sum())
    S1 = 0.5 * float((W + W.T).multiply(W + W.T).sum())
    col_sum = np.asarray(W.sum(axis=0)).ravel()
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    S2 = float(((col_sum + row_sum) ** 2).sum())
    K = (n * np.sum(x ** 4)) / (sum_x2 ** 2)
    S02 = S0 * S0
    n1, n2, n3 = n - 1, n - 2, n - 3
    VI_rand_numer = n * (S1 * (n * n - 3 * n + 3) - n * S2 + 3 * S02) - K * (
        S1 * (n * n - n) - 2 * n * S2 + 6 * S02
    )
    VI_rand = VI_rand_numer / (n1 * n2 * n3 * S02) - EI ** 2
    if VI_rand <= 0:
        VI_rand = (n * n * S1 - n * S2 + 3 * S02) / (S02 * (n * n - 1)) - EI ** 2
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
    """Per-gene Moran's I against a kNN or principal-graph neighbour list.

    Returns a ``DataFrame`` with columns ``status``, ``p_value``,
    ``morans_test_statistic``, ``morans_I``, ``q_value``, plus every
    existing column of ``adata.var``.
    """
    del cores, verbose
    if method != "Moran_I":
        raise NotImplementedError(
            "Only method='Moran_I' is implemented in this port"
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
        # Remove edges that span disconnected principal-graph components.
        aux = get_monocle_uns(adata, "principal_graph_aux", reduction_method,
                              default={})
        if "pr_graph_cell_proj_closest_vertex" not in aux:
            raise KeyError(
                "No principal graph projection. Run learn_graph first."
            )
        closest = aux["pr_graph_cell_proj_closest_vertex"]["V1"].to_numpy()
        pg = get_monocle_uns(adata, "principal_graph", reduction_method)
        pg_adj = pg.get_adjacency_sparse()
        # Add self-loops so "same vertex" pairs pass the mask.
        pg_adj = pg_adj + sp.eye(pg_adj.shape[0], format="csr")
        filtered = []
        for i, nbrs in enumerate(knn_list):
            source_pp = int(closest[i]) - 1
            keep = []
            for j in nbrs:
                target_pp = int(closest[j]) - 1
                if (
                    0 <= source_pp < pg_adj.shape[0]
                    and 0 <= target_pp < pg_adj.shape[1]
                    and pg_adj[source_pp, target_pp] > 0
                ):
                    keep.append(j)
            filtered.append(keep)
        knn_list = filtered

    W = _build_W(knn_list, n=adata.n_obs)

    # Transform counts → log10(x / sf + 0.1) for most families (matches R).
    X = adata.X
    sf = adata.obs["Size_Factor"].to_numpy(dtype=float)
    results: list[dict[str, Any]] = []
    gene_names = list(adata.var_names)
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
            I, Z, p = _moran_per_row(values, W, alternative)
            results.append(
                {
                    "status": "OK",
                    "p_value": p,
                    "morans_test_statistic": Z,
                    "morans_I": I,
                }
            )
        except Exception:
            results.append(
                {
                    "status": "FAIL",
                    "p_value": np.nan,
                    "morans_test_statistic": np.nan,
                    "morans_I": np.nan,
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
