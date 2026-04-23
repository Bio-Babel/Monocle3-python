"""Internal helpers shared across the monocle3 Python port.

These are deliberately minimal — each function isolates an R idiom whose
Python equivalent is not obvious, or provides a single source of truth
for how ``anndata.AnnData`` is used to back the legacy ``cell_data_set``
container. Nothing here is part of the public API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse as sp

__all__ = [
    "as_sparse_csc",
    "as_sparse_csr",
    "as_array_2d",
    "ensure_monocle_uns",
    "get_monocle_uns",
    "log1p_sparse",
    "sparse_col_sums",
    "sparse_row_sums",
    "sparse_col_means",
    "size_factor_normalize",
    "size_factor_normalize_then_log",
    "tidy_model_frame",
    "coerce_obs_categorical",
    "numeric_column",
]


def as_sparse_csr(x: Any) -> sp.csr_matrix:
    """Return *x* as a CSR sparse matrix (no copy when possible)."""
    if sp.isspmatrix_csr(x):
        return x
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(np.asarray(x))


def as_sparse_csc(x: Any) -> sp.csc_matrix:
    """Return *x* as a CSC sparse matrix (no copy when possible)."""
    if sp.isspmatrix_csc(x):
        return x
    if sp.issparse(x):
        return x.tocsc()
    return sp.csc_matrix(np.asarray(x))


def as_array_2d(x: Any) -> np.ndarray:
    """Return *x* as a 2D dense ``numpy.ndarray``."""
    if sp.issparse(x):
        return np.asarray(x.todense())
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def ensure_monocle_uns(adata: AnnData) -> dict:
    """Return ``adata.uns["monocle3"]`` dict, creating it if missing."""
    uns = adata.uns
    if "monocle3" not in uns or not isinstance(uns["monocle3"], dict):
        uns["monocle3"] = {}
    return uns["monocle3"]


def get_monocle_uns(adata: AnnData, *path: str, default: Any = None) -> Any:
    """Follow ``uns["monocle3"][path[0]][path[1]]...`` or return *default*."""
    node: Any = adata.uns.get("monocle3", {})
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def sparse_col_sums(X: Any) -> np.ndarray:
    """Column sums of *X* as a 1D ndarray."""
    if sp.issparse(X):
        return np.asarray(X.sum(axis=0)).ravel()
    return np.asarray(X).sum(axis=0)


def sparse_row_sums(X: Any) -> np.ndarray:
    """Row sums of *X* as a 1D ndarray."""
    if sp.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X).sum(axis=1)


def sparse_col_means(X: Any) -> np.ndarray:
    """Column means of *X* as a 1D ndarray."""
    n = X.shape[0]
    return sparse_col_sums(X) / max(n, 1)


def log1p_sparse(X: Any, pseudocount: float = 1.0) -> sp.csr_matrix:
    """Return ``log(X + pseudocount)`` preserving sparsity when pseudocount=1."""
    if pseudocount == 1.0 and sp.issparse(X):
        out = X.copy().astype(float)
        out.data = np.log(out.data + 1.0) / np.log(2.0)  # log2 to match R default
        return out.tocsr()
    dense = as_array_2d(X).astype(float)
    return sp.csr_matrix(np.log2(dense + pseudocount))


def size_factor_normalize(X: sp.spmatrix, size_factors: np.ndarray) -> sp.csr_matrix:
    """Return cells × genes counts divided by the per-cell size factor.

    Matches R ``monocle3::normalize_expr_data`` behaviour — divides each
    cell (row) by its size factor; produces a sparse CSR matrix.
    """
    X = as_sparse_csr(X).astype(float, copy=True)
    sf = np.asarray(size_factors, dtype=float)
    if X.shape[0] != sf.shape[0]:
        raise ValueError(
            f"size_factors shape {sf.shape} incompatible with X rows {X.shape[0]}"
        )
    # Row-scale: multiply by diag(1 / sf) from the left.
    D = sp.diags(1.0 / sf)
    return (D @ X).tocsr()


def size_factor_normalize_then_log(
    X: sp.spmatrix,
    size_factors: np.ndarray,
    pseudocount: float = 1.0,
) -> sp.csr_matrix:
    """Size-factor normalize, then log2(x + pseudocount)."""
    norm = size_factor_normalize(X, size_factors)
    return log1p_sparse(norm, pseudocount=pseudocount)


def tidy_model_frame(
    per_gene_results: list[dict],
    gene_ids: list[str],
    gene_short_names: list[str],
) -> pd.DataFrame:
    """Stack per-gene model outputs into a tidy long data frame.

    Each entry in *per_gene_results* is a dict mapping column name (e.g.
    ``term``, ``estimate``, ``std_error``) to an equal-length list of
    per-coefficient values for that gene. The output has the usual
    ``broom::tidy()`` columns plus ``gene_id`` and ``gene_short_name``.
    """
    frames: list[pd.DataFrame] = []
    for gid, gsn, res in zip(gene_ids, gene_short_names, per_gene_results):
        if res is None or len(next(iter(res.values()), ())) == 0:
            continue
        df = pd.DataFrame(res)
        df["gene_id"] = gid
        df["gene_short_name"] = gsn
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=[
                "term",
                "estimate",
                "std_error",
                "statistic",
                "p_value",
                "gene_id",
                "gene_short_name",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def coerce_obs_categorical(series: pd.Series) -> pd.Series:
    """Turn a ``pd.Series`` into an ordered categorical factor if not numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.astype("category")


def numeric_column(df: pd.DataFrame, column: str) -> np.ndarray:
    """Return ``df[column]`` as float numpy array with NaNs for non-numeric."""
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
