"""Internal helpers shared across the monocle3 Python port.

These are deliberately minimal — each function isolates an R idiom whose
Python equivalent is not obvious, or provides a single source of truth
for how ``anndata.AnnData`` is used to back the legacy ``cell_data_set``
container. Nothing here is part of the public API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from anndata import AnnData
from scipy import sparse as sp

__all__ = [
    "as_sparse_csr",
    "ensure_monocle_uns",
    "get_monocle_uns",
    "log1p_sparse",
    "sparse_col_sums",
    "sparse_row_sums",
    "size_factor_normalize",
]


def as_sparse_csr(x: Any) -> sp.csr_matrix:
    """Return *x* as a CSR sparse matrix (no copy when possible)."""
    if sp.isspmatrix_csr(x):
        return x
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(np.asarray(x))


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


def log1p_sparse(X: Any, pseudocount: float = 1.0) -> sp.csr_matrix:
    """Return ``log(X + pseudocount)`` preserving sparsity when pseudocount=1."""
    if pseudocount == 1.0 and sp.issparse(X):
        out = X.copy().astype(float)
        out.data = np.log(out.data + 1.0) / np.log(2.0)  # log2 to match R default
        return out.tocsr()
    if sp.issparse(X):
        dense = np.asarray(X.todense()).astype(float)
    else:
        arr = np.asarray(X).astype(float)
        dense = arr if arr.ndim == 2 else arr.reshape(-1, 1)
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
