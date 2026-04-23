"""Pure-Python ports of the two ``Rcpp``-exported helpers in
``monocle3/src/clustering.cpp``.

The originals are ~90 LOC of numerical Rcpp. Re-implementing in NumPy /
SciPy keeps the Python package install path free of a C toolchain and
carries no performance penalty at the sizes monocle3 uses (tens of
thousands of cells, k ≲ 20 for the Jaccard graph; N ≲ a few hundred
partitions for the pnorm matrix).

Output shapes and indexing conventions mirror R for drop-in callers.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp
from scipy.stats import norm

__all__ = ["jaccard_coeff", "pnorm_over_mat"]


def jaccard_coeff(idx: np.ndarray, weight: bool) -> np.ndarray:
    """Port of ``src/clustering.cpp::jaccard_coeff``.

    Given a ``(N, k)`` integer kNN index matrix *idx* (1-based indices,
    R convention; each row is the kNN set of one cell including itself
    as the first entry), return a ``(N*k, 3)`` float array whose rows
    are ``(i+1, j+1, w)`` with ``w`` the normalized Jaccard weight of
    the edge ``i → j``. When ``weight`` is ``False`` all weights are 1.

    The output matches ``jaccard_coeff(idx, weight)`` from R bit-for-bit.

    Parameters
    ----------
    idx : numpy.ndarray
        Integer ``(N, k)`` kNN index matrix (1-based).
    weight : bool
        Compute Jaccard weight if ``True``; otherwise emit all-ones
        weights (later divided by their max, i.e. 1).

    Returns
    -------
    numpy.ndarray
        ``(N*k, 3)`` float array of ``[i+1, j+1, w]`` edge rows, where
        ``w`` has been normalized by its per-call max so the largest
        weight is 1.
    """
    idx = np.asarray(idx, dtype=np.int64)
    n, k = idx.shape

    # Pre-compute row-set membership so intersection is vectorisable.
    # Build a 0/1 CSR adjacency A where A[i, j] = 1 if j+1 is in row i's kNN.
    # Then intersections = A @ A.T.
    rows = np.repeat(np.arange(n), k)
    cols = (idx.ravel() - 1)  # convert to 0-based
    if cols.min() < 0 or cols.max() >= n:
        raise ValueError(
            "jaccard_coeff: kNN indices must be 1-based and in [1, N]"
        )
    data = np.ones_like(rows, dtype=np.float64)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    out = np.empty((n * k, 3), dtype=np.float64)
    out[:, 0] = np.repeat(np.arange(1, n + 1), k)
    out[:, 1] = idx.ravel().astype(np.float64)

    if not weight:
        # R divides by max weight at the end; with all-ones that is 1.
        out[:, 2] = 1.0
        return out

    # |A ∩ B|  — only edges present in idx need this.
    i_idx = np.repeat(np.arange(n), k)
    j_idx = idx.ravel() - 1
    # Row-wise dot product between A[i] and A[j].
    AAt = A @ A.T
    inter = np.asarray(AAt[i_idx, j_idx]).ravel()
    # |A ∪ B| = 2k - |A ∩ B|
    union = 2.0 * k - inter

    w = np.zeros(n * k, dtype=np.float64)
    mask = inter > 0
    w[mask] = inter[mask] / union[mask]
    max_w = w.max() if w.max() > 0 else 1.0
    out[:, 2] = w / max_w
    return out


def pnorm_over_mat(num_links_ij: np.ndarray, var_null_num_links: np.ndarray) -> np.ndarray:
    """Port of ``src/clustering.cpp::pnorm_over_mat``.

    Element-wise ``R::pnorm(x, mean=0, sd=sqrt(v), lower=FALSE,
    log=FALSE)``. With SciPy this is ``norm.sf(x / sqrt(v))``.

    Parameters
    ----------
    num_links_ij : numpy.ndarray
        ``(N, N)`` observed link-count matrix between partitions.
    var_null_num_links : numpy.ndarray
        ``(N, N)`` null-distribution variance matrix (same shape).

    Returns
    -------
    numpy.ndarray
        Upper-tail p-values with the same shape as the inputs.
    """
    num = np.asarray(num_links_ij, dtype=float)
    var = np.asarray(var_null_num_links, dtype=float)
    if num.shape != var.shape:
        raise ValueError(
            f"pnorm_over_mat: shape mismatch {num.shape} vs {var.shape}"
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        z = num / np.sqrt(var)
    # R::pnorm(..., sd=0) returns 0 when x > 0, 1 otherwise. Match that.
    out = norm.sf(z)
    zero_var = var == 0
    if zero_var.any():
        out = np.where(zero_var & (num > 0), 0.0, out)
        out = np.where(zero_var & (num <= 0), 1.0, out)
    return out
