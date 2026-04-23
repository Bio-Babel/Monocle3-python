"""Preprocessing (constructor, size factors, detect_genes, PCA/LSI).

Direct translation of the R source files ``cell_data_set.R``,
``utils.R`` (size-factor / detect-gene helpers), ``preprocess_cds.R``,
and ``pca.R``. The R container ``cell_data_set`` is replaced by
``anndata.AnnData`` everywhere — see essential-suggestions §0.1.
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse.linalg import LinearOperator, svds

from ._utils import (
    as_sparse_csr,
    ensure_monocle_uns,
    log1p_sparse,
    size_factor_normalize,
    sparse_col_sums,
)

__all__ = [
    "new_cell_data_set",
    "estimate_size_factors",
    "detect_genes",
    "preprocess_cds",
]


# ---------------------------------------------------------------------------
# new_cell_data_set — R: cell_data_set.R::new_cell_data_set
# ---------------------------------------------------------------------------


def new_cell_data_set(
    expression_data: Any,
    cell_metadata: pd.DataFrame | None = None,
    gene_metadata: pd.DataFrame | None = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Create an AnnData from an R-convention (genes × cells) expression matrix.

    The R ``cell_data_set`` class inherited from ``SingleCellExperiment``
    which is genes × cells. ``anndata.AnnData`` is cells × genes, so the
    input matrix is transposed before being stored in ``adata.X``.

    Parameters
    ----------
    expression_data : array-like, scipy.sparse matrix, or pandas.DataFrame
        Gene-by-cell expression matrix (R convention).
    cell_metadata : pandas.DataFrame, optional
        One row per cell, in the same order as ``expression_data``'s columns.
        Its index becomes ``adata.obs_names``.
    gene_metadata : pandas.DataFrame, optional
        One row per gene, in the same order as ``expression_data``'s rows.
        Must contain a ``gene_short_name`` column for downstream plotting.
    verbose : bool, optional
        Unused; retained for R-signature parity.

    Returns
    -------
    anndata.AnnData
        Cells × genes. Raw counts in ``X``. Size factors computed and stored
        in ``adata.obs["Size_Factor"]``.
    """
    del verbose  # unused, kept for signature parity

    if not (sp.issparse(expression_data) or hasattr(expression_data, "__array__")):
        raise TypeError(
            "expression_data must be a numpy array, scipy sparse matrix, or pandas DataFrame"
        )

    # Extract cell/gene names from dense inputs where possible.
    if isinstance(expression_data, pd.DataFrame):
        gene_names: Sequence | None = expression_data.index
        cell_names: Sequence | None = expression_data.columns
        X = sp.csr_matrix(expression_data.to_numpy())
    elif sp.issparse(expression_data):
        gene_names = None
        cell_names = None
        X = as_sparse_csr(expression_data)
    else:
        arr = np.asarray(expression_data)
        gene_names = None
        cell_names = None
        X = sp.csr_matrix(arr)

    n_genes, n_cells = X.shape

    # Validate cell_metadata.
    if cell_metadata is not None:
        if cell_metadata.shape[0] != n_cells:
            raise ValueError(
                "cell_metadata must have the same number of rows as "
                "expression_data has columns"
            )
        if cell_metadata.index is None:
            raise ValueError("cell_metadata must have row names")
        if cell_names is not None and list(cell_metadata.index) != list(cell_names):
            raise ValueError(
                "row.names of cell_metadata must equal colnames of expression_data"
            )
        obs = cell_metadata.copy()
    else:
        labels = list(cell_names) if cell_names is not None else [
            f"cell_{i}" for i in range(n_cells)
        ]
        obs = pd.DataFrame({"cell": labels}, index=labels)

    # Validate gene_metadata.
    if gene_metadata is not None:
        if gene_metadata.shape[0] != n_genes:
            raise ValueError(
                "gene_metadata must have the same number of rows as "
                "expression_data has rows"
            )
        if gene_metadata.index is None:
            raise ValueError("gene_metadata must have row names")
        if gene_names is not None and list(gene_metadata.index) != list(gene_names):
            raise ValueError(
                "row.names of gene_metadata must equal row.names of expression_data"
            )
        var = gene_metadata.copy()
    else:
        labels = list(gene_names) if gene_names is not None else [
            f"gene_{i}" for i in range(n_genes)
        ]
        var = pd.DataFrame({"id": labels}, index=labels)

    if "gene_short_name" not in var.columns:
        warnings.warn(
            "gene_metadata must contain a column named 'gene_short_name' "
            "for certain functions.",
            stacklevel=2,
        )

    # Transpose: AnnData is cells × genes.
    adata = ad.AnnData(X=X.T.tocsr(), obs=obs, var=var)

    # Default monocle3 uns namespace.
    uns = ensure_monocle_uns(adata)
    uns["cds_version"] = "1.4.26"

    # Mirror R: estimate_size_factors is called in the constructor.
    estimate_size_factors(adata)
    return adata


# ---------------------------------------------------------------------------
# estimate_size_factors — R: utils.R::estimate_size_factors
# ---------------------------------------------------------------------------


def estimate_size_factors(
    adata: ad.AnnData,
    round_exprs: bool = True,
    method: str = "mean-geometric-mean-total",
) -> ad.AnnData:
    """Attach ``Size_Factor`` to ``adata.obs`` using the R convention.

    Parameters
    ----------
    adata : anndata.AnnData
        Cells × genes. ``adata.X`` is treated as raw counts.
    round_exprs : bool, default True
        Round counts to integers (mirrors R's ``round_exprs``).
    method : {"mean-geometric-mean-total", "mean-geometric-mean-log-total"}
        The R default is the first.

    Returns
    -------
    anndata.AnnData
        The same object, with ``adata.obs["Size_Factor"]`` populated.
    """
    if method not in {"mean-geometric-mean-total", "mean-geometric-mean-log-total"}:
        raise ValueError(
            "method must be 'mean-geometric-mean-total' or "
            "'mean-geometric-mean-log-total'"
        )

    counts = adata.X
    if sp.issparse(counts):
        if round_exprs:
            # Round in-place on a copy.
            counts = counts.copy()
            counts.data = np.round(counts.data)
        cell_total = np.asarray(counts.sum(axis=1)).ravel()
    else:
        counts = np.asarray(counts, dtype=float)
        if round_exprs:
            counts = np.round(counts)
        cell_total = counts.sum(axis=1)

    if np.any(cell_total == 0):
        warnings.warn(
            "Your AnnData contains cells with zero reads. "
            "Remove them before calling estimate_size_factors.",
            stacklevel=2,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "mean-geometric-mean-total":
            sfs = cell_total / np.exp(np.mean(np.log(cell_total)))
        else:  # mean-geometric-mean-log-total
            sfs = np.log(cell_total) / np.exp(np.mean(np.log(np.log(cell_total))))
    sfs = np.where(np.isfinite(sfs), sfs, 1.0)

    adata.obs["Size_Factor"] = sfs.astype(float)
    return adata


# ---------------------------------------------------------------------------
# detect_genes — R: utils.R::detect_genes
# ---------------------------------------------------------------------------


def detect_genes(adata: ad.AnnData, min_expr: float = 0) -> ad.AnnData:
    """Count cells-per-gene and genes-per-cell above *min_expr*.

    Mirrors R ``detect_genes``. Writes:

    - ``adata.var["num_cells_expressed"]``
    - ``adata.obs["num_genes_expressed"]``
    """
    X = adata.X
    if sp.issparse(X):
        mask = X > min_expr
        num_cells_expressed = np.asarray(mask.sum(axis=0)).ravel()
        num_genes_expressed = np.asarray(mask.sum(axis=1)).ravel()
    else:
        arr = np.asarray(X)
        num_cells_expressed = (arr > min_expr).sum(axis=0)
        num_genes_expressed = (arr > min_expr).sum(axis=1)

    adata.var["num_cells_expressed"] = num_cells_expressed.astype(np.int64)
    adata.obs["num_genes_expressed"] = num_genes_expressed.astype(np.int64)
    return adata


# ---------------------------------------------------------------------------
# normalized_counts — R: utils.R::normalized_counts
# ---------------------------------------------------------------------------


def normalized_counts(
    adata: ad.AnnData,
    norm_method: str = "log",
    pseudocount: float = 1.0,
) -> sp.csr_matrix:
    """Size-factor-normalize the raw counts, optionally log-transformed.

    Matches R's ``normalized_counts``: log uses base 10 (distinct from the
    base-2 log in ``preprocess_cds`` — this mismatch is an R quirk that we
    preserve for parity).
    """
    if norm_method not in {"log", "binary", "size_only"}:
        raise ValueError("norm_method must be 'log', 'binary', or 'size_only'")

    X = adata.X
    if norm_method == "binary":
        if sp.issparse(X):
            out = (X > 0).astype(np.float64).tocsr()
        else:
            out = sp.csr_matrix((np.asarray(X) > 0).astype(np.float64))
        return out

    size_factors = adata.obs["Size_Factor"].to_numpy(dtype=float)

    if sp.issparse(X):
        if norm_method == "log" and pseudocount != 1:
            raise ValueError(
                "normalized_counts: pseudocount must be 1 for sparse matrices "
                "with norm_method='log'"
            )
        out = size_factor_normalize(X, size_factors)
        if norm_method == "log":
            out.data = np.log10(out.data + pseudocount)
        return out

    dense = np.asarray(X, dtype=float)
    # Broadcast size factors across columns: dense is cells × genes, sf per cell.
    dense = dense / size_factors.reshape(-1, 1)
    if norm_method == "log":
        dense = np.log10(dense + pseudocount)
    return sp.csr_matrix(dense)


# ---------------------------------------------------------------------------
# sparse PCA — R: pca.R::sparse_prcomp_irlba
# ---------------------------------------------------------------------------


def _sparse_col_mean_var(X: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Column means and sample variances for a sparse matrix."""
    n = X.shape[0]
    col_sum = np.asarray(X.sum(axis=0)).ravel()
    col_mean = col_sum / max(n, 1)
    # E[X^2]
    sq = X.multiply(X)
    col_sum_sq = np.asarray(sq.sum(axis=0)).ravel()
    col_mean_sq = col_sum_sq / max(n, 1)
    var_pop = col_mean_sq - col_mean**2
    var_sample = var_pop * n / max(1, n - 1)
    return col_mean, var_sample


def sparse_prcomp(
    X: sp.csr_matrix | np.ndarray,
    n: int,
    center: bool = True,
    scale: bool = True,
) -> dict:
    """Truncated PCA mirroring R's ``sparse_prcomp_irlba``.

    Parameters
    ----------
    X : scipy.sparse or numpy.ndarray
        Cells × features.
    n : int
        Number of components.
    center : bool, default True
        Subtract per-feature means (done implicitly for sparse inputs).
    scale : bool, default True
        Divide by per-feature sample standard deviation.

    Returns
    -------
    dict
        ``x`` (cells × n projected coordinates), ``rotation`` (features × n
        loadings), ``sdev`` (component standard deviations), ``center``, and
        ``svd_scale`` (either a vector or ``None``). Names match R's
        ``sparse_prcomp_irlba`` return list.
    """
    X_sparse = sp.issparse(X)
    if X_sparse:
        Xcsr: sp.csr_matrix = as_sparse_csr(X).astype(np.float64, copy=False)
    else:
        Xcsr = np.asarray(X, dtype=np.float64)

    n_rows, n_cols = Xcsr.shape
    k = min(int(n), min(n_rows, n_cols) - 1)
    if k <= 0:
        raise ValueError("n must be at least 1 and less than min(dim(X))")

    if center:
        if X_sparse:
            col_mean, col_var = _sparse_col_mean_var(Xcsr)
        else:
            col_mean = Xcsr.mean(axis=0)
            col_var = Xcsr.var(axis=0, ddof=1)
    else:
        col_mean = np.zeros(n_cols)
        if X_sparse:
            _, col_var = _sparse_col_mean_var(Xcsr)
        else:
            col_var = Xcsr.var(axis=0, ddof=1)

    if scale:
        col_std = np.sqrt(col_var)
        col_std[col_std == 0] = 1.0
    else:
        col_std = np.ones(n_cols)

    # Fast dense path for small problems; LinearOperator for large sparse.
    if not X_sparse or (n_rows * n_cols < 5_000_000):
        dense = Xcsr.toarray() if X_sparse else Xcsr
        A = dense - col_mean
        if scale:
            A = A / col_std
        # Use numpy SVD for determinism.
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        u = u[:, :k]
        s = s[:k]
        vt = vt[:k]
    else:
        def matvec(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=np.float64).ravel()
            scaled_v = v / col_std
            # Xcsr is (n_rows, n_cols); returns length n_rows.
            return Xcsr @ scaled_v - float(col_mean @ scaled_v)

        def rmatvec(u_vec: np.ndarray) -> np.ndarray:
            u_vec = np.asarray(u_vec, dtype=np.float64).ravel()
            # Returns length n_cols.
            return (Xcsr.T @ u_vec - col_mean * u_vec.sum()) / col_std

        op = LinearOperator(
            shape=(n_rows, n_cols),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float64,
        )
        # svds returns ascending — ask for the largest k and then reverse.
        u, s, vt = svds(op, k=k, which="LM")
        order = np.argsort(-s)
        u = u[:, order]
        s = s[order]
        vt = vt[order]

    sdev = s / np.sqrt(max(1, n_rows - 1))
    x = u * s  # cells × k projected coords
    rotation = vt.T  # features × k loadings

    return {
        "x": np.asarray(x),
        "rotation": np.asarray(rotation),
        "sdev": np.asarray(sdev),
        "center": np.asarray(col_mean) if center else None,
        "svd_scale": np.asarray(col_std) if scale else None,
    }


def _tfidf(X: sp.csr_matrix, log_scale_tf: bool = True, scale_factor: float = 1e5) -> dict:
    """TF-IDF transform matching R's ``tfidf`` used by LSI preprocessing."""
    Xcsr = as_sparse_csr(X).astype(np.float64, copy=False)
    n_cells, n_genes = Xcsr.shape

    # R convention (genes × cells) treats col_sums as per-cell totals. In
    # Python (cells × genes) that's row sums.
    cell_totals = np.asarray(Xcsr.sum(axis=1)).ravel()
    safe_totals = np.where(cell_totals > 0, cell_totals, 1.0)
    tf = sp.diags(1.0 / safe_totals) @ Xcsr

    if log_scale_tf:
        # R does log1p(x * scale_factor) element-wise on non-zeros.
        tf = tf.copy()
        tf.data = np.log1p(tf.data * scale_factor)

    # R: row_sums = number of cells expressing each gene.
    gene_cell_counts = np.asarray((Xcsr > 0).sum(axis=0)).ravel()
    safe_gene_counts = np.where(gene_cell_counts > 0, gene_cell_counts, 1.0)
    idf = np.log(1.0 + n_cells / safe_gene_counts)

    tf_idf = tf @ sp.diags(idf)
    return {
        "tf_idf": tf_idf.tocsr(),
        "log_scale_tf": log_scale_tf,
        "scale_factor": scale_factor,
        "cell_totals": cell_totals,
        "gene_cell_counts": gene_cell_counts,
        "num_cells": n_cells,
        "frequencies": True,
    }


# ---------------------------------------------------------------------------
# preprocess_cds — R: preprocess_cds.R::preprocess_cds
# ---------------------------------------------------------------------------


def preprocess_cds(
    adata: ad.AnnData,
    method: str = "PCA",
    num_dim: int = 50,
    norm_method: str = "log",
    use_genes: Sequence[str] | None = None,
    pseudo_count: float | None = None,
    scaling: bool = True,
    verbose: bool = False,
    build_nn_index: bool = False,
    nn_control: dict | None = None,
) -> ad.AnnData:
    """Normalize counts then compute PCA or LSI.

    Mirrors R ``preprocess_cds``. Writes the projected matrix to
    ``adata.obsm["X_pca"]`` (or ``X_lsi``) and the fitted model components
    to ``adata.uns["monocle3"]["preprocess"][method]`` for later use by
    ``reduce_dimension`` and ``preprocess_transform``.

    Parameters
    ----------
    adata : anndata.AnnData
        Cells × genes with raw counts in ``X`` and a populated
        ``Size_Factor`` obs column.
    method : {"PCA", "LSI"}, default "PCA"
        Initial dimensionality-reduction method.
    num_dim : int, default 50
        Number of components to keep.
    norm_method : {"log", "size_only", "none"}, default "log"
        Pre-SVD normalization. R ``normalize_expr_data`` uses base-2 log.
    use_genes : sequence of str, optional
        Restrict to a subset of genes (must be in ``adata.var_names``).
    pseudo_count : float, optional
        Defaults to ``1`` for ``log`` and ``0`` otherwise.
    scaling : bool, default True
        Center and unit-scale features for PCA.
    verbose : bool, default False
        Retained for signature parity; currently unused.
    build_nn_index : bool, default False
        Build and cache a nearest-neighbour index on the reduced
        coordinates. Done post-hoc via ``nearest_neighbors.set_cds_nn_index``
        only when this argument is ``True`` (matches R).
    nn_control : dict, optional
        NN-index parameters passed through to ``make_nn_index``.

    Returns
    -------
    anndata.AnnData
        The same object, with new ``obsm`` and ``uns["monocle3"]`` entries.
    """
    del verbose  # hook for future logging
    if method not in {"PCA", "LSI"}:
        raise ValueError("method must be 'PCA' or 'LSI'")
    if norm_method not in {"log", "size_only", "none"}:
        raise ValueError("norm_method must be 'log', 'size_only', or 'none'")
    if not isinstance(num_dim, (int, np.integer)) or num_dim <= 0:
        raise ValueError("num_dim must be a positive integer")

    if use_genes is not None:
        missing = [g for g in use_genes if g not in adata.var_names]
        if missing:
            raise ValueError(
                "use_genes must be a subset of adata.var_names; "
                f"missing: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

    if "Size_Factor" not in adata.obs.columns:
        raise ValueError("Call estimate_size_factors before preprocess_cds.")
    if adata.obs["Size_Factor"].isna().any():
        raise ValueError("One or more cells has a size factor of NA.")

    # Log-norm or size-only.
    if pseudo_count is None:
        pseudo_count = 1.0 if norm_method == "log" else 0.0

    X = adata.X
    size_factors = adata.obs["Size_Factor"].to_numpy(dtype=float)

    if norm_method == "log":
        norm = size_factor_normalize(X, size_factors)
        norm = log1p_sparse(norm, pseudocount=float(pseudo_count))
    elif norm_method == "size_only":
        norm = size_factor_normalize(X, size_factors)
        if pseudo_count != 0.0:
            norm = norm + pseudo_count
        norm = as_sparse_csr(norm)
    else:  # "none"
        norm = as_sparse_csr(X).astype(np.float64, copy=False)

    # Filter all-zero genes.
    gene_sums = sparse_col_sums(norm)
    keep_gene_mask = np.isfinite(gene_sums) & (gene_sums != 0)
    if use_genes is not None:
        subset_mask = np.array(
            [g in set(use_genes) for g in adata.var_names], dtype=bool
        )
        keep_gene_mask = keep_gene_mask & subset_mask

    if not keep_gene_mask.any():
        raise RuntimeError("all genes have standard deviation zero")

    norm_sub = norm[:, keep_gene_mask]
    kept_gene_names = list(np.asarray(adata.var_names)[keep_gene_mask])

    uns = ensure_monocle_uns(adata)
    uns.setdefault("preprocess", {})

    if method == "PCA":
        pca = sparse_prcomp(norm_sub, n=int(num_dim), center=bool(scaling),
                            scale=bool(scaling))
        adata.obsm["X_pca"] = pca["x"].astype(np.float64)
        model = {
            "num_dim": int(num_dim),
            "norm_method": norm_method,
            "use_genes": None if use_genes is None else list(use_genes),
            "pseudo_count": float(pseudo_count),
            "svd_v": pca["rotation"],
            "svd_sdev": pca["sdev"],
            "svd_center": pca["center"],
            "svd_scale": pca["svd_scale"],
            "prop_var_expl": (pca["sdev"] ** 2) / np.sum(pca["sdev"] ** 2),
            "gene_names": kept_gene_names,
        }
        uns["preprocess"]["PCA"] = model
    else:  # LSI
        tfidf_res = _tfidf(norm_sub)
        tfidf_mat = tfidf_res["tf_idf"]
        n_rows = tfidf_mat.shape[0]
        k = min(int(num_dim), min(tfidf_mat.shape) - 1)
        # R runs irlba on t(tfidf) → equivalent to SVD of tfidf with swapped roles.
        u, s, vt = svds(tfidf_mat, k=k, which="LM")
        order = np.argsort(-s)
        u = u[:, order]
        s = s[order]
        vt = vt[order]
        adata.obsm["X_lsi"] = (u * s).astype(np.float64)
        model = {
            "num_dim": int(num_dim),
            "norm_method": norm_method,
            "use_genes": None if use_genes is None else list(use_genes),
            "pseudo_count": float(pseudo_count),
            "log_scale_tf": tfidf_res["log_scale_tf"],
            "frequencies": tfidf_res["frequencies"],
            "scale_factor": tfidf_res["scale_factor"],
            "col_sums": tfidf_res["cell_totals"],
            "row_sums": tfidf_res["gene_cell_counts"],
            "num_cols": tfidf_res["num_cells"],
            "svd_v": vt.T,
            "svd_sdev": s / np.sqrt(max(1, n_rows - 1)),
            "gene_names": kept_gene_names,
        }
        uns["preprocess"]["LSI"] = model

    # Clear stale Aligned beta from prior align_cds.
    uns.setdefault("Aligned", {})
    if isinstance(uns["Aligned"], dict) and uns["Aligned"].get("beta") is not None:
        uns["Aligned"]["beta"] = None

    if build_nn_index:
        # Deferred to nearest_neighbors to avoid a circular import.
        from .nearest_neighbors import make_nn_index, set_cds_nn_index

        reduction_key = "X_pca" if method == "PCA" else "X_lsi"
        nn_index = make_nn_index(adata.obsm[reduction_key], nn_control=nn_control)
        set_cds_nn_index(adata, reduction_method=method, nn_index=nn_index)

    return adata
