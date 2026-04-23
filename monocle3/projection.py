"""preprocess_transform / reduce_dimension_transform — project new cells
onto an existing PCA, LSI, or UMAP model.

R stores its models on ``cds@reduce_dim_aux``; Python keeps them on
``ref_adata.uns["monocle3"]["preprocess"][<method>]`` (for PCA / LSI)
and ``ref_adata.uns["monocle3"]["reduce_dim"][<method>]["model"]`` (for
UMAP / tSNE). The transform functions take a *query* AnnData plus the
reference AnnData that holds the fitted models.
"""
from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse as sp

from ._utils import (
    as_sparse_csr,
    ensure_monocle_uns,
    get_monocle_uns,
    log1p_sparse,
    size_factor_normalize,
)

__all__ = ["preprocess_transform", "reduce_dimension_transform"]


def _normalize_query_counts(
    adata: ad.AnnData, norm_method: str, pseudo_count: float
) -> sp.csr_matrix:
    """Apply the same normalisation the reference used, yielding a
    cells × genes sparse matrix."""
    if "Size_Factor" not in adata.obs.columns:
        raise KeyError(
            "preprocess_transform: estimate_size_factors must be called on "
            "the query before projection."
        )
    if adata.obs["Size_Factor"].isna().any():
        raise ValueError(
            "preprocess_transform: one or more query cells has Size_Factor=NA."
        )

    X = adata.X
    sf = adata.obs["Size_Factor"].to_numpy(dtype=float)
    if norm_method == "log":
        norm = size_factor_normalize(X, sf)
        norm = log1p_sparse(norm, pseudocount=float(pseudo_count))
    elif norm_method == "size_only":
        norm = size_factor_normalize(X, sf)
        if pseudo_count != 0.0:
            norm = norm + pseudo_count
        norm = as_sparse_csr(norm)
    elif norm_method == "none":
        norm = as_sparse_csr(X).astype(np.float64, copy=False)
    else:
        raise ValueError(
            f"preprocess_transform: unknown norm_method {norm_method!r}"
        )
    return norm


def _align_query_genes_to_model(
    norm: sp.csr_matrix,
    query_var_names: list[str],
    model_gene_names: list[str],
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Sub-select query columns to the intersection with the model's
    stored gene set, preserving the model's gene order.

    Returns ``(norm_sub, intersect_in_model, intersect_gene_names)``
    where ``intersect_in_model`` is a 1-d ``int64`` array of positions
    into the model's rotation/center/scale arrays.
    """
    q_idx = {g: i for i, g in enumerate(query_var_names)}
    model_pos: list[int] = []
    query_pos: list[int] = []
    intersect_names: list[str] = []
    for m_pos, g in enumerate(model_gene_names):
        j = q_idx.get(g)
        if j is not None:
            model_pos.append(m_pos)
            query_pos.append(j)
            intersect_names.append(g)

    if not intersect_names:
        raise ValueError(
            "preprocess_transform: no genes in the query overlap the "
            "reference model gene set."
        )
    if len(intersect_names) / max(len(query_var_names), 1) < 0.5:
        import warnings
        warnings.warn(
            "preprocess_transform: fewer than half the query genes are in "
            "the reference — are both prepared from the same gene set?",
            stacklevel=2,
        )

    norm_sub = norm[:, np.asarray(query_pos, dtype=np.int64)]
    return norm_sub, np.asarray(model_pos, dtype=np.int64), np.asarray(intersect_names)


def _pca_transform(
    query_adata: ad.AnnData, model: dict
) -> np.ndarray:
    """Project the query cells onto the reference PCA space.

    Mirrors R ``sparse_apply_transform`` — subset to intersecting genes,
    apply stored centering / scaling, multiply by the stored rotation.
    """
    norm = _normalize_query_counts(
        query_adata,
        norm_method=str(model["norm_method"]),
        pseudo_count=float(model["pseudo_count"]),
    )

    norm_sub, model_idx, _ = _align_query_genes_to_model(
        norm,
        list(query_adata.var_names),
        list(model["gene_names"]),
    )

    # Pull the (possibly sub-selected) center / scale / rotation slices.
    rotation = np.asarray(model["svd_v"])[model_idx]
    center = model.get("svd_center")
    scale = model.get("svd_scale")

    # Apply centering / scaling. The reference stored vectors are over the
    # full gene set that survived preprocess_cds; select the intersection.
    dense = np.asarray(norm_sub.todense() if sp.issparse(norm_sub) else norm_sub,
                       dtype=np.float64)
    if center is not None:
        dense = dense - np.asarray(center)[model_idx]
    if scale is not None:
        scale_arr = np.asarray(scale)[model_idx]
        scale_arr = np.where(scale_arr == 0, 1.0, scale_arr)
        dense = dense / scale_arr
    return dense @ rotation


def _lsi_transform(
    query_adata: ad.AnnData, model: dict
) -> np.ndarray:
    """Project query cells onto reference LSI space.

    Mirrors R ``preprocess_transform`` LSI branch — applies the stored
    TF-IDF parameters (row_sums / col_sums / num_cols / scale_factor /
    log_scale_tf / frequencies) and multiplies by the stored rotation.
    """
    norm = _normalize_query_counts(
        query_adata,
        norm_method=str(model["norm_method"]),
        pseudo_count=float(model["pseudo_count"]),
    )

    norm_sub, model_idx, _ = _align_query_genes_to_model(
        norm,
        list(query_adata.var_names),
        list(model["gene_names"]),
    )
    # norm_sub shape: (n_query_cells, n_intersect_genes). R's TF-IDF
    # operates gene × cell, so work on the transpose throughout.
    tf = norm_sub.T.tocsr().astype(np.float64, copy=True)  # (n_genes, n_cells)

    frequencies = bool(model["frequencies"])
    scale_factor = float(model["scale_factor"])
    log_scale_tf = bool(model["log_scale_tf"])

    if frequencies:
        col_sums_query = np.asarray(tf.sum(axis=0)).ravel()
        col_sums_query = np.where(col_sums_query == 0, 1.0, col_sums_query)
        tf = tf @ sp.diags(1.0 / col_sums_query)

    if log_scale_tf:
        if frequencies:
            tf.data = np.log1p(tf.data * scale_factor)
        else:
            tf.data = np.log1p(tf.data)

    # IDF uses the *reference* row_sums and num_cols.
    row_sums_ref = np.asarray(model["row_sums"], dtype=np.float64)[model_idx]
    num_cols_ref = float(model["num_cols"])
    # Avoid division by zero.
    denom = np.where(row_sums_ref == 0, 1.0, row_sums_ref)
    idf = np.log(1.0 + num_cols_ref / denom)

    tfidf = sp.diags(idf) @ tf  # (n_genes, n_cells)
    # Project: cells × components = cells × genes @ genes × k
    rotation = np.asarray(model["svd_v"])[model_idx]
    return (tfidf.T @ rotation)


def preprocess_transform(
    query_adata: ad.AnnData,
    ref_adata: ad.AnnData,
    method: str = "PCA",
    verbose: bool = False,
) -> ad.AnnData:
    """Project *query_adata*'s counts onto the PCA or LSI space stored in
    *ref_adata*.

    The reference must have been produced by :func:`preprocess_cds` so its
    ``uns["monocle3"]["preprocess"][method]`` slot carries ``svd_v``,
    ``svd_center`` (PCA), ``svd_scale`` (PCA), ``norm_method``,
    ``pseudo_count``, ``gene_names`` and — for LSI — the stored TF-IDF
    parameters (``row_sums``, ``num_cols``, ``log_scale_tf``, etc.).

    The query must have ``obs["Size_Factor"]`` populated via
    :func:`estimate_size_factors`. The resulting projection is written
    to ``query_adata.obsm["X_pca"]`` or ``query_adata.obsm["X_lsi"]``.
    """
    del verbose
    if method not in {"PCA", "LSI"}:
        raise ValueError("method must be 'PCA' or 'LSI'")

    model = get_monocle_uns(ref_adata, "preprocess", method)
    if model is None:
        raise KeyError(
            f"preprocess_transform: reference has no {method} model. "
            f"Run preprocess_cds(method={method!r}) on the reference."
        )

    np.random.seed(2016)
    if method == "PCA":
        coords = _pca_transform(query_adata, model)
        query_adata.obsm["X_pca"] = np.asarray(coords, dtype=np.float64)
    else:
        coords = _lsi_transform(query_adata, model)
        query_adata.obsm["X_lsi"] = np.asarray(coords, dtype=np.float64)
    return query_adata


def reduce_dimension_transform(
    query_adata: ad.AnnData,
    ref_adata: ad.AnnData,
    reduction_method: str = "UMAP",
    preprocess_method: str | None = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Project query cells onto the reference UMAP or tSNE embedding.

    Assumes the query has already been projected via
    :func:`preprocess_transform` (so ``query_adata.obsm["X_<preprocess_method>"]``
    exists and is in the reference feature space). Looks up the fitted
    ``umap.UMAP`` / ``openTSNE.TSNEEmbedding`` on
    ``ref_adata.uns["monocle3"]["reduce_dim"][reduction_method]["model"]``
    and calls its ``.transform()`` method.

    Stores the projection in ``query_adata.obsm["X_<reduction>"]``.
    """
    del verbose
    if reduction_method not in {"UMAP", "tSNE"}:
        raise ValueError("reduction_method must be 'UMAP' or 'tSNE'")

    reduce_dim_meta = get_monocle_uns(ref_adata, "reduce_dim", reduction_method,
                                      default={})
    if preprocess_method is None:
        preprocess_method = reduce_dim_meta.get("preprocess_method", "PCA")
    if preprocess_method not in {"PCA", "LSI", "Aligned"}:
        raise ValueError(
            "preprocess_method must be 'PCA', 'LSI', or 'Aligned'"
        )

    src_key = f"X_{preprocess_method.lower()}"
    if src_key not in query_adata.obsm:
        raise KeyError(
            f"reduce_dimension_transform: query missing {src_key!r}. "
            f"Call preprocess_transform(method={preprocess_method!r}) first."
        )
    X = np.asarray(query_adata.obsm[src_key], dtype=np.float64)

    np.random.seed(2016)

    if reduction_method == "UMAP":
        umap_model = reduce_dim_meta.get("model")
        if umap_model is None:
            raise KeyError(
                "reduce_dimension_transform: reference has no fitted UMAP "
                "model. Run reduce_dimension(reduction_method='UMAP') first."
            )
        embedding = umap_model.transform(X)
        query_adata.obsm["X_umap"] = np.asarray(embedding, dtype=np.float64)
    else:
        # tSNE — openTSNE's TSNEEmbedding object has a `.transform()` method.
        tsne_embedding = reduce_dim_meta.get("model")
        if tsne_embedding is None:
            raise KeyError(
                "reduce_dimension_transform: reference has no fitted tSNE "
                "embedding. Run reduce_dimension(reduction_method='tSNE') first."
            )
        new_emb = tsne_embedding.transform(X)
        query_adata.obsm["X_tsne"] = np.asarray(new_emb, dtype=np.float64)

    return query_adata
