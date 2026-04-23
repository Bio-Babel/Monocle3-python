"""transfer_cell_labels / fix_missing_cell_labels — kNN-based label
transfer.

Mirrors R ``transfer_cell_labels`` and ``fix_missing_cell_labels``
(``label_transfer.R``) exactly. For each query cell the k nearest
reference cells are found in the chosen reduced-dim space; for discrete
labels a threshold-based majority vote is taken, for continuous values
the mean is transferred.

Assumes the query has been projected into the reference reduced-dim
space via :func:`preprocess_transform` + :func:`reduce_dimension_transform`
— the two embeddings must be directly comparable.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from .nearest_neighbors import make_nn_index, search_nn_index

__all__ = ["transfer_cell_labels", "fix_missing_cell_labels"]


# R ``label_transfer.R::which_mode`` — return the modal label when it
# either dominates the neighbour set (``top_frac_threshold``) or its
# count is ``top_next_ratio_threshold`` × the second most frequent
# neighbour label. Otherwise return NaN / ``None``.
def _which_mode(
    labels: Sequence[Any],
    top_frac_threshold: float = 0.5,
    top_next_ratio_threshold: float = 1.5,
) -> Any:
    counts = Counter(labels)
    if not counts:
        return None
    sorted_counts = counts.most_common()
    top_label, top_count = sorted_counts[0]
    frac = top_count / len(labels)
    if frac > top_frac_threshold:
        return top_label
    if len(sorted_counts) < 2:
        # Only one label present but it didn't clear the fraction bar.
        # R reaches this path via `ta[2] = NA`; the ratio is `Inf` so the
        # label *is* returned.
        return top_label
    second_count = sorted_counts[1][1]
    if second_count == 0:
        return top_label
    if (top_count / second_count) >= top_next_ratio_threshold:
        return top_label
    return None


def _resolve_default_metric(reduction_method: str) -> str:
    """Match R's ``nn_control`` defaults: UMAP/tSNE → euclidean,
    PCA/LSI/Aligned → cosine."""
    if reduction_method in {"UMAP", "tSNE"}:
        return "euclidean"
    return "cosine"


def _build_or_reuse_ref_index(
    ref_adata: ad.AnnData,
    reduction_method: str,
    nn_control: dict | None,
) -> tuple[dict, np.ndarray]:
    """Build (or reuse) a nearest-neighbour index on the reference's
    reduced-dim coords, and return ``(index, ref_coords)``."""
    key = f"X_{reduction_method.lower()}"
    if key not in ref_adata.obsm:
        raise KeyError(
            f"transfer_cell_labels: reference has no {reduction_method} "
            f"reduction."
        )
    ref_coords = np.asarray(ref_adata.obsm[key], dtype=np.float64)

    uns_idx = ref_adata.uns.get("monocle3", {}).get("nn_index", {}).get(
        reduction_method
    )
    if uns_idx is not None:
        return uns_idx, ref_coords

    metric = _resolve_default_metric(reduction_method)
    ctrl = {"method": "annoy", "metric": metric}
    if nn_control:
        ctrl.update(nn_control)
    nn_index = make_nn_index(ref_coords, nn_control=ctrl)
    return nn_index, ref_coords


def transfer_cell_labels(
    query_adata: ad.AnnData,
    ref_adata: ad.AnnData,
    ref_column_name: str,
    reduction_method: str = "UMAP",
    query_column_name: str | None = None,
    k: int = 10,
    top_frac_threshold: float = 0.5,
    top_next_ratio_threshold: float = 1.5,
    nn_control: dict | None = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Copy a reference ``obs`` column to the query by kNN majority vote.

    Parameters
    ----------
    query_adata : anndata.AnnData
        Must have ``obsm[f"X_{reduction_method.lower()}"]`` populated by
        :func:`reduce_dimension_transform` (or by training in the same
        reference space).
    ref_adata : anndata.AnnData
        Reference with the column-to-transfer in its ``obs`` and a
        matching reduced-dim embedding.
    ref_column_name : str
        Column to transfer. When values are numeric (``float``-like) the
        k-neighbour mean is used; otherwise the threshold-based mode.
    reduction_method : {"UMAP", "PCA", "LSI", "Aligned", "tSNE"}
        Which stored embedding to search.
    query_column_name : str, optional
        Destination column in ``query_adata.obs``. Defaults to
        ``ref_column_name``.
    k : int, default 10
    top_frac_threshold / top_next_ratio_threshold : float
        Majority-vote thresholds from ``label_transfer.R::which_mode``.
    nn_control : dict, optional
        Override the default kNN backend (annoy + euclidean/cosine).
    verbose : bool, default False
    """
    del verbose
    if query_column_name is None:
        query_column_name = ref_column_name
    if ref_column_name not in ref_adata.obs.columns:
        raise KeyError(
            f"transfer_cell_labels: ref_column_name {ref_column_name!r} "
            f"missing from ref_adata.obs."
        )
    if reduction_method not in {"UMAP", "PCA", "LSI", "Aligned", "tSNE"}:
        raise ValueError(
            "reduction_method must be one of 'UMAP', 'PCA', 'LSI', "
            "'Aligned', 'tSNE'"
        )

    qkey = f"X_{reduction_method.lower()}"
    if qkey not in query_adata.obsm:
        raise KeyError(
            f"transfer_cell_labels: query has no {reduction_method} "
            f"reduction. Project the query first with "
            f"preprocess_transform + reduce_dimension_transform."
        )
    query_coords = np.asarray(query_adata.obsm[qkey], dtype=np.float64)

    nn_index, _ref_coords = _build_or_reuse_ref_index(
        ref_adata, reduction_method, nn_control
    )
    merged_ctrl = {"method": "annoy",
                   "metric": _resolve_default_metric(reduction_method)}
    if nn_control:
        merged_ctrl.update(nn_control)
    nn_res = search_nn_index(
        query_coords, nn_index=nn_index, k=int(k), nn_control=merged_ctrl
    )
    # nn.idx is 1-based; convert to 0-based to index ref.obs.
    neighbor_idx = nn_res["nn.idx"].astype(np.int64) - 1

    ref_labels = ref_adata.obs[ref_column_name]
    is_numeric = pd.api.types.is_numeric_dtype(ref_labels)

    labels_arr = ref_labels.to_numpy()
    if is_numeric:
        # Mean of k neighbours, skipping NaN (matches R `mean(x, na.rm=TRUE)`).
        gathered = labels_arr[neighbor_idx]
        with np.errstate(invalid="ignore"):
            transferred = np.nanmean(gathered.astype(float), axis=1)
    else:
        transferred = np.empty(query_coords.shape[0], dtype=object)
        for i, row in enumerate(neighbor_idx):
            labels = labels_arr[row].tolist()
            transferred[i] = _which_mode(
                labels,
                top_frac_threshold=float(top_frac_threshold),
                top_next_ratio_threshold=float(top_next_ratio_threshold),
            )

    query_adata.obs[query_column_name] = pd.Series(
        transferred, index=query_adata.obs_names
    )
    return query_adata


def fix_missing_cell_labels(
    query_adata: ad.AnnData,
    reduction_method: str = "UMAP",
    from_column_name: str | None = None,
    to_column_name: str | None = None,
    k: int = 10,
    top_frac_threshold: float = 0.5,
    top_next_ratio_threshold: float = 1.5,
    nn_control: dict | None = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Fill in NaN / None labels in ``query_adata.obs[from_column_name]``
    using intra-query kNN majority vote.

    Mirrors R ``fix_missing_cell_labels`` — partition the cells into a
    not-null set (used to build a local kNN index) and a null set (whose
    labels are replaced via :func:`_which_mode`).
    """
    del verbose
    if from_column_name is None:
        raise ValueError("from_column_name is required")
    if to_column_name is None:
        to_column_name = from_column_name
    if from_column_name not in query_adata.obs.columns:
        raise KeyError(
            f"fix_missing_cell_labels: from_column_name {from_column_name!r} "
            f"missing."
        )

    col = query_adata.obs[from_column_name]
    is_na = col.isna().to_numpy()
    if not is_na.any():
        if to_column_name != from_column_name:
            query_adata.obs[to_column_name] = col.copy()
        return query_adata

    qkey = f"X_{reduction_method.lower()}"
    if qkey not in query_adata.obsm:
        raise KeyError(
            f"fix_missing_cell_labels: query has no {reduction_method} "
            f"reduction."
        )
    coords = np.asarray(query_adata.obsm[qkey], dtype=np.float64)

    notna_coords = coords[~is_na]
    notna_labels = col[~is_na].to_numpy()
    na_coords = coords[is_na]

    metric = _resolve_default_metric(reduction_method)
    merged_ctrl = {"method": "annoy", "metric": metric}
    if nn_control:
        merged_ctrl.update(nn_control)
    index = make_nn_index(notna_coords, nn_control=merged_ctrl)
    # R requests k+1 then discards the self neighbour (for the query==ref
    # case). Here query cells are a strict subset *disjoint* from the
    # notna set, so self is not present — still request exactly k.
    k_eff = min(int(k), notna_coords.shape[0])
    nn_res = search_nn_index(
        na_coords, nn_index=index, k=k_eff, nn_control=merged_ctrl
    )
    neighbour_idx = nn_res["nn.idx"].astype(np.int64) - 1

    replacements = np.empty(na_coords.shape[0], dtype=object)
    for i, row in enumerate(neighbour_idx):
        labels = notna_labels[row].tolist()
        replacements[i] = _which_mode(
            labels,
            top_frac_threshold=float(top_frac_threshold),
            top_next_ratio_threshold=float(top_next_ratio_threshold),
        )

    out = col.copy()
    if to_column_name != from_column_name:
        out.name = to_column_name
    out = out.astype(object)
    out_arr = out.to_numpy(copy=True)
    out_arr[is_na] = replacements
    query_adata.obs[to_column_name] = pd.Series(
        out_arr, index=query_adata.obs_names
    )
    return query_adata
