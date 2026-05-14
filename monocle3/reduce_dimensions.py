"""Low-dimensional projection of cells.

UMAP delegated to umap-learn, t-SNE to openTSNE; PCA / LSI / Aligned are
passthroughs that expose existing ``obsm`` coords under a new key.

On successful reduction the function clears stale principal-graph slots
for the same reduction so downstream learn_graph / order_cells do not
see stale coordinates.
"""

from __future__ import annotations

import warnings
from typing import Any

import anndata as ad
import numpy as np
import openTSNE
import umap as umap_module

from ._utils import ensure_monocle_uns, get_monocle_uns

__all__ = ["reduce_dimension"]


# UMAP-metric → annoy-metric mapping. R uwot uses annoy "angular" for
# cosine/correlation and the same name otherwise.
_ANNOY_METRIC_MAP = {
    "cosine": "angular",
    "correlation": "angular",
    "euclidean": "euclidean",
    "manhattan": "manhattan",
    "hamming": "hamming",
}


def _build_annoy_precomputed_knn(
    X: np.ndarray,
    n_neighbors: int,
    metric: str,
    seed: int,
    n_trees: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a kNN graph with uwot's exact ``nn_method='annoy'`` parameters.

    Uwot defaults (uwot::umap formals): ``n_trees=50`` and
    ``search_k = 2 * n_neighbors * n_trees``. Cosine becomes annoy
    'angular', whose distance is ``sqrt(2*(1-cos))``; we square-and-halve
    so the values match umap-learn's expected ``metric='cosine'``
    distance (``1 - cos``).

    Returns
    -------
    (indices, distances)
        Both shape ``(N, n_neighbors + 1)`` with self at column 0,
        suitable for ``umap.UMAP(precomputed_knn=...)``.
    """
    from annoy import AnnoyIndex  # lazy: keep annoy optional

    if metric not in _ANNOY_METRIC_MAP:
        raise ValueError(
            f"umap_nn_method='annoy' does not support metric={metric!r}; "
            f"supported: {sorted(_ANNOY_METRIC_MAP)}. "
            f"Pass umap_nn_method='default' to fall back to umap-learn's kNN."
        )
    am = _ANNOY_METRIC_MAP[metric]
    n, d = X.shape

    idx = AnnoyIndex(d, am)
    idx.set_seed(int(seed))
    for i in range(n):
        idx.add_item(i, X[i].tolist())
    idx.build(n_trees)

    search_k = 2 * int(n_neighbors) * int(n_trees)
    k_plus = int(n_neighbors) + 1
    knn_idx = np.zeros((n, k_plus), dtype=np.int64)
    knn_dist = np.zeros((n, k_plus), dtype=np.float64)
    for i in range(n):
        nn_i, nn_d = idx.get_nns_by_item(
            i, k_plus, search_k=search_k, include_distances=True,
        )
        knn_idx[i] = nn_i
        knn_dist[i] = nn_d

    if am == "angular":
        # annoy angular distance = sqrt(2*(1-cos)); umap-learn metric='cosine'
        # expects (1 - cos).
        knn_dist = (knn_dist ** 2) / 2.0
    return knn_idx, knn_dist


def reduce_dimension(
    adata: ad.AnnData,
    max_components: int = 2,
    reduction_method: str = "UMAP",
    preprocess_method: str | None = None,
    umap_metric: str = "cosine",
    umap_min_dist: float = 0.1,
    umap_n_neighbors: int = 15,
    umap_nn_method: str = "annoy",
    verbose: bool = False,
    cores: int = 1,
    build_nn_index: bool = False,
    nn_control: dict | None = None,
    **kwargs: Any,
) -> ad.AnnData:
    """Low-dimensional projection of cells onto UMAP, t-SNE, PCA, LSI, or Aligned.

    Parameters
    ----------
    adata : anndata.AnnData
        Cells × genes.
    max_components : int, default 2
        Output dimensionality.
    reduction_method : {"UMAP", "tSNE", "PCA", "LSI", "Aligned"}
        Algorithm. ``PCA``, ``LSI``, and ``Aligned`` are passthroughs —
        they simply expose an existing ``obsm`` matrix under a new key.
    preprocess_method : {"PCA", "LSI", "Aligned", None}, optional
        Source coordinates for UMAP / t-SNE. If ``None``, prefer
        ``Aligned`` if present, otherwise ``PCA``.
    umap_metric, umap_min_dist, umap_n_neighbors
        Mapped to ``umap.UMAP`` arguments. ``umap.fast_sgd`` is a
        ``uwot``-specific extension that ``umap-learn`` does not expose
        and is not available here.
    umap_nn_method : {"annoy", "default"}, default "annoy"
        kNN backend for UMAP. ``"annoy"`` (the default, matching R uwot's
        ``nn_method="annoy"``) precomputes the kNN graph with the real
        ``annoy`` library using uwot's exact parameters
        (``n_trees=50``, ``search_k = 2 * n_neighbors * n_trees``) and
        feeds it to umap-learn via ``precomputed_knn``. ``"default"``
        delegates to umap-learn's internal kNN (pynndescent for n>4096,
        sklearn balltree otherwise) — historical behaviour.
    verbose : bool, default False
        Forwarded to ``umap.UMAP`` / ``openTSNE``.
    cores : int, default 1
        Number of threads (UMAP only).
    build_nn_index : bool, default False
        When ``True``, also build a nearest-neighbour index from the
        reduced coords and store it in uns via ``set_cds_nn_index``.
    nn_control : dict, optional
        Parameters for ``make_nn_index``.
    **kwargs
        Passed through to the backend algorithm.

    Returns
    -------
    anndata.AnnData
        The same object, with new ``adata.obsm[f"X_{reduction_method.lower()}"]``.
    """
    if reduction_method not in {"UMAP", "tSNE", "PCA", "LSI", "Aligned"}:
        raise ValueError(
            "reduction_method must be one of 'UMAP', 'tSNE', 'PCA', 'LSI', 'Aligned'"
        )
    if not isinstance(max_components, (int, np.integer)) or max_components < 1:
        raise ValueError("max_components must be a positive integer")

    if preprocess_method is None:
        if "X_aligned" in adata.obsm:
            preprocess_method = "Aligned"
            warnings.warn(
                "No preprocess_method specified, and aligned coordinates "
                "have been computed previously. Using preprocess_method = 'Aligned'",
                stacklevel=2,
            )
        else:
            preprocess_method = "PCA"

    if preprocess_method not in {"PCA", "LSI", "Aligned"}:
        raise ValueError(
            "preprocess_method must be 'PCA', 'LSI', or 'Aligned'"
        )

    preprocess_key = f"X_{preprocess_method.lower()}"
    if preprocess_key not in adata.obsm:
        raise KeyError(
            f"Preprocessed '{preprocess_method}' matrix not found. "
            f"Run preprocess_cds / align_cds first."
        )

    np.random.seed(2016)

    preprocess_mat = np.asarray(adata.obsm[preprocess_key], dtype=np.float64)
    out_key = f"X_{reduction_method.lower()}"
    uns = ensure_monocle_uns(adata)
    uns.setdefault("reduce_dim", {})

    if reduction_method == "PCA":
        if preprocess_method != "PCA":
            raise ValueError(
                "preprocess_method must be 'PCA' when reduction_method='PCA'"
            )
        # Passthrough — R behaviour.
        adata.obsm["X_pca"] = preprocess_mat.copy()
    elif reduction_method == "LSI":
        if preprocess_method != "LSI":
            raise ValueError(
                "preprocess_method must be 'LSI' when reduction_method='LSI'"
            )
        adata.obsm["X_lsi"] = preprocess_mat.copy()
    elif reduction_method == "Aligned":
        if preprocess_method != "Aligned":
            raise ValueError(
                "preprocess_method must be 'Aligned' when reduction_method='Aligned'"
            )
        adata.obsm["X_aligned"] = preprocess_mat.copy()
    elif reduction_method == "tSNE":
        # Pass perplexity through unchanged. openTSNE will raise on invalid
        # values; we emit a pre-emptive warning so the cause is obvious.
        n_samples = preprocess_mat.shape[0]
        perp = float(kwargs.pop("perplexity", 30))
        max_perp = max(5, (n_samples - 1) // 3)
        if perp > max_perp:
            warnings.warn(
                f"perplexity={perp} exceeds openTSNE's sanity bound "
                f"(n_samples-1)/3 = {max_perp}; openTSNE will raise. "
                "Reduce perplexity or provide more cells.",
                stacklevel=2,
            )
        tsne = openTSNE.TSNE(
            n_components=int(max_components),
            perplexity=perp,
            random_state=2016,
            n_jobs=int(cores),
            verbose=bool(verbose),
            **kwargs,
        )
        embedding = tsne.fit(preprocess_mat)
        adata.obsm["X_tsne"] = np.asarray(embedding, dtype=np.float64)
        # Persist the TSNEEmbedding object so reduce_dimension_transform can
        # call `embedding.transform(X_new)` to project new cells later.
        uns["reduce_dim"]["tSNE"] = {
            "preprocess_method": preprocess_method,
            "max_components": int(max_components),
            "perplexity": perp,
            "model": embedding,
        }
    else:  # UMAP
        if cores > 1:
            warnings.warn(
                "reduce_dimension may produce slightly different output each "
                "time unless cores=1 (umap-learn's parallel SGD is non-deterministic).",
                stacklevel=2,
            )

        n_neighbors = int(umap_n_neighbors)

        umap_kwargs: dict[str, Any] = dict(
            n_components=int(max_components),
            n_neighbors=n_neighbors,
            min_dist=float(umap_min_dist),
            metric=str(umap_metric),
            random_state=2016,
            transform_seed=2016,
            n_jobs=int(cores),
            verbose=bool(verbose),
        )
        if umap_nn_method == "annoy":
            knn_idx, knn_dist = _build_annoy_precomputed_knn(
                preprocess_mat,
                n_neighbors=n_neighbors,
                metric=str(umap_metric),
                seed=2016,
            )
            umap_kwargs["precomputed_knn"] = (knn_idx, knn_dist, None)
        elif umap_nn_method != "default":
            raise ValueError(
                f"umap_nn_method must be 'annoy' or 'default', got {umap_nn_method!r}"
            )
        umap_kwargs.update(kwargs)

        umap_model = umap_module.UMAP(**umap_kwargs)
        # fit_transform is deterministic in umap-learn >= 0.5 with
        # n_jobs=1 and random_state set; no manual re-seeding needed.
        embedding = umap_model.fit_transform(preprocess_mat)
        adata.obsm["X_umap"] = np.asarray(embedding, dtype=np.float64)
        uns["reduce_dim"]["UMAP"] = {
            "preprocess_method": preprocess_method,
            "max_components": int(max_components),
            "umap_metric": umap_metric,
            "umap_min_dist": float(umap_min_dist),
            "umap_n_neighbors": n_neighbors,
            "umap_nn_method": umap_nn_method,
            "model": umap_model,
        }

    # Clear stale principal graph entries for this reduction.
    for slot in ("principal_graph", "principal_graph_aux", "clusters"):
        if slot in uns and reduction_method in uns[slot]:
            del uns[slot][reduction_method]

    if build_nn_index and reduction_method in {"PCA", "LSI", "Aligned", "UMAP", "tSNE"}:
        from .nearest_neighbors import make_nn_index, set_cds_nn_index

        nn_index = make_nn_index(
            adata.obsm[out_key], nn_control=nn_control
        )
        set_cds_nn_index(adata, reduction_method=reduction_method, nn_index=nn_index)

    return adata
