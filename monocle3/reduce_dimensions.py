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

from ._utils import ensure_monocle_uns, get_monocle_uns

__all__ = ["reduce_dimension"]


def reduce_dimension(
    adata: ad.AnnData,
    max_components: int = 2,
    reduction_method: str = "UMAP",
    preprocess_method: str | None = None,
    umap_metric: str = "cosine",
    umap_min_dist: float = 0.1,
    umap_n_neighbors: int = 15,
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
        Mapped to ``umap.UMAP`` arguments. ``umap.fast_sgd`` and
        ``umap.nn_method`` are ``uwot``-specific extensions that
        ``umap-learn`` does not expose, so they are not available here.
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
        import openTSNE

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
        uns["reduce_dim"]["tSNE"] = {
            "preprocess_method": preprocess_method,
            "max_components": int(max_components),
            "perplexity": perp,
        }
    else:  # UMAP
        import umap as umap_module

        if cores > 1:
            warnings.warn(
                "reduce_dimension may produce slightly different output each "
                "time unless cores=1 (umap-learn's parallel SGD is non-deterministic).",
                stacklevel=2,
            )

        n_neighbors = int(umap_n_neighbors)

        umap_model = umap_module.UMAP(
            n_components=int(max_components),
            n_neighbors=n_neighbors,
            min_dist=float(umap_min_dist),
            metric=str(umap_metric),
            random_state=2016,
            n_jobs=int(cores),
            verbose=bool(verbose),
            **kwargs,
        )
        # Two-step fit + reseed + transform: `.transform()` produces a
        # consistent re-embedding when the RNG is reset, whereas the
        # `.fit_transform()` embedding includes SGD randomness that varies
        # across calls even with the same `random_state`.
        umap_model = umap_model.fit(preprocess_mat)
        np.random.seed(2016)
        embedding = umap_model.transform(preprocess_mat)
        adata.obsm["X_umap"] = np.asarray(embedding, dtype=np.float64)
        uns["reduce_dim"]["UMAP"] = {
            "preprocess_method": preprocess_method,
            "max_components": int(max_components),
            "umap_metric": umap_metric,
            "umap_min_dist": float(umap_min_dist),
            "umap_n_neighbors": n_neighbors,
        }
        uns["UMAP_model"] = umap_model

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
