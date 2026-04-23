"""Nearest-neighbour indices — port of R/nearest_neighbors.R.

The R implementation dispatches on ``nn_control[['method']]``:

- ``nn2`` → ``RANN::nn2`` (exact)           → ``sklearn.neighbors.NearestNeighbors``
- ``annoy`` → ``RcppAnnoy``                  → ``pynndescent`` (approximate)
- ``hnsw`` → ``RcppHNSW``                    → ``hnswlib``

Return shape mirrors R's ``nn2``-style list ``{'nn.idx', 'nn.dists'}`` with
1-based integer indices so downstream callers (clustering, graph_test)
don't have to translate between conventions.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np

from ._utils import ensure_monocle_uns

__all__ = [
    "make_nn_index",
    "search_nn_index",
    "search_nn_matrix",
    "set_cds_nn_index",
]

_DEFAULT_NN_CONTROL: dict[str, Any] = {
    "method": "annoy",
    "metric": "euclidean",
    "n_trees": 50,
    "M": 48,
    "ef_construction": 200,
    "ef": 150,
    "cores": 1,
    "grain_size": 1,
    "annoy_random_seed": 2016,
}


def _resolve_nn_control(nn_control: dict | None) -> dict:
    merged = dict(_DEFAULT_NN_CONTROL)
    if nn_control:
        merged.update(nn_control)
    return merged


def make_nn_index(
    subject_matrix: Any,
    nn_control: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Build a nearest-neighbour index over *subject_matrix* (rows = items).

    Parameters
    ----------
    subject_matrix : array-like
        ``(n_items, n_features)`` dense matrix.
    nn_control : dict, optional
        Merged with the monocle3 defaults. Required keys if supplied:
        ``method`` (``"annoy"``, ``"hnsw"``, ``"nn2"``), ``metric``
        (``"euclidean"`` / ``"cosine"`` / ``"manhattan"``).
    verbose : bool, default False
        Currently unused (kept for R signature parity).

    Returns
    -------
    dict
        ``{'method', 'metric', 'index', 'nrow', 'ncol'}``. Callers should
        not introspect the ``index`` object — pass it back to
        ``search_nn_index``.
    """
    del verbose
    nn_control = _resolve_nn_control(nn_control)
    method = nn_control["method"]

    X = np.asarray(subject_matrix, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("subject_matrix must be 2D")

    n_items, n_features = X.shape
    metric = nn_control["metric"]

    if method == "nn2":
        raise ValueError(
            "make_nn_index is not valid for method 'nn2'; use search_nn_matrix "
            "with nn_control={'method': 'nn2'}"
        )

    if method == "annoy":
        # pynndescent is the closest Python equivalent of RcppAnnoy for
        # our purposes — both are approximate, tree-based, and fast.
        import pynndescent

        index = pynndescent.NNDescent(
            X,
            metric=metric,
            n_neighbors=min(max(nn_control.get("n_trees", 50), 5), n_items),
            random_state=int(nn_control.get("annoy_random_seed", 2016)),
            n_jobs=int(nn_control.get("cores", 1)),
        )
        index.prepare()
        return {
            "method": "annoy",
            "metric": metric,
            "index": index,
            "nrow": n_items,
            "ncol": n_features,
            "matrix": X,
        }

    if method == "hnsw":
        import hnswlib

        space = {"euclidean": "l2", "cosine": "cosine", "ip": "ip"}.get(
            metric, "l2"
        )
        idx = hnswlib.Index(space=space, dim=n_features)
        idx.init_index(
            max_elements=n_items,
            ef_construction=int(nn_control.get("ef_construction", 200)),
            M=int(nn_control.get("M", 48)),
        )
        idx.add_items(X, np.arange(n_items))
        idx.set_ef(int(nn_control.get("ef", 150)))
        return {
            "method": "hnsw",
            "metric": metric,
            "index": idx,
            "nrow": n_items,
            "ncol": n_features,
            "matrix": X,
        }

    raise ValueError(
        f"make_nn_index: unsupported nearest neighbor index type {method!r}"
    )


def search_nn_index(
    query_matrix: Any,
    nn_index: dict,
    k: int = 25,
    nn_control: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Query an index built by :func:`make_nn_index`.

    Returns
    -------
    dict
        ``{'nn.idx': (n_query, k) int64 1-based indices,
           'nn.dists': (n_query, k) float64 distances}``.
    """
    del verbose
    nn_control = _resolve_nn_control(nn_control)
    Q = np.asarray(query_matrix, dtype=np.float64)
    if Q.ndim != 2:
        raise ValueError("query_matrix must be 2D")

    k = min(int(k), nn_index["nrow"])
    method = nn_index["method"]

    if method == "annoy":
        idx_index = nn_index["index"]
        neighbor_idx, neighbor_dist = idx_index.query(Q, k=k)
        return {
            "nn.idx": (neighbor_idx + 1).astype(np.int64),
            "nn.dists": neighbor_dist.astype(np.float64),
        }
    if method == "hnsw":
        idx_index = nn_index["index"]
        labels, dists = idx_index.knn_query(Q, k=k)
        # hnswlib returns squared L2 for "l2"; take sqrt for Euclidean parity.
        if nn_index["metric"] == "euclidean":
            dists = np.sqrt(dists)
        return {
            "nn.idx": (labels + 1).astype(np.int64),
            "nn.dists": dists.astype(np.float64),
        }

    raise ValueError(f"search_nn_index: unsupported nn_index method {method!r}")


def search_nn_matrix(
    subject_matrix: Any,
    query_matrix: Any,
    k: int = 25,
    nn_control: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Exact or approximate kNN search without a pre-built index.

    For ``method='nn2'`` this uses ``sklearn.neighbors.NearestNeighbors``
    (the R ``RANN::nn2`` call); otherwise it builds an index and searches.
    """
    del verbose
    nn_control = _resolve_nn_control(nn_control)
    method = nn_control["method"]
    k = min(int(k), np.asarray(subject_matrix).shape[0])

    if method == "nn2":
        from sklearn.neighbors import NearestNeighbors

        S = np.asarray(subject_matrix, dtype=np.float64)
        Q = np.asarray(query_matrix, dtype=np.float64)
        nn = NearestNeighbors(
            n_neighbors=k,
            metric=nn_control.get("metric", "euclidean"),
            algorithm="auto",
            n_jobs=int(nn_control.get("cores", 1)),
        ).fit(S)
        dist, idx = nn.kneighbors(Q, n_neighbors=k, return_distance=True)
        return {
            "nn.idx": (idx + 1).astype(np.int64),
            "nn.dists": dist.astype(np.float64),
        }

    nn_index = make_nn_index(subject_matrix, nn_control=nn_control)
    return search_nn_index(query_matrix, nn_index=nn_index, k=k, nn_control=nn_control)


def set_cds_nn_index(
    adata: ad.AnnData,
    reduction_method: str,
    nn_index: dict,
    verbose: bool = False,
) -> ad.AnnData:
    """Attach ``nn_index`` to ``adata.uns["monocle3"]["nn_index"]``.

    Matches R ``set_cds_nn_index``. Indices are stored in-memory only —
    they are not written to the h5ad file.
    """
    del verbose
    valid = {"UMAP", "PCA", "LSI", "Aligned", "tSNE"}
    if reduction_method not in valid:
        raise ValueError(f"reduction_method must be one of {sorted(valid)}")

    key = f"X_{reduction_method.lower()}"
    if key not in adata.obsm:
        raise KeyError(
            f"Reduced matrix for {reduction_method} not found on adata.obsm"
        )

    uns = ensure_monocle_uns(adata)
    uns.setdefault("nn_index", {})
    uns["nn_index"][reduction_method] = nn_index
    return adata
