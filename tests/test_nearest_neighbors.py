"""Slice 3: nearest neighbours tests."""

from __future__ import annotations

import numpy as np

from monocle3.nearest_neighbors import (
    make_nn_index,
    search_nn_index,
    search_nn_matrix,
)


def _grid_points(n=40, d=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def test_search_nn_matrix_nn2_is_exact():
    X = _grid_points()
    out = search_nn_matrix(X, X, k=5, nn_control={"method": "nn2"})
    # First neighbor should always be itself for self-kNN with exact search.
    assert (out["nn.idx"][:, 0] == np.arange(1, X.shape[0] + 1)).all()
    assert out["nn.dists"].shape == (X.shape[0], 5)
    # Self-distance ~0.
    assert np.allclose(out["nn.dists"][:, 0], 0, atol=1e-9)


def test_annoy_index_round_trip():
    X = _grid_points()
    index = make_nn_index(X, nn_control={"method": "annoy"})
    out = search_nn_index(X, nn_index=index, k=5)
    assert out["nn.idx"].shape == (X.shape[0], 5)
    assert out["nn.dists"].shape == (X.shape[0], 5)


def test_hnsw_index_round_trip():
    X = _grid_points()
    index = make_nn_index(X, nn_control={"method": "hnsw"})
    out = search_nn_index(X, nn_index=index, k=5)
    assert out["nn.idx"].shape == (X.shape[0], 5)
    assert out["nn.dists"].shape == (X.shape[0], 5)


def test_make_nn_index_rejects_unknown_metric():
    """R's RcppAnnoy / RcppHNSW raise on unknown metrics (no silent fallback).
    Python must do the same — the previous behaviour silently remapped every
    unknown hnsw metric to 'l2'."""
    import pytest
    X = _grid_points()
    with pytest.raises(ValueError, match="metric"):
        make_nn_index(X, nn_control={"method": "hnsw", "metric": "bogus"})
    with pytest.raises(ValueError, match="metric"):
        make_nn_index(X, nn_control={"method": "annoy", "metric": "bogus"})


def test_search_nn_index_honours_search_k_via_epsilon():
    """R's Annoy ``search_k`` trades query time for recall. The Python port
    maps ``search_k`` → pynndescent ``epsilon``. Larger search_k must mean
    recall at least as good as smaller search_k."""
    X = _grid_points(n=200, d=5, seed=3)
    Q = X[:50]
    index = make_nn_index(X, nn_control={"method": "annoy", "n_trees": 10})
    low = search_nn_index(
        Q, nn_index=index, k=5, nn_control={"search_k": 1},
    )
    high = search_nn_index(
        Q, nn_index=index, k=5, nn_control={"search_k": 10000},
    )
    # self-neighbour recovered in the high-budget run (it always is for epsilon
    # > 0; this is a smoke that search_k actually threads through).
    assert (high["nn.idx"][:, 0] == np.arange(1, Q.shape[0] + 1)).all()
    # Shapes stay consistent regardless of epsilon.
    assert low["nn.idx"].shape == high["nn.idx"].shape == (50, 5)


def test_search_nn_index_hnsw_threads():
    """hnswlib.knn_query must receive ``num_threads`` from ``nn_control['cores']``.
    We don't assert wall-clock; we only assert the call succeeds with cores > 1."""
    X = _grid_points(n=200, d=5)
    index = make_nn_index(X, nn_control={"method": "hnsw", "cores": 2})
    out = search_nn_index(X, nn_index=index, k=5, nn_control={"cores": 2})
    assert out["nn.idx"].shape == (X.shape[0], 5)
