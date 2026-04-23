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
