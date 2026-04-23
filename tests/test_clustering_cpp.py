"""Unit tests for the Python re-host of src/clustering.cpp."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from monocle3._clustering_cpp import jaccard_coeff, pnorm_over_mat


def test_jaccard_coeff_weight_false_is_all_ones():
    # 4 cells, k=3 neighbours each (1-based R indices).
    idx = np.array(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 1, 4],
            [4, 2, 3],
        ],
        dtype=np.int64,
    )
    out = jaccard_coeff(idx, weight=False)
    assert out.shape == (12, 3)
    assert np.all(out[:, 2] == 1.0)
    # First two columns should be (i+1) repeats and idx raveled.
    assert np.array_equal(out[:, 0], np.repeat(np.arange(1, 5), 3))
    assert np.array_equal(out[:, 1], idx.ravel())


def test_jaccard_coeff_weight_true_normalized():
    idx = np.array(
        [
            [1, 2, 3],
            [2, 1, 3],
            [3, 1, 2],
            [4, 1, 2],
        ],
        dtype=np.int64,
    )
    out = jaccard_coeff(idx, weight=True)
    # All neighbours overlap — max Jaccard should be 1 after normalization.
    assert out.shape == (12, 3)
    assert out[:, 2].max() == 1.0
    # Weight is non-negative.
    assert out[:, 2].min() >= 0.0


def test_pnorm_over_mat_matches_scipy():
    rng = np.random.default_rng(0)
    num = rng.uniform(-1, 3, size=(5, 5))
    var = rng.uniform(0.5, 2.0, size=(5, 5))
    out = pnorm_over_mat(num, var)
    expected = norm.sf(num / np.sqrt(var))
    np.testing.assert_allclose(out, expected)


def test_pnorm_over_mat_zero_variance():
    num = np.array([[1.0, -1.0], [0.0, 2.0]])
    var = np.zeros((2, 2))
    out = pnorm_over_mat(num, var)
    # R::pnorm(x, sd=0) -> 0 when x > 0, else 1.
    assert out[0, 0] == 0.0
    assert out[0, 1] == 1.0
    assert out[1, 0] == 1.0
    assert out[1, 1] == 0.0
