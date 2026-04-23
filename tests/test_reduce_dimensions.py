"""Slice 2: reduce_dimension tests."""

from __future__ import annotations

import pytest

from monocle3 import preprocess_cds, reduce_dimension


def test_reduce_dimension_umap_shape(synthetic_adata):
    preprocess_cds(synthetic_adata, num_dim=10)
    reduce_dimension(synthetic_adata, max_components=2)
    assert synthetic_adata.obsm["X_umap"].shape == (synthetic_adata.n_obs, 2)


def test_reduce_dimension_requires_preprocess(tiny_adata):
    with pytest.raises(KeyError, match="Preprocessed"):
        reduce_dimension(tiny_adata, reduction_method="UMAP")


def test_reduce_dimension_pca_passthrough(synthetic_adata):
    preprocess_cds(synthetic_adata, num_dim=5)
    reduce_dimension(
        synthetic_adata,
        reduction_method="PCA",
        preprocess_method="PCA",
    )
    assert synthetic_adata.obsm["X_pca"].shape == (synthetic_adata.n_obs, 5)
