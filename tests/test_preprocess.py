"""Slice 1: preprocessing core tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from monocle3 import (
    detect_genes,
    estimate_size_factors,
    new_cell_data_set,
    normalized_counts,
    preprocess_cds,
    size_factors,
)


def test_new_cell_data_set_from_sparse(synthetic_adata):
    # Reconstruct R-convention input: genes × cells.
    X_gc = synthetic_adata.X.T.tocsc()
    cell_md = synthetic_adata.obs.copy()
    gene_md = synthetic_adata.var.copy()

    cds = new_cell_data_set(X_gc, cell_metadata=cell_md, gene_metadata=gene_md)
    assert cds.shape == synthetic_adata.shape
    assert "Size_Factor" in cds.obs.columns
    np.testing.assert_allclose(cds.X.sum(), X_gc.sum(), rtol=1e-9)


def test_new_cell_data_set_dense_warns_without_gene_short_name():
    X = sp.csr_matrix(np.array([[1, 0], [0, 3], [2, 1]], dtype=float))  # 3 genes × 2 cells
    gene_md = pd.DataFrame(
        {"id": ["g0", "g1", "g2"]}, index=["g0", "g1", "g2"]
    )
    with pytest.warns(UserWarning, match="gene_short_name"):
        new_cell_data_set(X, gene_metadata=gene_md)


def test_estimate_size_factors_matches_geometric_mean(synthetic_adata):
    estimate_size_factors(synthetic_adata)
    cell_total = np.asarray(synthetic_adata.X.sum(axis=1)).ravel()
    expected = cell_total / np.exp(np.mean(np.log(cell_total)))
    np.testing.assert_allclose(
        synthetic_adata.obs["Size_Factor"].to_numpy(), expected
    )


def test_detect_genes_counts(synthetic_adata):
    detect_genes(synthetic_adata, min_expr=0)
    assert (synthetic_adata.var["num_cells_expressed"] >= 0).all()
    # A highly expressed gene in our fixture should be in many cells.
    assert synthetic_adata.var["num_cells_expressed"].max() > 10


def test_normalized_counts_log(synthetic_adata):
    nc = normalized_counts(synthetic_adata, norm_method="log")
    assert sp.issparse(nc)
    assert nc.shape == synthetic_adata.shape
    # log10(x + 1) should be non-negative.
    assert nc.data.min() >= 0.0


def test_preprocess_cds_pca_shape(synthetic_adata):
    preprocess_cds(synthetic_adata, num_dim=10)
    assert "X_pca" in synthetic_adata.obsm
    assert synthetic_adata.obsm["X_pca"].shape == (synthetic_adata.n_obs, 10)
    model = synthetic_adata.uns["monocle3"]["preprocess"]["PCA"]
    assert model["num_dim"] == 10
    assert model["svd_sdev"].shape == (10,)
    assert "prop_var_expl" in model


def test_preprocess_cds_requires_size_factor():
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    import anndata as ad

    adata = ad.AnnData(X=X)
    with pytest.raises(ValueError, match="estimate_size_factors"):
        preprocess_cds(adata, num_dim=1)


def test_size_factors_accessor(synthetic_adata):
    import pandas as pd

    estimate_size_factors(synthetic_adata)
    sf = size_factors(synthetic_adata)
    # R's ``size_factors`` returns a named numeric vector
    # (``methods-cell_data_set.R:36-40``); Python returns a pandas Series
    # indexed by cell name so R's ``sf["AAACGG..."]`` idiom still works.
    assert isinstance(sf, pd.Series)
    assert sf.shape == (synthetic_adata.n_obs,)
    assert list(sf.index) == list(synthetic_adata.obs_names)
    assert (sf > 0).all()
    # Named lookup must resolve.
    first_cell = synthetic_adata.obs_names[0]
    assert float(sf.loc[first_cell]) == float(
        synthetic_adata.obs.loc[first_cell, "Size_Factor"]
    )


def test_preprocess_cds_lsi_shape(synthetic_adata):
    preprocess_cds(synthetic_adata, method="LSI", num_dim=8)
    assert "X_lsi" in synthetic_adata.obsm
    assert synthetic_adata.obsm["X_lsi"].shape == (synthetic_adata.n_obs, 8)
