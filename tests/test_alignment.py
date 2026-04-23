"""Slice 2: alignment tests."""

from __future__ import annotations

import numpy as np

from monocle3 import align_cds, preprocess_cds


def test_align_cds_residual_reduces_covariate_variance(synthetic_adata):
    preprocess_cds(synthetic_adata, num_dim=10)
    # Inject a synthetic continuous covariate that biases PC1.
    synthetic_adata.obs["cov"] = synthetic_adata.obsm["X_pca"][:, 0] * 1.0

    before = synthetic_adata.obsm["X_pca"][:, 0].copy()
    align_cds(synthetic_adata, residual_model_formula_str="~ cov")

    aligned = synthetic_adata.obsm["X_aligned"][:, 0]
    assert aligned.shape == before.shape
    # PC1 is exactly the covariate → residual on PC1 should be ~0.
    np.testing.assert_allclose(aligned, 0.0, atol=1e-10)
    # Fitted beta stored in uns.
    assert synthetic_adata.uns["monocle3"]["Aligned"]["beta"].shape == (1, 10)


def test_align_cds_batch_path(synthetic_adata):
    preprocess_cds(synthetic_adata, num_dim=10)
    align_cds(synthetic_adata, alignment_group="batch", alignment_k=10)
    assert synthetic_adata.obsm["X_aligned"].shape == (
        synthetic_adata.n_obs,
        10,
    )
