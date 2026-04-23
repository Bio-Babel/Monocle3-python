"""Synthetic AnnData fixtures shared across tests.

Design constraints:

- Deterministic — fixed seed.
- Small enough to run every test in < 1s.
- Rich enough to exercise PCA, UMAP, and Leiden (at least two latent
  clusters plus a batch covariate).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp

__all__ = ["make_synthetic_adata", "make_tiny_adata"]


def _sample_negbin(
    rng: np.random.Generator, mean: np.ndarray, size: float = 1.0
) -> np.ndarray:
    # scipy's NB parameterization: p = size / (size + mean)
    p = size / (size + mean)
    return rng.negative_binomial(size, p)


def make_synthetic_adata(
    n_cells: int = 120,
    n_genes: int = 80,
    seed: int = 0,
) -> ad.AnnData:
    """A (cells × genes) AnnData with two planted clusters and a batch."""
    rng = np.random.default_rng(seed)

    half = n_cells // 2
    # Two cluster means; first cluster enriches first 20 genes, second
    # cluster enriches next 20 genes.
    mean = np.ones((n_cells, n_genes), dtype=float) * 0.2
    mean[:half, :20] = 4.0
    mean[half:, 20:40] = 4.0

    # Batch effect on another 10 genes for half the cells of each cluster.
    batch = np.zeros(n_cells, dtype=int)
    batch[::2] = 1
    mean[batch == 1, 40:50] *= 2.5

    counts = _sample_negbin(rng, mean, size=1.0).astype(np.int64)
    X = sp.csr_matrix(counts)

    # Pre-compute size factors so tests can skip calling estimate_size_factors
    # when they only exercise a downstream function.
    cell_total = np.asarray(counts.sum(axis=1), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        size_factor = cell_total / np.exp(np.mean(np.log(cell_total)))
    size_factor = np.where(np.isfinite(size_factor), size_factor, 1.0)

    barcodes = [f"cell_{i}" for i in range(n_cells)]
    gene_ids = [f"g_{j}" for j in range(n_genes)]
    obs = pd.DataFrame(
        {
            "cluster_truth": ["A"] * half + ["B"] * (n_cells - half),
            "batch": batch.astype(str),
            "n_umi": counts.sum(axis=1),
            "Size_Factor": size_factor,
        },
        index=barcodes,
    )
    var = pd.DataFrame(
        {
            "gene_short_name": gene_ids,
            "num_cells_expressed": np.asarray((counts > 0).sum(axis=0)).ravel(),
        },
        index=gene_ids,
    )
    obs.index.name = "barcode"
    var.index.name = "feature_id"
    return ad.AnnData(X=X, obs=obs, var=var)


def make_tiny_adata(seed: int = 0) -> ad.AnnData:
    """Very small fixture (30 × 20) for quick round-trip tests."""
    return make_synthetic_adata(n_cells=30, n_genes=20, seed=seed)
