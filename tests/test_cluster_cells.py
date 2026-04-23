"""Slice 4: cluster_cells tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monocle3 import (
    cluster_cells,
    preprocess_cds,
    reduce_dimension,
)
from monocle3.cluster_cells import clusters, partitions


def test_cluster_cells_leiden_smoke(synthetic_adata):
    preprocess_cds(synthetic_adata, num_dim=10)
    reduce_dimension(synthetic_adata, max_components=2)
    cluster_cells(synthetic_adata, resolution=1e-3, random_seed=1)
    assert "monocle3_clusters" in synthetic_adata.obs.columns
    assert "monocle3_partitions" in synthetic_adata.obs.columns
    c = clusters(synthetic_adata)
    p = partitions(synthetic_adata)
    assert len(c) == synthetic_adata.n_obs
    assert len(p) == synthetic_adata.n_obs
    # Should find at least one cluster.
    assert c.nunique() >= 1


def test_cluster_cells_requires_reduction(tiny_adata):
    import pytest

    with pytest.raises(KeyError, match="Dimensionality reduction"):
        cluster_cells(tiny_adata)


def test_cluster_cells_louvain(synthetic_adata):
    preprocess_cds(synthetic_adata, num_dim=10)
    reduce_dimension(synthetic_adata, max_components=2)
    cluster_cells(
        synthetic_adata, cluster_method="louvain", random_seed=0, weight=True
    )
    assert "monocle3_clusters" in synthetic_adata.obs.columns
