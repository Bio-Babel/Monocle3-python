"""Slice 6: graph_test / cluster_genes smoke tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monocle3 import (
    aggregate_gene_expression,
    cluster_cells,
    find_gene_modules,
    graph_test,
    preprocess_cds,
    reduce_dimension,
)


@pytest.fixture(scope="function")
def prepared(synthetic_adata):
    preprocess_cds(synthetic_adata, num_dim=10)
    reduce_dimension(synthetic_adata, max_components=2)
    cluster_cells(synthetic_adata, resolution=1e-3, random_seed=1)
    return synthetic_adata


def test_graph_test_knn_returns_per_gene_results(prepared):
    df = graph_test(prepared, neighbor_graph="knn", k=10)
    assert "morans_I" in df.columns
    assert "q_value" in df.columns
    assert df.shape[0] == prepared.n_vars
    assert df["q_value"].between(0, 1).all()


def test_find_gene_modules_columns(prepared):
    modules = find_gene_modules(prepared, k=10, resolution=1e-2)
    assert {"id", "module", "supermodule", "dim_1"}.issubset(modules.columns)
    assert modules.shape[0] <= prepared.n_vars


def test_aggregate_gene_expression_shapes(prepared):
    gene_modules = find_gene_modules(prepared, k=10, resolution=1e-2)
    gene_group_df = gene_modules[["id", "module"]].copy()
    cell_group_df = pd.DataFrame(
        {
            "cell": prepared.obs_names,
            "group": prepared.obs["cluster_truth"].astype(str).to_numpy(),
        }
    )
    agg = aggregate_gene_expression(prepared, gene_group_df, cell_group_df)
    assert agg.shape[0] == gene_modules["module"].nunique()
    assert agg.shape[1] == prepared.obs["cluster_truth"].nunique()


def test_find_gene_modules_umap_nn_method_validation(prepared):
    """R's ``umap.nn_method`` accepts a fixed set of backends
    (``cluster_genes.R:108``). Unknown values must raise rather than silently
    fall through."""
    with pytest.raises(ValueError, match="umap_nn_method"):
        find_gene_modules(prepared, k=10, resolution=1e-2, umap_nn_method="bogus")


def test_find_gene_modules_umap_nn_method_fnn(prepared):
    """``umap_nn_method='fnn'`` must return a valid module table — umap-learn
    receives an sklearn-precomputed kNN graph through its ``precomputed_knn``
    slot."""
    modules = find_gene_modules(
        prepared, k=10, resolution=1e-2, umap_nn_method="fnn",
    )
    assert {"id", "module", "dim_1", "dim_2"}.issubset(modules.columns)
    assert modules.shape[0] <= prepared.n_vars
