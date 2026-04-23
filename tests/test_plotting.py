"""Slice 8: plotting smoke tests (do not render)."""

from __future__ import annotations

import numpy as np
import pytest

import ggplot2_py as gg

from monocle3 import (
    cluster_cells,
    learn_graph,
    order_cells,
    plot_cells,
    plot_cells_3d,
    plot_genes_by_group,
    plot_genes_in_pseudotime,
    plot_genes_violin,
    plot_pc_variance_explained,
    plot_percent_cells_positive,
    preprocess_cds,
    reduce_dimension,
)


@pytest.fixture(scope="module")
def prepared(synthetic_adata):
    preprocess_cds(synthetic_adata, num_dim=10)
    reduce_dimension(synthetic_adata, max_components=2)
    cluster_cells(synthetic_adata, resolution=1e-3, random_seed=1)
    learn_graph(
        synthetic_adata,
        learn_graph_control={"ncenter": 20, "minimal_branch_len": 3,
                             "maxiter": 5},
    )
    names = synthetic_adata.uns["monocle3"]["principal_graph"]["UMAP"].vs["name"]
    order_cells(synthetic_adata, root_pr_nodes=[names[0]])
    return synthetic_adata


def test_plot_cells_color_by_cluster(prepared):
    p = plot_cells(prepared, color_cells_by="monocle3_clusters",
                   group_cells_by="cluster", show_trajectory_graph=False)
    assert isinstance(p, gg.GGPlot)


def test_plot_cells_color_by_genes(prepared):
    genes = list(prepared.var_names[:3])
    p = plot_cells(prepared, genes=genes, show_trajectory_graph=False)
    assert isinstance(p, gg.GGPlot)


def test_plot_cells_with_trajectory(prepared):
    p = plot_cells(prepared, color_cells_by="cluster_truth",
                   show_trajectory_graph=True)
    assert isinstance(p, gg.GGPlot)


def test_plot_pc_variance_explained(prepared):
    p = plot_pc_variance_explained(prepared)
    assert isinstance(p, gg.GGPlot)


def test_plot_percent_cells_positive(prepared):
    p = plot_percent_cells_positive(
        prepared[:, prepared.var_names[:3]],
        group_cells_by="cluster_truth",
    )
    assert isinstance(p, gg.GGPlot)


def test_plot_genes_in_pseudotime(prepared):
    p = plot_genes_in_pseudotime(
        prepared[:, prepared.var_names[:3]],
        color_cells_by="cluster_truth",
    )
    assert isinstance(p, gg.GGPlot)


def test_plot_genes_violin(prepared):
    p = plot_genes_violin(
        prepared[:, prepared.var_names[:3]],
        group_cells_by="cluster_truth",
    )
    assert isinstance(p, gg.GGPlot)


def test_plot_genes_by_group(prepared):
    p = plot_genes_by_group(
        prepared,
        markers=list(prepared.var_names[:3]),
        group_cells_by="cluster_truth",
    )
    assert isinstance(p, gg.GGPlot)


def test_plot_cells_3d(synthetic_adata):
    # Separate fixture to avoid collision with prepared UMAP dimensionality.
    preprocess_cds(synthetic_adata, num_dim=10)
    reduce_dimension(synthetic_adata, max_components=3)
    fig = plot_cells_3d(synthetic_adata, color_cells_by="cluster_truth",
                        show_trajectory_graph=False)
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)
