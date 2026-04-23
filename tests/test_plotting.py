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


@pytest.fixture(scope="function")
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


def test_plot_cells_genes_module_df(prepared):
    """When ``genes`` is a (id, module) DataFrame, plot_cells must aggregate
    per module (R path: ``dim(genes)[[2]] >= 2``), producing one facet per
    module rather than one per gene — otherwise notebook renders explode.
    """
    import pandas as pd

    ids = list(prepared.var_names[:9])
    modules = ["m1"] * 3 + ["m2"] * 3 + ["m3"] * 3
    module_df = pd.DataFrame({"id": ids, "module": modules})

    p = plot_cells(
        prepared, genes=module_df, show_trajectory_graph=False,
        scale_to_range=True,
    )
    assert isinstance(p, gg.GGPlot)
    # 3 modules × n_cells rows in the plot data (not 9 × n_cells).
    assert p.data["feature_label"].nunique() == 3
    assert len(p.data) == 3 * prepared.n_obs


def test_plot_cells_with_trajectory(prepared):
    p = plot_cells(prepared, color_cells_by="cluster_truth",
                   show_trajectory_graph=True)
    assert isinstance(p, gg.GGPlot)


def test_plot_cells_label_branch_points_leaves_roots_add_layers(prepared):
    """R ``plot_cells`` draws filled-circle + numbered-text layers for
    label_leaves / label_branch_points / label_roots — Python previously
    accepted these flags but never used them.
    """
    baseline = plot_cells(
        prepared, color_cells_by="cluster_truth",
        show_trajectory_graph=True,
        label_branch_points=False, label_leaves=False, label_roots=False,
        label_cell_groups=False,
    )
    annotated = plot_cells(
        prepared, color_cells_by="cluster_truth",
        show_trajectory_graph=True,
        label_branch_points=True, label_leaves=True, label_roots=True,
        label_cell_groups=False,
    )
    # Annotated plot should carry strictly more layers (filled-circle +
    # text pair per label type that resolves to a non-empty node set).
    assert len(annotated.layers) > len(baseline.layers)


def test_plot_cells_label_cell_groups_adds_text_layer(prepared):
    """R ``plot_cells`` adds ``ggrepel::geom_text_repel`` at the per-group
    median coords when ``label_cell_groups=TRUE`` and cells are colored by
    a categorical column.
    """
    with_labels = plot_cells(
        prepared, color_cells_by="cluster_truth",
        show_trajectory_graph=False,
        label_cell_groups=True, labels_per_group=1,
    )
    without = plot_cells(
        prepared, color_cells_by="cluster_truth",
        show_trajectory_graph=False,
        label_cell_groups=False,
    )
    assert len(with_labels.layers) == len(without.layers) + 1


def test_plot_pc_variance_explained(prepared):
    p = plot_pc_variance_explained(prepared)
    assert isinstance(p, gg.GGPlot)


def test_plot_percent_cells_positive(prepared):
    p = plot_percent_cells_positive(
        prepared[:, prepared.var_names[:3]],
        group_cells_by="cluster_truth",
    )
    assert isinstance(p, gg.GGPlot)


def test_plot_percent_cells_positive_adds_ci(prepared):
    """R emits ``geom_linerange`` with ``target_fraction_low`` / ``_high``
    bootstrap quantiles (``plotting.R:1642-1677``). The plot data must carry
    those four extra columns when the default bootstrap is on.
    """
    p = plot_percent_cells_positive(
        prepared[:, prepared.var_names[:3]],
        group_cells_by="cluster_truth",
        bootstrap_samples=20,
        random_seed=0,
    )
    for col in (
        "target_fraction_mean", "target_fraction_low", "target_fraction_high",
        "target_mean", "target_low", "target_high",
    ):
        assert col in p.data.columns, col
    # 2+ layers: geom_bar + geom_linerange (and theme/facet don't count as
    # layers in ggplot2_py).
    layer_classes = [type(layer.geom).__name__ for layer in p.layers]
    assert "GeomBar" in layer_classes
    assert "GeomLinerange" in layer_classes


def test_plot_percent_cells_positive_null_group(prepared):
    """R defaults ``group_cells_by=NULL`` and collapses to a single 'All'
    bar per gene (``plotting.R:1609-1612``). Python must do the same."""
    p = plot_percent_cells_positive(prepared[:, prepared.var_names[:3]])
    assert isinstance(p, gg.GGPlot)
    assert "all_cell" in p.data.columns
    assert set(p.data["all_cell"].unique()) == {"All"}


def test_plot_percent_cells_positive_panel_order(prepared):
    """``panel_order`` must make ``feature_label`` an ordered Categorical."""
    import pandas as pd

    genes = list(prepared.var_names[:3])
    short = (
        prepared.var.loc[genes, "gene_short_name"].astype(str).tolist()
        if "gene_short_name" in prepared.var.columns
        else genes
    )
    order = list(reversed(short))
    p = plot_percent_cells_positive(
        prepared[:, genes], group_cells_by="cluster_truth",
        panel_order=order, bootstrap_samples=10,
    )
    assert isinstance(p.data["feature_label"].dtype, pd.CategoricalDtype)
    assert list(p.data["feature_label"].cat.categories) == order


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


def test_plot_genes_violin_layers_match_r(prepared):
    """R ``plot_genes_violin`` (``plotting.R:1290-1308``) adds exactly three
    stat layers: violin + stat_summary(median) + facet_wrap. ``log_scale``
    wires in ``scale_y_log10`` (axis transform, not data-space pre-log)."""
    p = plot_genes_violin(
        prepared[:, prepared.var_names[:3]],
        group_cells_by="cluster_truth",
        log_scale=True,
    )
    names = [type(layer.geom).__name__ for layer in p.layers]
    # GeomViolin + StatSummary's point layer (same geom_point-like class).
    assert "GeomViolin" in names
    # scale_y_log10 registers a position scale with a log-10 transform.
    y_scales = [s for s in p.scales.scales if "y" in getattr(s, "aesthetics", [])]
    assert any("log-10" in repr(getattr(s, "trans", "")) for s in y_scales)


def test_plot_genes_violin_null_group(prepared):
    """Matches R (``plotting.R:1199-1310``): a default-None ``group_cells_by``
    collapses all cells into a single ``"All"`` bin."""
    p = plot_genes_violin(prepared[:, prepared.var_names[:3]])
    assert "all_cell" in p.data.columns
    assert set(p.data["all_cell"].unique()) == {"All"}


def test_plot_genes_by_group(prepared):
    p = plot_genes_by_group(
        prepared,
        markers=list(prepared.var_names[:3]),
        group_cells_by="cluster_truth",
    )
    assert isinstance(p, gg.GGPlot)


def test_plot_genes_by_group_ordering_modes(prepared):
    """Each ordering_type must yield a deterministic Gene/Group factor order
    (R lines 1843-1881). 'none' preserves input marker order.
    """
    markers = list(prepared.var_names[:6])
    p_cluster = plot_genes_by_group(
        prepared, markers=markers, group_cells_by="cluster_truth",
        ordering_type="cluster_row_col",
    )
    p_diag = plot_genes_by_group(
        prepared, markers=markers, group_cells_by="cluster_truth",
        ordering_type="maximal_on_diag",
    )
    p_none = plot_genes_by_group(
        prepared, markers=markers, group_cells_by="cluster_truth",
        ordering_type="none",
    )
    # "none" keeps input marker order (via gene_short_name, defaulting to id).
    gene_levels_none = list(p_none.data["Gene"].cat.categories)
    assert gene_levels_none == markers
    for p in (p_cluster, p_diag, p_none):
        assert isinstance(p, gg.GGPlot)
        assert p.data["Gene"].cat.ordered
        assert p.data["Group"].cat.ordered


def test_plot_cells_3d(synthetic_adata):
    # Separate fixture to avoid collision with prepared UMAP dimensionality.
    preprocess_cds(synthetic_adata, num_dim=10)
    reduce_dimension(synthetic_adata, max_components=3)
    fig = plot_cells_3d(synthetic_adata, color_cells_by="cluster_truth",
                        show_trajectory_graph=False)
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)


def test_plot_cells_3d_uses_set2_by_default(synthetic_adata):
    """R defaults categorical coloring to ``RColorBrewer::brewer.pal(N, "Set2")``.
    Python must use those exact hex codes when ``color_palette`` is not
    supplied — otherwise the 3D plot looks visibly different from R.
    """
    import scales

    preprocess_cds(synthetic_adata, num_dim=10)
    reduce_dimension(synthetic_adata, max_components=3)
    fig = plot_cells_3d(
        synthetic_adata, color_cells_by="cluster_truth",
        show_trajectory_graph=False,
    )
    # First trace color(s) should come from Set2.
    levels = synthetic_adata.obs["cluster_truth"].astype("category").cat.categories
    expected = scales.brewer_pal(palette="Set2")(max(3, len(levels)))
    expected = [c for c in expected if c is not None][: len(levels)]
    for i, trace in enumerate(fig.data[: len(levels)]):
        assert trace.marker.color == expected[i], (
            f"trace {i}: got {trace.marker.color}, want {expected[i]}"
        )


def test_plot_cells_3d_trajectory_uses_trajectory_graph_color(synthetic_adata):
    """R draws every trajectory edge with ``trajectory_graph_color``; Python
    must pass the exact value through to the plotly line trace color.
    """
    import numpy as np
    from monocle3 import learn_graph, cluster_cells

    preprocess_cds(synthetic_adata, num_dim=10)
    reduce_dimension(synthetic_adata, max_components=3)
    cluster_cells(synthetic_adata, resolution=1e-3, random_seed=0)
    learn_graph(
        synthetic_adata,
        learn_graph_control={"ncenter": 20, "minimal_branch_len": 3,
                             "maxiter": 5},
    )
    fig = plot_cells_3d(
        synthetic_adata, color_cells_by="cluster_truth",
        show_trajectory_graph=True, trajectory_graph_color="red",
        trajectory_graph_segment_size=3,
    )
    line_traces = [t for t in fig.data if t.mode == "lines"]
    assert len(line_traces) >= 1
    assert line_traces[0].line.color == "red"
    assert line_traces[0].line.width == 3
