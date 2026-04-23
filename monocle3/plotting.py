"""plotting — port of R/plotting.R.

All plot entry points return ``ggplot2_py.ggplot`` objects per
essential-suggestions §3. Only ``plot_cells_3d`` (in
``plotting_3d.py``) uses ``plotly``. No ``matplotlib`` / ``seaborn`` /
``scanpy.pl`` imports.
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp

from ._utils import get_monocle_uns
from .cluster_cells import clusters as _clusters, partitions as _partitions
from .order_cells import (
    _branch_nodes,
    _leaf_nodes,
    _root_nodes,
    pseudotime as _pseudotime,
)

__all__ = [
    "plot_cells",
    "plot_genes_by_group",
    "plot_genes_in_pseudotime",
    "plot_genes_violin",
    "plot_percent_cells_positive",
    "plot_pc_variance_explained",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cells_coord_df(
    adata: ad.AnnData, reduction_method: str, x: int = 1, y: int = 2,
) -> pd.DataFrame:
    key = f"X_{reduction_method.lower()}"
    if key not in adata.obsm:
        raise KeyError(
            f"No {reduction_method} reduction. Run reduce_dimension first."
        )
    coords = np.asarray(adata.obsm[key])
    if coords.shape[1] < max(x, y):
        raise ValueError("x and y must be dimensions in the reduced space.")
    df = pd.DataFrame(
        coords[:, [x - 1, y - 1]],
        columns=["data_dim_1", "data_dim_2"],
        index=adata.obs_names,
    )
    df["sample_name"] = adata.obs_names.astype(str)
    df = df.join(adata.obs, how="left")
    return df


def _ica_space_df(
    adata: ad.AnnData,
    reduction_method: str,
    x: int = 1,
    y: int = 2,
) -> pd.DataFrame | None:
    """Per-centroid coord frame (R: ``ica_space_df`` from ``dp_mst``)."""
    aux = get_monocle_uns(
        adata, "principal_graph_aux", reduction_method, default={}
    )
    pg = get_monocle_uns(adata, "principal_graph", reduction_method)
    if not aux or pg is None or pg.vcount() == 0:
        return None
    Y = np.asarray(aux["dp_mst"])  # D × K
    return pd.DataFrame(
        {
            "prin_graph_dim_1": Y[x - 1],
            "prin_graph_dim_2": Y[y - 1],
            "sample_name": list(pg.vs["name"]),
        }
    )


def _trajectory_edges(
    adata: ad.AnnData,
    reduction_method: str,
    x: int = 1,
    y: int = 2,
) -> pd.DataFrame | None:
    pg = get_monocle_uns(adata, "principal_graph", reduction_method)
    dim_df = _ica_space_df(adata, reduction_method, x, y)
    if dim_df is None or pg is None or pg.vcount() == 0:
        return None
    name_to_idx = {n: i for i, n in enumerate(dim_df["sample_name"].tolist())}
    rows = []
    for a, b in pg.get_edgelist():
        src_name = pg.vs[a]["name"]
        tgt_name = pg.vs[b]["name"]
        src_idx = name_to_idx.get(src_name)
        tgt_idx = name_to_idx.get(tgt_name)
        if src_idx is None or tgt_idx is None:
            continue
        rows.append(
            {
                "source": src_name,
                "target": tgt_name,
                "source_x": dim_df.loc[src_idx, "prin_graph_dim_1"],
                "target_x": dim_df.loc[tgt_idx, "prin_graph_dim_1"],
                "source_y": dim_df.loc[src_idx, "prin_graph_dim_2"],
                "target_y": dim_df.loc[tgt_idx, "prin_graph_dim_2"],
            }
        )
    return pd.DataFrame(rows)


def _build_text_df(
    data_df: pd.DataFrame,
    label_groups_by_cluster: bool,
    labels_per_group: int,
) -> pd.DataFrame | None:
    """Port of R ``plot_cells`` ``label_cell_groups`` text_df construction
    (lines 658-717 of ``plotting.R``).

    - ``label_groups_by_cluster=TRUE``: group by (cell_group, cell_color),
      keep top ``labels_per_group`` colors per group.
    - ``label_groups_by_cluster=FALSE``: group by cell_color only, with
      ``per = 1`` for every row.
    """
    if "cell_color" not in data_df.columns:
        return None
    colors = data_df["cell_color"]
    if not (
        colors.dtype.kind == "O" or pd.api.types.is_categorical_dtype(colors)
    ):
        return None  # R: "Cells aren't colored in a way that allows grouping."

    df = data_df.copy()
    df["cell_color"] = df["cell_color"].astype(str)

    if label_groups_by_cluster and "cell_group" in df.columns:
        df["cell_group"] = df["cell_group"].astype(str)
        group_sizes = df.groupby("cell_group").size().rename("n_group")
        cc_sizes = (
            df.groupby(["cell_group", "cell_color"])
            .size().rename("n_cc").reset_index()
        )
        cc_sizes = cc_sizes.join(group_sizes, on="cell_group")
        cc_sizes["per"] = cc_sizes["n_cc"] / cc_sizes["n_group"]
        median_coord = (
            df.groupby(["cell_group", "cell_color"])[["data_dim_1", "data_dim_2"]]
            .median()
            .rename(columns={"data_dim_1": "text_x", "data_dim_2": "text_y"})
            .reset_index()
        )
        merged = cc_sizes.merge(median_coord, on=["cell_group", "cell_color"])
        merged = (
            merged.sort_values(["cell_group", "per"], ascending=[True, False])
            .groupby("cell_group", as_index=False)
            .head(max(1, int(labels_per_group)))
        )
    else:
        # R: per = 1 for every color row; one median per color.
        median_coord = (
            df.groupby("cell_color")[["data_dim_1", "data_dim_2"]]
            .median()
            .rename(columns={"data_dim_1": "text_x", "data_dim_2": "text_y"})
            .reset_index()
        )
        merged = median_coord.assign(per=1.0)
        merged = (
            merged.sort_values(["cell_color"])
            .groupby("cell_color", as_index=False)
            .head(max(1, int(labels_per_group)))
        )

    merged["label"] = merged["cell_color"]
    keep = [c for c in ("cell_group", "cell_color", "per", "text_x", "text_y",
                        "label") if c in merged.columns]
    return merged[keep]


def _gene_ids_for(adata: ad.AnnData, genes: Sequence[str] | pd.DataFrame) -> list[str]:
    if isinstance(genes, pd.DataFrame):
        genes = list(genes.iloc[:, 0])
    genes = list(genes)
    gene_ids = list(adata.var_names)
    short_names = (
        adata.var["gene_short_name"].astype(str).tolist()
        if "gene_short_name" in adata.var.columns
        else gene_ids
    )
    by_short = {s: gid for gid, s in zip(gene_ids, short_names)}
    resolved = []
    for g in genes:
        g = str(g)
        if g in gene_ids:
            resolved.append(g)
        elif g in by_short:
            resolved.append(by_short[g])
    if not resolved:
        raise ValueError("None of the provided genes were found in adata")
    return resolved


def _is_module_genes_df(genes: object) -> bool:
    """Return True for the R ``plot_cells(genes=DataFrame)`` module form.

    R dispatches on ``!is.null(dim(genes)) && dim(genes)[[2]] >= 2`` — a
    two-column (or wider) table where col 1 is gene id and col 2 is the
    module / group label.
    """
    return isinstance(genes, pd.DataFrame) and genes.shape[1] >= 2


def _module_expression_long(
    adata: ad.AnnData,
    genes_df: pd.DataFrame,
    data_df: pd.DataFrame,
    norm_method: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Return a long ``(sample_name, feature_label, expression)`` frame plus
    the facet ordering, by aggregating per module as R's ``plot_cells``
    does when given a 2-col DataFrame.
    """
    from .cluster_genes import aggregate_gene_expression

    agg = aggregate_gene_expression(
        adata,
        gene_group_df=genes_df,
        norm_method=norm_method,
        gene_agg_fun="mean",
        scale_agg_values=False,
    )  # modules × cells
    # Preserve module order from the second column of the user-supplied df.
    module_col = genes_df.columns[1]
    module_order = list(pd.unique(genes_df[module_col].astype(str)))
    module_order = [m for m in module_order if m in agg.index]
    agg = agg.loc[module_order]

    long = agg.T.reset_index(names="sample_name").melt(
        id_vars="sample_name", var_name="feature_label", value_name="expression",
    )
    long["data_dim_1"] = data_df.loc[long["sample_name"], "data_dim_1"].to_numpy()
    long["data_dim_2"] = data_df.loc[long["sample_name"], "data_dim_2"].to_numpy()
    return long, module_order


def _per_gene_expression_long(
    adata: ad.AnnData,
    genes: Sequence[str],
    data_df: pd.DataFrame,
    norm_method: str,
    min_expr: float,
) -> tuple[pd.DataFrame, list[str]]:
    gene_ids = _gene_ids_for(adata, genes)
    expr_df = _log_norm_expr(
        adata, gene_ids, norm_method=norm_method, min_expr=min_expr
    )
    gene_short = (
        adata.var["gene_short_name"].astype(str)
        if "gene_short_name" in adata.var.columns
        else pd.Series(adata.var_names, index=adata.var_names)
    )
    long = expr_df.reset_index(names="sample_name").melt(
        id_vars="sample_name", var_name="gene_id", value_name="expression",
    )
    long["feature_label"] = long["gene_id"].map(gene_short).fillna(long["gene_id"])
    long["data_dim_1"] = data_df.loc[long["sample_name"], "data_dim_1"].to_numpy()
    long["data_dim_2"] = data_df.loc[long["sample_name"], "data_dim_2"].to_numpy()
    # R floors sub-min_expr values to NA so they draw as gray80.
    long.loc[long["expression"] < min_expr, "expression"] = np.nan
    # Facet order follows the user-supplied gene order.
    requested = [str(g) for g in genes]
    label_order: list[str] = []
    for g in requested:
        if g in gene_short.values:
            label_order.append(g)
        elif g in gene_ids:
            label_order.append(str(gene_short.get(g, g)))
    label_order = list(dict.fromkeys(label_order))
    return long, label_order


def _log_norm_expr(
    adata: ad.AnnData,
    gene_ids: Sequence[str],
    norm_method: str = "log",
    min_expr: float = 0.1,
) -> pd.DataFrame:
    """Return a (cells × genes) DataFrame of normalized expression values."""
    gene_idx = [adata.var_names.get_loc(g) for g in gene_ids]
    X = adata.X[:, gene_idx]
    if sp.issparse(X):
        mat = np.asarray(X.todense())
    else:
        mat = np.asarray(X, dtype=float)
    sf = adata.obs["Size_Factor"].to_numpy(dtype=float)
    mat = mat / sf[:, None]
    if norm_method == "log":
        mat = np.log10(mat + 1)
    # min_expr is used downstream for floor/viridis; we don't clip here.
    return pd.DataFrame(mat, index=adata.obs_names, columns=gene_ids)


# ---------------------------------------------------------------------------
# plot_cells
# ---------------------------------------------------------------------------


def plot_cells(
    adata: ad.AnnData,
    x: int = 1,
    y: int = 2,
    reduction_method: str = "UMAP",
    color_cells_by: str = "cluster",
    group_cells_by: str = "cluster",
    genes: Sequence[str] | pd.DataFrame | None = None,
    show_trajectory_graph: bool = True,
    trajectory_graph_color: str = "grey28",
    trajectory_graph_segment_size: float = 0.75,
    norm_method: str = "log",
    label_cell_groups: bool = True,
    label_groups_by_cluster: bool = True,
    group_label_size: float = 2,
    labels_per_group: int = 1,
    label_branch_points: bool = True,
    label_roots: bool = True,
    label_leaves: bool = True,
    graph_label_size: float = 2,
    cell_size: float = 0.35,
    cell_stroke: float | None = None,
    alpha: float = 1.0,
    min_expr: float = 0.1,
    rasterize: bool = False,
    scale_to_range: bool = True,
    label_principal_points: bool = False,
):
    """Monocle3's umbrella 2D cell scatter (``plot_cells``)."""
    import ggplot2_py as gg
    from ggrepel_py import geom_text_repel

    if rasterize:
        warnings.warn(
            "rasterize=True is not supported in v1 (ggrastr-python port pending); "
            "falling back to vector geom_point.",
            stacklevel=2,
        )

    # R: cell_stroke = I(cell_size / 2) as formal default.
    if cell_stroke is None:
        cell_stroke = cell_size / 2.0

    # R: label_principal_points pre-empts the per-type toggles.
    pg_for_labels = get_monocle_uns(adata, "principal_graph", reduction_method)
    if label_principal_points and (pg_for_labels is None or pg_for_labels.vcount() == 0):
        label_principal_points = False
    if label_principal_points:
        label_branch_points = False
        label_leaves = False
        label_roots = False

    data_df = _cells_coord_df(adata, reduction_method=reduction_method, x=x, y=y)

    # Assign group labels.
    if group_cells_by == "cluster":
        try:
            data_df["cell_group"] = _clusters(
                adata, reduction_method=reduction_method
            ).reindex(data_df.index).astype(str).values
        except KeyError:
            data_df["cell_group"] = None
    elif group_cells_by == "partition":
        try:
            data_df["cell_group"] = _partitions(
                adata, reduction_method=reduction_method
            ).reindex(data_df.index).astype(str).values
        except KeyError:
            data_df["cell_group"] = None
    else:
        if group_cells_by in adata.obs.columns:
            data_df["cell_group"] = adata.obs[group_cells_by].astype(str).to_numpy()
        else:
            raise KeyError(f"group_cells_by '{group_cells_by}' not in obs")

    # Determine the colour channel.
    color_is_numeric = False
    if genes is not None:
        if _is_module_genes_df(genes):
            long, label_order = _module_expression_long(
                adata, genes, data_df, norm_method=norm_method,
            )
            legend_label = "Expression score"
        else:
            long, label_order = _per_gene_expression_long(
                adata, genes, data_df,
                norm_method=norm_method, min_expr=min_expr,
            )
            legend_label = (
                "Expression" if norm_method == "size_only" else "log10(Expression)"
            )

        if scale_to_range and len(long) > 0:
            # R: per-facet rescale to 0-100, then legend label flips to '% Max'.
            grp = long.groupby("feature_label")["expression"]
            lo = grp.transform("min")
            hi = grp.transform("max")
            span = (hi - lo).where(lambda s: s > 0, 1.0)
            long["expression"] = 100.0 * (long["expression"] - lo) / span
            legend_label = "% Max"

        # Enforce facet ordering (matches R's levels(markers) / levels(modules)).
        if label_order:
            long["feature_label"] = pd.Categorical(
                long["feature_label"], categories=label_order, ordered=True,
            )

        # R: draws NA (sub-min_expr) cells grey first, then the colored layer on top.
        p = gg.ggplot(long, gg.aes(x="data_dim_1", y="data_dim_2"))
        na_sub = long[long["expression"].isna()]
        ya_sub = long[~long["expression"].isna()].sort_values("expression")
        if len(na_sub) > 0:
            p = p + gg.geom_point(
                data=na_sub, size=cell_size, stroke=cell_stroke,
                color="grey80", alpha=alpha,
            )
        p = p + gg.geom_point(
            gg.aes(color="expression"),
            data=ya_sub, size=cell_size, stroke=cell_stroke, alpha=alpha,
        )
        p = p + gg.scale_color_viridis_c(
            option="D", end=0.8, na_value="gray80", name=legend_label,
        )
        p = p + gg.facet_wrap("feature_label")
        p = p + gg.theme_bw()
        p = p + gg.xlab(f"{reduction_method} {x}")
        p = p + gg.ylab(f"{reduction_method} {y}")
    else:
        if color_cells_by == "cluster":
            try:
                data_df["cell_color"] = _clusters(
                    adata, reduction_method=reduction_method
                ).reindex(data_df.index).astype(str).values
            except KeyError:
                data_df["cell_color"] = None
        elif color_cells_by == "partition":
            try:
                data_df["cell_color"] = _partitions(
                    adata, reduction_method=reduction_method
                ).reindex(data_df.index).astype(str).values
            except KeyError:
                data_df["cell_color"] = None
        elif color_cells_by == "pseudotime":
            data_df["cell_color"] = _pseudotime(
                adata, reduction_method=reduction_method
            ).reindex(data_df.index).to_numpy()
            color_is_numeric = True
        elif color_cells_by in adata.obs.columns:
            col = adata.obs[color_cells_by]
            color_is_numeric = pd.api.types.is_numeric_dtype(col)
            data_df["cell_color"] = (
                col.to_numpy() if color_is_numeric else col.astype(str).to_numpy()
            )
        else:
            raise KeyError(
                f"color_cells_by '{color_cells_by}' not valid"
            )

        p = gg.ggplot(data_df, gg.aes(x="data_dim_1", y="data_dim_2"))
        p = p + gg.geom_point(
            gg.aes(color="cell_color"),
            size=cell_size, stroke=cell_stroke, alpha=alpha,
        )
        if color_is_numeric:
            p = p + gg.scale_color_viridis_c(option="C", name=color_cells_by)
        p = p + gg.theme_bw()
        p = p + gg.xlab(f"{reduction_method} {x}")
        p = p + gg.ylab(f"{reduction_method} {y}")

    # Trajectory edges + optional node labels.
    if show_trajectory_graph:
        edges = _trajectory_edges(adata, reduction_method=reduction_method, x=x, y=y)
        ica_df = _ica_space_df(adata, reduction_method=reduction_method, x=x, y=y)
        if edges is None or edges.empty:
            warnings.warn(
                "No trajectory to plot. Has learn_graph() been called yet?",
                stacklevel=2,
            )
        else:
            p = p + gg.geom_segment(
                gg.aes(
                    x="source_x", xend="target_x",
                    y="source_y", yend="target_y",
                ),
                data=edges,
                size=float(trajectory_graph_segment_size),
                color=str(trajectory_graph_color),
            )
            marker_size = float(graph_label_size) * 1.5
            marker_stroke = float(trajectory_graph_segment_size)
            if label_principal_points and ica_df is not None:
                # R: union of branch / leaf / root nodes, label by vertex name.
                names = pd.Index(
                    list(_branch_nodes(adata, reduction_method).index)
                    + list(_leaf_nodes(adata, reduction_method).index)
                    + list(_root_nodes(adata, reduction_method).index)
                )
                princ_df = ica_df[ica_df["sample_name"].isin(names)].reset_index(drop=True)
                if len(princ_df):
                    p = _draw_node_labels(
                        p, princ_df,
                        label_col="sample_name",
                        fill="black", color="white",
                        text_color="black",
                        marker_size=marker_size,
                        marker_stroke=marker_stroke,
                        text_size=marker_size,
                        use_repel=True,
                    )
            if label_branch_points and ica_df is not None:
                nodes = _branch_nodes(adata, reduction_method).index
                sub = ica_df[ica_df["sample_name"].isin(nodes)].reset_index(drop=True)
                if len(sub):
                    sub = sub.assign(idx=np.arange(1, len(sub) + 1).astype(str))
                    p = _draw_node_labels(
                        p, sub, label_col="idx",
                        fill="black", color="white", text_color="white",
                        marker_size=marker_size, marker_stroke=marker_stroke,
                        text_size=float(graph_label_size),
                    )
            if label_leaves and ica_df is not None:
                nodes = _leaf_nodes(adata, reduction_method).index
                sub = ica_df[ica_df["sample_name"].isin(nodes)].reset_index(drop=True)
                if len(sub):
                    sub = sub.assign(idx=np.arange(1, len(sub) + 1).astype(str))
                    p = _draw_node_labels(
                        p, sub, label_col="idx",
                        fill="lightgray", color="black", text_color="black",
                        marker_size=marker_size, marker_stroke=marker_stroke,
                        text_size=float(graph_label_size),
                    )
            if label_roots and ica_df is not None:
                nodes = _root_nodes(adata, reduction_method).index
                sub = ica_df[ica_df["sample_name"].isin(nodes)].reset_index(drop=True)
                if len(sub):
                    sub = sub.assign(idx=np.arange(1, len(sub) + 1).astype(str))
                    p = _draw_node_labels(
                        p, sub, label_col="idx",
                        fill="white", color="black", text_color="black",
                        marker_size=marker_size, marker_stroke=marker_stroke,
                        text_size=float(graph_label_size),
                    )

    # Cell-group labels via ggrepel — only when colored by group-like aesthetic.
    if (
        label_cell_groups
        and genes is None
        and not color_is_numeric
        and "cell_color" in data_df.columns
    ):
        text_df = _build_text_df(
            data_df,
            label_groups_by_cluster=label_groups_by_cluster,
            labels_per_group=labels_per_group,
        )
        if text_df is not None and len(text_df):
            p = p + geom_text_repel(
                mapping=gg.aes(x="text_x", y="text_y", label="label"),
                data=text_df,
                size=float(group_label_size),
                inherit_aes=False,
            )

    return p


def _draw_node_labels(
    p,
    df: pd.DataFrame,
    label_col: str,
    fill: str,
    color: str,
    text_color: str,
    marker_size: float,
    marker_stroke: float,
    text_size: float,
    use_repel: bool = False,
):
    """Add a filled-circle + integer/text label layer to a principal-graph plot."""
    import ggplot2_py as gg

    p = p + gg.geom_point(
        gg.aes(x="prin_graph_dim_1", y="prin_graph_dim_2"),
        data=df,
        shape=21, fill=fill, color=color,
        size=marker_size, stroke=marker_stroke,
        inherit_aes=False,
    )
    if use_repel:
        from ggrepel_py import geom_text_repel
        p = p + geom_text_repel(
            mapping=gg.aes(
                x="prin_graph_dim_1", y="prin_graph_dim_2", label=label_col,
            ),
            data=df, size=text_size, color=text_color, inherit_aes=False,
        )
    else:
        p = p + gg.geom_text(
            gg.aes(
                x="prin_graph_dim_1", y="prin_graph_dim_2", label=label_col,
            ),
            data=df, size=text_size, color=text_color, inherit_aes=False,
        )
    return p


# ---------------------------------------------------------------------------
# plot_pc_variance_explained
# ---------------------------------------------------------------------------


def plot_pc_variance_explained(adata: ad.AnnData):
    import ggplot2_py as gg

    model = get_monocle_uns(adata, "preprocess", "PCA")
    if model is None:
        raise KeyError("Run preprocess_cds(method='PCA') first.")
    sdev = np.asarray(model["svd_sdev"], dtype=float)
    df = pd.DataFrame(
        {
            "pc": np.arange(1, len(sdev) + 1),
            "variance_explained": sdev ** 2 / np.sum(sdev ** 2),
        }
    )
    return (
        gg.ggplot(df, gg.aes(x="pc", y="variance_explained"))
        + gg.geom_point()
        + gg.theme_bw()
        + gg.xlab("PC")
        + gg.ylab("Variance explained")
    )


# ---------------------------------------------------------------------------
# plot_percent_cells_positive
# ---------------------------------------------------------------------------


def plot_percent_cells_positive(
    adata: ad.AnnData,
    group_cells_by: str | None = None,
    min_expr: float = 0.0,
    nrow: int | None = None,
    ncol: int = 1,
    panel_order: Sequence[str] | None = None,
    plot_as_count: bool = False,
    label_by_short_name: bool = True,
    normalize: bool = True,
    plot_limits: Sequence[float] | None = None,
    bootstrap_samples: int = 100,
    conf_int_alpha: float = 0.95,
    random_seed: int | None = 0,
):
    """Bar plot with bootstrapped CI of % cells expressing each gene per group.

    Port of R ``plot_percent_cells_positive`` (``plotting.R:1539-1681``).

    - ``group_cells_by=None`` groups every cell into a single "All" bar
      (R line 1609-1612).
    - ``normalize`` divides counts by ``size_factors`` before thresholding
      (R lines 1576-1583) — it is NOT a percent-conversion flag.
    - ``plot_as_count`` switches the y-axis between raw target count and
      percent (``target_fraction_mean * 100``).
    - CIs are computed by ``bootstrap_samples`` row-level resamples of the
      melted ``(gene, cell)`` frame, matching R's ``rsample::bootstraps``
      workflow.
    - ``conf_int_alpha`` is passed straight through per R's wiring, which
      uses ``quantile(target_fraction, (1-alpha)/2)`` and its complement
      for the fraction linerange (lines 1642-1643). The raw-count linerange
      uses the narrower ``alpha/2`` / ``1 - alpha/2`` quantiles exactly as
      R does — we mirror R here rather than "fix" it.
    """
    import ggplot2_py as gg

    if adata.n_vars > 100:
        raise ValueError(
            "adata has more than 100 genes — pass only the subset of the CDS "
            "to be plotted."
        )

    if group_cells_by is None:
        group_vals = np.array(["All"] * adata.n_obs)
        group_col_name = "all_cell"
    else:
        if group_cells_by in adata.obs.columns:
            group_vals = adata.obs[group_cells_by].astype(str).to_numpy()
        else:
            raise KeyError(f"group_cells_by '{group_cells_by}' not in obs")
        group_col_name = group_cells_by

    X = adata.X
    mat = np.asarray(X.todense() if sp.issparse(X) else X, dtype=float)
    if normalize:
        sf = adata.obs["Size_Factor"].to_numpy(dtype=float)
        mat = np.round(mat / sf[:, None], decimals=4)  # R: round(..., digits=4)

    # Feature labels — R lines 1592-1602.
    if label_by_short_name and "gene_short_name" in adata.var.columns:
        short = adata.var["gene_short_name"].astype(str).to_numpy()
        feature_labels = np.where(
            pd.isna(adata.var["gene_short_name"].to_numpy()),
            adata.var_names.to_numpy(),
            short,
        )
    else:
        feature_labels = adata.var_names.to_numpy()

    # Melt: (n_genes, n_cells) → long array of (feature_idx, cell_idx,
    # binary-expressed, group).
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    binary = (mat > float(min_expr)).astype(np.int64)  # cells × genes
    # feature_idx[i] = gene index for row i; cell_idx[i] = cell index.
    gene_idx_flat = np.repeat(np.arange(n_genes), n_cells)
    cell_idx_flat = np.tile(np.arange(n_cells), n_genes)
    binary_flat = binary.T.reshape(-1)  # same ordering as above.
    group_flat = group_vals[cell_idx_flat]

    # Unique (feature, group) pair index for vectorised sums.
    n_rows = binary_flat.size
    groups_unique = list(pd.unique(group_vals))
    group_to_i = {g: i for i, g in enumerate(groups_unique)}
    group_idx_flat = np.array(
        [group_to_i[g] for g in group_flat], dtype=np.int64
    )
    pair_idx_flat = gene_idx_flat * len(groups_unique) + group_idx_flat
    n_pairs = n_genes * len(groups_unique)

    rng = np.random.default_rng(random_seed)
    B = max(1, int(bootstrap_samples))
    target_boot = np.zeros((B, n_pairs), dtype=np.float64)
    count_boot = np.zeros((B, n_pairs), dtype=np.float64)
    for b in range(B):
        draw = rng.integers(0, n_rows, size=n_rows)
        pairs = pair_idx_flat[draw]
        vals = binary_flat[draw]
        target_boot[b] = np.bincount(pairs, weights=vals, minlength=n_pairs)
        count_boot[b] = np.bincount(pairs, minlength=n_pairs)
    # target_fraction per bootstrap.
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_boot = np.where(count_boot > 0, target_boot / count_boot, 0.0)

    # R lines 1636-1643 summarise across bootstraps.
    alpha = float(conf_int_alpha)
    target_mean = target_boot.mean(axis=0)
    frac_mean = frac_boot.mean(axis=0)
    # R uses alpha/2 and 1 - alpha/2 for count CI (narrow — likely a bug in R
    # but we mirror it to stay faithful) and (1-alpha)/2 / 1-(1-alpha)/2 for
    # fraction CI (standard symmetric 95% CI when alpha=0.95).
    target_low = np.quantile(target_boot, alpha / 2, axis=0)
    target_high = np.quantile(target_boot, 1 - alpha / 2, axis=0)
    frac_low = np.quantile(frac_boot, (1 - alpha) / 2, axis=0)
    frac_high = np.quantile(frac_boot, 1 - (1 - alpha) / 2, axis=0)

    out = []
    for gi in range(n_genes):
        for grp, grpi in group_to_i.items():
            pi = gi * len(groups_unique) + grpi
            if count_boot[:, pi].mean() == 0:
                continue
            out.append({
                "feature_label": feature_labels[gi],
                group_col_name: grp,
                "target_mean": float(target_mean[pi]),
                "target_fraction_mean": float(frac_mean[pi]),
                "target_low": float(target_low[pi]),
                "target_high": float(target_high[pi]),
                "target_fraction_low": float(frac_low[pi]),
                "target_fraction_high": float(frac_high[pi]),
            })
    marker_counts = pd.DataFrame(out)

    # Panel order — R lines 1604-1607.
    if panel_order is not None:
        marker_counts["feature_label"] = pd.Categorical(
            marker_counts["feature_label"],
            categories=list(panel_order), ordered=True,
        )

    if not plot_as_count:
        # R lines 1651-1658: convert fraction CI to percent.
        for col in ("target_fraction_mean", "target_fraction_low",
                    "target_fraction_high"):
            marker_counts[col] = marker_counts[col] * 100.0
        y_col = "target_fraction_mean"
        ymin_col = "target_fraction_low"
        ymax_col = "target_fraction_high"
        y_label = "Cells (percent)"
    else:
        y_col = "target_mean"
        ymin_col = "target_low"
        ymax_col = "target_high"
        y_label = "Cells"

    p = (
        gg.ggplot(
            marker_counts,
            gg.aes(x=group_col_name, y=y_col, fill=group_col_name),
        )
        + gg.geom_bar(stat="identity")
        + gg.geom_linerange(gg.aes(ymin=ymin_col, ymax=ymax_col))
        + gg.theme_bw()
        + gg.ylab(y_label)
    )
    if group_col_name == "all_cell":
        # R removes the legend title when grouping is 'All'.
        p = p + gg.theme(legend_title=gg.element_blank()) + gg.xlab("")
    else:
        p = p + gg.xlab(group_col_name)

    if plot_limits is not None:
        p = p + gg.scale_y_continuous(limits=list(plot_limits))

    facet_kwargs = {"ncol": int(ncol), "scales": "free_y"}
    if nrow is not None:
        facet_kwargs["nrow"] = int(nrow)
    p = p + gg.facet_wrap("feature_label", **facet_kwargs)
    return p


# ---------------------------------------------------------------------------
# plot_genes_in_pseudotime
# ---------------------------------------------------------------------------


def plot_genes_in_pseudotime(
    adata_subset: ad.AnnData,
    min_expr: float | None = None,
    cell_size: float = 0.75,
    nrow: int | None = None,
    ncol: int = 1,
    panel_order: Sequence[str] | None = None,
    color_cells_by: str = "pseudotime",
    trend_formula: str = "~ splines::ns(pseudotime, df=3)",
    label_by_short_name: bool = True,
    vertical_jitter: float | None = None,
    horizontal_jitter: float | None = None,
    reduction_method: str = "UMAP",
):
    """Line plot of gene expression along pseudotime (R: ``plot_genes_in_pseudotime``).

    Fits ``trend_formula`` via :func:`fit_models` and overlays the model
    expectation on the per-cell scatter, matching R's default trend line.
    """
    import ggplot2_py as gg
    from .expr_models import fit_models, model_predictions

    if "monocle3_pseudotime" not in adata_subset.obs.columns:
        raise KeyError("Run order_cells first.")
    if adata_subset.n_vars > 100:
        raise ValueError(
            "adata_subset has more than 100 genes — pass only the subset to be plotted."
        )
    if color_cells_by not in ("cluster", "partition"):
        if color_cells_by not in adata_subset.obs.columns and color_cells_by != "pseudotime":
            raise KeyError(
                f"color_cells_by '{color_cells_by}' not in obs"
            )

    # R: subset to cells with finite pseudotime.
    pseudo_series = adata_subset.obs["monocle3_pseudotime"].astype(float)
    finite_mask = np.isfinite(pseudo_series.to_numpy())
    sub = adata_subset[finite_mask].copy()
    sub.obs["pseudotime"] = sub.obs["monocle3_pseudotime"].astype(float)

    # R: min_expr NULL → 0 after the data is prepared.
    if min_expr is None:
        min_expr = 0.0

    # Per-cell normalised expression (R: counts / size_factors, no log).
    X = sub.X
    mat = np.asarray(X.todense() if sp.issparse(X) else X, dtype=float)
    sf = sub.obs["Size_Factor"].to_numpy(dtype=float)
    norm = np.round(mat / sf[:, None])

    short_names = (
        sub.var["gene_short_name"].astype(str).tolist()
        if "gene_short_name" in sub.var.columns
        else list(sub.var_names)
    )
    feature_labels = (
        short_names if label_by_short_name else list(sub.var_names)
    )

    # Long (cell × gene) scatter data.
    n_cells, n_genes = norm.shape
    rep_cells = np.tile(sub.obs_names, n_genes)
    rep_genes = np.repeat(list(sub.var_names), n_cells)
    rep_labels = np.repeat(feature_labels, n_cells)
    rep_pseudo = np.tile(sub.obs["pseudotime"].to_numpy(float), n_genes)
    rep_expr = norm.T.reshape(-1)
    df = pd.DataFrame(
        {
            "sample_name": rep_cells,
            "f_id": rep_genes,
            "feature_label": rep_labels,
            "pseudotime": rep_pseudo,
            "expression": rep_expr,
        }
    )

    # Fit trend + model expectation (R: fit_models + model_predictions).
    new_data = sub.obs.copy()
    new_data["Size_Factor"] = 1.0
    model_tbl = fit_models(sub, model_formula_str=trend_formula)
    model_exp_arr = np.asarray(
        model_predictions(model_tbl, new_data=new_data)
    )  # (n_genes, n_cells)
    # Row order follows model_tbl (same gene order as sub.var_names when
    # fit_models iterates var_names in order).
    exp_long = pd.DataFrame(
        model_exp_arr,
        index=list(model_tbl["gene_id"]) if "gene_id" in model_tbl.columns
        else list(sub.var_names),
        columns=list(sub.obs_names),
    ).reset_index(names="f_id").melt(
        id_vars="f_id", var_name="sample_name", value_name="expectation",
    )
    df = df.merge(exp_long, on=["f_id", "sample_name"], how="left")

    # R: floor both observed and fitted at min_expr.
    df.loc[df["expression"] < min_expr, "expression"] = min_expr
    df.loc[df["expectation"] < min_expr, "expectation"] = min_expr

    # Facet / panel order.
    if panel_order is not None:
        df["feature_label"] = pd.Categorical(
            df["feature_label"], categories=list(panel_order), ordered=True,
        )

    # Color column.
    color_is_numeric = False
    if color_cells_by == "pseudotime":
        df["cell_color"] = df["pseudotime"]
        color_is_numeric = True
    elif color_cells_by == "cluster":
        df["cell_color"] = _clusters(
            sub, reduction_method=reduction_method
        ).astype(str).reindex(df["sample_name"]).to_numpy()
    elif color_cells_by == "partition":
        df["cell_color"] = _partitions(
            sub, reduction_method=reduction_method
        ).astype(str).reindex(df["sample_name"]).to_numpy()
    else:
        col = sub.obs[color_cells_by]
        color_is_numeric = pd.api.types.is_numeric_dtype(col)
        values = (
            col.to_numpy(float) if color_is_numeric else col.astype(str).to_numpy()
        )
        df["cell_color"] = pd.Series(values, index=sub.obs_names).reindex(
            df["sample_name"]
        ).to_numpy()

    # Build the plot.
    jitter = gg.position_jitter(
        width=float(horizontal_jitter or 0.0),
        height=float(vertical_jitter or 0.0),
    )
    p = (
        gg.ggplot(df, gg.aes(x="pseudotime", y="expression"))
        + gg.geom_point(
            gg.aes(color="cell_color"), size=cell_size, position=jitter,
        )
    )
    if color_is_numeric:
        p = p + gg.scale_color_viridis_c(option="C", name=color_cells_by)
    # Trend line per facet.
    p = p + gg.geom_line(gg.aes(x="pseudotime", y="expectation"))

    # log10 y scale + bottom of axis at [min_expr, 1].
    p = p + gg.scale_y_log10()
    if min_expr < 1:
        p = p + gg.expand_limits(y=[float(min_expr), 1.0])

    facet_kwargs = {"ncol": int(ncol), "scales": "free_y"}
    if nrow is not None:
        facet_kwargs["nrow"] = int(nrow)
    p = p + gg.facet_wrap("feature_label", **facet_kwargs)
    p = p + gg.theme_bw() + gg.xlab("pseudotime") + gg.ylab("Expression")
    return p


# ---------------------------------------------------------------------------
# plot_genes_violin
# ---------------------------------------------------------------------------


def plot_genes_violin(
    adata_subset: ad.AnnData,
    group_cells_by: str | None = None,
    min_expr: float = 0.0,
    nrow: int | None = None,
    ncol: int = 1,
    panel_order: Sequence[str] | None = None,
    label_by_short_name: bool = True,
    normalize: bool = True,
    log_scale: bool = True,
    pseudocount: float = 0.0,
):
    """Violin plot of per-group gene expression.

    Port of R ``plot_genes_violin`` (``plotting.R:1199-1310``). The R source
    is deliberately minimal — it does **not** use ``ggforce::geom_sina`` or
    ``ggdist::stat_histinterval`` (those live in ``plot_genes_hybrid``).
    The signature mirrors R exactly, including:

    - ``group_cells_by=None`` → all cells in one ``"All"`` bin.
    - ``normalize=True`` divides counts by ``size_factors`` (not a
      percent-conversion flag).
    - ``pseudocount > 0`` adds a constant 1 to counts (R: ``counts + 1``),
      matching R's "pseudocount > 0 is reset to 1" docstring.
    - ``min_expr`` floors (not filters) values below it; violin shape on the
      original population.
    - ``log_scale`` applies ``scale_y_log10()`` as an axis transform (R
      behavior) — it does NOT pre-log the data.
    - ``stat_summary(fun=median, geom="point", size=1, color="black")``
      draws a small black dot at each violin's median.
    """
    import ggplot2_py as gg

    if adata_subset.n_vars > 100:
        raise ValueError(
            "adata_subset has more than 100 genes — pass only the subset "
            "of the CDS to be plotted."
        )
    if group_cells_by is not None and group_cells_by not in adata_subset.obs.columns:
        raise KeyError(f"group_cells_by '{group_cells_by}' not in obs")

    X = adata_subset.X
    mat = np.asarray(X.todense() if sp.issparse(X) else X, dtype=float)
    # R: if pseudocount > 0 → counts + 1 (NOT counts + pseudocount).
    if pseudocount > 0:
        mat = mat + 1.0
    if normalize:
        sf = adata_subset.obs["Size_Factor"].to_numpy(dtype=float)
        mat = mat / sf[:, None]

    short_names = (
        adata_subset.var["gene_short_name"].astype(str).tolist()
        if "gene_short_name" in adata_subset.var.columns
        else list(adata_subset.var_names)
    )
    feature_labels = (
        short_names if label_by_short_name else list(adata_subset.var_names)
    )

    if group_cells_by is None:
        group_vals = np.array(["All"] * adata_subset.n_obs)
        group_col_name = "all_cell"
    else:
        group_vals = adata_subset.obs[group_cells_by].astype(str).to_numpy()
        group_col_name = group_cells_by

    n_cells, n_genes = mat.shape
    rep_group = np.tile(group_vals, n_genes)
    rep_labels = np.repeat(feature_labels, n_cells)
    rep_expr = mat.T.reshape(-1)
    # R floors sub-min_expr (does not filter them out).
    rep_expr = np.where(rep_expr < min_expr, float(min_expr), rep_expr)
    df = pd.DataFrame({
        group_col_name: rep_group,
        "feature_label": rep_labels,
        "expression": rep_expr,
    })

    if panel_order is not None:
        df["feature_label"] = pd.Categorical(
            df["feature_label"], categories=list(panel_order), ordered=True,
        )

    p = (
        gg.ggplot(df, gg.aes(x=group_col_name, y="expression"))
        + gg.geom_violin(gg.aes(fill=group_col_name), scale="width")
        + gg.guides(fill="none")
        + gg.stat_summary(fun=np.median, geom="point", size=1, color="black")
    )

    facet_kwargs = {"ncol": int(ncol), "scales": "free_y"}
    if nrow is not None:
        facet_kwargs["nrow"] = int(nrow)
    p = p + gg.facet_wrap("feature_label", **facet_kwargs)

    if min_expr < 1:
        p = p + gg.expand_limits(y=[float(min_expr), 1.0])

    p = p + gg.ylab("Expression")
    p = p + gg.xlab("" if group_cells_by is None else group_cells_by)

    if log_scale:
        p = p + gg.scale_y_log10()

    return p


# ---------------------------------------------------------------------------
# plot_genes_by_group
# ---------------------------------------------------------------------------


def plot_genes_by_group(
    adata: ad.AnnData,
    markers: Sequence[str],
    group_cells_by: str = "cluster",
    reduction_method: str = "UMAP",
    norm_method: str = "log",
    lower_threshold: float = 0.0,
    max_size: float = 10,
    ordering_type: str = "cluster_row_col",
    axis_order: str = "group_marker",
    flip_percentage_mean: bool = False,
    pseudocount: float = 1.0,
    scale_max: float = 3.0,
    scale_min: float = -3.0,
):
    """Dot plot of mean expression and fraction expressing per group.

    Port of R ``plot_genes_by_group``. Matches R semantics including:

    - ``mean = mean(log(normalized_counts + pseudocount))`` — R applies a
      natural-log wrapper on top of the already-size-factor-normalised
      expression (``normalized_counts(norm_method="log")`` returns
      ``log10(counts/sf + 1)``).
    - ``percentage`` as a fraction (0–1) of cells with
      ``Expression > lower_threshold``.
    - Dot size is scaled to ``[0, max_size]``.
    - ``ordering_type`` = ``cluster_row_col`` (pheatmap Ward.D2 on
      correlation distance), ``maximal_on_diag``, or ``none``.
    - ``axis_order='marker_group'`` applies ``coord_flip``.
    - ``scale_max``/``scale_min`` clamp the mean.
    - ``flip_percentage_mean`` swaps which axis (major/minor) carries mean
      vs. percentage.
    """
    import ggplot2_py as gg
    from .preprocess import normalized_counts as _norm_counts
    import pheatmap as _ph

    if ordering_type not in {"cluster_row_col", "maximal_on_diag", "none"}:
        raise ValueError(
            f"ordering_type must be one of 'cluster_row_col', "
            f"'maximal_on_diag', or 'none', got {ordering_type!r}"
        )
    if axis_order not in {"group_marker", "marker_group"}:
        raise ValueError(
            f"axis_order must be 'group_marker' or 'marker_group', got {axis_order!r}"
        )
    if norm_method not in {"log", "size_only"}:
        raise ValueError(
            f"norm_method must be 'log' or 'size_only', got {norm_method!r}"
        )

    marker_ids = _gene_ids_for(adata, markers)
    if group_cells_by == "cluster":
        cell_group = _clusters(
            adata, reduction_method=reduction_method
        ).astype(str)
    elif group_cells_by == "partition":
        cell_group = _partitions(
            adata, reduction_method=reduction_method
        ).astype(str)
    elif group_cells_by in adata.obs.columns:
        cell_group = adata.obs[group_cells_by].astype(str)
    else:
        raise KeyError(f"group_cells_by '{group_cells_by}' not valid")
    if cell_group.nunique() < 2:
        raise ValueError(
            "Only one type in group_cells_by. plot_genes_by_group needs >=2."
        )

    # R: exprs_mat = normalized_counts(cds, norm_method)[gene_ids, ]  (gene × cell)
    #    then melt; we do the algebra directly on the dense subset for speed.
    gene_idx = [adata.var_names.get_loc(g) for g in marker_ids]
    expr_sparse = _norm_counts(adata, norm_method=norm_method, pseudocount=1.0)
    expr_gene_cell = expr_sparse[:, gene_idx]
    expr = np.asarray(
        expr_gene_cell.todense() if sp.issparse(expr_gene_cell) else expr_gene_cell,
        dtype=float,
    )  # (cells, markers)

    # R:  mean(log(Expression + pseudocount)),  percentage = frac(E > lower_threshold).
    log_wrapped = np.log(expr + pseudocount)
    binary = (expr > float(lower_threshold)).astype(float)

    short_name_lookup = (
        adata.var["gene_short_name"].astype(str).to_dict()
        if "gene_short_name" in adata.var.columns
        else {g: g for g in adata.var_names}
    )

    group_arr = cell_group.to_numpy()
    groups = list(pd.unique(cell_group))
    rows = []
    for gi, gid in enumerate(marker_ids):
        feat = short_name_lookup.get(gid, gid)
        for grp in groups:
            mask = group_arr == grp
            n = int(mask.sum())
            if n == 0:
                continue
            rows.append(
                {
                    "Gene": feat,
                    "gene_id": gid,
                    "Group": str(grp),
                    "mean": float(log_wrapped[mask, gi].mean()),
                    "percentage": float(binary[mask, gi].sum() / n),
                }
            )
    exp_val = pd.DataFrame(rows)
    exp_val["mean"] = exp_val["mean"].clip(
        lower=float(scale_min), upper=float(scale_max)
    )

    # Wide matrix for clustering — R uses dcast(Group ~ Gene).
    value_var = "mean" if not flip_percentage_mean else "percentage"
    res = exp_val.pivot(index="Group", columns="Gene", values=value_var)

    # Orderings.
    if ordering_type == "cluster_row_col":
        ph = _ph.pheatmap(
            res,
            cluster_cols=True, cluster_rows=True,
            clustering_method="ward.D2",
            silent=True,
        )
        gene_levels = [list(res.columns)[i] for i in ph.tree_col.order]
        group_levels = [list(res.index)[i] for i in ph.tree_row.order]
    elif ordering_type == "maximal_on_diag":
        # Which row holds the per-column max? (R: dplyr::mutate is_max = max == value)
        with_max = exp_val.assign(
            max_value=exp_val.groupby("Gene")[value_var].transform("max"),
        )
        with_max["is_max"] = with_max[value_var] == with_max["max_value"]
        group_ord_df = (
            with_max[with_max["is_max"]]
            .groupby("Group", as_index=False).size()
            .rename(columns={"size": "num_genes"})
        )
        ordered_rows = (
            with_max.merge(group_ord_df, on="Group", how="left")
            .sort_values(
                ["num_genes", "max_value"], ascending=[False, False],
            )
        )
        group_levels = list(pd.unique(ordered_rows["Group"]))
        gene_levels = (
            ordered_rows[ordered_rows["is_max"]]
            .drop_duplicates("Gene", keep="first")
            .assign(
                Group=lambda d: pd.Categorical(
                    d["Group"], categories=group_levels, ordered=True,
                ),
            )
            .sort_values(["Group", "max_value"], ascending=[True, False])
            ["Gene"].tolist()
        )
    else:  # "none" — preserve input marker order.
        gene_levels = [short_name_lookup.get(g, g) for g in marker_ids]
        group_levels = groups

    exp_val["Gene"] = pd.Categorical(
        exp_val["Gene"], categories=gene_levels, ordered=True,
    )
    exp_val["Group"] = pd.Categorical(
        exp_val["Group"], categories=group_levels, ordered=True,
    )

    if flip_percentage_mean:
        # R: color=percentage, size=mean (with size legend label "log(mean + pc)").
        size_aes, color_aes = "mean", "percentage"
        size_name = f"log(mean + {pseudocount})"
        color_name = "percentage"
    else:
        size_aes, color_aes = "percentage", "mean"
        size_name = "percentage"
        color_name = f"log(mean + {pseudocount})"

    p = (
        gg.ggplot(exp_val, gg.aes(y="Gene", x="Group"))
        + gg.geom_point(gg.aes(color=color_aes, size=size_aes))
        + gg.scale_color_viridis_c(name=color_name)
        + gg.scale_size(name=size_name, range=[0.0, float(max_size)])
        + gg.theme_bw()
        + gg.theme(axis_text_x=gg.element_text(angle=30, hjust=1))
    )
    if group_cells_by == "cluster":
        p = p + gg.xlab("Cluster")
    elif group_cells_by == "partition":
        p = p + gg.xlab("Partition")
    else:
        p = p + gg.xlab(group_cells_by)
    p = p + gg.ylab("Gene")
    if axis_order == "marker_group":
        p = p + gg.coord_flip()
    return p
