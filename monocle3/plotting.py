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
from .order_cells import pseudotime as _pseudotime

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


def _trajectory_edges(
    adata: ad.AnnData,
    reduction_method: str,
    x: int = 1,
    y: int = 2,
) -> pd.DataFrame | None:
    aux = get_monocle_uns(adata, "principal_graph_aux", reduction_method,
                         default={})
    pg = get_monocle_uns(adata, "principal_graph", reduction_method)
    if not aux or pg is None or pg.vcount() == 0:
        return None
    Y = aux["dp_mst"]  # D × K
    dim_df = pd.DataFrame(
        {
            "prin_graph_dim_1": Y[x - 1],
            "prin_graph_dim_2": Y[y - 1],
            "sample_name": pg.vs["name"],
        }
    )
    name_to_idx = {n: i for i, n in enumerate(dim_df["sample_name"].tolist())}
    edges = pg.get_edgelist()
    rows = []
    for a, b in edges:
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

    if rasterize:
        warnings.warn(
            "rasterize=True is not supported in v1 (ggrastr-python port pending); "
            "falling back to vector geom_point.",
            stacklevel=2,
        )

    data_df = _cells_coord_df(adata, reduction_method=reduction_method, x=x, y=y)

    # Assign group labels.
    if group_cells_by == "cluster":
        try:
            data_df["cell_group"] = _clusters(adata, reduction_method=reduction_method).reindex(data_df.index).astype(str).values
        except KeyError:
            data_df["cell_group"] = "1"
    elif group_cells_by == "partition":
        try:
            data_df["cell_group"] = _partitions(adata, reduction_method=reduction_method).reindex(data_df.index).astype(str).values
        except KeyError:
            data_df["cell_group"] = "1"
    else:
        if group_cells_by in adata.obs.columns:
            data_df["cell_group"] = adata.obs[group_cells_by].astype(str).to_numpy()
        else:
            raise KeyError(f"group_cells_by '{group_cells_by}' not in obs")

    # Determine the colour channel.
    if genes is not None:
        gene_ids = _gene_ids_for(adata, genes)
        expr_df = _log_norm_expr(
            adata, gene_ids, norm_method=norm_method, min_expr=min_expr
        )
        gene_short = (
            adata.var["gene_short_name"].astype(str)
            if "gene_short_name" in adata.var.columns
            else pd.Series(adata.var_names, index=adata.var_names)
        )
        # Long form so we can facet.
        long = expr_df.reset_index().melt(
            id_vars="barcode" if expr_df.index.name == "barcode" else adata.obs.index.name or "index",
            var_name="gene_id",
            value_name="expression",
        )
        long = long.rename(columns={long.columns[0]: "sample_name"})
        long["feature_label"] = long["gene_id"].map(gene_short)
        long["data_dim_1"] = data_df.loc[long["sample_name"], "data_dim_1"].to_numpy()
        long["data_dim_2"] = data_df.loc[long["sample_name"], "data_dim_2"].to_numpy()
        # Floor expression at min_expr.
        long.loc[long["expression"] < min_expr, "expression"] = np.nan

        p = (
            gg.ggplot(long, gg.aes(x="data_dim_1", y="data_dim_2"))
            + gg.geom_point(gg.aes(color="expression"), size=cell_size, alpha=alpha)
            + gg.scale_color_viridis_c(option="C", na_value="gray80")
            + gg.facet_wrap("feature_label")
            + gg.theme_bw()
            + gg.xlab(f"{reduction_method} {x}")
            + gg.ylab(f"{reduction_method} {y}")
        )
    else:
        if color_cells_by == "cluster":
            data_df["cell_color"] = _clusters(
                adata, reduction_method=reduction_method
            ).reindex(data_df.index).astype(str).values
        elif color_cells_by == "partition":
            data_df["cell_color"] = _partitions(
                adata, reduction_method=reduction_method
            ).reindex(data_df.index).astype(str).values
        elif color_cells_by == "pseudotime":
            data_df["cell_color"] = _pseudotime(
                adata, reduction_method=reduction_method
            ).reindex(data_df.index).to_numpy()
        elif color_cells_by in adata.obs.columns:
            data_df["cell_color"] = adata.obs[color_cells_by].astype(str).to_numpy()
        else:
            raise KeyError(
                f"color_cells_by '{color_cells_by}' not valid"
            )

        p = gg.ggplot(data_df, gg.aes(x="data_dim_1", y="data_dim_2"))
        p = p + gg.geom_point(
            gg.aes(color="cell_color"), size=cell_size, alpha=alpha
        )
        if color_cells_by == "pseudotime":
            p = p + gg.scale_color_viridis_c(option="C")
        p = p + gg.theme_bw()
        p = p + gg.xlab(f"{reduction_method} {x}")
        p = p + gg.ylab(f"{reduction_method} {y}")

    # Add trajectory edges if available.
    if show_trajectory_graph:
        edges = _trajectory_edges(adata, reduction_method=reduction_method, x=x, y=y)
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
    group_cells_by: str = "cluster",
    ncol: int = 1,
    plot_as_count: bool = False,
    nrow: int | None = None,
    normalize: bool = True,
    min_expr: float = 0,
    reduction_method: str = "UMAP",
):
    """Bar plot of % cells with non-zero expression per gene per group."""
    import ggplot2_py as gg

    if group_cells_by == "cluster":
        group = _clusters(adata, reduction_method=reduction_method).astype(str)
    elif group_cells_by == "partition":
        group = _partitions(adata, reduction_method=reduction_method).astype(str)
    elif group_cells_by in adata.obs.columns:
        group = adata.obs[group_cells_by].astype(str)
    else:
        raise KeyError(f"group_cells_by '{group_cells_by}' not valid")

    X = adata.X
    if sp.issparse(X):
        mat = np.asarray(X.todense())
    else:
        mat = np.asarray(X, dtype=float)

    rows = []
    group_arr = group.to_numpy()
    gene_short = (
        adata.var["gene_short_name"].astype(str).tolist()
        if "gene_short_name" in adata.var.columns
        else list(adata.var_names)
    )
    for j, gid in enumerate(adata.var_names):
        for grp in pd.unique(group_arr):
            mask = group_arr == grp
            n = mask.sum()
            if n == 0:
                continue
            pos = float((mat[mask, j] > min_expr).sum())
            pct = 100.0 * pos / n if normalize else pos
            rows.append(
                {"gene_id": gid, "feature_label": gene_short[j],
                 "cell_group": grp, "pct_positive": pct}
            )
    df = pd.DataFrame(rows)
    p = (
        gg.ggplot(df, gg.aes(x="cell_group", y="pct_positive", fill="cell_group"))
        + gg.geom_bar(stat="identity")
        + gg.facet_wrap("feature_label", ncol=int(ncol))
        + gg.theme_bw()
        + gg.xlab(group_cells_by)
        + gg.ylab("% cells positive")
    )
    return p


# ---------------------------------------------------------------------------
# plot_genes_in_pseudotime
# ---------------------------------------------------------------------------


def plot_genes_in_pseudotime(
    adata_subset: ad.AnnData,
    min_expr: float = 0.5,
    cell_size: float = 0.75,
    nrow: int | None = None,
    ncol: int = 1,
    panel_order: Sequence[str] | None = None,
    color_cells_by: str = "cluster",
    trend_formula: str = "~ splines::ns(pseudotime, df=3)",
    label_by_short_name: bool = True,
    vertical_jitter: float | None = None,
    horizontal_jitter: float | None = None,
    reduction_method: str = "UMAP",
):
    """Line plot of gene expression along pseudotime."""
    import ggplot2_py as gg

    if "monocle3_pseudotime" not in adata_subset.obs.columns:
        raise KeyError("Run order_cells first.")

    pseudo = adata_subset.obs["monocle3_pseudotime"].to_numpy(dtype=float)
    X = adata_subset.X
    if sp.issparse(X):
        mat = np.asarray(X.todense())
    else:
        mat = np.asarray(X, dtype=float)
    sf = adata_subset.obs["Size_Factor"].to_numpy(dtype=float)
    norm = mat / sf[:, None]

    short_names = (
        adata_subset.var["gene_short_name"].astype(str).tolist()
        if "gene_short_name" in adata_subset.var.columns
        else list(adata_subset.var_names)
    )

    rows = []
    for j, gid in enumerate(adata_subset.var_names):
        feat = short_names[j] if label_by_short_name else gid
        for i, cell in enumerate(adata_subset.obs_names):
            rows.append(
                {
                    "pseudotime": pseudo[i],
                    "expression": norm[i, j],
                    "feature_label": feat,
                    "sample_name": cell,
                }
            )
    df = pd.DataFrame(rows)
    if color_cells_by in adata_subset.obs.columns:
        df["cell_color"] = (
            adata_subset.obs[color_cells_by].astype(str).loc[df["sample_name"]]
            .to_numpy()
        )

    df = df[df["expression"] >= min_expr]
    p = (
        gg.ggplot(df, gg.aes(x="pseudotime", y="expression"))
        + gg.geom_point(
            gg.aes(color="cell_color")
            if color_cells_by in adata_subset.obs.columns
            else None,
            size=cell_size,
        )
        + gg.facet_wrap("feature_label", ncol=int(ncol))
        + gg.theme_bw()
    )
    return p


# ---------------------------------------------------------------------------
# plot_genes_violin
# ---------------------------------------------------------------------------


def plot_genes_violin(
    adata_subset: ad.AnnData,
    group_cells_by: str = "cluster",
    min_expr: float = 0,
    nrow: int | None = None,
    ncol: int = 1,
    normalize: bool = True,
    log_scale: bool = True,
    pseudocount: float = 0.0,
    label_by_short_name: bool = True,
    relative_expr: bool = True,
    reduction_method: str = "UMAP",
):
    """Violin-style plot of per-cell-group gene expression."""
    import ggplot2_py as gg

    if group_cells_by == "cluster":
        group = _clusters(adata_subset, reduction_method=reduction_method).astype(str)
    elif group_cells_by == "partition":
        group = _partitions(adata_subset, reduction_method=reduction_method).astype(str)
    elif group_cells_by in adata_subset.obs.columns:
        group = adata_subset.obs[group_cells_by].astype(str)
    else:
        raise KeyError(f"group_cells_by '{group_cells_by}' not valid")

    X = adata_subset.X
    mat = np.asarray(X.todense() if sp.issparse(X) else X, dtype=float)
    sf = adata_subset.obs["Size_Factor"].to_numpy(dtype=float)
    if relative_expr:
        mat = mat / sf[:, None]
    if log_scale:
        mat = np.log10(mat + max(pseudocount, 1e-9))

    short_names = (
        adata_subset.var["gene_short_name"].astype(str).tolist()
        if "gene_short_name" in adata_subset.var.columns
        else list(adata_subset.var_names)
    )
    rows = []
    for j, gid in enumerate(adata_subset.var_names):
        feat = short_names[j] if label_by_short_name else gid
        for i in range(adata_subset.n_obs):
            rows.append(
                {
                    "feature_label": feat,
                    "cell_group": str(group.iloc[i]),
                    "expression": mat[i, j],
                }
            )
    df = pd.DataFrame(rows)
    df = df[df["expression"] >= min_expr]

    return (
        gg.ggplot(df, gg.aes(x="cell_group", y="expression", fill="cell_group"))
        + gg.geom_violin()
        + gg.geom_jitter(size=0.5, alpha=0.4)
        + gg.facet_wrap("feature_label", ncol=int(ncol))
        + gg.theme_bw()
    )


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
    axis_order: str = "marker_group",
    flip_percentage_mean: bool = False,
    pseudocount: float = 1.0,
    scale_max: float = 3.0,
    scale_min: float = -3.0,
):
    """Dot plot of mean expression and fraction expressing per group."""
    import ggplot2_py as gg

    marker_ids = _gene_ids_for(adata, markers)
    if group_cells_by == "cluster":
        group = _clusters(adata, reduction_method=reduction_method).astype(str)
    elif group_cells_by == "partition":
        group = _partitions(adata, reduction_method=reduction_method).astype(str)
    elif group_cells_by in adata.obs.columns:
        group = adata.obs[group_cells_by].astype(str)
    else:
        raise KeyError(f"group_cells_by '{group_cells_by}' not valid")

    gene_idx = [adata.var_names.get_loc(g) for g in marker_ids]
    X = adata.X[:, gene_idx]
    mat = np.asarray(X.todense() if sp.issparse(X) else X, dtype=float)
    sf = adata.obs["Size_Factor"].to_numpy(dtype=float)
    mat = mat / sf[:, None]
    if norm_method == "log":
        mat = np.log10(mat + pseudocount)
    mat_binary = (mat > lower_threshold).astype(float)

    groups = pd.unique(group)
    rows = []
    for gi, gid in enumerate(marker_ids):
        feat = gid
        if "gene_short_name" in adata.var.columns:
            feat = str(adata.var.loc[gid, "gene_short_name"])
        for grp in groups:
            mask = group.values == grp
            n = mask.sum()
            if n == 0:
                continue
            pct = 100.0 * mat_binary[mask, gi].sum() / n
            mean_expr = float(mat[mask, gi].mean())
            rows.append(
                {
                    "feature_label": feat,
                    "gene_id": gid,
                    "cell_group": str(grp),
                    "pct": pct,
                    "mean_expr": mean_expr,
                }
            )
    df = pd.DataFrame(rows)

    return (
        gg.ggplot(df, gg.aes(x="cell_group", y="feature_label"))
        + gg.geom_point(gg.aes(size="pct", color="mean_expr"))
        + gg.scale_color_viridis_c(option="D")
        + gg.theme_bw()
        + gg.xlab(group_cells_by)
        + gg.ylab("marker")
    )
