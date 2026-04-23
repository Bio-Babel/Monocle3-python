"""plot_cells_3d — 3D trajectory visualisation.

This is the ONLY location in ``monocle3-python`` where ``plotly`` is
imported, matching essential-suggestions §3.4. Implemented as a direct
translation of the R ``plotly::plot_ly(...) %>% add_trace(...) %>%
layout(...)`` pipeline (``plotting.R:54-329``).
"""

from __future__ import annotations

from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from ._utils import get_monocle_uns
from .cluster_cells import clusters as _clusters, partitions as _partitions
from .order_cells import pseudotime as _pseudotime

__all__ = ["plot_cells_3d"]


def _set2_palette(n: int) -> list[str]:
    """R equivalent: ``RColorBrewer::brewer.pal(n, "Set2")``. ``scales.brewer_pal``
    is a faithful port; for ``n > 8`` we recycle the 8 available Set2 colors
    the way R does once the RColorBrewer warning fires and plotly recycles.
    """
    import scales

    raw = scales.brewer_pal(palette="Set2")(max(3, n))
    # scales returns None entries when n exceeds the palette.
    base = [c for c in raw if c is not None]
    if n <= len(base):
        return base[:n]
    # Recycle — matches plot_ly's color-recycling when colors < levels.
    return [base[i % len(base)] for i in range(n)]


def plot_cells_3d(
    adata: ad.AnnData,
    dims: Sequence[int] = (1, 2, 3),
    reduction_method: str = "UMAP",
    color_cells_by: str = "cluster",
    genes: Sequence[str] | pd.DataFrame | None = None,
    show_trajectory_graph: bool = True,
    trajectory_graph_color: str = "black",
    trajectory_graph_segment_size: float = 5,
    norm_method: str = "log",
    color_palette: Any | None = None,
    color_scale: str = "Viridis",
    cell_size: float = 25,
    alpha: float = 1.0,
    min_expr: float = 0.1,
    # Back-compat shims: earlier Python signature used x/y/z singletons.
    x: int | None = None,
    y: int | None = None,
    z: int | None = None,
):
    """Return a ``plotly.graph_objects.Figure`` of the 3D cell embedding.

    Faithful to R ``plot_cells_3d``: categorical ``color_cells_by`` uses the
    supplied ``color_palette`` (default ``RColorBrewer::brewer.pal(N, "Set2")``);
    numeric ``color_cells_by`` and ``genes`` use the ``color_scale`` viridis
    colorbar; trajectory edges are drawn with ``trajectory_graph_color`` /
    ``trajectory_graph_segment_size``; NA-valued expression is rendered in
    lightgrey with opacity 0.4.
    """
    import plotly.graph_objects as go

    if norm_method not in {"log", "size_only"}:
        raise ValueError(f"norm_method must be 'log' or 'size_only', got {norm_method!r}")
    if x is not None or y is not None or z is not None:
        dims = (x or dims[0], y or dims[1], z or dims[2])
    if len(dims) != 3:
        raise ValueError("dims must have 3 entries for plot_cells_3d")
    dx, dy, dz = int(dims[0]), int(dims[1]), int(dims[2])

    key = f"X_{reduction_method.lower()}"
    if key not in adata.obsm:
        raise KeyError(
            f"No {reduction_method} reduction. Run reduce_dimension(max_components=3) first."
        )
    coords = np.asarray(adata.obsm[key])
    if coords.shape[1] < max(dx, dy, dz):
        raise ValueError(
            "reduction_method must have at least 3 components for plot_cells_3d"
        )

    # ---- R lines 121-137: pick cell_color ----
    if color_cells_by == "cluster":
        cell_color = _clusters(
            adata, reduction_method=reduction_method
        )
    elif color_cells_by == "partition":
        cell_color = _partitions(
            adata, reduction_method=reduction_method
        )
    elif color_cells_by == "pseudotime":
        cell_color = _pseudotime(
            adata, reduction_method=reduction_method
        )
    else:
        if color_cells_by not in adata.obs.columns:
            raise KeyError(
                f"color_cells_by '{color_cells_by}' not in obs"
            )
        cell_color = adata.obs[color_cells_by]

    pts_x = coords[:, dx - 1]
    pts_y = coords[:, dy - 1]
    pts_z = coords[:, dz - 1]

    fig = go.Figure()

    # ---- R lines 139-194: optional marker-gene coloring ----
    if genes is not None:
        from .plotting import _gene_ids_for, _is_module_genes_df
        from .cluster_genes import aggregate_gene_expression as _agg

        if _is_module_genes_df(genes):
            agg = _agg(
                adata, gene_group_df=genes, norm_method=norm_method,
                gene_agg_fun="mean", scale_agg_values=False,
            )  # modules × cells
            # R: log10(agg+1) then t(scale(...)) then clip [-2, 2].
            m = np.log10(np.asarray(agg.values, dtype=float) + 1.0)
            row_mean = m.mean(axis=1, keepdims=True)
            row_std = m.std(axis=1, keepdims=True, ddof=1)
            row_std = np.where(row_std == 0, 1.0, row_std)
            m = np.clip((m - row_mean) / row_std, -2.0, 2.0)
            # Aggregate across module panels → average across modules for the
            # 3D scatter (R's 3D marker-gene code uses single-module agg_mat).
            expr_per_cell = m.mean(axis=0)
        else:
            gene_ids = _gene_ids_for(adata, genes)
            col_idx = [adata.var_names.get_loc(g) for g in gene_ids]
            X = adata.X[:, col_idx]
            mat = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)
            sf = adata.obs["Size_Factor"].to_numpy(dtype=float)
            mat = mat / sf[:, None]
            expr_per_cell = mat.mean(axis=1)  # R: melt per-gene then add_markers per facet; we colour by mean across requested genes

        # R lines 199-235: NA → lightgrey, valid → color by log10(expr + min_expr) if norm_method=log else raw.
        valid = np.isfinite(expr_per_cell) & (expr_per_cell >= min_expr)
        if norm_method == "log":
            val = np.log10(np.where(valid, expr_per_cell, 0) + min_expr)
            colorbar_title = "Log10\nExpression"
        else:
            val = np.where(valid, expr_per_cell, 0.0)
            colorbar_title = "Expression"
        fig.add_trace(
            go.Scatter3d(
                x=pts_x[valid], y=pts_y[valid], z=pts_z[valid],
                mode="markers",
                marker=dict(
                    size=cell_size, opacity=alpha,
                    color=val[valid], colorscale=color_scale,
                    colorbar=dict(title=colorbar_title, len=0.5),
                ),
                name=colorbar_title,
            )
        )
        if (~valid).any():
            fig.add_trace(
                go.Scatter3d(
                    x=pts_x[~valid], y=pts_y[~valid], z=pts_z[~valid],
                    mode="markers",
                    marker=dict(
                        size=cell_size, color="lightgrey", opacity=0.4,
                    ),
                    showlegend=False,
                )
            )
    elif cell_color is not None and pd.api.types.is_numeric_dtype(cell_color):
        # R lines 256-267: numeric → Viridis colorbar.
        fig.add_trace(
            go.Scatter3d(
                x=pts_x, y=pts_y, z=pts_z,
                mode="markers",
                marker=dict(
                    size=cell_size, opacity=alpha,
                    color=np.asarray(cell_color, dtype=float),
                    colorscale=color_scale,
                    colorbar=dict(title=color_cells_by, len=0.5),
                ),
                name=color_cells_by,
            )
        )
    else:
        # R lines 237-278: categorical → Set2 (or user palette). One trace per
        # level keeps plotly's categorical legend working.
        cats = (
            list(cell_color.cat.categories)
            if hasattr(cell_color, "cat")
            else list(pd.unique(cell_color))
        )
        palette = (
            list(color_palette) if color_palette is not None
            else _set2_palette(len(cats))
        )
        level_arr = np.asarray(cell_color)
        for i, lvl in enumerate(cats):
            mask = level_arr == lvl
            if not mask.any():
                continue
            color = palette[i % len(palette)] if palette else None
            fig.add_trace(
                go.Scatter3d(
                    x=pts_x[mask], y=pts_y[mask], z=pts_z[mask],
                    mode="markers",
                    marker=dict(
                        size=cell_size, color=color, opacity=alpha,
                    ),
                    name=str(lvl),
                )
            )

    # ---- R lines 285-326: trajectory edges ----
    if show_trajectory_graph:
        aux = get_monocle_uns(
            adata, "principal_graph_aux", reduction_method, default={},
        )
        pg = get_monocle_uns(adata, "principal_graph", reduction_method)
        if aux and pg is not None and pg.vcount() > 0:
            Y = np.asarray(aux["dp_mst"])
            name_to_idx = {n: i for i, n in enumerate(pg.vs["name"])}
            # R iterates edges, one add_trace per edge; plotly.py renders the
            # same line segments as a single trace with None-separated pairs.
            xs, ys, zs = [], [], []
            for a, b in pg.get_edgelist():
                src_i = name_to_idx.get(pg.vs[a]["name"])
                tgt_i = name_to_idx.get(pg.vs[b]["name"])
                if src_i is None or tgt_i is None:
                    continue
                xs.extend([Y[dx - 1, src_i], Y[dx - 1, tgt_i], None])
                ys.extend([Y[dy - 1, src_i], Y[dy - 1, tgt_i], None])
                zs.extend([Y[dz - 1, src_i], Y[dz - 1, tgt_i], None])
            if xs:
                fig.add_trace(
                    go.Scatter3d(
                        x=xs, y=ys, z=zs,
                        mode="lines",
                        line=dict(
                            color=trajectory_graph_color,
                            width=trajectory_graph_segment_size,
                        ),
                        showlegend=False,
                        name="principal_graph",
                    )
                )

    # R line 281-283: axis titles use "Component <dim>".
    fig.update_layout(
        scene=dict(
            xaxis_title=f"Component {dx}",
            yaxis_title=f"Component {dy}",
            zaxis_title=f"Component {dz}",
        ),
        showlegend=True,
    )
    return fig
