"""plot_cells_3d — 3D trajectory visualisation.

This is the ONLY location in ``monocle3-python`` where ``plotly`` is
imported, matching essential-suggestions §3.4. Implemented as a direct
translation of the R ``plotly::plot_ly(...) %>% add_trace(...) %>%
layout(...)`` pipeline.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from ._utils import get_monocle_uns
from .cluster_cells import clusters as _clusters, partitions as _partitions
from .order_cells import pseudotime as _pseudotime

__all__ = ["plot_cells_3d"]


def plot_cells_3d(
    adata: ad.AnnData,
    x: int = 1,
    y: int = 2,
    z: int = 3,
    reduction_method: str = "UMAP",
    color_cells_by: str = "partition",
    color_palette: Any | None = None,
    color_scale: str = "Viridis",
    cell_size: float = 2,
    trajectory_graph_color: str = "#c5c5c5",
    trajectory_graph_segment_size: float = 5,
    show_trajectory_graph: bool = True,
):
    """Return a ``plotly.graph_objects.Figure`` showing 3D cell embedding."""
    import plotly.graph_objects as go

    key = f"X_{reduction_method.lower()}"
    if key not in adata.obsm:
        raise KeyError(
            f"No {reduction_method} reduction. Run reduce_dimension(max_components=3) first."
        )
    coords = np.asarray(adata.obsm[key])
    if coords.shape[1] < max(x, y, z):
        raise ValueError(
            "reduction_method must have at least 3 components for plot_cells_3d"
        )

    if color_cells_by == "cluster":
        colors = _clusters(adata, reduction_method=reduction_method).astype(str)
    elif color_cells_by == "partition":
        colors = _partitions(adata, reduction_method=reduction_method).astype(str)
    elif color_cells_by == "pseudotime":
        colors = _pseudotime(adata, reduction_method=reduction_method)
    else:
        colors = adata.obs[color_cells_by]

    fig = go.Figure()
    if pd.api.types.is_numeric_dtype(colors):
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, x - 1],
                y=coords[:, y - 1],
                z=coords[:, z - 1],
                mode="markers",
                marker=dict(
                    size=cell_size,
                    color=colors.to_numpy(),
                    colorscale=color_scale,
                    showscale=True,
                ),
                name=color_cells_by,
            )
        )
    else:
        for level in pd.unique(colors):
            mask = np.asarray(colors) == level
            fig.add_trace(
                go.Scatter3d(
                    x=coords[mask, x - 1],
                    y=coords[mask, y - 1],
                    z=coords[mask, z - 1],
                    mode="markers",
                    marker=dict(size=cell_size),
                    name=str(level),
                )
            )

    if show_trajectory_graph:
        aux = get_monocle_uns(adata, "principal_graph_aux", reduction_method,
                             default={})
        pg = get_monocle_uns(adata, "principal_graph", reduction_method)
        if aux and pg is not None and pg.vcount() > 0:
            Y = aux["dp_mst"]
            name_to_idx = {n: i for i, n in enumerate(pg.vs["name"])}
            xs, ys, zs = [], [], []
            for a, b in pg.get_edgelist():
                src = pg.vs[a]["name"]
                tgt = pg.vs[b]["name"]
                if src not in name_to_idx or tgt not in name_to_idx:
                    continue
                si, ti = name_to_idx[src], name_to_idx[tgt]
                xs.extend([Y[x - 1, si], Y[x - 1, ti], None])
                ys.extend([Y[y - 1, si], Y[y - 1, ti], None])
                zs.extend([Y[z - 1, si], Y[z - 1, ti], None])
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines",
                    line=dict(
                        color=trajectory_graph_color,
                        width=trajectory_graph_segment_size,
                    ),
                    name="principal_graph",
                )
            )

    fig.update_layout(
        scene=dict(
            xaxis_title=f"{reduction_method} {x}",
            yaxis_title=f"{reduction_method} {y}",
            zaxis_title=f"{reduction_method} {z}",
        ),
        showlegend=True,
    )
    return fig
