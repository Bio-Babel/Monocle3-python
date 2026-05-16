"""Recipe: canonical Monocle3 pseudotime trajectory on an AnnData.

Mirrors workflows/basic_trajectory.yaml step-by-step. Requires a raw-counts
AnnData and at least one root barcode supplied by the caller (Monocle3's
interactive UI is not available outside R Shiny).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import anndata as ad

import monocle3


def main(
    adata: ad.AnnData,
    *,
    root_cells: Sequence[str],
    num_dim: int = 50,
    reduction_method: str = "UMAP",
    out_path: Path | None = None,
) -> ad.AnnData:
    """Run the full 6-step pipeline; return the mutated AnnData.

    Parameters
    ----------
    adata : AnnData
        Cells × genes with raw UMI counts in ``adata.X``.
    root_cells : sequence of str
        Cell barcodes used to root pseudotime.
    num_dim : int, default 50
        PCA dimensions for preprocess_cds.
    reduction_method : str, default "UMAP"
        Reduction passed to reduce_dimension / cluster_cells / order_cells.
    out_path : Path, optional
        If given, write the resulting AnnData to this path as h5ad.

    Returns
    -------
    AnnData
        With obs.Size_Factor, obs.monocle3_pseudotime, obs.monocle3_clusters,
        obs.monocle3_partitions, obsm.X_pca, obsm.X_umap, and uns.monocle3
        populated. The pseudotime column is `monocle3_pseudotime` (not
        `pseudotime`), matching `monocle3.order_cells._PSEUDOTIME_COL`.
    """
    monocle3.estimate_size_factors(adata)
    monocle3.preprocess_cds(adata, num_dim=num_dim)
    monocle3.reduce_dimension(adata, reduction_method=reduction_method)
    monocle3.cluster_cells(adata, reduction_method=reduction_method)
    monocle3.learn_graph(adata)
    monocle3.order_cells(
        adata,
        reduction_method=reduction_method,
        root_cells=list(root_cells),
    )
    if out_path is not None:
        adata.write_h5ad(out_path)
    return adata


if __name__ == "__main__":
    # Smoke entry: load a tiny built-in dataset and a known root barcode.
    a = monocle3.load_packer_embryo()  # ships with the package
    main(a, root_cells=[a.obs_names[0]])
