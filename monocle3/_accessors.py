"""Module-level accessors that mirror R S4 generics.

Per ``monocle3_porting_essential_suggestions.md`` §1, the R generics
``exprs``, ``pData``, ``fData``, ``counts``, ``reducedDims`` are *not*
ported. The ones we do expose (``clusters``, ``partitions``,
``pseudotime``, ``principal_graph``, ``size_factors``,
``normalized_counts``) are kept as module-level functions (not properties
on a bespoke class).
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from ._utils import ensure_monocle_uns, get_monocle_uns

__all__ = [
    "size_factors",
    "counts_row_order",
    "set_cds_row_order_matrix",
    "set_matrix_control",
    "saveRDS",
]


def size_factors(adata: ad.AnnData) -> np.ndarray:
    """Return the per-cell ``Size_Factor`` column as a 1D array.

    Raises
    ------
    KeyError
        If ``estimate_size_factors`` has not been called.
    """
    if "Size_Factor" not in adata.obs.columns:
        raise KeyError(
            "Size_Factor not found on adata.obs; call estimate_size_factors first."
        )
    return adata.obs["Size_Factor"].to_numpy(dtype=float)


def counts_row_order(adata: ad.AnnData) -> None:
    """Return the row-order counts matrix (BPCells-only; no-op for AnnData).

    The R method returns a BPCells IterableMatrix representing the
    transposed counts for faster row iteration. AnnData stores the counts
    in memory, so this accessor always returns ``None``.
    """
    del adata
    return None


def set_cds_row_order_matrix(
    adata: ad.AnnData,
    matrix: Any | None = None,
) -> ad.AnnData:
    """Attach a gene-major counts matrix under the monocle3 uns namespace.

    The R version materializes a BPCells row-order view. In the Python
    port we simply park a user-supplied matrix under
    ``adata.uns["monocle3"]["row_order_matrix"]`` so downstream callers
    can read it; most callers pass ``None`` and this is a no-op.
    """
    if matrix is not None:
        ensure_monocle_uns(adata)["row_order_matrix"] = matrix
    return adata


def set_matrix_control(nn_control: dict | None = None, **kwargs: Any) -> dict:
    """Return the nn_control dict merged with any provided overrides."""
    base: dict = {} if nn_control is None else dict(nn_control)
    base.update(kwargs)
    return base


def saveRDS(obj: Any, file: str | None = None, **kwargs: Any) -> None:
    """R ``saveRDS`` is not available in Python.

    Per the essential-suggestions doc, persist an ``AnnData`` via
    ``adata.write_h5ad(path)``. This shim raises a clear error so R code
    ported mechanically surfaces the missing idiom at the right point.
    """
    del obj, file, kwargs
    raise NotImplementedError(
        "saveRDS is not available in Python. Use adata.write_h5ad(path) "
        "to persist an AnnData."
    )
