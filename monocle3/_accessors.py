"""Module-level accessors that mirror R S4 generics.

Per ``monocle3_porting_essential_suggestions.md`` §1, the R generics
``exprs``, ``pData``, ``fData``, ``counts``, ``reducedDims`` are *not*
ported. The one kept here (``size_factors``) is exposed as a module-level
function rather than a property on a bespoke class.
"""

from __future__ import annotations

import anndata as ad
import pandas as pd

__all__ = ["size_factors"]


def size_factors(adata: ad.AnnData) -> pd.Series:
    """Return per-cell ``Size_Factor`` as a ``pandas.Series`` indexed by
    cell name.

    Matches R ``methods-cell_data_set.R:36-40`` which returns a named numeric
    vector (``names(sf) <- colnames(counts(cds))``). Downstream callers that
    want the raw array can call ``.to_numpy()`` or pass through pandas ops.

    Raises
    ------
    KeyError
        If ``estimate_size_factors`` has not been called.
    """
    if "Size_Factor" not in adata.obs.columns:
        raise KeyError(
            "Size_Factor not found on adata.obs; call estimate_size_factors first."
        )
    return pd.Series(
        adata.obs["Size_Factor"].to_numpy(dtype=float),
        index=adata.obs_names.astype(str),
        name="Size_Factor",
    )
