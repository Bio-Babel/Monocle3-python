"""Alignment — port of R/alignment.R::align_cds.

Residual model subtraction → statsmodels OLS per PC.
Batch alignment → :func:`monocle3._batchelor.reduced_mnn`, our pure-Python
port of ``batchelor::reducedMNN`` restricted to the monocle3 call.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import patsy

from ._batchelor import reduced_mnn
from ._utils import ensure_monocle_uns

__all__ = ["align_cds"]


def _sanitize_columns(df: pd.DataFrame, formula: str) -> tuple[pd.DataFrame, str]:
    """Rename dotted columns to underscore form so patsy accepts the formula."""
    mapping = {c: c.replace(".", "_") for c in df.columns if "." in c}
    if not mapping:
        return df, formula
    renamed = df.rename(columns=mapping)
    fmt = formula
    for old, new in mapping.items():
        fmt = fmt.replace(old, new)
    return renamed, fmt


def _residual_subtract(
    coords: np.ndarray,
    model_df: pd.DataFrame,
    formula: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a linear model to *coords* using *formula*, then subtract all
    non-intercept effects.

    Returns the cleaned coords and the fitted beta coefficients
    (non-intercept columns × PCs).
    """
    model_df, formula = _sanitize_columns(model_df, formula)
    design = patsy.dmatrix(
        formula, data=model_df, return_type="dataframe", NA_action="raise"
    )
    X = design.to_numpy(dtype=float)
    # lstsq handles rank-deficient design matrices gracefully.
    beta, *_ = np.linalg.lstsq(X, coords, rcond=None)

    col_names = design.design_info.column_names
    keep_mask = np.array(
        [name != "Intercept" and name != "(Intercept)" for name in col_names]
    )
    beta_no_int = np.where(np.isfinite(beta[keep_mask]), beta[keep_mask], 0.0)
    X_no_int = X[:, keep_mask]
    cleaned = coords - X_no_int @ beta_no_int
    return cleaned, beta_no_int


def align_cds(
    adata: ad.AnnData,
    preprocess_method: str = "PCA",
    alignment_group: str | None = None,
    alignment_k: int = 20,
    residual_model_formula_str: str | None = None,
    verbose: bool = False,
    build_nn_index: bool = False,
    nn_control: dict | None = None,
    **kwargs: Any,
) -> ad.AnnData:
    """Align ``adata.obsm[preprocess_method]`` to remove batch / covariates.

    Writes aligned coordinates to ``adata.obsm["X_aligned"]`` and a
    summary of the fit to ``adata.uns["monocle3"]["Aligned"]``.

    Parameters
    ----------
    adata : anndata.AnnData
    preprocess_method : {"PCA", "LSI"}
        Which reduced matrix to align. Must be present in ``obsm``.
    alignment_group : str, optional
        ``obs`` column with a categorical batch label. If provided,
        batch effects are removed by :func:`monocle3._batchelor.reduced_mnn`,
        a port of ``batchelor::reducedMNN`` matching the monocle3 R
        caller exactly (k from ``alignment_k``, all other parameters at
        batchelor defaults: ``ndist=3``, sequential merge order, exact
        KmknnParam-style kNN).
    alignment_k : int, default 20
        ``k`` forwarded to ``reduced_mnn``; matches batchelor's default.
    residual_model_formula_str : str, optional
        R-style formula without response, e.g.
        ``"~ bg.300.loading + batch"``. Uses patsy to build the design
        matrix, then subtracts the non-intercept effects.
    verbose : bool, default False
        Unused here.
    build_nn_index : bool, default False
        Ignored until a nn-index is built elsewhere.
    nn_control : dict, optional
        Passed through to ``make_nn_index`` when ``build_nn_index`` is
        set.
    **kwargs
        Unused (kept for R-signature parity).
    """
    del verbose, kwargs
    if preprocess_method not in {"PCA", "LSI"}:
        raise ValueError("preprocess_method must be 'PCA' or 'LSI'")

    key = f"X_{preprocess_method.lower()}"
    if key not in adata.obsm:
        raise KeyError(
            f"{preprocess_method} coordinates not found. "
            f"Run preprocess_cds(method='{preprocess_method}') first."
        )
    coords = np.asarray(adata.obsm[key], dtype=float).copy()

    np.random.seed(2016)
    beta = None

    if residual_model_formula_str is not None:
        coords, beta = _residual_subtract(
            coords, adata.obs, residual_model_formula_str
        )

    if alignment_group is not None:
        if alignment_group not in adata.obs.columns:
            raise KeyError(
                f"alignment_group '{alignment_group}' not found in adata.obs"
            )
        batch = adata.obs[alignment_group].astype(str).to_numpy()
        result = reduced_mnn(coords, batch, k=int(alignment_k))
        coords = result["corrected"]

    adata.obsm["X_aligned"] = coords.astype(np.float64)

    uns = ensure_monocle_uns(adata)
    uns.setdefault("Aligned", {})
    uns["Aligned"]["preprocess_method"] = preprocess_method
    uns["Aligned"]["alignment_group"] = alignment_group
    uns["Aligned"]["alignment_k"] = int(alignment_k)
    uns["Aligned"]["residual_model_formula_str"] = residual_model_formula_str
    uns["Aligned"]["beta"] = beta

    if build_nn_index:
        from .nearest_neighbors import make_nn_index, set_cds_nn_index

        nn_index = make_nn_index(coords, nn_control=nn_control)
        set_cds_nn_index(adata, reduction_method="Aligned", nn_index=nn_index)

    return adata
