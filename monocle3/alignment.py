"""Alignment — port of R/alignment.R::align_cds.

Residual model subtraction → statsmodels OLS per PC.
Batch alignment → scanorama (Batchelor's MNN replacement, see
``monocle3_porting_essential_suggestions.md`` §2).
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import patsy
import scanorama

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
    scanorama_sigma: float = 15.0,
    scanorama_alpha: float = 0.10,
    scanorama_approx: bool = False,
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
        scanorama.correct removes between-batch effects (MNN-style).
    alignment_k : int, default 20
        k for the MNN stage.
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
    scanorama_sigma : float, default 15.0
        Scanorama Gaussian-kernel bandwidth for the MNN correction.
        No batchelor analogue; we expose scanorama's default.
    scanorama_alpha : float, default 0.10
        Scanorama alignment-strength scalar for the MNN correction.
        No batchelor analogue; we expose scanorama's default.
    scanorama_approx : bool, default False
        When ``False``, use exact nearest-neighbour search (closer to
        batchelor's behaviour at a speed cost). When ``True``, use
        scanorama's annoy-backed approximate search.
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
        coords = _scanorama_correct(
            coords,
            batch,
            k=int(alignment_k),
            sigma=float(scanorama_sigma),
            alpha=float(scanorama_alpha),
            approx=bool(scanorama_approx),
        )

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


def _scanorama_correct(
    coords: np.ndarray,
    batch: np.ndarray,
    k: int,
    sigma: float = 15.0,
    alpha: float = 0.10,
    approx: bool = False,
) -> np.ndarray:
    """MNN correction via scanorama.

    scanorama.correct operates on per-batch expression matrices; we feed
    the PC coords directly (each row a cell, each column a PC).

    ``sigma`` / ``alpha`` are scanorama-specific and exposed unchanged.
    ``approx=False`` flips scanorama from annoy to exact kNN so the
    MNN-pair computation has no index-approximation noise.

    Scanorama always internally L2-normalises its input rows ("cos.norm"),
    with no toggle to disable it, so the magnitude of corrections will
    differ from a batchelor::reducedMNN run on the same data; the
    direction of correction (which cells move towards which batch) is
    preserved.
    """
    unique_batches = pd.unique(pd.Series(batch))
    if len(unique_batches) < 2:
        return coords.astype(np.float64, copy=True)

    order: list[np.ndarray] = []
    datasets: list[np.ndarray] = []
    gene_lists: list[list[str]] = []
    for b in unique_batches:
        mask = batch == b
        order.append(np.where(mask)[0])
        datasets.append(coords[mask].astype(np.float64))
        gene_lists.append([f"pc_{i}" for i in range(coords.shape[1])])

    corrected, _ = scanorama.correct(
        datasets,
        gene_lists,
        knn=int(k),
        sigma=float(sigma),
        alpha=float(alpha),
        approx=bool(approx),
        return_dimred=False,
    )
    out = np.empty_like(coords, dtype=np.float64)
    for cidx, idx in zip(corrected, order):
        out[idx] = np.asarray(cidx.todense() if hasattr(cidx, "todense") else cidx)
    return out
