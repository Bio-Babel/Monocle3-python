"""expr_models — port of R/expr_models.R.

Fits one GLM per gene over a shared model formula and provides tidy
summaries that mirror R's ``broom::tidy`` / ``glance`` via ``coefficient_table``
and ``evaluate_fits``. Family dispatch matches the R table (see
``monocle3_porting_essential_suggestions.md`` §2.1):

- ``"quasipoisson"`` (default) → ``statsmodels.GLM(Poisson, offset=log(SF))``
  with dispersion estimated post-fit (R's quasi-family is a dispersion
  scale of the Poisson fit).
- ``"poisson"`` / ``"gaussian"`` / ``"binomial"`` → ``statsmodels.GLM``
  with the matching family.
- ``"negbinomial"`` → ``statsmodels.discrete.count_model.NegativeBinomial``
  with ``offset=log(SF)``.
- ``"zipoisson"`` / ``"zinegbinomial"`` →
  ``statsmodels.discrete.count_model.ZeroInflatedPoisson``
  / ``ZeroInflatedNegativeBinomialP``.
- ``"mixed-negbinomial"`` raises ``NotImplementedError``.

The ``model`` column of the fit table holds the fitted
``statsmodels.regression.generalized_linear_model.GLMResultsWrapper``
(or equivalent) — downstream callers reach into its attributes. The
``model_summary`` column is ``None`` in this port; ``coefficient_table``
reads the estimates directly from the model object.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import patsy
from scipy import sparse as sp
from scipy.stats import chi2, false_discovery_control
import statsmodels.api as sm
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP,
    ZeroInflatedPoisson,
)
from statsmodels.discrete.discrete_model import NegativeBinomial as _NegBin

__all__ = [
    "fit_models",
    "coefficient_table",
    "compare_models",
    "evaluate_fits",
    "model_predictions",
]


_SUPPORTED_FAMILIES = {
    "quasipoisson",
    "poisson",
    "gaussian",
    "binomial",
    "negbinomial",
    "zipoisson",
    "zinegbinomial",
    "mixed-negbinomial",
}


_R_FORMULA_REPLACEMENTS = (
    ("splines::ns", "cr"),        # patsy provides `cr` (natural cubic splines).
    ("stats::offset", "offset"),
)


def _sanitize_columns(df: pd.DataFrame, formula: str) -> tuple[pd.DataFrame, str]:
    """Rename dotted columns to underscore form so patsy accepts the formula."""
    mapping = {c: c.replace(".", "_") for c in df.columns if "." in c}
    fmt = formula
    for old, new in _R_FORMULA_REPLACEMENTS:
        fmt = fmt.replace(old, new)
    if not mapping:
        return df, fmt
    renamed = df.rename(columns=mapping)
    # Prefer longer replacements first to avoid partial matches.
    for old in sorted(mapping, key=len, reverse=True):
        fmt = fmt.replace(old, mapping[old])
    return renamed, fmt


def _design(formula: str, data: pd.DataFrame) -> pd.DataFrame:
    """Build a patsy design matrix for ``model_formula_str``."""
    data_san, formula_san = _sanitize_columns(data, formula)
    dm = patsy.dmatrix(
        formula_san, data=data_san, return_type="dataframe", NA_action="raise"
    )
    return dm


def _expression_column(adata: ad.AnnData, gene_idx: int) -> np.ndarray:
    col = adata.X[:, gene_idx]
    if sp.issparse(col):
        col = np.asarray(col.todense()).ravel()
    else:
        col = np.asarray(col, dtype=float).ravel()
    return col.astype(np.float64)


def _fit_one(
    y: np.ndarray,
    design: np.ndarray,
    offset: np.ndarray,
    expression_family: str,
):
    """Fit a single-gene GLM. Returns (model_obj, status)."""
    try:
        if expression_family == "quasipoisson":
            mod = sm.GLM(
                y, design, family=sm.families.Poisson(), offset=offset
            ).fit()
            # Store dispersion estimate (Pearson chi²/df) for quasi-scaling.
            resid = mod.resid_pearson
            dof = max(mod.df_resid, 1)
            mod._quasi_dispersion = float(np.sum(resid ** 2) / dof)
            return mod, "OK"
        if expression_family == "poisson":
            mod = sm.GLM(
                y, design, family=sm.families.Poisson(), offset=offset
            ).fit()
            return mod, "OK"
        if expression_family == "gaussian":
            mod = sm.GLM(
                y, design, family=sm.families.Gaussian(), offset=offset
            ).fit()
            return mod, "OK"
        if expression_family == "binomial":
            mod = sm.GLM(
                y, design, family=sm.families.Binomial(), offset=offset
            ).fit()
            return mod, "OK"
        if expression_family == "negbinomial":
            mod = _NegBin(y, design, offset=offset).fit(disp=0, maxiter=100)
            return mod, "OK"
        if expression_family == "zipoisson":
            mod = ZeroInflatedPoisson(y, design, offset=offset).fit(disp=0, maxiter=100)
            return mod, "OK"
        if expression_family == "zinegbinomial":
            mod = ZeroInflatedNegativeBinomialP(
                y, design, offset=offset
            ).fit(disp=0, maxiter=100)
            return mod, "OK"
    except Exception:
        return None, "FAIL"
    return None, "FAIL"


def fit_models(
    adata: ad.AnnData,
    model_formula_str: str,
    expression_family: str = "quasipoisson",
    reduction_method: str = "UMAP",
    cores: int = 1,
    clean_model: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """One GLM per gene; returns a tidy DataFrame.

    Rows correspond to ``adata.var_names``; columns include every
    existing ``adata.var`` column, plus ``gene_id``, ``model``,
    ``model_summary`` (placeholder ``None``) and ``status``.
    """
    del cores, clean_model, verbose
    if expression_family not in _SUPPORTED_FAMILIES:
        raise ValueError(
            f"expression_family must be one of {sorted(_SUPPORTED_FAMILIES)}"
        )
    if expression_family == "mixed-negbinomial":
        raise NotImplementedError(
            "mixed-negbinomial requires lme4::glmer.nb; not available in Python"
        )

    # Assemble the obs frame with the canonical extra columns.
    obs = adata.obs.copy()
    if "Size_Factor" not in obs.columns:
        raise KeyError("Size_Factor not found; run estimate_size_factors first.")
    if "monocle3_clusters" in obs.columns:
        obs["cluster"] = obs["monocle3_clusters"]
    if "monocle3_partitions" in obs.columns:
        obs["partition"] = obs["monocle3_partitions"]
    if "monocle3_pseudotime" in obs.columns:
        obs["pseudotime"] = obs["monocle3_pseudotime"]

    formula = model_formula_str.lstrip("~").strip()
    # Prepend response placeholder name consistent with R.
    design = _design(f"~ {formula}", obs)
    design_matrix = design.to_numpy(dtype=float)

    offset = np.log(obs["Size_Factor"].to_numpy(dtype=float))

    gene_names = list(adata.var_names)
    gene_short_names = (
        adata.var["gene_short_name"].astype(str).tolist()
        if "gene_short_name" in adata.var.columns
        else gene_names.copy()
    )
    num_cells_expressed = (
        adata.var.get("num_cells_expressed", pd.Series(-1, index=adata.var_names))
        .to_numpy()
    )

    rows: list[dict[str, Any]] = []
    for i, gid in enumerate(gene_names):
        y = _expression_column(adata, i)
        model, status = _fit_one(y, design_matrix, offset, expression_family)
        row = {
            "id": gid,
            "gene_id": gid,
            "gene_short_name": gene_short_names[i],
            "num_cells_expressed": num_cells_expressed[i],
            "model": model,
            "model_summary": None,
            "status": status,
            "_design_info": design.design_info,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def coefficient_table(model_tbl: pd.DataFrame, pseudo_count: float = 0.01) -> pd.DataFrame:
    """Extract Wald coefficients and BH q-values from a fit table.

    Output columns mirror ``broom::tidy(model)``:
    ``term, estimate, std_err, test_val, p_value, normalized_effect,
    model_component, gene_id, gene_short_name, num_cells_expressed,
    status, q_value``.
    """
    frames: list[pd.DataFrame] = []
    for _, row in model_tbl.iterrows():
        model = row["model"]
        if model is None:
            continue
        try:
            params = np.asarray(model.params, dtype=float)
            std_err = np.asarray(model.bse, dtype=float)
            tvals = np.asarray(model.tvalues, dtype=float)
            pvals = np.asarray(model.pvalues, dtype=float)
            names = list(row["_design_info"].column_names)
        except Exception:
            continue
        # statsmodels inserts NB's dispersion alpha for NegativeBinomial; skip it.
        if params.size > len(names):
            params = params[: len(names)]
            std_err = std_err[: len(names)]
            tvals = tvals[: len(names)]
            pvals = pvals[: len(names)]
        intercept_idx = next(
            (k for k, n in enumerate(names) if n in ("Intercept", "(Intercept)")),
            None,
        )
        normalized_effect = np.zeros_like(params)
        if intercept_idx is not None:
            intercept = params[intercept_idx]
            family_inv = getattr(model, "family", None)
            if family_inv is not None and hasattr(family_inv, "link"):
                try:
                    link = family_inv.link.inverse
                except AttributeError:
                    link = lambda x: np.exp(x)
            else:
                link = lambda x: np.exp(x)
            with np.errstate(invalid="ignore", divide="ignore"):
                normalized_effect = np.log2(
                    (link(params + intercept) + pseudo_count)
                    / (link(intercept) + pseudo_count)
                )
            normalized_effect[intercept_idx] = 0.0

        df = pd.DataFrame(
            {
                "term": names,
                "estimate": params,
                "std_err": std_err,
                "test_val": tvals,
                "p_value": pvals,
                "normalized_effect": normalized_effect,
                "model_component": "count",
            }
        )
        df["gene_id"] = row["gene_id"]
        df["gene_short_name"] = row["gene_short_name"]
        df["num_cells_expressed"] = row["num_cells_expressed"]
        df["status"] = row["status"]
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "term", "estimate", "std_err", "test_val", "p_value",
                "normalized_effect", "model_component",
                "gene_id", "gene_short_name", "num_cells_expressed",
                "status", "q_value",
            ]
        )

    out = pd.concat(frames, ignore_index=True)
    out["q_value"] = 1.0
    for (_mc, _term), idx in out.groupby(["model_component", "term"]).groups.items():
        subset_idx = np.asarray(idx)
        pvals = out.loc[subset_idx, "p_value"].to_numpy(dtype=float)
        valid = ~np.isnan(pvals)
        if valid.any():
            qvals = false_discovery_control(pvals[valid], method="bh")
            valid_idx = subset_idx[valid]
            out.loc[valid_idx, "q_value"] = qvals
    return out


def _glance_one(model) -> dict:
    """Return a ``broom::glance``-style row for a single fitted model."""
    if model is None:
        return {
            "null_deviance": np.nan,
            "df_null": np.nan,
            "logLik": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "deviance": np.nan,
            "df_residual": np.nan,
        }
    try:
        return {
            "null_deviance": float(getattr(model, "null_deviance", np.nan)),
            "df_null": float(getattr(model, "df_null", np.nan)),
            "logLik": float(getattr(model, "llf", np.nan)),
            "AIC": float(getattr(model, "aic", np.nan)),
            "BIC": float(getattr(model, "bic", np.nan))
            if getattr(model, "bic", None) is not None
            else np.nan,
            "deviance": float(getattr(model, "deviance", np.nan))
            if getattr(model, "deviance", None) is not None
            else np.nan,
            "df_residual": float(getattr(model, "df_resid", np.nan)),
        }
    except Exception:
        return {
            "null_deviance": np.nan,
            "df_null": np.nan,
            "logLik": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "deviance": np.nan,
            "df_residual": np.nan,
        }


def evaluate_fits(model_tbl: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of per-gene glance stats + the original var columns."""
    glanced = [_glance_one(m) for m in model_tbl["model"]]
    frame = pd.DataFrame(glanced)
    keep_cols = [
        c for c in model_tbl.columns if c not in {"model", "model_summary", "_design_info"}
    ]
    merged = pd.concat(
        [model_tbl[keep_cols].reset_index(drop=True), frame], axis=1
    )
    return merged


def compare_models(
    model_tbl_full: pd.DataFrame, model_tbl_reduced: pd.DataFrame
) -> pd.DataFrame:
    """Likelihood-ratio test between two ``fit_models`` outputs."""
    full = evaluate_fits(model_tbl_full)
    reduced = evaluate_fits(model_tbl_reduced)
    merge_keys = [
        c for c in ("gene_id", "gene_short_name", "num_cells_expressed")
        if c in full.columns and c in reduced.columns
    ]
    if "gene_id" not in merge_keys:
        raise KeyError("'gene_id' missing from model tables")

    joined = full.merge(reduced, on=merge_keys, suffixes=(".x", ".y"))
    dfs = np.round(np.abs(joined["df_residual.x"] - joined["df_residual.y"]))
    LLR = 2.0 * np.abs(joined["logLik.x"] - joined["logLik.y"])
    p_value = chi2.sf(LLR.to_numpy(), dfs.to_numpy().clip(min=1))
    joined = joined.assign(p_value=p_value)
    q_value = np.ones_like(p_value)
    valid = ~np.isnan(p_value)
    if valid.any():
        q_value[valid] = false_discovery_control(p_value[valid], method="bh")
    joined["q_value"] = q_value
    keep_cols = ["gene_id"]
    for col in ("gene_short_name", "num_cells_expressed"):
        if col in joined.columns:
            keep_cols.append(col)
    keep_cols += ["p_value", "q_value"]
    return joined[keep_cols]


def model_predictions(
    model_tbl: pd.DataFrame,
    new_data: pd.DataFrame,
    type: str = "response",
) -> np.ndarray:
    """Predict expression for each gene at *new_data*.

    Returns a ``(n_genes, n_new_rows)`` ndarray.
    """
    if not isinstance(new_data, pd.DataFrame):
        raise TypeError("new_data must be a pandas.DataFrame")
    # Sanitise columns in new_data so dotted names line up with the
    # design_info built on the training frame.
    new_san, _ = _sanitize_columns(new_data, "")
    preds: list[np.ndarray] = []
    for _, row in model_tbl.iterrows():
        model = row["model"]
        design_info = row["_design_info"]
        try:
            new_design = patsy.dmatrix(
                design_info, data=new_san, return_type="dataframe"
            )
            pred = np.asarray(model.predict(new_design.to_numpy(dtype=float)))
        except Exception:
            pred = np.full(new_data.shape[0], np.nan)
        preds.append(pred)
    return np.asarray(preds)
