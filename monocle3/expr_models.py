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
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP,
    ZeroInflatedPoisson,
)
from statsmodels.discrete.discrete_model import NegativeBinomial as _NegBin
from statsmodels.stats.multitest import multipletests

__all__ = [
    "fit_models",
    "coefficient_table",
    "compare_models",
    "evaluate_fits",
    "model_predictions",
]


import re as _re

# Translate patsy-style term names ("x[T.lvl]", "Intercept") to the names R's
# ``model.matrix`` produces ("xlvl", "(Intercept)") so downstream callers
# filter by R-compatible strings.
_PATSY_FACTOR = _re.compile(r"([^\[\]\s]+?)\[T\.([^\]]+)\]")


def _to_r_term(name: str) -> str:
    if name == "Intercept":
        return "(Intercept)"
    return _PATSY_FACTOR.sub(r"\1\2", name)


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


def _log2_normalized_effect(
    params: np.ndarray,
    intercept_idx: int | None,
    link_inv,
    pseudo_count: float,
) -> np.ndarray:
    """log2((linkinv(β + β_0) + pc) / (linkinv(β_0) + pc)), 0 at intercept."""
    out = np.zeros_like(params)
    if intercept_idx is None:
        return out
    intercept = params[intercept_idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:] = np.log2(
            (link_inv(params + intercept) + pseudo_count)
            / (link_inv(intercept) + pseudo_count)
        )
    out[intercept_idx] = 0.0
    return out


def _split_zi_params(model, design_names: list[str]):
    """For a ZeroInflated* model, partition params into ``(count_slice,
    zero_slice)`` lists of ``(names, params, bse, tvalues, pvalues)``.

    statsmodels orders ZIP params as ``[inflate_const, inflate_x1, ...,
    const, x1, ...]`` — inflation (zero) submodel first, count submodel
    second. Returns per-submodel name lists matching R's
    ``model_summary$coefficients$count`` / ``$zero`` layout.
    """
    params = np.asarray(model.params, dtype=float)
    bse = np.asarray(model.bse, dtype=float)
    tvalues = np.asarray(model.tvalues, dtype=float)
    pvalues = np.asarray(model.pvalues, dtype=float)

    k_inflate = int(model.model.k_inflate)
    exog_names = list(model.model.exog_names)
    zero_names_raw = [n.removeprefix("inflate_") for n in exog_names[:k_inflate]]
    # Rename patsy factor contrasts → R naming style.
    zero_names = [_to_r_term(
        "(Intercept)" if n == "const" else n
    ) for n in zero_names_raw]
    count_names = list(design_names)

    zero = {
        "names": zero_names,
        "params": params[:k_inflate],
        "std_err": bse[:k_inflate],
        "test_val": tvalues[:k_inflate],
        "p_value": pvalues[:k_inflate],
    }
    count = {
        "names": count_names,
        "params": params[k_inflate : k_inflate + len(count_names)],
        "std_err": bse[k_inflate : k_inflate + len(count_names)],
        "test_val": tvalues[k_inflate : k_inflate + len(count_names)],
        "p_value": pvalues[k_inflate : k_inflate + len(count_names)],
    }
    return count, zero


def coefficient_table(model_tbl: pd.DataFrame, pseudo_count: float = 0.01) -> pd.DataFrame:
    """Extract Wald coefficients and Holm-adjusted q-values from a fit table.

    Output columns mirror ``broom::tidy(model)``:
    ``term, estimate, std_err, test_val, p_value, normalized_effect,
    model_component, gene_id, gene_short_name, num_cells_expressed,
    status, q_value``. Zero-inflated models emit both ``count`` and
    ``zero`` rows per gene (matching R's ``model_component`` column).
    """
    frames: list[pd.DataFrame] = []
    for _, row in model_tbl.iterrows():
        model = row["model"]
        if model is None:
            continue
        raw_names = list(row["_design_info"].column_names)
        names = [_to_r_term(n) for n in raw_names]

        if isinstance(model, (ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP)):
            try:
                count, zero = _split_zi_params(model, names)
            except Exception:
                continue
            # Count submodel — normalize effect via exp (Poisson link).
            intercept_idx_c = next(
                (k for k, n in enumerate(count["names"])
                 if n in ("Intercept", "(Intercept)")),
                None,
            )
            df_c = pd.DataFrame({
                "term": count["names"],
                "estimate": count["params"],
                "std_err": count["std_err"],
                "test_val": count["test_val"],
                "p_value": count["p_value"],
                "normalized_effect": _log2_normalized_effect(
                    count["params"], intercept_idx_c, np.exp, pseudo_count
                ),
                "model_component": "count",
            })
            # Zero submodel — R sets normalized_effect = NA.
            df_z = pd.DataFrame({
                "term": zero["names"],
                "estimate": zero["params"],
                "std_err": zero["std_err"],
                "test_val": zero["test_val"],
                "p_value": zero["p_value"],
                "normalized_effect": np.nan,
                "model_component": "zero",
            })
            df = pd.concat([df_c, df_z], ignore_index=True)
        else:
            try:
                params = np.asarray(model.params, dtype=float)
                std_err = np.asarray(model.bse, dtype=float)
                tvals = np.asarray(model.tvalues, dtype=float)
                pvals = np.asarray(model.pvalues, dtype=float)
            except Exception:
                continue
            # statsmodels NegativeBinomial appends a dispersion alpha — drop it.
            if params.size > len(names):
                params = params[: len(names)]
                std_err = std_err[: len(names)]
                tvals = tvals[: len(names)]
                pvals = pvals[: len(names)]
            intercept_idx = next(
                (k for k, n in enumerate(names)
                 if n in ("Intercept", "(Intercept)")),
                None,
            )
            family_inv = getattr(model, "family", None)
            link_inv = (
                family_inv.link.inverse
                if family_inv is not None and hasattr(family_inv, "link")
                else np.exp
            )
            df = pd.DataFrame({
                "term": names,
                "estimate": params,
                "std_err": std_err,
                "test_val": tvals,
                "p_value": pvals,
                "normalized_effect": _log2_normalized_effect(
                    params, intercept_idx, link_inv, pseudo_count
                ),
                "model_component": "count",
            })

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
    # R: `group_by(model_component, term) %>% mutate(q_value = p.adjust(p_value))`.
    # Default p.adjust() method is Holm step-down (not BH).
    for (_mc, _term), idx in out.groupby(["model_component", "term"]).groups.items():
        subset_idx = np.asarray(idx)
        pvals = out.loc[subset_idx, "p_value"].to_numpy(dtype=float)
        valid = ~np.isnan(pvals)
        if valid.any():
            qvals = multipletests(pvals[valid], method="holm")[1]
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
        # R `compare_models` adjusts with `stats::p.adjust()` default (Holm).
        q_value[valid] = multipletests(p_value[valid], method="holm")[1]
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

    Parameters
    ----------
    type : {"response", "link"}, default "response"
        ``"response"`` returns predictions on the mean scale; ``"link"``
        returns predictions on the linear-predictor scale. Matches R
        ``stats::predict(..., type=)``.

    Returns
    -------
    numpy.ndarray
        ``(n_genes, n_new_rows)`` array of predictions.
    """
    if not isinstance(new_data, pd.DataFrame):
        raise TypeError("new_data must be a pandas.DataFrame")
    if type not in {"response", "link"}:
        raise ValueError("type must be 'response' or 'link'")
    new_san, _ = _sanitize_columns(new_data, "")
    preds: list[np.ndarray] = []
    for _, row in model_tbl.iterrows():
        model = row["model"]
        design_info = row["_design_info"]
        try:
            new_design = patsy.dmatrix(
                design_info, data=new_san, return_type="dataframe"
            )
            X = new_design.to_numpy(dtype=float)
            if type == "link":
                # statsmodels GLM has linear= param; ZI / NB expose
                # `which="linear"` or require manual β·X evaluation.
                pred_fn = getattr(model, "predict", None)
                try:
                    pred = np.asarray(model.predict(X, linear=True))
                except TypeError:
                    # Manual linear predictor for backends that don't
                    # expose `linear=True` (e.g. discrete count models).
                    beta = np.asarray(model.params, dtype=float)[: X.shape[1]]
                    pred = X @ beta
            else:
                pred = np.asarray(model.predict(X))
        except Exception:
            pred = np.full(new_data.shape[0], np.nan)
        preds.append(pred)
    return np.asarray(preds)
