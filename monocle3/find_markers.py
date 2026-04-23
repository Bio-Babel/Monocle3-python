"""find_markers — port of R/find_markers.R.

``top_markers`` follows the R algorithm exactly: mean (size-only) and
binary (any positive count) per-group aggregated matrices → JS
specificity → marker score (binary × specificity) → top-N per group.

Per-gene marker significance is a logistic regression of group
membership against log-expression + 0.1 per R, with pseudo-R²
computed from the log-likelihoods of the null vs. full model (matches
``test_marker_for_cell_group``).

``generate_garnett_marker_file`` writes the Garnett marker list used by
the L2 tutorial.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.stats import chi2
from statsmodels.api import GLM as _SMGLM
from statsmodels.genmod.families import Binomial as _SMBinomial

from ._accessors import size_factors
from .cluster_cells import clusters, partitions
from .cluster_genes import aggregate_gene_expression

__all__ = ["top_markers", "generate_garnett_marker_file"]


def _shannon_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    if p.min() < 0 or p.sum() <= 0:
        return np.inf
    p_norm = p[p > 0] / p[p > 0].sum()
    return -float(np.sum(np.log2(p_norm) * p_norm))


def _js_dist(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2
    jsdiv = _shannon_entropy(m) - 0.5 * (_shannon_entropy(p) + _shannon_entropy(q))
    if not np.isfinite(jsdiv):
        return 1.0
    jsdiv = max(0.0, jsdiv)
    return float(np.sqrt(jsdiv))


def _specificity_matrix(agg: pd.DataFrame) -> pd.DataFrame:
    """Per-(gene, group) 1 - JS-distance against the one-hot indicator vector."""
    mat = agg.to_numpy(dtype=float)
    n_groups = mat.shape[1]
    eye = np.eye(n_groups)
    spec = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        vec = mat[i]
        total = vec.sum()
        prob = vec / total if total > 0 else np.zeros_like(vec)
        for j in range(n_groups):
            spec[i, j] = 1.0 - _js_dist(prob, eye[:, j])
    return pd.DataFrame(spec, index=agg.index, columns=agg.columns)


def _fit_logistic(x: np.ndarray, y: np.ndarray, maxiter: int = 25):
    design = np.column_stack([np.ones_like(x), x]).astype(float)
    model = _SMGLM(y.astype(float), design, family=_SMBinomial()).fit(
        maxiter=maxiter, disp=0
    )
    return model


def _fit_null_logistic(y: np.ndarray, maxiter: int = 25):
    design = np.ones((y.size, 1), dtype=float)
    return _SMGLM(y.astype(float), design, family=_SMBinomial()).fit(
        maxiter=maxiter, disp=0
    )


def _test_marker(
    adata: ad.AnnData,
    gene_id: str,
    cell_group: str,
    cell_group_series: pd.Series,
    reference_cells: Sequence[str] | None,
    maxiter: int,
) -> tuple[float, float]:
    try:
        gene_idx = list(adata.var_names).index(gene_id)
        counts = adata.X[:, gene_idx]
        if sp.issparse(counts):
            counts = np.asarray(counts.todense()).ravel()
        else:
            counts = np.asarray(counts, dtype=float).ravel()
        sf = size_factors(adata)
        expression = np.log(counts / sf + 0.1)
        membership = (cell_group_series.astype(str) == str(cell_group)).to_numpy()

        if reference_cells is not None:
            keep = membership | np.isin(adata.obs_names, list(reference_cells))
            expression = expression[keep]
            membership = membership[keep]

        if np.isnan(expression).any() or np.isnan(membership.astype(float)).any():
            raise ValueError("NA expression or membership")

        full = _fit_logistic(expression, membership.astype(float), maxiter=maxiter)
        null = _fit_null_logistic(membership.astype(float), maxiter=maxiter)
        LR = 2.0 * (full.llf - null.llf)
        df = full.df_model
        p = float(chi2.sf(LR, df=max(df, 1)))
        n = len(membership)
        pseudo_r2 = float(
            (1.0 - np.exp(-LR / n)) / (1.0 - np.exp(2.0 * null.llf / n))
        )
        return pseudo_r2, p
    except Exception:
        return 0.0, 1.0


def top_markers(
    adata: ad.AnnData,
    group_cells_by: str = "cluster",
    genes_to_test_per_group: int = 25,
    reduction_method: str = "UMAP",
    marker_sig_test: bool = True,
    reference_cells: int | Sequence[str] | None = None,
    speedglm_maxiter: int = 25,
    cores: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Identify specific marker genes per cell group.

    Columns of the returned DataFrame match R top_markers:
    ``gene_id, gene_short_name, cell_group, marker_score,
    mean_expression, fraction_expressing, specificity, pseudo_R2,
    marker_test_p_value, marker_test_q_value``.
    """
    del cores, verbose
    if group_cells_by == "cluster":
        cell_group_series = clusters(adata, reduction_method=reduction_method).astype(str)
    elif group_cells_by == "partition":
        cell_group_series = partitions(adata, reduction_method=reduction_method).astype(str)
    else:
        if group_cells_by not in adata.obs.columns:
            raise KeyError(
                f"group_cells_by '{group_cells_by}' not in adata.obs"
            )
        cell_group_series = adata.obs[group_cells_by].astype(str)

    cell_group_df = pd.DataFrame(
        {
            "cell": adata.obs_names.astype(str),
            "cell_group": cell_group_series.to_numpy(),
        }
    )

    mean_expr = aggregate_gene_expression(
        adata, cell_group_df=cell_group_df,
        norm_method="size_only", scale_agg_values=False,
    )
    binary_expr = aggregate_gene_expression(
        adata, cell_group_df=cell_group_df,
        norm_method="binary", scale_agg_values=False,
    )

    spec_mat = _specificity_matrix(mean_expr)
    marker_score_mat = binary_expr * spec_mat

    long = marker_score_mat.stack().reset_index()
    long.columns = ["gene_id", "cell_group", "marker_score"]

    long["specificity"] = spec_mat.stack().reindex(
        list(zip(long["gene_id"], long["cell_group"]))
    ).to_numpy()
    long["mean_expression"] = mean_expr.stack().reindex(
        list(zip(long["gene_id"], long["cell_group"]))
    ).to_numpy()
    long["fraction_expressing"] = binary_expr.stack().reindex(
        list(zip(long["gene_id"], long["cell_group"]))
    ).to_numpy()

    # Top genes_to_test_per_group per group by marker_score.
    long = (
        long.sort_values(["cell_group", "marker_score"], ascending=[True, False])
        .groupby("cell_group", as_index=False)
        .head(int(genes_to_test_per_group))
    )

    if marker_sig_test:
        if isinstance(reference_cells, (int, np.integer)):
            n_per_group = max(1, int(reference_cells) // cell_group_df["cell_group"].nunique())
            ref_cells = (
                cell_group_df.groupby("cell_group", group_keys=False)
                .apply(
                    lambda g, m=n_per_group: g.sample(
                        n=min(m, len(g)), random_state=2016
                    )
                )["cell"]
                .tolist()
            )
        else:
            ref_cells = list(reference_cells) if reference_cells is not None else None

        pseudo_r2_list = []
        p_list = []
        for _, row in long.iterrows():
            r2, p = _test_marker(
                adata,
                gene_id=row["gene_id"],
                cell_group=row["cell_group"],
                cell_group_series=cell_group_series,
                reference_cells=ref_cells,
                maxiter=int(speedglm_maxiter),
            )
            pseudo_r2_list.append(r2)
            p_list.append(p)
        long["pseudo_R2"] = pseudo_r2_list
        long["marker_test_p_value"] = p_list
        long["marker_test_q_value"] = np.clip(
            long["marker_test_p_value"].to_numpy(float) * spec_mat.size, 0, 1
        )
    else:
        long["pseudo_R2"] = np.nan
        long["marker_test_p_value"] = np.nan
        long["marker_test_q_value"] = np.nan

    if "gene_short_name" in adata.var.columns:
        name_map = dict(zip(adata.var_names, adata.var["gene_short_name"].astype(str)))
        long["gene_short_name"] = long["gene_id"].map(name_map)
    return long.reset_index(drop=True)


def generate_garnett_marker_file(
    marker_test_res: pd.DataFrame,
    file: str | Path = "./marker_file.txt",
    max_genes_per_group: int = 10,
    remove_duplicate_genes: bool = False,
) -> None:
    """Write a Garnett-format marker file."""
    df = marker_test_res.copy()
    if "group_name" not in df.columns:
        df["group_name"] = "Cell type " + df["cell_group"].astype(str)
    df = (
        df.sort_values(["group_name", "marker_score"], ascending=[True, False])
        .groupby("group_name", as_index=False)
        .head(int(max_genes_per_group))
    )
    dups = df["gene_id"][df["gene_id"].duplicated(keep=False)].unique().tolist()
    if remove_duplicate_genes:
        df = df[~df["gene_id"].isin(dups)]
    elif dups:
        warnings.warn(
            "The following marker genes mark multiple cell groups: "
            + ", ".join(dups),
            stacklevel=2,
        )

    ordered_groups = list(pd.unique(df["group_name"]))
    entries: list[str] = []
    for group in ordered_groups:
        sub = df[df["group_name"] == group]
        if sub.empty:
            continue
        safe_name = re.sub(r"[\(\):>,#]", ".", str(group))
        if safe_name != group:
            warnings.warn(
                f"Group name '{group}' contained illegal characters; using '{safe_name}'.",
                stacklevel=2,
            )
        if "gene_short_name" in sub.columns:
            genes = ", ".join(sub["gene_short_name"].astype(str))
        else:
            genes = ", ".join(sub["gene_id"].astype(str))
        entries.append(f"> {safe_name}\nexpressed: {genes}\n")
    Path(file).write_text("\n".join(entries))
