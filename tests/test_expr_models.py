"""Slice 7: expr_models + find_markers smoke tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monocle3 import (
    cluster_cells,
    coefficient_table,
    compare_models,
    evaluate_fits,
    fit_models,
    generate_garnett_marker_file,
    model_predictions,
    preprocess_cds,
    reduce_dimension,
    top_markers,
)


@pytest.fixture(scope="module")
def prepared(synthetic_adata):
    preprocess_cds(synthetic_adata, num_dim=10)
    reduce_dimension(synthetic_adata, max_components=2)
    cluster_cells(synthetic_adata, resolution=1e-3, random_seed=1)
    # Add a continuous covariate for fit_models.
    rng = np.random.default_rng(0)
    synthetic_adata.obs["time"] = rng.uniform(0, 1, size=synthetic_adata.n_obs)
    return synthetic_adata


def test_fit_models_returns_table(prepared):
    tbl = fit_models(prepared, model_formula_str="~time")
    assert "model" in tbl.columns
    assert "status" in tbl.columns
    assert tbl.shape[0] == prepared.n_vars
    # At least most genes converged on our synthetic fixture.
    assert (tbl["status"] == "OK").mean() > 0.5


def test_coefficient_table_has_q_value(prepared):
    tbl = fit_models(prepared, model_formula_str="~time")
    coefs = coefficient_table(tbl)
    assert "term" in coefs.columns
    assert "q_value" in coefs.columns
    assert set(coefs["term"].unique()) >= {"Intercept", "time"}


def test_evaluate_and_compare(prepared):
    full = fit_models(prepared, model_formula_str="~time")
    reduced = fit_models(prepared, model_formula_str="~1")
    glance = evaluate_fits(full)
    assert {"logLik", "AIC", "df_residual"}.issubset(glance.columns)
    cmp = compare_models(full, reduced)
    assert {"gene_id", "p_value", "q_value"}.issubset(cmp.columns)


def test_model_predictions(prepared):
    tbl = fit_models(prepared, model_formula_str="~time")
    # Keep only OK fits for clean predictions.
    ok = tbl[tbl["status"] == "OK"]
    new_df = pd.DataFrame({"time": np.linspace(0, 1, 5)})
    pred = model_predictions(ok, new_df)
    assert pred.shape == (len(ok), 5)


def test_fit_models_mixed_negbinomial_raises(prepared):
    with pytest.raises(NotImplementedError):
        fit_models(
            prepared, model_formula_str="~time",
            expression_family="mixed-negbinomial",
        )


def test_top_markers_shape(prepared):
    df = top_markers(
        prepared, group_cells_by="cluster_truth",
        genes_to_test_per_group=5,
        marker_sig_test=False,
    )
    assert {"gene_id", "cell_group", "marker_score"}.issubset(df.columns)
    assert df.shape[0] <= prepared.n_vars * prepared.obs["cluster_truth"].nunique()


def test_generate_garnett_marker_file(prepared):
    df = top_markers(
        prepared, group_cells_by="cluster_truth",
        genes_to_test_per_group=3,
        marker_sig_test=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "markers.txt"
        generate_garnett_marker_file(df, file=out, max_genes_per_group=2)
        content = out.read_text()
    assert ">" in content
    assert "expressed:" in content
