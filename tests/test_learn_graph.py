"""Slice 5: learn_graph + order_cells smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from monocle3 import (
    cluster_cells,
    learn_graph,
    order_cells,
    preprocess_cds,
    reduce_dimension,
)
from monocle3.order_cells import pseudotime


@pytest.fixture(scope="module")
def prepared(synthetic_adata):
    preprocess_cds(synthetic_adata, num_dim=10)
    reduce_dimension(synthetic_adata, max_components=2)
    cluster_cells(synthetic_adata, resolution=1e-3, random_seed=1)
    return synthetic_adata


def test_learn_graph_creates_principal_graph(prepared):
    learn_graph(
        prepared,
        learn_graph_control={"ncenter": 20, "minimal_branch_len": 3,
                             "maxiter": 5},
    )
    uns = prepared.uns["monocle3"]
    assert "principal_graph" in uns
    g = uns["principal_graph"]["UMAP"]
    assert g.vcount() > 0
    aux = uns["principal_graph_aux"]["UMAP"]
    assert aux["dp_mst"].shape[1] == g.vcount()
    assert "pr_graph_cell_proj_closest_vertex" in aux


def test_order_cells_produces_pseudotime(prepared):
    # Reuse the learned graph from the previous test.
    node_names = prepared.uns["monocle3"]["principal_graph"]["UMAP"].vs["name"]
    root = node_names[0]
    order_cells(prepared, root_pr_nodes=[root])
    pt = pseudotime(prepared)
    assert len(pt) == prepared.n_obs
    assert pt.min() >= 0.0
    # Pseudotime at the root vertex's cells should be 0.
    closest = prepared.uns["monocle3"]["principal_graph_aux"]["UMAP"][
        "pr_graph_cell_proj_closest_vertex"
    ]["V1"].to_numpy()
    root_idx = node_names.index(root) + 1
    cells_at_root = np.where(closest == root_idx)[0]
    if cells_at_root.size:
        assert float(pt.iloc[cells_at_root].min()) == 0.0


def test_order_cells_requires_root(prepared):
    with pytest.raises(ValueError, match="root_pr_nodes or root_cells"):
        order_cells(prepared)
