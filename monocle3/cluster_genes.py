"""cluster_genes — port of R/cluster_genes.R (``find_gene_modules`` and
``aggregate_gene_expression``).

UMAP on gene loadings → Leiden → ``module`` labels per gene. R's gene
loadings come from ``SVD.v @ diag(SVD.sdev)`` stored under
``uns["monocle3"]["preprocess"]["PCA"]``.

``aggregate_gene_expression`` builds the module × cell-group matrix used
by the embryo tutorial's heatmap + ``plot_cells(genes=gene_module_df)``.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import hnswlib
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import umap as umap_module
from scipy import sparse as sp
from sklearn.neighbors import NearestNeighbors

from ._utils import get_monocle_uns
from .cluster_cells import _compute_partitions, _make_knn_graph, _run_leiden
from .preprocess import normalized_counts as _normalized_counts

__all__ = ["find_gene_modules", "aggregate_gene_expression"]


def _gene_loadings(adata: ad.AnnData, preprocess_method: str) -> pd.DataFrame:
    model = get_monocle_uns(adata, "preprocess", preprocess_method)
    if model is None:
        raise KeyError(
            f"No {preprocess_method} preprocess model. Run preprocess_cds first."
        )
    svd_v = model["svd_v"]  # (n_genes_kept, n_dim)
    svd_sdev = model["svd_sdev"]
    gene_names = model["gene_names"]
    loadings = svd_v * svd_sdev[None, :]
    df = pd.DataFrame(loadings, index=gene_names)
    df.columns = [f"PC_{i + 1}" for i in range(df.shape[1])]
    return df


def _precomputed_knn_for_umap(
    X: np.ndarray,
    n_neighbors: int,
    metric: str,
    nn_method: str,
    random_seed: int | None,
    cores: int,
) -> tuple[np.ndarray, np.ndarray, None] | None:
    """Build ``(knn_indices, knn_dists, None)`` for umap-learn's
    ``precomputed_knn`` slot when the caller asked for an NN backend other
    than pynndescent (umap-learn's built-in).

    - ``"annoy"`` / ``"nndescent"``  → ``None`` (let umap-learn use pynndescent).
    - ``"fnn"``                        → exact brute-force via sklearn.
    - ``"hnsw"``                       → hnswlib.
    """
    if nn_method in ("annoy", "nndescent"):
        return None
    if nn_method == "fnn":
        nn = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm="brute", metric=metric,
            n_jobs=int(cores),
        ).fit(X)
        dist, idx = nn.kneighbors(X, n_neighbors=n_neighbors)
        return idx.astype(np.int64), dist.astype(np.float64), None
    if nn_method == "hnsw":
        space = {"euclidean": "l2", "cosine": "cosine"}.get(metric)
        if space is None:
            raise ValueError(
                f"find_gene_modules: umap_nn_method='hnsw' only supports "
                f"metric in {{'euclidean', 'cosine'}}, got {metric!r}."
            )
        idx_obj = hnswlib.Index(space=space, dim=X.shape[1])
        idx_obj.init_index(
            max_elements=X.shape[0], ef_construction=200, M=48,
        )
        idx_obj.set_num_threads(int(cores))
        idx_obj.add_items(X, np.arange(X.shape[0]))
        idx_obj.set_ef(max(n_neighbors, 150))
        labels, dists = idx_obj.knn_query(X, k=n_neighbors)
        if space == "l2":
            dists = np.sqrt(dists)
        return labels.astype(np.int64), dists.astype(np.float64), None
    raise ValueError(
        f"umap_nn_method must be one of {{'annoy', 'nndescent', 'fnn', 'hnsw'}}, "
        f"got {nn_method!r}."
    )


def find_gene_modules(
    adata: ad.AnnData,
    reduction_method: str = "UMAP",
    max_components: int = 2,
    umap_metric: str = "cosine",
    umap_min_dist: float = 0.1,
    umap_n_neighbors: int = 15,
    umap_fast_sgd: bool = False,
    umap_nn_method: str = "annoy",
    k: int = 20,
    leiden_iter: int = 1,
    partition_qval: float = 0.05,
    weight: bool = False,
    resolution: float | None = None,
    random_seed: int = 0,
    cores: int = 1,
    verbose: bool = False,
    preprocess_method: str = "PCA",
    nn_control: dict | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """UMAP-on-gene-loadings then Leiden to produce gene modules.

    ``umap_nn_method`` mirrors R's ``umap.nn_method`` (default ``"annoy"``
    per ``cluster_genes.R:108``). umap-learn's built-in pynndescent covers
    ``"annoy"`` and ``"nndescent"`` natively; ``"fnn"`` and ``"hnsw"`` are
    realised by pre-computing the kNN graph and handing it to umap-learn
    through its ``precomputed_knn`` slot.
    """
    del verbose, kwargs
    if reduction_method != "UMAP":
        raise NotImplementedError("Only reduction_method='UMAP' is supported")

    # Subset gene loadings to the genes present in adata (handles post-subset CDS).
    loadings = _gene_loadings(adata, preprocess_method)
    loadings = loadings.loc[loadings.index.intersection(adata.var_names)]
    if loadings.shape[0] < 5:
        raise ValueError(
            "find_gene_modules needs at least 5 genes with loadings."
        )

    if random_seed != 0:
        np.random.seed(int(random_seed))
    n_neighbors = min(int(umap_n_neighbors), max(2, loadings.shape[0] - 1))
    loading_arr = loadings.to_numpy()
    precomputed = _precomputed_knn_for_umap(
        loading_arr,
        n_neighbors=n_neighbors,
        metric=str(umap_metric),
        nn_method=str(umap_nn_method),
        random_seed=random_seed if random_seed > 0 else None,
        cores=int(cores),
    )
    umap_kwargs = dict(
        n_components=int(max_components),
        n_neighbors=n_neighbors,
        min_dist=float(umap_min_dist),
        metric=str(umap_metric),
        random_state=random_seed if random_seed > 0 else None,
        n_jobs=int(cores),
    )
    if precomputed is not None:
        umap_kwargs["precomputed_knn"] = precomputed
    u = umap_module.UMAP(**umap_kwargs)
    embedding = u.fit_transform(loading_arr)
    del cores

    # Leiden on a kNN graph built from the gene UMAP.
    nn_ctrl = {"method": "annoy", "metric": "euclidean"}
    if nn_control:
        nn_ctrl.update(nn_control)
    g, _nbr, _dists = _make_knn_graph(
        embedding, k=int(k), nn_control=nn_ctrl,
        cell_names=list(loadings.index), weight=bool(weight),
    )
    membership, _modularity = _run_leiden(
        g,
        resolution=resolution,
        num_iter=int(leiden_iter),
        random_seed=random_seed if random_seed > 0 else None,
    )
    partition_arr = (
        _compute_partitions(g, membership, qval_thresh=float(partition_qval))
        if len(set(membership)) > 1
        else np.ones_like(membership, dtype=np.int64)
    )

    df = pd.DataFrame(
        {
            "id": list(loadings.index),
            "module": pd.Categorical(
                [str(m + 1) for m in membership],
                categories=[str(x) for x in sorted(set(membership + 1))],
            ),
            "supermodule": pd.Categorical(
                [str(p) for p in partition_arr],
                categories=[str(x) for x in sorted(set(int(v) for v in partition_arr))],
            ),
            "dim_1": embedding[:, 0],
        }
    )
    if embedding.shape[1] > 1:
        df["dim_2"] = embedding[:, 1]
    return df


def aggregate_gene_expression(
    adata: ad.AnnData,
    gene_group_df: pd.DataFrame | None = None,
    cell_group_df: pd.DataFrame | None = None,
    norm_method: str = "log",
    pseudocount: float = 1.0,
    scale_agg_values: bool = True,
    max_agg_value: float = 3.0,
    min_agg_value: float = -3.0,
    exclude_na: bool = True,
    gene_agg_fun: str = "sum",
    cell_agg_fun: str = "mean",
) -> pd.DataFrame:
    """Return a (gene_group × cell_group) aggregated expression matrix.

    Mirrors R ``aggregate_gene_expression``. Input DataFrames must have
    the columns in the first and second positions (R accesses them by
    integer column index).
    """
    if gene_group_df is None and cell_group_df is None:
        raise ValueError(
            "one of either gene_group_df or cell_group_df must not be None"
        )

    # cells × genes. AnnData is cells × genes, R returns genes × cells.
    norm_cells_x_genes = _normalized_counts(
        adata, norm_method=norm_method, pseudocount=pseudocount
    )
    # Transpose to genes × cells to match R's convention for downstream ops.
    agg = norm_cells_x_genes.T.tocsr()
    gene_ids = list(adata.var_names)
    cell_ids = list(adata.obs_names)
    gene_short_names = (
        adata.var["gene_short_name"].astype(str).tolist()
        if "gene_short_name" in adata.var.columns
        else list(adata.var_names)
    )

    if gene_group_df is not None:
        gg = pd.DataFrame(gene_group_df).copy()
        # Lookup column labels by integer position (matches R indexing).
        c0, c1 = gg.columns[0], gg.columns[1]
        # Keep rows whose first column references a valid gene id or gene_short_name.
        valid_ids = set(gene_ids)
        valid_short = set(gene_short_names)
        keep = gg[c0].astype(str).isin(valid_ids) | gg[c0].astype(str).isin(
            valid_short
        )
        gg = gg.loc[keep].copy()
        # Convert any short-name entries to gene ids.
        short_mask = gg[c0].astype(str).isin(valid_short) & (~gg[c0].astype(str).isin(valid_ids))
        if short_mask.any():
            name_to_id = {
                sn: gid for gid, sn in zip(gene_ids, gene_short_names) if sn
            }
            gg.loc[short_mask, c0] = gg.loc[short_mask, c0].astype(str).map(name_to_id)

        unique_ids = list(pd.unique(gg[c0].astype(str)))
        id_to_row = {g: i for i, g in enumerate(gene_ids)}
        rows = [id_to_row[g] for g in unique_ids if g in id_to_row]
        agg = agg[rows]
        local_gene_ids = [gene_ids[r] for r in rows]

        groups = list(pd.unique(gg[c1].astype(str)))
        # Build sparse group × gene indicator X; X @ agg sums genes per group.
        gene_to_idx = {g: i for i, g in enumerate(local_gene_ids)}
        row_idx = []
        col_idx = []
        for _, r in gg.iterrows():
            gid = str(r[c0])
            grp = str(r[c1])
            if gid not in gene_to_idx:
                continue
            row_idx.append(groups.index(grp))
            col_idx.append(gene_to_idx[gid])
        X = sp.csr_matrix(
            (np.ones(len(row_idx)), (row_idx, col_idx)),
            shape=(len(groups), len(local_gene_ids)),
        )
        agg = X @ agg
        if gene_agg_fun == "mean":
            counts_per_group = np.asarray(X.sum(axis=1)).ravel()
            counts_per_group = np.where(counts_per_group == 0, 1.0, counts_per_group)
            agg = sp.diags(1.0 / counts_per_group) @ agg
        gene_group_labels = groups
    else:
        gene_group_labels = gene_ids

    if cell_group_df is not None:
        cg = pd.DataFrame(cell_group_df).copy()
        c0, c1 = cg.columns[0], cg.columns[1]
        valid_cells = set(cell_ids)
        cg = cg.loc[cg[c0].astype(str).isin(valid_cells)].copy()
        ordered_cells = list(cg[c0].astype(str))
        # Reorder agg columns to match cg row order.
        agg = agg[:, [cell_ids.index(c) for c in ordered_cells]]

        groups = list(pd.unique(cg[c1].astype(str)))
        row_idx = []
        col_idx = []
        for pos, grp in enumerate(cg[c1].astype(str)):
            row_idx.append(groups.index(str(grp)))
            col_idx.append(pos)
        Y = sp.csr_matrix(
            (np.ones(len(row_idx)), (row_idx, col_idx)),
            shape=(len(groups), len(ordered_cells)),
        )
        agg = agg @ Y.T
        if cell_agg_fun == "mean":
            counts_per_group = np.asarray(Y.sum(axis=1)).ravel()
            counts_per_group = np.where(counts_per_group == 0, 1.0, counts_per_group)
            agg = agg @ sp.diags(1.0 / counts_per_group)
        cell_group_labels = groups
    else:
        cell_group_labels = cell_ids

    dense = np.asarray(agg.todense() if sp.issparse(agg) else agg)

    if scale_agg_values:
        # z-score across cell groups (columns), per gene group (row) — matches
        # R's ``t(scale(t(agg_mat)))``.
        row_mean = dense.mean(axis=1, keepdims=True)
        row_std = dense.std(axis=1, keepdims=True, ddof=1)
        row_std = np.where(row_std == 0, 1.0, row_std)
        dense = (dense - row_mean) / row_std
        dense[np.isnan(dense)] = 0.0
        dense = np.clip(dense, min_agg_value, max_agg_value)

    out = pd.DataFrame(dense, index=gene_group_labels, columns=cell_group_labels)
    if exclude_na:
        out = out.loc[out.index != "NA", out.columns != "NA"]
    return out
