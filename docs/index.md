# monocle3-python

Python port of the R [monocle3](https://cole-trapnell-lab.github.io/monocle3/) package. Monocle 3 performs clustering, differential expression, and trajectory analysis for single-cell RNA-seq data — this port preserves the semantics of the R orchestration while replacing the `cell_data_set` S4 container with `anndata.AnnData` and delegating every non-Monocle-specific algorithm to the scverse ecosystem (umap-learn, leidenalg, statsmodels, …).

## Quick start

```python
import monocle3 as m3

adata = m3.load_packer_embryo()          # AnnData (6188 × 20222), Zenodo-cached
m3.preprocess_cds(adata, num_dim=50)      # size-factor norm + truncated PCA
m3.align_cds(adata, alignment_group='batch')
m3.reduce_dimension(adata)                # UMAP via umap-learn
m3.cluster_cells(adata)                    # Leiden (leidenalg)
m3.learn_graph(adata)                      # SimplePPT principal graph
m3.order_cells(adata, root_pr_nodes=['Y_1'])  # pseudotime
m3.plot_cells(adata, color_cells_by='pseudotime')
```

## Design

- **Container**: `anndata.AnnData` throughout. No `cell_data_set` shim.
- **Accessors**: R generics (`exprs`, `pData`, `fData`, `counts`, `reducedDims`) are removed. Use `adata.X` / `obs` / `var` / `layers` / `obsm` directly.
- **Algorithms**: UMAP via `umap-learn`, t-SNE via `openTSNE`, Leiden via `leidenalg`, GLMs via `statsmodels`, MNN via `scanorama`, Moran's I via a self-contained Python implementation (matches R's randomisation-based variance), kNN via `pynndescent` / `hnswlib` / `sklearn.neighbors`.
- **Visualisation**: Bio-Babel stack (`ggplot2_py`, `pheatmap`, `grid_py`, `scales`, `ggrepel_py`). `plotly` is used only inside `plot_cells_3d`.

## Installation

```bash
pip install -e .
```

The runtime does not require an R interpreter. Remote data assets are cached automatically on first use.

## Tutorials

- [*C. elegans* embryo](tutorials/c_elegans_embryo_v2.ipynb) — Packer et al. 2019, full trajectory analysis.
- [*C. elegans* L2](tutorials/c_elegans_L2_v2.ipynb) — Cao et al. 2017, clustering + marker detection + Garnett export.

## Reference implementation

The canonical R source lives at <https://github.com/cole-trapnell-lab/monocle3>. For the deviations from R-side behaviour introduced by this port, see [`08_validation.md`](https://github.com/cole-trapnell-lab/monocle3/blob/main/port_reports/monocle3/08_validation.md) in the porting report tree.
