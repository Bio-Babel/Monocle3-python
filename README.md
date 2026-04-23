# monocle3-python

[![PyPI](https://img.shields.io/pypi/v/monocle3-python)](https://pypi.org/project/monocle3-python/)

Python port of the R [**monocle3**](https://github.com/cole-trapnell-lab/monocle3) single-cell trajectory toolkit. Tracks upstream `1.4.26` at commit `4f4239a`.

The port keeps monocle3's orchestration and R-side numerical behaviour while replacing the `cell_data_set` S4 container with `anndata.AnnData` and delegating non-monocle-specific algorithms to the scverse ecosystem (`umap-learn`, `openTSNE`, `leidenalg`, `statsmodels`, `scanorama`, …).

## Installation

```bash
pip install monocle3-python                # from PyPI
```

For local development:

```bash
git clone https://github.com/Bio-Babel/Monocle3-python.git
cd Monocle3-python
pip install -e ".[dev]"
```

No R interpreter is required at runtime. Tutorial datasets are downloaded on first use and cached under `~/.cache/monocle3-python/`.

## Quickstart

```python
import monocle3 as m3

adata = m3.load_packer_embryo()                  # AnnData (6188 × 20222)
m3.preprocess_cds(adata, num_dim=50)              # size-factor norm + truncated PCA
m3.align_cds(adata, alignment_group="batch")
m3.reduce_dimension(adata)                        # UMAP via umap-learn
m3.cluster_cells(adata)                           # Leiden
m3.learn_graph(adata)                             # SimplePPT principal graph
m3.order_cells(adata, root_pr_nodes=["Y_1"])      # pseudotime
m3.plot_cells(adata, color_cells_by="pseudotime")
```

## Tutorials

Runnable notebooks that reproduce the R monocle3 vignettes live under [`tutorials/`](tutorials/):

| Notebook | Dataset |
|---|---|
| `c_elegans_embryo_v2.ipynb` | Packer et al. 2019 — *C. elegans* embryo trajectory |
| `c_elegans_L2_v2.ipynb`     | Cao et al. 2017 — *C. elegans* L2 clustering + Garnett markers |

## License

Artistic-2.0, matching the upstream R monocle3.
