"""Tutorial dataset loaders.

The originals (``load_worm_embryo``, ``load_worm_l2``, ``load_a549``,
``load_mm_data``) are not ported — see ``monocle3_porting_essential_suggestions.md``
§5. They are replaced by two Zenodo-hosted AnnData files:

- ``packer_embryo.h5ad`` — the Packer et al. 2019 *C. elegans* embryo
  dataset used by ``c_elegans_embryo_v2.ipynb``.
- ``cao_l2.h5ad`` — the Cao et al. 2017 *C. elegans* L2 dataset used by
  ``c_elegans_L2_v2.ipynb``.

Both are resolved through the three-tier ``_download.resolve_data_path``
loader: cwd-local staging → ``~/.cache/monocle3-python/`` → registry
download.
"""

from __future__ import annotations

import anndata as ad

from ._download import resolve_data_path

__all__ = ["load_packer_embryo", "load_cao_l2"]


def load_packer_embryo() -> ad.AnnData:
    """Load the Packer et al. 2019 *C. elegans* embryo dataset.

    Returns
    -------
    anndata.AnnData
        ``(6188, 20222)`` — cells × genes. Raw UMI counts in ``X`` as
        ``scipy.sparse.csr_matrix``. ``obs`` holds per-cell metadata
        (time point, batch, loading-background covariates, Packer
        ``cell.type`` annotation); ``var`` holds gene ids
        (``WBGene00...``), ``gene_short_name``, and
        ``num_cells_expressed``.
    """
    path = resolve_data_path("packer_embryo.h5ad")
    return ad.read_h5ad(path)


def load_cao_l2() -> ad.AnnData:
    """Load the Cao et al. 2017 *C. elegans* L2 dataset.

    Returns
    -------
    anndata.AnnData
        ``(42035, 20271)`` — cells × genes. Raw counts in ``X`` as
        ``scipy.sparse.csr_matrix``. ``obs`` holds ``plate``,
        ``cao_cluster``, ``cao_cell_type``, ``cao_tissue``. ``var``
        holds ``gene_short_name``.
    """
    path = resolve_data_path("cao_l2.h5ad")
    return ad.read_h5ad(path)
