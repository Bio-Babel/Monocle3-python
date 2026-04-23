"""Tests for dataset loaders (require the staged h5ad files)."""

from __future__ import annotations

from pathlib import Path

import pytest

import monocle3.datasets as datasets


def _has_staged(filename: str) -> bool:
    # Look in the default local staging dir used by the three-tier resolver.
    root = Path(__file__).resolve().parent.parent
    return (root / "monocle3_data" / filename).exists()


@pytest.mark.skipif(
    not _has_staged("packer_embryo.h5ad"),
    reason="packer_embryo.h5ad not staged",
)
def test_load_packer_embryo_shape():
    adata = datasets.load_packer_embryo()
    assert adata.shape == (6188, 20222)
    assert "cell.type" in adata.obs.columns
    assert "gene_short_name" in adata.var.columns


@pytest.mark.skipif(
    not _has_staged("cao_l2.h5ad"),
    reason="cao_l2.h5ad not staged",
)
def test_load_cao_l2_shape():
    adata = datasets.load_cao_l2()
    assert adata.shape == (42035, 20271)
    assert "plate" in adata.obs.columns
    assert "gene_short_name" in adata.var.columns
