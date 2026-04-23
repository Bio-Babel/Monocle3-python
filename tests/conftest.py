"""Shared pytest fixtures for monocle3-python."""

from __future__ import annotations

import pytest

from ._fixtures import make_synthetic_adata, make_tiny_adata


@pytest.fixture(scope="function")
def synthetic_adata():
    return make_synthetic_adata()


@pytest.fixture(scope="function")
def tiny_adata():
    return make_tiny_adata()
