"""monocle3-python — Python port of the R monocle3 package.

See ``docs/index.md`` for an overview and ``tutorials/`` for runnable
notebooks. The public surface is re-exported here; submodules are free
for callers to import directly as well.
"""

from __future__ import annotations

__version__ = "1.4.26+4f4239a"

from ._accessors import (
    counts_row_order,
    saveRDS,
    set_cds_row_order_matrix,
    set_matrix_control,
    size_factors,
)
from .alignment import align_cds
from .cluster_cells import cluster_cells, clusters, partitions
from .cluster_genes import aggregate_gene_expression, find_gene_modules
from .datasets import load_cao_l2, load_packer_embryo
from .expr_models import (
    coefficient_table,
    compare_models,
    evaluate_fits,
    fit_models,
    model_predictions,
)
from .find_markers import generate_garnett_marker_file, top_markers
from .graph_test import graph_test
from .learn_graph import learn_graph
from .nearest_neighbors import (
    make_nn_index,
    search_nn_index,
    search_nn_matrix,
    set_cds_nn_index,
)
from .order_cells import order_cells, principal_graph, pseudotime
from .plotting import (
    plot_cells,
    plot_genes_by_group,
    plot_genes_in_pseudotime,
    plot_genes_violin,
    plot_pc_variance_explained,
    plot_percent_cells_positive,
)
from .plotting_3d import plot_cells_3d
from .preprocess import (
    detect_genes,
    estimate_size_factors,
    new_cell_data_set,
    normalized_counts,
    preprocess_cds,
)
from .reduce_dimensions import reduce_dimension

__all__ = [
    "__version__",
    "aggregate_gene_expression",
    "align_cds",
    "cluster_cells",
    "clusters",
    "coefficient_table",
    "compare_models",
    "counts_row_order",
    "detect_genes",
    "estimate_size_factors",
    "evaluate_fits",
    "find_gene_modules",
    "fit_models",
    "generate_garnett_marker_file",
    "graph_test",
    "learn_graph",
    "load_cao_l2",
    "load_packer_embryo",
    "make_nn_index",
    "model_predictions",
    "new_cell_data_set",
    "normalized_counts",
    "order_cells",
    "partitions",
    "plot_cells",
    "plot_cells_3d",
    "plot_genes_by_group",
    "plot_genes_in_pseudotime",
    "plot_genes_violin",
    "plot_pc_variance_explained",
    "plot_percent_cells_positive",
    "preprocess_cds",
    "principal_graph",
    "pseudotime",
    "reduce_dimension",
    "saveRDS",
    "search_nn_index",
    "search_nn_matrix",
    "set_cds_nn_index",
    "set_cds_row_order_matrix",
    "set_matrix_control",
    "size_factors",
    "top_markers",
]
