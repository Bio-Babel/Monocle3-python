"""monocle3-python — Python port of the R monocle3 package.

See ``docs/index.md`` for an overview and ``tutorials/`` for runnable
notebooks. The public surface is re-exported here; submodules are free
for callers to import directly as well.
"""

from __future__ import annotations

__version__ = "1.4.26"
__r_commit__ = "4f4239a"

from ._accessors import size_factors
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
from .label_transfer import fix_missing_cell_labels, transfer_cell_labels
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
from .projection import preprocess_transform, reduce_dimension_transform
from .reduce_dimensions import reduce_dimension

__all__ = [
    "__version__",
    "__r_commit__",
    "aggregate_gene_expression",
    "align_cds",
    "cluster_cells",
    "clusters",
    "coefficient_table",
    "compare_models",
    "detect_genes",
    "estimate_size_factors",
    "evaluate_fits",
    "find_gene_modules",
    "fit_models",
    "fix_missing_cell_labels",
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
    "preprocess_transform",
    "principal_graph",
    "pseudotime",
    "reduce_dimension",
    "reduce_dimension_transform",
    "search_nn_index",
    "search_nn_matrix",
    "set_cds_nn_index",
    "size_factors",
    "top_markers",
    "transfer_cell_labels",
]
