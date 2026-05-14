"""Port of batchelor::reducedMNN restricted to the monocle3 R caller.

Gold-standard target — the only batchelor entry monocle3's alignment.R hits:

    batchelor::reducedMNN(matrix, batch=batch_vec, k=alignment_k)

All other parameters take their batchelor defaults: ``prop.k=NULL``,
``restrict=NULL``, ``ndist=3``, ``merge.order=NULL`` (so the binary
merge tree degenerates to a sequential left-fold), ``auto.merge=FALSE``,
``min.batch.skip=0``, and ``BNPARAM=KmknnParam()`` (exact k-means kNN).

The full ``.fast_mnn_core`` machinery in batchelor/R/fastMNN.R supports
auto-merge tree search, MNN-restriction subsetting, prop.k scaling,
batch-magnitude skipping, and lost-variance bookkeeping — **none** of
which monocle3 reaches. Porting only the predefined sequential path
shrinks the surface from ~1300 LOC of R/C++ to the ~280 LOC below
while preserving exact algorithmic parity for our caller.

Helper-by-helper correspondence (R source → Python port):

    divideIntoBatches                       -> _divide_into_batches
    BiocNeighbors::findMutualNN             -> _find_mutual_nn
    BiocNeighbors/src/find_mutual_nns.cpp   -> _find_mutual_nns
    fastMNN.R::.average_correction          -> _average_correction
    fastMNN.R::.get_batch_magnitude         -> _get_batch_magnitude
    fastMNN.R::.center_along_batch_vector   -> _center_along_batch_vector
    fastMNN.R::.orthogonalize_other         -> _orthogonalize_other
    fastMNN.R::.tricube_weighted_correction -> _tricube_weighted_correction
    utils_tricube.R::.compute_tricube_average -> _compute_tricube_average
    fastMNN.R::.fast_mnn_core (subset)      -> reduced_mnn

The C++ kernel ``find_mutual_nns`` is re-implemented in numpy
(vectorised, chunked over the left side). At monocle3 scale (≤ 100 k
cells, k ≤ 50) the cost is dominated by the kNN search; the MNN
intersection itself is sub-second either way.

kNN backend: ``sklearn.neighbors.NearestNeighbors(algorithm='auto')``.
R uses ``BiocNeighbors::KmknnParam()`` which is also exact (k-means
based). Both return identical neighbour sets for our PCA-reduced
inputs; tie-breaking differences in 50-dim float64 distances are
essentially impossible.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

__all__ = ["reduced_mnn"]


# ---------------------------------------------------------------------------
# Port of BiocNeighbors/src/find_mutual_nns.cpp.
# ---------------------------------------------------------------------------


def _find_mutual_nns(
    left_knn: np.ndarray, right_knn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised numpy port of ``find_mutual_nns.cpp``.

    Parameters
    ----------
    left_knn : (n_left, k2) ndarray, 0-indexed
        For each row ``l`` in the left batch, the k2 nearest right rows.
    right_knn : (n_right, k1) ndarray, 0-indexed
        For each row ``r`` in the right batch, the k1 nearest left rows.

    Returns
    -------
    (mutual_left, mutual_right) : 0-indexed paired ndarrays
        Row-major emission order matches the C++ original (outer over
        ``l``, inner over the columns of ``left_knn[l]``).

    The C++ uses sort + binary search for O(n*k*log k). For numpy a
    direct equality test ``right_knn[r] == l`` over the chunk is
    equivalent in time and trivial to read. Memory is capped via
    chunking to ~100 MB to stay safe on larger datasets.
    """
    n_left, k2 = left_knn.shape
    if n_left == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    k1 = right_knn.shape[1]
    chunk = max(1, 100_000_000 // (k2 * k1 * 8))

    out_l: list[np.ndarray] = []
    out_r: list[np.ndarray] = []
    for start in range(0, n_left, chunk):
        end = min(start + chunk, n_left)
        sub = left_knn[start:end]
        l_rep = np.repeat(np.arange(start, end, dtype=np.int64), k2)
        r_rep = sub.ravel()
        # For each candidate (l, r) pair, check whether l appears among
        # right_knn[r]. Vectorised broadcast: (n*k2, k1) bool matrix.
        is_mutual = (right_knn[r_rep] == l_rep[:, None]).any(axis=1)
        out_l.append(l_rep[is_mutual])
        out_r.append(r_rep[is_mutual])
    return np.concatenate(out_l), np.concatenate(out_r)


# ---------------------------------------------------------------------------
# Port of BiocNeighbors/R/findMutualNN.R::findMutualNN.
# ---------------------------------------------------------------------------


def _find_mutual_nn(
    data1: np.ndarray, data2: np.ndarray, k1: int, k2: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest neighbours between two datasets.

    Mirrors ``BiocNeighbors::findMutualNN`` with default ``KmknnParam``.
    For each row of ``data1`` find its k2 nearest in ``data2``; for each
    row of ``data2`` find its k1 nearest in ``data1``; return the pairs
    that hold in both directions.

    Returns 0-indexed ``(first, second)`` arrays — ``first`` indexes
    ``data1``, ``second`` indexes ``data2``.
    """
    safe_k2 = max(1, min(int(k2), data2.shape[0]))
    safe_k1 = max(1, min(int(k1), data1.shape[0]))
    if data1.shape[0] == 0 or data2.shape[0] == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

    nn2 = NearestNeighbors(n_neighbors=safe_k2, algorithm="auto").fit(data2)
    W21 = nn2.kneighbors(data1, return_distance=False)  # (n_data1, k2)

    nn1 = NearestNeighbors(n_neighbors=safe_k1, algorithm="auto").fit(data1)
    W12 = nn1.kneighbors(data2, return_distance=False)  # (n_data2, k1)

    return _find_mutual_nns(W21, W12)


# ---------------------------------------------------------------------------
# Port of batchelor/R/utils_tricube.R::.compute_tricube_average.
# ---------------------------------------------------------------------------


def _compute_tricube_average(
    vals: np.ndarray,
    indices: np.ndarray,
    distances: np.ndarray,
    bandwidth: np.ndarray | None = None,
    ndist: float = 3.0,
) -> np.ndarray:
    """Tricube-weighted local average — port of ``.compute_tricube_average``.

    Bandwidth defaults to ``ndist * middle-distance`` per row where
    ``middle = ceiling(k/2)`` (R, 1-indexed; column ``(k+1)//2 - 1`` in
    0-indexed Python). The ``pmax(1e-8, bandwidth)`` floor is preserved
    to guard against zero distances.

    Parameters
    ----------
    vals : (n_src, d) ndarray
        Source values to average.
    indices : (n_query, k) ndarray, 0-indexed
        For each query row, the k source rows to read from ``vals``.
    distances : (n_query, k) ndarray
        Corresponding distances driving the tricube weights.
    bandwidth : (n_query,) ndarray, optional
        Per-row bandwidth. If None, computed from distances.
    ndist : float, default 3.0
        Bandwidth scale factor.

    Returns
    -------
    (n_query, d) ndarray of weighted averages.
    """
    n_query, k = indices.shape
    if k == 0:
        # R returns matrix(0, nrow(vals), ncol(vals)); we mirror.
        return np.zeros((n_query, vals.shape[1]), dtype=np.float64)

    if bandwidth is None:
        middle_0 = (k + 1) // 2 - 1  # ceiling(k/2L) - 1, 0-indexed
        bandwidth = distances[:, middle_0] * float(ndist)
    bandwidth = np.maximum(1e-8, bandwidth)

    rel_dist = distances / bandwidth[:, None]
    rel_dist = np.minimum(rel_dist, 1.0)
    tricube = (1.0 - rel_dist ** 3) ** 3
    row_sums = tricube.sum(axis=1)
    # rel_dist <= 1 → tricube >= 0. Row sum could be zero only if every
    # rel_dist equals 1 (all neighbours at the bandwidth radius), in
    # which case fall back to uniform weighting.
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    weight = tricube / row_sums[:, None]

    output = np.zeros((n_query, vals.shape[1]), dtype=np.float64)
    for kdx in range(k):
        output += vals[indices[:, kdx]] * weight[:, kdx : kdx + 1]
    return output


# ---------------------------------------------------------------------------
# Port of batchelor/R/fastMNN.R helpers used by .fast_mnn_core.
# ---------------------------------------------------------------------------


def _average_correction(
    refdata: np.ndarray,
    mnn1: np.ndarray,
    curdata: np.ndarray,
    mnn2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of fastMNN.R:567 ``.average_correction``.

    For each unique cell in ``mnn2``, average the ``(refdata[mnn1] -
    curdata[mnn2])`` correction vectors over all MNN pairs that touch
    that cell.

    Inputs are 0-indexed. Output ``second`` is sorted ascending (matches
    R's ``rowsum`` / ``table`` semantics).

    Returns
    -------
    averaged : (n_unique_mnn2, d) ndarray
    second   : (n_unique_mnn2,)   ndarray of unique 0-indexed cur indices
    """
    corvec = refdata[mnn1] - curdata[mnn2]
    unique_mnn2, inverse = np.unique(mnn2, return_inverse=True)
    n_unique = unique_mnn2.size
    summed = np.zeros((n_unique, refdata.shape[1]), dtype=np.float64)
    np.add.at(summed, inverse, corvec)
    counts = np.bincount(inverse, minlength=n_unique).astype(np.float64)
    averaged = summed / counts[:, None]
    return averaged, unique_mnn2


def _get_batch_magnitude(
    correction: np.ndarray, ave: np.ndarray | None = None,
) -> float:
    """Port of fastMNN.R:582 ``.get_batch_magnitude``.

    Standardised magnitude of the average batch vector against per-pair
    magnitudes. Unused by monocle3 (``min.batch.skip=0`` so no skipping)
    but kept for parity / future opt-in.
    """
    if ave is None:
        ave = correction.mean(axis=0)
    ave_l2sq = float((correction ** 2).mean(axis=0).sum())
    if ave_l2sq == 0:
        return 0.0
    l2sq = float((ave ** 2).sum())
    return float(np.sqrt(l2sq / ave_l2sq))


def _center_along_batch_vector(
    mat: np.ndarray, batch_vec: np.ndarray,
) -> np.ndarray:
    """Port of fastMNN.R:626 ``.center_along_batch_vector`` (no restrict).

    Project each row of ``mat`` onto ``batch_vec``, then shift every row
    to the batch-mean position along that direction. Removes the
    along-vec variance component.

    R's division-by-zero behaviour is preserved: if ``batch_vec`` is the
    zero vector, the unit-normalisation produces NaN and the output is
    NaN. With ``min.batch.skip=0`` monocle3 never enters this branch
    intentionally; an exactly-zero ``overall.batch`` only arises with
    perfectly cancelling correction pairs, vanishingly rare in
    PCA-reduced float64 data.
    """
    norm = np.sqrt(float((batch_vec ** 2).sum()))
    bv = batch_vec / norm
    batch_loc = mat @ bv
    central_loc = float(batch_loc.mean())
    return mat + np.outer(central_loc - batch_loc, bv)


def _orthogonalize_other(
    data: np.ndarray, vectors: list[np.ndarray],
) -> np.ndarray:
    """Port of fastMNN.R:642 ``.orthogonalize_other`` (no restrict).

    Sequentially centre ``data`` along each vector in ``vectors``.
    Empty list is a no-op.
    """
    for vec in vectors:
        data = _center_along_batch_vector(data, vec)
    return data


def _tricube_weighted_correction(
    curdata: np.ndarray,
    correction: np.ndarray,
    in_mnn: np.ndarray,
    k: int,
    ndist: float = 3.0,
) -> np.ndarray:
    """Port of fastMNN.R:599 ``.tricube_weighted_correction``.

    For each row of ``curdata``, find its k nearest among
    ``curdata[in_mnn]`` (the MNN-participating subset), then apply a
    tricube-weighted average of their ``correction`` vectors.

    Parameters
    ----------
    curdata : (n, d) ndarray
        All current-batch cells.
    correction : (n_in_mnn, d) ndarray
        Per-in-MNN-cell correction (averaged across MNN pairs).
    in_mnn : (n_in_mnn,) ndarray, 0-indexed
        Indices of MNN-participating cells in ``curdata``.
    k : int
        Tricube-kNN.
    ndist : float
        Tricube bandwidth scale.
    """
    cur_uniq = curdata[in_mnn]
    safe_k = max(1, min(int(k), cur_uniq.shape[0]))
    nn = NearestNeighbors(n_neighbors=safe_k, algorithm="auto").fit(cur_uniq)
    closest_dist, closest_idx = nn.kneighbors(curdata, return_distance=True)
    weighted = _compute_tricube_average(
        correction, closest_idx, closest_dist, ndist=ndist,
    )
    return curdata + weighted


# ---------------------------------------------------------------------------
# Port of batchelor/R/divideIntoBatches.R (byrow=TRUE, no restrict).
# ---------------------------------------------------------------------------


def _divide_into_batches(
    X: np.ndarray, batch,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Port of ``divideIntoBatches.R::divideIntoBatches`` (byrow=TRUE).

    Returns
    -------
    batches : list of (n_b, d) ndarrays, in level order (np.unique → sorted asc).
    reorder : (n_cells,) 0-indexed positions. After running the merge
              algorithm on ``vstack(batches)`` to produce ``output``,
              ``output[reorder]`` recovers the input cell order. (R uses
              the same indexing semantics, just 1-indexed; we shift.)
    levels  : (n_levels,) ndarray, unique batch labels in level order.
    """
    batch_arr = np.asarray(batch)
    levels = np.unique(batch_arr)
    batches: list[np.ndarray] = []
    reorder = np.empty(X.shape[0], dtype=np.int64)
    covered = np.zeros(X.shape[0], dtype=bool)
    last = 0
    for lvl in levels:
        mask = batch_arr == lvl
        N = int(mask.sum())
        batches.append(X[mask])
        reorder[mask] = last + np.arange(N, dtype=np.int64)
        covered |= mask
        last += N
    if not covered.all():
        n_missing = int((~covered).sum())
        raise ValueError(
            f"_divide_into_batches: {n_missing} cells have batch labels "
            "that do not appear in np.unique(batch) (e.g. NaN/None)."
        )
    return batches, reorder, levels


# ---------------------------------------------------------------------------
# Main entry: port of batchelor::reducedMNN restricted to monocle3's call.
# ---------------------------------------------------------------------------


def reduced_mnn(
    X: np.ndarray,
    batch,
    k: int = 20,
    ndist: float = 3.0,
) -> dict:
    """Port of ``batchelor::reducedMNN(matrix, batch=batch, k=k)``.

    Restricted to monocle3's R caller (``alignment.R::align_cds``):
    no auto.merge, no restrict, no prop.k, no merge.order, no
    min.batch.skip; exact ``KmknnParam`` kNN; ``ndist=3``.

    The merge tree built by ``.create_tree_predefined`` with
    ``merge.order=NULL`` degenerates to a sequential left-fold:
    ``(((((B0, B1), B2), B3), ...), Bn)``. We replicate that fold
    in a single loop.

    Parameters
    ----------
    X : (n_cells, n_dims) ndarray
        Already-PCA-reduced coordinates. Modified in place is not
        performed; a fresh float64 copy is returned.
    batch : array-like, length n_cells
        Batch labels. Must not contain NaN/None.
    k : int, default 20
        Number of mutual neighbours.
    ndist : float, default 3.0
        Tricube bandwidth multiplier.

    Returns
    -------
    dict with
        corrected : (n_cells, n_dims) ndarray, same cell order as ``X``.
        batch     : (n_cells,) ndarray, batch labels echoed back.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if len(batch) != X.shape[0]:
        raise ValueError("len(batch) must equal X.shape[0]")
    if int(k) < 1:
        raise ValueError("k must be >= 1")

    batches, reorder, _levels = _divide_into_batches(X, batch)
    if len(batches) < 2:
        return {"corrected": X.copy(), "batch": np.asarray(batch)}

    left = batches[0].astype(np.float64, copy=True)
    # `extras` mirrors batchelor's `extras` slot on the accumulating left
    # merge-tree node: the running list of overall batch vectors from
    # previous merges. In a sequential left-fold the right side is always
    # a fresh leaf with empty extras, so only `extras` (=left.extras)
    # contributes to orthogonalisation each step.
    extras: list[np.ndarray] = []

    for right_orig in batches[1:]:
        right = right_orig.astype(np.float64, copy=True)

        # R fastMNN.R:473-474: orthogonalise each side against the *other*
        # side's extras. In sequential merge right.extras is always [], so
        # the corresponding left orthogonalisation is a no-op and is
        # omitted; the right side is centred against the accumulated
        # left.extras = `extras`.
        right = _orthogonalize_other(right, extras)

        mnn_l, mnn_r = _find_mutual_nn(left, right, k1=int(k), k2=int(k))

        if mnn_l.size == 0:
            # batchelor: with min.batch.skip=NA the always-correct branch
            # would still try .average_correction on empty input and crash.
            # With min.batch.skip=0 (our setting), an empty MNN set is
            # effectively a skipped merge: we concatenate without
            # correction and we do NOT extend `extras`.
            left = np.vstack([left, right])
            continue

        # R fastMNN.R:480-481: per-cell average correction, then overall.
        avg, _in_mnn = _average_correction(left, mnn_l, right, mnn_r)
        overall = avg.mean(axis=0)

        # R fastMNN.R:496-497: centre both sides along the batch vector.
        left = _center_along_batch_vector(left, overall)
        right = _center_along_batch_vector(right, overall)

        # R fastMNN.R:505-507: recompute correction after centring and
        # tricube-smooth it over all right cells.
        avg2, in_mnn2 = _average_correction(left, mnn_l, right, mnn_r)
        right = _tricube_weighted_correction(
            right, avg2, in_mnn2, k=int(k), ndist=float(ndist),
        )

        # R fastMNN.R:520-525: rbind left, right and extend extras.
        left = np.vstack([left, right])
        extras.append(overall)

    # Recover the input cell order. After divideIntoBatches we know
    # original cell `i` lives at `reorder[i]` in the vstack output, so the
    # inverse mapping is just `left[reorder]`.
    return {
        "corrected": left[reorder],
        "batch": np.asarray(batch),
    }
