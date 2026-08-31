"""Graclus multilevel graph coarsening, ported from Defferrard's ``cnn_graph``.

This is what makes ``mpool1`` meaningful. Max-pooling with stride 2 over a graph
only makes sense if node *2i* and node *2i+1* are the two children of one node in
the coarser graph. ``coarsen`` builds that hierarchy by repeated heavy-edge
matching, and ``compute_perm`` reorders every layer so that the binary-tree
structure becomes plain adjacency in memory — after which pooling is literally a
stride-2 max over the node axis.

Fake nodes are the subtle part. A node that fails to pair (a singleton) is given
one fake sibling; a node in the coarse graph with no children is given two. Fake
nodes are appended after the real ones and their features are filled with zeros,
so max-pooling selects the real sibling. Get this wrong and the hierarchy
misaligns silently — the model still trains, it just pools unrelated electrodes.

Deviation from the reference, deliberate
----------------------------------------
``metis_one_level`` in ``cnn_graph`` computes its row lengths with an off-by-one:
``rowlength[count]`` is incremented before ``count`` advances, so every row except
the last is one too long and the neighbour scan reads the first entry of the
*next* row. That makes the matching consider a node that is not a neighbour. Here
the row offsets are computed directly from the sorted row indices, which is
correct. The consequence is a slightly different (valid) matching, not a
different algorithm.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse


def _row_offsets(rr: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Start index and length of each row in a row-sorted COO triplet list.

    Computed from the data rather than by the reference's incremental scan; see
    the module docstring for why.
    """
    lengths = np.bincount(rr, minlength=n_nodes).astype(np.int64)
    starts = np.zeros(n_nodes, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    return starts, lengths


def graclus_one_level(rr: np.ndarray, cc: np.ndarray, vv: np.ndarray,
                      rid: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """One level of normalised-cut heavy-edge matching.

    Visits nodes in the order given by ``rid``. Each unmarked node is paired with
    the unmarked neighbour maximising the normalised cut weight

        w(i, j) * (1/d_i + 1/d_j)

    which is the Graclus criterion: prefer heavy edges between low-degree nodes.
    A node with no unmarked neighbour becomes a singleton cluster.

    Returns:
        ``cluster_id``, mapping each node to its cluster in the coarser graph.
    """
    n_nodes = len(rid)
    marked = np.zeros(n_nodes, dtype=bool)
    cluster_id = np.zeros(n_nodes, dtype=np.int64)
    rowstart, rowlength = _row_offsets(rr, n_nodes)

    clustercount = 0
    for ii in range(n_nodes):
        tid = int(rid[ii])
        if marked[tid]:
            continue

        marked[tid] = True
        wmax = 0.0
        bestneighbor = -1
        rs = rowstart[tid]

        for jj in range(rowlength[tid]):
            nid = int(cc[rs + jj])
            if marked[nid]:
                tval = 0.0
            else:
                # Guard against a zero-degree node; the reference divides blindly.
                di, dj = weights[tid], weights[nid]
                tval = 0.0 if di == 0 or dj == 0 else vv[rs + jj] * (1.0 / di + 1.0 / dj)
            if tval > wmax:
                wmax = tval
                bestneighbor = nid

        cluster_id[tid] = clustercount
        if bestneighbor > -1:
            cluster_id[bestneighbor] = clustercount
            marked[bestneighbor] = True
        clustercount += 1

    return cluster_id


def graclus(W: scipy.sparse.spmatrix, levels: int,
            seed: int | None = 42) -> tuple[list, list[np.ndarray]]:
    """Coarsen a graph ``levels`` times.

    Returns:
        ``(graphs, parents)`` where ``graphs[0]`` is the input and ``graphs[k]``
        is the k-times-coarsened graph, and ``parents[k]`` maps nodes of
        ``graphs[k]`` to nodes of ``graphs[k+1]``.
    """
    W = scipy.sparse.csr_matrix(W)
    n_nodes = W.shape[0]

    # Visit order. The reference randomises the first pass, then orders by degree
    # at each subsequent level. Seeded here so a coarsening is reproducible.
    rng = np.random.default_rng(seed)
    rid = rng.permutation(n_nodes)

    parents: list[np.ndarray] = []
    graphs = [W]
    degree = np.asarray(W.sum(axis=0)).squeeze() - W.diagonal()

    for _ in range(levels):
        weights = np.asarray(degree).squeeze()

        idx_row, idx_col, val = scipy.sparse.find(W)
        order = np.argsort(idx_row, kind="stable")
        rr, cc, vv = idx_row[order], idx_col[order], val[order]

        cluster_id = graclus_one_level(rr, cc, vv, rid, weights)
        parents.append(cluster_id)

        # Contract: edge weights between clusters sum over the merged edges.
        nrr, ncc = cluster_id[rr], cluster_id[cc]
        n_new = int(cluster_id.max()) + 1
        W = scipy.sparse.csr_matrix((vv, (nrr, ncc)), shape=(n_new, n_new))
        W.eliminate_zeros()
        graphs.append(W)

        degree = np.asarray(W.sum(axis=0)).squeeze()
        # Next pass visits low-degree nodes first (the reference's strategy).
        rid = np.argsort(np.asarray(W.sum(axis=0)).squeeze())

    return graphs, parents


def compute_perm(parents: list[np.ndarray]) -> list[list[int]]:
    """Node orderings that turn the coarsening hierarchy into a binary tree.

    After permutation, nodes ``2i`` and ``2i+1`` of layer ``k`` are the children
    of node ``i`` of layer ``k+1``, so pooling is a stride-2 max. Fake nodes are
    appended past the real ones; :func:`perm_data` fills them with zeros.
    """
    indices: list[list[int]] = []
    if len(parents) > 0:
        m_last = int(max(parents[-1])) + 1
        indices.append(list(range(m_last)))

    for parent in parents[::-1]:
        pool_singletons = len(parent)
        indices_layer: list[int] = []

        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            if len(indices_node) > 2:
                raise RuntimeError(
                    f"cluster {i} has {len(indices_node)} children; Graclus "
                    f"matching must produce at most 2"
                )

            if len(indices_node) == 1:
                # Singleton: give it one fake sibling.
                indices_node.append(pool_singletons)
                pool_singletons += 1
            elif len(indices_node) == 0:
                # Childless coarse node: two fake children.
                indices_node.extend([pool_singletons, pool_singletons + 1])
                pool_singletons += 2

            indices_layer.extend(int(x) for x in indices_node)
        indices.append(indices_layer)

    # Each layer must be a clean permutation of range(m_last * 2**i), or the
    # binary-tree invariant is broken and pooling would mix unrelated nodes.
    for i, indices_layer in enumerate(indices):
        expected = m_last * 2 ** i
        if len(indices_layer) != expected:
            raise RuntimeError(f"layer {i} has {len(indices_layer)} nodes, expected {expected}")
        if sorted(indices_layer) != list(range(expected)):
            raise RuntimeError(f"layer {i} ordering is not a permutation of range({expected})")

    return indices[::-1]


def perm_data(X: np.ndarray, indices: list[int] | None) -> np.ndarray:
    """Reorder the **channel** axis of ``(n_trials, n_channels, n_samples)``.

    Fake nodes are filled with zeros so max-pooling selects the real sibling.
    Note this is a change from the reference's signature, which permutes an
    ``(N, M)`` matrix — our data keeps an explicit time axis.
    """
    if indices is None:
        return X
    if X.ndim != 3:
        raise ValueError(f"expected (n_trials, n_channels, n_samples), got {X.shape}")

    n_trials, n_channels, n_samples = X.shape
    n_new = len(indices)
    if n_new < n_channels:
        raise ValueError(f"permutation has {n_new} slots for {n_channels} channels")

    out = np.zeros((n_trials, n_new, n_samples), dtype=X.dtype)
    idx = np.asarray(indices)
    real = idx < n_channels
    out[:, np.flatnonzero(real), :] = X[:, idx[real], :]
    return out


def perm_adjacency(A: scipy.sparse.spmatrix, indices: list[int] | None):
    """Apply the same reordering to an adjacency matrix, padding with isolates."""
    if indices is None:
        return A

    A = scipy.sparse.coo_matrix(A)
    m = A.shape[0]
    n_new = len(indices)
    if n_new < m:
        raise ValueError(f"permutation has {n_new} slots for {m} nodes")

    if n_new > m:
        rows = scipy.sparse.coo_matrix((n_new - m, m), dtype=A.dtype)
        cols = scipy.sparse.coo_matrix((n_new, n_new - m), dtype=A.dtype)
        A = scipy.sparse.vstack([A, rows])
        A = scipy.sparse.hstack([A, cols])

    perm = np.argsort(indices)
    A = A.tocsr()[perm, :][:, perm].tocoo()

    if np.abs(A - A.T).mean() > 1e-9:
        raise RuntimeError("permuted adjacency lost symmetry")
    return A


def coarsen(A: np.ndarray | scipy.sparse.spmatrix, levels: int,
            self_connections: bool = False, seed: int | None = 42
            ) -> tuple[list, list[int] | None]:
    """Full pipeline: coarsen, then permute every layer into binary-tree order.

    Args:
        A: ``(n, n)`` adjacency, dense or sparse.
        levels: number of coarsening steps. The reference uses 5, giving 6 graphs
            (64, 32, 16, 8, 4, 2 nodes for a 64-electrode montage).
        self_connections: keep the diagonal. The reference passes False.

    Returns:
        ``(graphs, perm)`` — ``graphs[k]`` for each layer, and the permutation to
        apply to input data with :func:`perm_data`.
    """
    graphs, parents = graclus(A, levels, seed=seed)
    perms = compute_perm(parents)

    out = []
    for i, G in enumerate(graphs):
        G = scipy.sparse.lil_matrix(G)
        if not self_connections:
            G.setdiag(0)
        if i < levels:
            G = perm_adjacency(G, perms[i])
        G = scipy.sparse.csr_matrix(G)
        G.eliminate_zeros()
        out.append(G)

    return out, (perms[0] if levels > 0 else None)
