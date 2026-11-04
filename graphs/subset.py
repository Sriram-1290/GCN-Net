"""Graphs and standardisers for an electrode subset, derived exactly from moments.

Why this module exists
----------------------
A model tested on a subset of electrodes must see the same *kind* of input it was
built for: a common-average-referenced (CAR) signal over **that subset**, a graph
whose edge weights are correlations of that signal, and a per-channel
standardiser fitted to it. All three come from **source subjects only** (the
subjects the model was pre-trained on); no target data, training or test, ever
enters them. That is the project's leakage rule applied to the graph and to the
normalisation statistics.

Recomputing them from ~92 subjects' raw epochs for every one of the hundreds of
subsets scored would be far too slow. The shortcut below is not an
approximation: it is the same number, reached through accumulated second
moments.

The algebra (why it is exact)
-----------------------------
Let ``X`` be the no-reference data, ``(trials, C, time)``, and let each
(trial, time) sample be a column vector ``x`` in R^C. Pool trials and time into
``n`` samples and accumulate, once per source set,

    s = sum_t x_t           (C,)      outer = sum_t x_t x_t^T   (C, C)      n

so that the population mean and covariance are ``mu = s / n`` and
``Sigma = outer / n - mu mu^T``.

Restrict to a kept subset of ``S`` channels: take the ``S``-vector ``x_K`` (the
sub-vector of ``x``). CAR over the subset subtracts the subset's own mean from
every kept channel at every sample, which is a *linear* map

    x_K  ->  H x_K,        H = I_S - (1/S) 1 1^T     (the centring matrix).

A linear map transforms the moments of a distribution in closed form:

    mean of H x_K        = H mu_K
    covariance of H x_K  = H Sigma_KK H^T = H Sigma_KK H     (H is symmetric).

``mu_K`` and ``Sigma_KK`` are just the kept rows/columns of ``mu`` and
``Sigma``, so every subset's statistics are read off one set of moments.
From them:

* the standardiser's mean is ``H mu_K`` and its std is ``sqrt(diag(H Sigma H))``
  (population std over trials x time, exactly what ``fit_standardizer``
  computes);
* the Pearson matrix is ``P_ij = C_ij / (std_i std_j)`` with ``C = H Sigma H``,
  which is the matrix ``np.corrcoef`` returns for the CAR'd data, so
  ``A = |P| - I`` clipped to [0, 1] is the adjacency ``pcc_adjacency`` returns;
* the normalised Laplacian, its rescaling and the Chebyshev basis are then the
  existing functions applied to that adjacency, unchanged.

Because the moments are sums, ``Moments`` objects add: the source set's moments
are the sum of its subjects' moments, and a target subject is excluded simply by
never adding theirs.

Numerics: moments are float64. ``Sigma = outer/n - mu mu^T`` is a difference of
two close numbers when the channel offsets dwarf their variance, so the variance
is clipped at zero before the square root; for EEG that has been band-pass
filtered the offsets are negligible and the result agrees with the direct
computation to well inside 1e-6 (tested).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.montage import region_vector
from data.normalize import Standardizer
from graphs.laplacian import chebyshev_basis, laplacian, rescale_laplacian

# Same floor as data.normalize.fit_standardizer, so a subset the direct path would
# refuse is refused here too.
_EPS = 1e-8


@dataclass(frozen=True)
class Moments:
    """Accumulated first and second moments of no-reference data.

    Attributes:
        sum: ``(C,)`` float64, sum over every (trial, time) sample.
        outer: ``(C, C)`` float64, sum of ``x x^T`` over the same samples.
        n: number of samples, i.e. trials x time samples (not trials).
    """

    sum: np.ndarray
    outer: np.ndarray
    n: int

    def __add__(self, other: "Moments") -> "Moments":
        if not isinstance(other, Moments):
            return NotImplemented
        if self.sum.shape != other.sum.shape:
            raise ValueError(
                f"cannot add moments over {self.sum.shape[0]} and "
                f"{other.sum.shape[0]} channels"
            )
        return Moments(self.sum + other.sum, self.outer + other.outer, self.n + other.n)


def moments(X_noref: np.ndarray) -> Moments:
    """Moments of ``(trials, channels, time)`` no-reference data, over trials and time."""
    X_noref = np.asarray(X_noref)
    if X_noref.ndim != 3:
        raise ValueError(f"expected (trials, channels, time), got {X_noref.shape}")
    trials, channels, time = X_noref.shape
    if trials == 0 or time == 0:
        raise ValueError("cannot take moments of an empty array")
    # (channels, trials*time) in float64: every channel pair shares the same samples.
    flat = X_noref.transpose(1, 0, 2).reshape(channels, trials * time).astype(np.float64)
    return Moments(flat.sum(axis=1), flat @ flat.T, trials * time)


@dataclass(frozen=True)
class SubsetRepresentation:
    """Everything a model needs to be scored on one electrode subset.

    ``basis``, ``regions`` and ``standardizer`` are all indexed by the kept
    channels *in montage order* (``np.flatnonzero(keep)``), so row ``i`` of each
    is the same electrode. Building them in different orders would make the model
    read one electrode's power as another region's, with no error raised.
    """

    keep: np.ndarray            # (C,) bool, over the full montage
    standardizer: Standardizer  # mean/std (S, 1)
    basis: np.ndarray           # (K, S, S) float32 Chebyshev basis
    regions: np.ndarray         # (S,) int64 region index of each kept channel


def subset_representation(m: Moments, keep: np.ndarray, ch_names: list[str],
                          K: int = 2, lmax: float = 2.0) -> SubsetRepresentation:
    """Standardiser, Chebyshev basis and region vector of the subset ``keep``.

    Equivalent to fitting ``fit_standardizer`` / ``pcc_adjacency`` on
    ``common_average(X, keep)`` for the data ``m`` was accumulated from; see the
    module docstring for the proof. Pass moments of **source subjects only**.

    ``Standardizer.n_trials_fit`` is set to ``m.n``, the number of trial x time
    samples, because the moments do not retain the trial count.

    Raises:
        ValueError: on a malformed ``keep``/``ch_names``, or if any kept channel
            has a standard deviation below 1e-8 after centring.
    """
    n_channels = m.sum.shape[0]
    keep = np.asarray(keep)
    if keep.dtype != bool or keep.shape != (n_channels,):
        raise ValueError(
            f"keep must be a boolean vector of length {n_channels}; "
            f"got dtype {keep.dtype}, shape {keep.shape}"
        )
    if len(ch_names) != n_channels:
        raise ValueError(f"ch_names has {len(ch_names)} entries for {n_channels} channels")
    if not keep.any():
        raise ValueError("keep selects no channels")
    if m.n <= 0:
        raise ValueError("moments hold no samples")

    idx = np.flatnonzero(keep)  # montage order: the one order used throughout
    size = idx.size

    mu = m.sum[idx] / m.n
    sigma = m.outer[np.ix_(idx, idx)] / m.n - np.outer(mu, mu)

    # CAR over the subset is x -> H x with H the centring matrix (module docstring).
    H = np.eye(size) - np.full((size, size), 1.0 / size)
    mean = H @ mu
    cov = H @ sigma @ H
    cov = 0.5 * (cov + cov.T)

    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    dead = np.flatnonzero(std < _EPS)
    if dead.size:
        raise ValueError(
            f"channels {idx[dead].tolist()} have ~zero variance after centring "
            f"over the subset"
        )

    corr = cov / np.outer(std, std)
    A = np.clip(np.abs(corr) - np.eye(size), 0.0, 1.0)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)

    basis = chebyshev_basis(rescale_laplacian(laplacian(A), lmax), K).astype(np.float32)
    standardizer = Standardizer(mean=mean[:, None], std=std[:, None], n_trials_fit=int(m.n))
    regions = region_vector([ch_names[i] for i in idx])
    return SubsetRepresentation(keep=keep.copy(), standardizer=standardizer,
                                basis=basis, regions=regions)
