"""Node features derived from the signal, rather than raw voltage.

Motivation is Gate A (project_documentation.md 9.5): CSP+LDA beats the network,
and CSP is mechanically *a spatial filter followed by log band power*. The network
is currently handed raw microvolts and asked to infer band power from a waveform
through six graph convolutions. Making it an explicit node feature gives the
network the representation its winning competitor uses.

This is also what the recent literature does — SAMGCN's differential entropy is
log band power for a Gaussian, and MHD-GCN's multi-level wavelet blocks produce
per-band representations (8.2).

Band power is computed from the **unfiltered** cache. A band-passed cache would
have no power outside its passband, so the filter bank has to see the raw signal.
"""

from __future__ import annotations

import numpy as np

#: Filter-bank style bands, 4 Hz wide, spanning theta through low gamma. Follows
#: the FBCSP convention rather than being hand-picked, so the choice is defensible
#: without a sweep. `mu_beta` is the narrower, more standard motor-imagery pair.
BAND_SETS: dict[str, tuple[tuple[float, float], ...]] = {
    "fbcsp": ((4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32),
              (32, 36), (36, 40)),
    "mu_beta": ((8, 13), (13, 20), (20, 30)),
    "mu_beta_fine": ((8, 10), (10, 13), (13, 16), (16, 20), (20, 25), (25, 30)),
    "broad": ((4, 8), (8, 13), (13, 30), (30, 45)),
}


def band_power_features(X: np.ndarray, sfreq: float,
                        bands: tuple[tuple[float, float], ...],
                        window: int, stride: int, *, log: bool = True,
                        eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Log band power per channel per window.

    A Hann taper is applied before the FFT to limit spectral leakage between
    adjacent bands — without it a strong mu rhythm bleeds into the beta bins and
    the bands stop being independent features.

    Args:
        X: ``(n_trials, n_channels, n_samples)``, unfiltered.
        sfreq: sampling rate, 160 Hz for this dataset.
        bands: ``((low, high), ...)`` in Hz, half-open.
        window: samples per window. Frequency resolution is ``sfreq / window``,
            so 160 (1 s) gives 1 Hz and 64 (0.4 s) gives 2.5 Hz — too coarse to
            separate 4 Hz bands cleanly.
        stride: hop between windows.
        log: take the log, which makes the roughly log-normal power distribution
            approximately Gaussian. This is what differential entropy amounts to.

    Returns:
        ``(features, trial_index)`` where features is
        ``(n_trials * n_windows, n_channels, n_bands)`` and trial_index maps each
        row back to its trial.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (trials, channels, samples), got {X.shape}")
    n_trials, n_channels, n_time = X.shape
    if not 1 <= window <= n_time:
        raise ValueError(f"window {window} must be in [1, {n_time}]")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    resolution = sfreq / window
    narrowest = min(hi - lo for lo, hi in bands)
    if narrowest < 2 * resolution:
        raise ValueError(
            f"window {window} gives {resolution:.2f} Hz resolution, too coarse for "
            f"a {narrowest} Hz band; use a longer window"
        )

    starts = np.arange(0, n_time - window + 1, stride)
    freqs = np.fft.rfftfreq(window, d=1.0 / sfreq)
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in bands]
    for (lo, hi), m in zip(bands, masks):
        if not m.any():
            raise ValueError(f"band {lo}-{hi} Hz contains no FFT bin at this resolution")

    taper = np.hanning(window).astype(np.float64)
    out = np.empty((n_trials, len(starts), n_channels, len(bands)), dtype=np.float32)

    for w, s in enumerate(starts):
        seg = X[:, :, s:s + window].astype(np.float64) * taper
        psd = np.abs(np.fft.rfft(seg, axis=-1)) ** 2
        for b, m in enumerate(masks):
            power = psd[..., m].sum(axis=-1)
            out[:, w, :, b] = np.log(power + eps) if log else power

    features = out.reshape(n_trials * len(starts), n_channels, len(bands))
    trial_index = np.repeat(np.arange(n_trials, dtype=np.int64), len(starts))
    return features, trial_index


def resolve_bands(name_or_bands: str | tuple[tuple[float, float], ...]
                  ) -> tuple[tuple[float, float], ...]:
    """Look up a named band set, or pass an explicit one through."""
    if isinstance(name_or_bands, str):
        try:
            return BAND_SETS[name_or_bands]
        except KeyError:
            raise ValueError(
                f"unknown band set {name_or_bands!r}; known: {sorted(BAND_SETS)}"
            ) from None
    return tuple(tuple(b) for b in name_or_bands)
