"""Tests for the shared tangent-space helpers in ``data/riemann.py``.

The helpers moved out of ``scripts/run_tangent_space.py`` so that the group-transfer
runner (A1, A6) and the confound audit (A3, A4) import them from one place. The move
must not change behaviour: the subject-dependent tangent-space runner keeps working on
exactly the same functions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import riemann  # noqa: E402


def _spd(rng, c=5):
    A = rng.normal(size=(c, c))
    return A @ A.T + c * np.eye(c)


def test_covariances_are_shrunk_spd_with_the_default_of_0_1():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 6, 50)).astype(np.float32)
    covs = riemann.covariances(X)
    assert covs.shape == (4, 6, 6)
    for x, c in zip(X, covs):
        s = np.cov(x.astype(np.float64))
        expect = 0.9 * s + 0.1 * (np.trace(s) / 6) * np.eye(6)
        assert np.allclose(c, expect)
        assert np.linalg.eigvalsh(c).min() > 0


def test_covariances_stay_positive_definite_on_rank_deficient_input():
    """Common average reference leaves rank C-1; shrinkage must restore full rank."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 8, 200))
    X -= X.mean(axis=1, keepdims=True)
    assert np.linalg.matrix_rank(np.cov(X[0])) == 7
    assert np.linalg.eigvalsh(riemann.covariances(X)).min() > 0


def test_riemannian_mean_of_copies_is_the_matrix():
    rng = np.random.default_rng(2)
    S = _spd(rng)
    assert np.allclose(riemann.riemannian_mean(np.stack([S, S, S])), S)


def test_riemannian_mean_of_commuting_matrices_is_the_geometric_mean():
    a, b = np.diag([1.0, 4.0, 9.0]), np.diag([4.0, 1.0, 1.0])
    M = riemann.riemannian_mean(np.stack([a, b]))
    assert np.allclose(M, np.diag([2.0, 2.0, 3.0]), atol=1e-6)


def test_tangent_space_of_the_reference_is_zero_and_norm_preserving():
    rng = np.random.default_rng(3)
    ref, C = _spd(rng), _spd(rng)
    z = riemann.tangent_space(np.stack([ref, C]), ref)
    assert z.shape == (2, 5 * 6 // 2)
    assert np.allclose(z[0], 0.0, atol=1e-10)
    _, Ris = riemann.sqrtm_inv(ref)
    L = riemann.logm_spd(Ris @ C @ Ris)
    assert np.isclose(np.linalg.norm(z[1]), np.linalg.norm(L, "fro"))


def test_expm_inverts_logm():
    rng = np.random.default_rng(4)
    S = _spd(rng)
    assert np.allclose(riemann.expm_spd(riemann.logm_spd(S)), S)


def _load_tangent_space():
    spec = importlib.util.spec_from_file_location(
        "run_tangent_space", ROOT / "scripts" / "run_tangent_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_subject_dependent_runner_uses_the_shared_helpers():
    """``run_tangent_space.py`` keeps its old names, now bound to ``data.riemann``."""
    m = _load_tangent_space()
    assert m._covariances is riemann.covariances
    assert m.riemannian_mean is riemann.riemannian_mean
    assert m.tangent_space is riemann.tangent_space
    assert m._sqrtm_inv is riemann.sqrtm_inv
    assert m._logm_spd is riemann.logm_spd
    assert m._expm_spd is riemann.expm_spd


def test_subject_dependent_runner_still_decodes_a_separable_problem():
    m = _load_tangent_space()
    rng = np.random.default_rng(5)
    y = np.tile([0, 1], 40)
    X = rng.normal(size=(80, 6, 200))
    X[y == 0, 0] *= 6.0
    X[y == 1, 3] *= 6.0
    mean, folds = m.tangent_accuracy(X.astype(np.float32), y, n_splits=5, seed=42)
    assert len(folds) == 5 and mean > 0.9
