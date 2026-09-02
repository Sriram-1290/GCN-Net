"""Tests for the A2 baselines (project_documentation.md 17.2).

A2's whole value is that it runs on **identical folds, cache, seeds and metric**
as the graph arms — otherwise it cannot discriminate the graph-prior explanation
from the sample-size one (17.1), it just adds a number. So the tests here are
less about the architectures (they are published and standard) and more about the
two properties that make the comparison valid:

* the graph-free path through ``prepare_fold`` leaves the leakage-sensitive steps
  untouched — standardiser fit on train only, windows never crossing a trial
  boundary, trial ids preserved for aggregation;
* no node permutation or fake-node padding is applied, so channels reach the
  convolution in montage order.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.pipeline import Trials, prepare_fold
from models.eegnet import EEGNet, ShallowConvNet, build_baseline


def _trials(n=12, c=64, t=640, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, c, t)).astype(np.float32)
    y = np.array([0, 1] * (n // 2), dtype=np.int64)
    return Trials(X=X, y=y)


# ------------------------------------------------------------------- shapes

@pytest.mark.parametrize("name", ["eegnet", "shallow"])
@pytest.mark.parametrize("n_samples", [32, 64, 160, 640])
def test_forward_shape_over_the_window_lengths_we_might_run(name, n_samples):
    model = build_baseline(name, n_channels=64, n_samples=n_samples, n_classes=2)
    out = model(torch.randn(5, 64, n_samples))
    assert out.shape == (5, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["eegnet", "shallow"])
def test_wrong_input_shape_fails_loudly(name):
    """CLAUDE.md: fail loudly on unexpected shapes; silent broadcasting hides bugs."""
    model = build_baseline(name, n_channels=64, n_samples=64, n_classes=2)
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(5, 32, 64))       # wrong channel count
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(5, 64, 64, 1))    # wrong rank


def test_eegnet_refuses_a_window_it_cannot_pool():
    with pytest.raises(ValueError, match="pools by 32"):
        EEGNet(n_channels=64, n_samples=16, n_classes=2)


def test_unknown_baseline_name_is_rejected():
    with pytest.raises(ValueError, match="unknown baseline"):
        build_baseline("resnet", 64, 64, 2)


# ------------------------------------------------- temporal sizing scales with T

def test_eegnet_recovers_the_published_kernel_at_the_published_window():
    """kern_length is min(sfreq // 2, T // 2): the paper's value once T is long enough."""
    short = EEGNet(64, 64, 2, sfreq=160.0)      # 0.4 s -- T // 2 binds
    long_ = EEGNet(64, 640, 2, sfreq=160.0)     # 4.0 s -- the paper's sfreq // 2 binds
    assert short.kern_length == 32
    assert long_.kern_length == 80


def test_shallow_pool_scales_and_stays_valid():
    for t in (32, 64, 160, 640):
        net = ShallowConvNet(64, t, 2)
        conv_out = t - net.temporal_kernel + 1
        assert 1 <= net.pool <= conv_out
        assert (conv_out - net.pool) // net.pool_stride + 1 >= 1


# ------------------------------------------------------------ regularisation

def test_max_norm_is_applied_in_training_mode_only():
    net = EEGNet(64, 64, 2)
    with torch.no_grad():
        net.classifier.weight.mul_(100.0)       # far above the 0.25 constraint
    before = net.classifier.weight.norm(2, dim=1).max().item()

    net.eval()
    net(torch.randn(2, 64, 64))
    assert net.classifier.weight.norm(2, dim=1).max().item() == pytest.approx(before)

    net.train()
    net(torch.randn(2, 64, 64))
    assert net.classifier.weight.norm(2, dim=1).max().item() <= 0.25 + 1e-5


@pytest.mark.parametrize("name", ["eegnet", "shallow"])
def test_l2_penalty_matches_the_gcn_convention(name):
    """The shared --regularization flag must mean the same thing for every arm."""
    model = build_baseline(name, 64, 64, 2)
    expected = 0.5 * ((model.classifier.weight ** 2).sum()
                      + (model.classifier.bias ** 2).sum())
    assert torch.allclose(model.l2_penalty(), expected)


# ------------------------------------------- the shared, graph-free fold path

def test_graph_free_fold_keeps_channels_unpermuted_and_unpadded():
    """The coarsening perm reorders channels and adds fake nodes; a CNN must not see that."""
    train, test = _trials(seed=1), _trials(n=4, seed=2)
    graphed = prepare_fold(train, test, window=64, stride=16, build_graph=True)
    flat = prepare_fold(train, test, window=64, stride=16, build_graph=False)

    assert graphed.train.X.shape[1] >= 64      # padded up to the coarsening tree
    assert flat.train.X.shape[1] == 64         # exactly the montage
    assert flat.perm == list(range(64))
    assert flat.bases == [] and flat.adjacencies == []


def test_graph_free_fold_preserves_the_leakage_sensitive_steps():
    train, test = _trials(seed=3), _trials(n=4, seed=4)
    fold = prepare_fold(train, test, window=64, stride=16, build_graph=False)

    # standardiser fit on train only -- train is centred, test is not forced to be
    assert abs(float(fold.train.X.mean())) < 0.05
    # windows never cross a trial boundary: 37 windows per trial at stride 16
    n_windows = (640 - 64) // 16 + 1
    assert len(fold.train.X) == len(train) * n_windows
    assert len(np.unique(fold.train.trial_id)) == len(train)
    # test trial ids stay in their own range so aggregation cannot collide
    assert fold.test.trial_id.min() >= 200_000


def test_graph_free_fold_rejects_band_power():
    """Band power is a graph-arm ablation (9.6); silently allowing it would mislead."""
    train, test = _trials(seed=5), _trials(n=4, seed=6)
    with pytest.raises(ValueError, match="raw features only"):
        prepare_fold(train, test, window=160, stride=32,
                     features="bandpower", build_graph=False)


def test_baseline_trains_one_step_on_a_real_fold():
    """End to end through the shared Trainer: same loss, same optimiser, same path."""
    from evaluation.trainer import TrainConfig, Trainer

    train, test = _trials(seed=7), _trials(n=4, seed=8)
    fold = prepare_fold(train, test, window=64, stride=16, build_graph=False)
    _, n_ch, n_tp = fold.train.X.shape
    model = build_baseline("eegnet", n_ch, n_tp, n_classes=2)

    trainer = Trainer(model, TrainConfig(epochs=1, batch_size=64, learning_rate=1e-3,
                                         device="cpu", verbose=False), n_classes=2)
    trainer.fit(fold)
    result = trainer.evaluate_samples(fold.test)
    assert set(result) == {"timepoint", "trial"}
    assert result["trial"].n_samples == len(test)
