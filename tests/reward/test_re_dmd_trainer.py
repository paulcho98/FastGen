"""Unit tests for OmniAvatarSelfForcingReDMD's reward-weighting logic.

Tests _apply_reward_weighting with a mocked scorer — no real model, no real
training loop, no GPU. Just: given an MSE-like loss and a mocked sync_c,
verify the weighting math and the logging dict.
"""
import math

import pytest
import torch


class _FakeScorer:
    """Returns a configurable constant sync_c for each sample in the batch."""
    def __init__(self, sync_c: float):
        self.sync_c = sync_c

    def reward_from_frames(self, videos, audios, prompts=None, use_norm=True):
        c = torch.full((len(videos),), self.sync_c, dtype=torch.float32)
        return {"sync_c": c, "MQ": c}


def _make_model(beta=0.25, center=False, clamp=None, scorer_sync_c=3.0):
    """Build an OmniAvatarSelfForcingReDMD skeleton for prep-only testing.

    Skips __init__ of the heavy base class; injects the config + scorer.
    """
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD

    model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)

    cfg = type("Cfg", (), {})()
    cfg.reward_beta = beta
    cfg.center_reward = center
    cfg.clamp_reward = clamp
    model.config = cfg
    model.reward_scorer = _FakeScorer(sync_c=scorer_sync_c)
    model._reward_running_mean = None

    return model


def test_weighted_loss_equals_exp_beta_r_times_unweighted():
    model = _make_model(beta=0.25, scorer_sync_c=4.0)
    videos = [torch.randint(0, 256, (81, 3, 224, 224), dtype=torch.uint8)]
    audios = [torch.randn(51840)]

    vsd_loss = torch.tensor(1.5)
    weighted, log_map = model._apply_reward_weighting(vsd_loss, videos, audios)

    expected_weight = math.exp(0.25 * 4.0)
    assert abs(weighted.item() - expected_weight * 1.5) < 1e-4
    assert abs(log_map["reward_sync_c_mean"] - 4.0) < 1e-6
    assert abs(log_map["reward_weight_mean"] - expected_weight) < 1e-4
    assert abs(log_map["vsd_loss_unweighted"] - 1.5) < 1e-6


def test_weighting_batched():
    """Multiple samples → sync_c mean is the batch mean."""
    model = _make_model(beta=0.25, scorer_sync_c=2.0)
    videos = [torch.randint(0, 256, (81, 3, 224, 224), dtype=torch.uint8) for _ in range(4)]
    audios = [torch.randn(51840) for _ in range(4)]
    vsd_loss = torch.tensor(1.0)

    weighted, log_map = model._apply_reward_weighting(vsd_loss, videos, audios)

    assert abs(log_map["reward_sync_c_mean"] - 2.0) < 1e-6
    assert log_map["reward_sync_c_min"] == 2.0
    assert log_map["reward_sync_c_max"] == 2.0


def test_centering_subtracts_ema_mean():
    model = _make_model(beta=0.25, center=True, scorer_sync_c=5.0)
    videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8)]
    audios = [torch.randn(51840)]
    vsd = torch.tensor(1.0)

    # First call seeds the running mean with 5.0 → centered reward = 0 → weight = exp(0) = 1
    _, log_map = model._apply_reward_weighting(vsd, videos, audios)
    assert abs(log_map["reward_weight_mean"] - 1.0) < 1e-4


def test_clamping_bounds_weight():
    model = _make_model(beta=1.0, clamp=(0.0, 2.0), scorer_sync_c=10.0)
    videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8)]
    audios = [torch.randn(51840)]
    vsd = torch.tensor(1.0)

    _, log_map = model._apply_reward_weighting(vsd, videos, audios)

    # sync_c=10 clamped to 2 → weight = exp(1*2) ≈ 7.39, NOT exp(10) ≈ 22026
    assert abs(log_map["reward_weight_mean"] - math.exp(2.0)) < 1e-3
    assert log_map["reward_sync_c_max"] == 2.0
