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


def test_student_update_step_integrates_reward():
    """_student_update_step should:
      (a) compute vsd_loss (mocked),
      (b) VAE-decode gen_data (mocked),
      (c) call reward scorer with decoded pixels + audio_waveform,
      (d) multiply vsd_loss by exp(beta * sync_c),
      (e) include reward/weight stats in the returned loss_map.
    """
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD

    model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)

    cfg = type("Cfg", (), {})()
    cfg.reward_beta = 0.25
    cfg.center_reward = False
    cfg.clamp_reward = None
    cfg.gan_loss_weight_gen = 0.0
    cfg.guidance_scale = None
    model.config = cfg
    model.reward_scorer = _FakeScorer(sync_c=3.0)
    model._reward_running_mean = None

    # Fake generator output + deps
    gen_latent = torch.randn(1, 16, 21, 8, 8, requires_grad=True)
    model.gen_data_from_net = lambda *a, **kw: gen_latent

    class _FakeSched:
        def forward_process(self, x, eps, t):
            return x + 0.1 * eps

    class _FakeVAE:
        def decode(self, x):
            # list in -> list out; one decoded pixel tensor per sample
            return [torch.zeros(3, 81, 64, 64) for _ in x]

    class _FakeNet:
        def __init__(self):
            self.noise_scheduler = _FakeSched()
            self.vae = _FakeVAE()

        def clear_caches(self):
            pass

    model.net = _FakeNet()
    model.fake_score = lambda x, t, condition, fwd_pred_type: torch.zeros_like(gen_latent)
    model._compute_teacher_prediction_gan_loss = lambda p, t, condition: (
        torch.zeros_like(gen_latent),
        torch.tensor(0.0),
    )
    model._get_outputs = lambda gen_data, input_student, condition=None: {}

    data = {"audio_waveform": torch.randn(1, 51840)}
    condition = {}
    neg_condition = {}
    input_s = torch.randn_like(gen_latent)
    t_student = torch.tensor([0.5])
    t = torch.tensor([0.5])
    eps = torch.randn_like(gen_latent)

    loss_map, outputs = model._student_update_step(
        input_s, t_student, t, eps, data,
        condition=condition, neg_condition=neg_condition,
    )

    assert "reward_sync_c_mean" in loss_map
    assert "reward_weight_mean" in loss_map
    assert "vsd_loss_unweighted" in loss_map
    assert "total_loss" in loss_map
    assert abs(float(loss_map["reward_sync_c_mean"]) - 3.0) < 1e-6


def test_student_update_step_bypasses_reward_when_scorer_missing():
    """If reward_scorer is None (config.reward.enabled=False), behave like base class:
    no reward_* keys, vsd_loss used directly.
    """
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD

    model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    cfg = type("Cfg", (), {})()
    cfg.gan_loss_weight_gen = 0.0
    cfg.guidance_scale = None
    model.config = cfg
    model.reward_scorer = None

    gen_latent = torch.randn(1, 16, 21, 8, 8, requires_grad=True)
    model.gen_data_from_net = lambda *a, **kw: gen_latent

    class _FakeSched:
        def forward_process(self, x, eps, t):
            return x + 0.1 * eps

    class _FakeNet:
        def __init__(self):
            self.noise_scheduler = _FakeSched()
        def clear_caches(self):
            pass

    model.net = _FakeNet()
    model.fake_score = lambda x, t, condition, fwd_pred_type: torch.zeros_like(gen_latent)
    model._compute_teacher_prediction_gan_loss = lambda p, t, condition: (
        torch.zeros_like(gen_latent),
        torch.tensor(0.0),
    )
    model._get_outputs = lambda gen_data, input_student, condition=None: {}

    data = {}
    loss_map, outputs = model._student_update_step(
        torch.randn_like(gen_latent),
        torch.tensor([0.5]), torch.tensor([0.5]),
        torch.randn_like(gen_latent), data,
        condition={}, neg_condition={},
    )

    assert "total_loss" in loss_map
    assert "reward_sync_c_mean" not in loss_map


def test_per_rank_log_keys_in_single_rank_path():
    """When torch.distributed is not initialized, per-rank keys fall back to
    per-batch-item keys (r0, r1, ...)."""
    model = _make_model(beta=0.25, scorer_sync_c=2.0)
    videos = [torch.randint(0, 256, (81, 3, 224, 224), dtype=torch.uint8) for _ in range(3)]
    audios = [torch.randn(51840) for _ in range(3)]
    vsd = torch.tensor(1.0)
    _, log_map = model._apply_reward_weighting(vsd, videos, audios)
    # 3 samples -> 3 per-rank keys
    assert "reward_sync_c_r0" in log_map
    assert "reward_sync_c_r1" in log_map
    assert "reward_sync_c_r2" in log_map
    assert "reward_weight_r0" in log_map
    assert abs(log_map["reward_sync_c_r0"] - 2.0) < 1e-6


def test_maybe_save_debug_video_disabled_by_default(tmp_path):
    """With save_reward_debug_video False (default), no file is written."""
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD

    model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    cfg = type("Cfg", (), {})()
    # no save_reward_debug_video -> defaults to False
    model.config = cfg

    pixels = torch.zeros(1, 3, 81, 64, 64)
    model._maybe_save_debug_video(pixels, iteration=0)

    # No files should have been created anywhere under tmp_path
    assert not any(tmp_path.rglob("*.mp4"))


def test_maybe_save_debug_video_writes_mp4_when_enabled(tmp_path):
    """With the flag on and a writable debug_dir, we get an mp4 file."""
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD

    model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    cfg = type("Cfg", (), {})()
    cfg.save_reward_debug_video = True
    cfg.reward_debug_dir = str(tmp_path)
    model.config = cfg

    # 5 frames is the minimum SyncNet window but here we just need a small video
    pixels = torch.zeros(1, 3, 10, 64, 64)  # [B=1, 3, T=10, H=64, W=64]
    model._maybe_save_debug_video(pixels, iteration=42)

    mp4s = list(tmp_path.glob("*.mp4"))
    assert len(mp4s) == 1
    assert mp4s[0].name == "gen_iter000042.mp4"
    assert mp4s[0].stat().st_size > 0
