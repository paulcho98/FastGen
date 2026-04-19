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


def _make_model(beta=0.25, center=False, clamp=None, scorer_sync_c=3.0, mode="per_sample"):
    """Build an OmniAvatarSelfForcingReDMD skeleton for prep-only testing.

    Skips __init__ of the heavy base class; injects the config + scorer.
    Default `mode="per_sample"` matches the production default.
    """
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD

    model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)

    cfg = type("Cfg", (), {})()
    cfg.reward_beta = beta
    cfg.center_reward = center
    cfg.clamp_reward = clamp
    cfg.reward_weighting_mode = mode
    model.config = cfg
    model.reward_scorer = _FakeScorer(sync_c=scorer_sync_c)
    model._reward_running_mean = None

    return model


def test_default_mode_is_per_sample():
    """Production config default: reward_weighting_mode='per_sample'."""
    from fastgen.configs.methods.config_omniavatar_sf import OmniAvatarModelConfig
    cfg = OmniAvatarModelConfig()
    assert cfg.reward_weighting_mode == "per_sample"


def test_per_sample_mode_at_b1():
    """Default mode: at B=1, weighted = exp(beta*r) * L.
    This matches what original Reward-Forcing at batch_size=1 produced
    (they always ran B=1, so `mean(w*L)` and `w*L` coincide trivially)."""
    model = _make_model(beta=0.25, scorer_sync_c=4.0, mode="per_sample")
    videos = [torch.randint(0, 256, (81, 3, 224, 224), dtype=torch.uint8)]
    audios = [torch.randn(51840)]

    vsd_loss = torch.tensor([1.5])  # [B=1]
    weighted, log_map = model._apply_reward_weighting(vsd_loss, videos, audios)

    expected = math.exp(0.25 * 4.0) * 1.5
    assert abs(weighted.item() - expected) < 1e-4
    assert abs(log_map["reward_sync_c_mean"] - 4.0) < 1e-6
    assert abs(log_map["reward_weight_mean"] - math.exp(0.25 * 4.0)) < 1e-4
    assert abs(log_map["vsd_loss_unweighted"] - 1.5) < 1e-6


def test_per_sample_mode_batched_varying():
    """Default mode at B>1 with non-uniform r, L: weighted = mean_i(w_i * L_i).
    This is the algorithmic fix from fc56e4a — per-sample coupling preserved
    (not collapsed to mean(w)*mean(L)), reward magnitude still scales loss."""
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    cfg = type("Cfg", (), {})()
    cfg.reward_beta = 0.2
    cfg.center_reward = False
    cfg.clamp_reward = None
    cfg.reward_weighting_mode = "per_sample"
    model.config = cfg
    model.reward_scorer = _VaryingScorer([0.0, 10.0])
    model._reward_running_mean = None

    videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8) for _ in range(2)]
    audios = [torch.randn(51840) for _ in range(2)]
    vsd = torch.tensor([10.0, 1.0], dtype=torch.float32)

    weighted, _ = model._apply_reward_weighting(vsd, videos, audios)

    w0 = math.exp(0.2 * 0.0)   # 1.0
    w1 = math.exp(0.2 * 10.0)  # ~7.389
    expected = (w0 * 10.0 + w1 * 1.0) / 2  # ~8.695
    assert abs(weighted.item() - expected) < 1e-3


def test_self_normalized_mode_at_b1_reduces_to_L():
    """Self-normalized IS at B=1: w cancels in numer/denom, weighted == L
    exactly. Reward has no effect at B=1 (nothing to normalize against)."""
    model = _make_model(beta=0.25, scorer_sync_c=4.0, mode="self_normalized")
    videos = [torch.randint(0, 256, (81, 3, 224, 224), dtype=torch.uint8)]
    audios = [torch.randn(51840)]

    vsd_loss = torch.tensor([1.5])  # [B=1]
    weighted, log_map = model._apply_reward_weighting(vsd_loss, videos, audios)

    assert abs(weighted.item() - 1.5) < 1e-4
    assert abs(log_map["reward_sync_c_mean"] - 4.0) < 1e-6
    # Raw weight still logged for diagnostic purposes
    assert abs(log_map["reward_weight_mean"] - math.exp(0.25 * 4.0)) < 1e-4
    assert abs(log_map["vsd_loss_unweighted"] - 1.5) < 1e-6


def test_self_normalized_uniform_sync_c_collapses_to_mean_L():
    """Self-normalized: uniform sync_c across a batch → all weights equal →
    normalized w_i = 1/B → weighted = mean(L). Reward has no effect when
    constant across the batch (no discriminative signal)."""
    model = _make_model(beta=0.25, scorer_sync_c=2.0, mode="self_normalized")
    videos = [torch.randint(0, 256, (81, 3, 224, 224), dtype=torch.uint8) for _ in range(4)]
    audios = [torch.randn(51840) for _ in range(4)]
    vsd_loss = torch.ones(4)

    weighted, log_map = model._apply_reward_weighting(vsd_loss, videos, audios)

    assert abs(log_map["reward_sync_c_mean"] - 2.0) < 1e-6
    assert log_map["reward_sync_c_min"] == 2.0
    assert log_map["reward_sync_c_max"] == 2.0
    assert abs(weighted.item() - 1.0) < 1e-4


def test_legacy_batch_mean_mode_decouples_reward_from_loss():
    """legacy_batch_mean: weighted = mean(w) * mean(L). At batch > 1 with
    non-uniform reward AND non-uniform loss that co-vary, this produces a
    different result from both per_sample (mean(w*L)) and self_normalized
    (sum(w*L)/sum(w)). Provided for direct comparison with the pre-fc56e4a
    state of Reward-Forcing."""
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    cfg = type("Cfg", (), {})()
    cfg.reward_beta = 0.2
    cfg.center_reward = False
    cfg.clamp_reward = None
    cfg.reward_weighting_mode = "legacy_batch_mean"
    model.config = cfg
    model.reward_scorer = _VaryingScorer([0.0, 10.0])
    model._reward_running_mean = None

    videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8) for _ in range(2)]
    audios = [torch.randn(51840) for _ in range(2)]
    vsd = torch.tensor([10.0, 1.0], dtype=torch.float32)

    weighted, _ = model._apply_reward_weighting(vsd, videos, audios)

    w0 = math.exp(0.2 * 0.0)   # 1.0
    w1 = math.exp(0.2 * 10.0)  # ~7.389
    # mean(w) * mean(L) = (1.0 + 7.389)/2 * (10 + 1)/2 = 4.194 * 5.5 = ~23.07
    expected = ((w0 + w1) / 2) * ((10.0 + 1.0) / 2)
    assert abs(weighted.item() - expected) < 1e-3


def test_unknown_reward_weighting_mode_raises():
    model = _make_model(beta=0.25, scorer_sync_c=4.0, mode="bogus_mode")
    videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8)]
    audios = [torch.randn(51840)]
    vsd = torch.tensor([1.0])
    with pytest.raises(ValueError, match="reward_weighting_mode"):
        model._apply_reward_weighting(vsd, videos, audios)


def test_centering_subtracts_ema_mean():
    model = _make_model(beta=0.25, center=True, scorer_sync_c=5.0)
    videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8)]
    audios = [torch.randn(51840)]
    vsd = torch.tensor([1.0])  # [B=1]

    # First call seeds the running mean with 5.0 → centered reward = 0 → weight = exp(0) = 1
    _, log_map = model._apply_reward_weighting(vsd, videos, audios)
    assert abs(log_map["reward_weight_mean"] - 1.0) < 1e-4


def test_clamping_bounds_weight():
    model = _make_model(beta=1.0, clamp=(0.0, 2.0), scorer_sync_c=10.0)
    videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8)]
    audios = [torch.randn(51840)]
    vsd = torch.tensor([1.0])  # [B=1]

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
    cfg.reward_weighting_mode = "per_sample"
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
    vsd = torch.ones(3)  # [B=3]
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


class _VaryingScorer:
    """Returns the configured per-sample sync_c tensor verbatim."""
    def __init__(self, sync_c_vec):
        self._vec = torch.as_tensor(sync_c_vec, dtype=torch.float32)

    def reward_from_frames(self, videos, audios, prompts=None, use_norm=True):
        assert len(videos) == self._vec.shape[0], "batch size must match sync_c vec"
        c = self._vec.clone()
        return {"sync_c": c, "MQ": c}


def test_reward_loss_coupling_is_per_sample_self_normalized():
    """Non-uniform rewards + non-uniform per-sample vsd_loss.

    Self-normalized IS: weighted_loss = sum_i(w_i * L_i) / sum_j(w_j), where
    w_i = exp(beta * r_i). High-reward samples get proportionally more weight
    in the batch combination; magnitude is bounded in [min(L), max(L)].
    """
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    cfg = type("Cfg", (), {})()
    cfg.reward_beta = 0.2
    cfg.center_reward = False
    cfg.clamp_reward = None
    cfg.reward_weighting_mode = "self_normalized"
    model.config = cfg
    model.reward_scorer = _VaryingScorer([0.0, 10.0])
    model._reward_running_mean = None

    videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8) for _ in range(2)]
    audios = [torch.randn(51840) for _ in range(2)]

    # Sample 0: low reward (r=0), high loss (10.0)
    # Sample 1: high reward (r=10), low loss (1.0)
    vsd_per_sample = torch.tensor([10.0, 1.0], dtype=torch.float32)

    weighted, log_map = model._apply_reward_weighting(vsd_per_sample, videos, audios)

    w0 = math.exp(0.2 * 0.0)    # 1.0
    w1 = math.exp(0.2 * 10.0)   # ~7.389
    expected = (w0 * 10.0 + w1 * 1.0) / (w0 + w1)   # ~2.073

    assert abs(weighted.item() - expected) < 1e-3, (
        f"self-normalized IS gave {weighted.item()}, expected {expected}"
    )
    # Must lie in [min(L), max(L)] = [1, 10] — convex combination property
    assert 1.0 <= weighted.item() <= 10.0

    # Logging entries report batch-mean stats sensibly
    assert abs(log_map["reward_sync_c_mean"] - 5.0) < 1e-6
    assert abs(log_map["vsd_loss_unweighted"] - 5.5) < 1e-6


def test_reward_weighting_shift_invariance():
    """Self-normalized IS is invariant under reward shifts:
        exp(beta * (r_i + c)) / Z_shifted = exp(beta*c) * w_i / (exp(beta*c) * Z) = w_i / Z
    So adding a constant to all sync_c values must NOT change the weighted loss.
    This is the property that makes additive combination with GAN losses work:
    weighted VSD magnitude doesn't drift with the absolute reward level.
    """
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD

    def compute_weighted(sync_c_vec, vsd_vec, beta):
        model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
        cfg = type("Cfg", (), {})()
        cfg.reward_beta = beta
        cfg.center_reward = False
        cfg.clamp_reward = None
        cfg.reward_weighting_mode = "self_normalized"
        model.config = cfg
        model.reward_scorer = _VaryingScorer(sync_c_vec)
        model._reward_running_mean = None
        n = len(sync_c_vec)
        videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8) for _ in range(n)]
        audios = [torch.randn(51840) for _ in range(n)]
        vsd = torch.as_tensor(vsd_vec, dtype=torch.float32)
        w, _ = model._apply_reward_weighting(vsd, videos, audios)
        return w.item()

    vsd_vec = [1.0, 5.0, 3.0, 0.5]
    base = [0.0, 2.0, 4.0, 6.0]
    shifted = [100.0, 102.0, 104.0, 106.0]  # +100, a large shift

    result_base = compute_weighted(base, vsd_vec, beta=0.5)
    result_shifted = compute_weighted(shifted, vsd_vec, beta=0.5)

    assert abs(result_base - result_shifted) < 1e-3, (
        f"shift invariance broken: base={result_base}, shifted={result_shifted}"
    )


def test_reward_weighting_bounded_by_loss_range():
    """Self-normalized IS is a convex combination of per-sample losses with
    non-negative weights → weighted_loss must lie in [min(L), max(L)]. This
    is what makes the loss magnitude stable at high beta or with outlier
    rewards — the GAN-clobbering scenario from the pre-fix code."""
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    cfg = type("Cfg", (), {})()
    cfg.reward_beta = 2.0   # high beta: unnormalized mean(w*L) would be O(exp(20))
    cfg.center_reward = False
    cfg.clamp_reward = None
    cfg.reward_weighting_mode = "self_normalized"
    model.config = cfg
    model.reward_scorer = _VaryingScorer([0.0, 5.0, 10.0, 15.0])
    model._reward_running_mean = None

    videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8) for _ in range(4)]
    audios = [torch.randn(51840) for _ in range(4)]
    vsd = torch.tensor([2.0, 3.0, 1.5, 4.0], dtype=torch.float32)

    weighted, _ = model._apply_reward_weighting(vsd, videos, audios)

    # Bounded in the convex hull of per-sample losses
    L_min, L_max = 1.5, 4.0
    assert L_min <= weighted.item() <= L_max, (
        f"weighted={weighted.item()} outside [{L_min}, {L_max}]; "
        f"self-normalized IS must be a convex combination"
    )


def test_reward_weighting_batch_size_1_matches_unweighted():
    """At B=1, self-normalized IS gives back the unweighted per-sample loss.
    Note this DIFFERS from the original Reward-Forcing behavior (which gave
    `exp(beta*r) * L`), because the paper's Z(c) normalization was dropped
    there. At batch_size=1 Z(c) trivially equals the single sample's weight,
    so the reward signal vanishes — this is the correct interpretation: at
    B=1 there's no relative ranking to do."""
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    cfg = type("Cfg", (), {})()
    cfg.reward_beta = 0.25
    cfg.center_reward = False
    cfg.clamp_reward = None
    cfg.reward_weighting_mode = "self_normalized"
    model.config = cfg
    model.reward_scorer = _VaryingScorer([4.0])
    model._reward_running_mean = None

    videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8)]
    audios = [torch.randn(51840)]
    vsd = torch.tensor([1.5], dtype=torch.float32)

    weighted, _ = model._apply_reward_weighting(vsd, videos, audios)
    assert abs(weighted.item() - 1.5) < 1e-4
