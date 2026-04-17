"""Re-DMD model selects decoder based on config.reward.decoder_kind.

Note: inside OmniAvatarSelfForcingReDMD, self.config IS the model sub-config
(it's passed as the `config=` kwarg during LazyCall instantiation), so the
access path is `self.config.reward.*`, NOT `self.config.model.reward.*`.
"""
import os
import types
import pytest
import torch


TAEW_CKPT = "/home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth"


def _make_minimal_config(decoder_kind="vae", taew_ckpt=""):
    """Build a minimal config object with just the bits the decoder-branch reads.

    Mimics the model sub-config layer (what the class sees as self.config).
    """
    from fastgen.configs.methods.config_omniavatar_sf import RewardConfig
    reward = RewardConfig(
        enabled=False,  # skip actually loading SyncNet for these unit tests
        checkpoint_path="",
        decoder_kind=decoder_kind,
        taew_checkpoint_path=taew_ckpt,
    )
    return types.SimpleNamespace(
        reward=reward,
        reward_beta=0.25,
        center_reward=False,
        clamp_reward=None,
    )


def test_build_model_skips_taew_when_decoder_kind_is_vae():
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    m = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    m.config = _make_minimal_config(decoder_kind="vae")
    m._maybe_init_taew_decoder()
    assert getattr(m, "_taew_decoder", None) is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(TAEW_CKPT),
    reason="CUDA and TAEW checkpoint required",
)
def test_build_model_loads_taew_when_decoder_kind_is_taew():
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    from fastgen.methods.reward.taehv_decoder import TAEHVDecoderWrapper
    m = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    m.config = _make_minimal_config(decoder_kind="taew", taew_ckpt=TAEW_CKPT)
    # _maybe_init_taew_decoder needs self.device to pick where to place TAEHV.
    # Stub as int 0 (cuda:0) since that's the pattern the existing build_model uses.
    m.device = 0
    m._maybe_init_taew_decoder()
    assert isinstance(m._taew_decoder, TAEHVDecoderWrapper)


def test_taew_kind_requires_checkpoint_path():
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    m = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    m.config = _make_minimal_config(decoder_kind="taew", taew_ckpt="")
    m.device = 0
    with pytest.raises(ValueError, match="taew_checkpoint_path"):
        m._maybe_init_taew_decoder()


def test_unknown_decoder_kind_raises():
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    m = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    m.config = _make_minimal_config(decoder_kind="bogus")
    m.device = 0
    with pytest.raises(ValueError, match="decoder_kind"):
        m._maybe_init_taew_decoder()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(TAEW_CKPT),
    reason="CUDA and TAEW checkpoint required",
)
def test_decode_gen_to_pixels_uses_taew_when_configured():
    """With _taew_decoder set, _decode_gen_to_pixels must go through it and
    NOT touch self.net.vae."""
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    from fastgen.methods.reward.taehv_decoder import TAEHVDecoderWrapper
    m = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    m._taew_decoder = TAEHVDecoderWrapper(checkpoint_path=TAEW_CKPT, device="cuda")
    # Give it a net with a .vae attribute that would blow up if touched.
    class _ExplodingVAE:
        def decode(self, *a, **k):
            raise AssertionError("vae.decode should NOT be called when _taew_decoder is set")
    m.net = types.SimpleNamespace(vae=_ExplodingVAE())
    lat = torch.randn(1, 16, 21, 64, 64, device="cuda", dtype=torch.float32)
    out = m._decode_gen_to_pixels(lat)
    assert out.shape == (1, 3, 81, 512, 512)
