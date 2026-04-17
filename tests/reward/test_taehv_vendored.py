"""Smoke test that the vendored TAEHV is importable and constructible.

Verifies the Wan 2.1-specific config (latent_channels=16, patch_size=1,
t_upscale=4, frames_to_trim=3) — these are load-bearing for the decoder
wrapper in taehv_decoder.py.
"""
import os
import pytest

TAEW_CKPT = "/home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth"


def test_taehv_imports():
    from fastgen.methods.reward.taehv import TAEHV
    assert TAEHV is not None


@pytest.mark.skipif(not os.path.exists(TAEW_CKPT), reason="TAEW checkpoint missing")
def test_taehv_loads_and_reports_config():
    from fastgen.methods.reward.taehv import TAEHV
    m = TAEHV(checkpoint_path=TAEW_CKPT)
    assert m.latent_channels == 16
    assert m.patch_size == 1
    assert m.t_upscale == 4
    assert m.frames_to_trim == 3
