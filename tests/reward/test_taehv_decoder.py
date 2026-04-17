"""Unit tests for TAEHVDecoderWrapper — the WanVideoVAE.decode-compatible shim."""
import os
import pytest
import torch

TAEW_CKPT = "/home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(TAEW_CKPT),
    reason="CUDA and TAEW checkpoint required",
)


@pytest.fixture
def wrapper():
    from fastgen.methods.reward.taehv_decoder import TAEHVDecoderWrapper
    return TAEHVDecoderWrapper(checkpoint_path=TAEW_CKPT, device="cuda")


def test_decode_matches_wan_contract_shape(wrapper):
    # Wan 2.1 latent for an 81-frame clip at 512x512:
    #   C=16, T_lat=21, H=64, W=64
    # Expected pixel shape matching WanVideoVAE: [N=1, 3, 81, 512, 512]
    lat = torch.randn(16, 21, 64, 64, device="cuda", dtype=torch.float32)
    out = wrapper.decode([lat])
    assert out.shape == (1, 3, 81, 512, 512), f"got {tuple(out.shape)}"
    assert out.dtype == torch.float32


def test_decode_output_range_matches_wan(wrapper):
    # Wan VAE decode output is in [-1, 1]; wrapper must rescale from TAEHV's [0, 1]
    lat = torch.randn(16, 21, 64, 64, device="cuda", dtype=torch.float32)
    out = wrapper.decode([lat])
    assert out.min() >= -1.01, f"range underflow: {out.min().item()}"
    assert out.max() <= 1.01, f"range overflow: {out.max().item()}"


def test_decode_frame_count_no_double_trim(wrapper):
    # Regression: an older wrapper in scripts/inference/inference_causal_taehv.py
    # applies its own `vid[:, frames_to_trim:]` AFTER decode_video — but our
    # vendored taehv.py already trims inside decode_video. Double-trim would
    # produce 78 frames for a 21-latent input instead of 81.
    lat = torch.randn(16, 21, 64, 64, device="cuda", dtype=torch.float32)
    out = wrapper.decode([lat])
    assert out.shape[2] == 81, (
        f"frame count mismatch: expected 81, got {out.shape[2]}. "
        f"If 78, the wrapper is double-trimming — remove the manual "
        f"frames_to_trim slice (decode_video already trims)."
    )


def test_decode_batch_of_two(wrapper):
    # list of 2 latents → stacked output of shape [2, 3, 81, H, W]
    latents = [torch.randn(16, 21, 64, 64, device="cuda", dtype=torch.float32) for _ in range(2)]
    out = wrapper.decode(latents)
    assert out.shape == (2, 3, 81, 512, 512), f"got {tuple(out.shape)}"


def test_decode_runs_under_no_grad_even_if_called_in_grad_context(wrapper):
    # Re-DMD calls VAE decode inside torch.no_grad() upstream. Wrapper must
    # not break if accidentally called with a leaf latent that has requires_grad.
    lat = torch.randn(16, 21, 64, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    out = wrapper.decode([lat])
    assert out.requires_grad is False, "wrapper should produce a detached tensor"
