"""Equivalence tests for the `expand_audio_checkpoint_scope` WanModel kwarg.

The production flag moves the audio-conditioning add (audio_cond_tmp + x) INSIDE
the `torch.utils.checkpoint.checkpoint` boundary, so its activations are
recomputed during backward rather than retained. Mathematically this is a
no-op: same ops in same order over the same tensors, only the activation-
saving strategy differs.

These tests guarantee that numerical equivalence: forward outputs and gradients
must match to floating-point round-off under both toggle values. If this ever
regresses, the flag is unsafe and must be kept off.
"""

import copy

import pytest
import torch

from fastgen.networks.OmniAvatar.wan_model import WanModel


def _make_tiny_model_and_inputs(seed: int = 42):
    """Construct a tiny WanModel and matching inputs.

    Shape accounting:
      - Latent x: [B=1, C_x=16, T=4, H=8, W=8]
      - V2V y:   [B=1, C_y=16, T=4, H=8, W=8]   (concat'd with x to 32 ch)
      - patch_size=(1,2,2), so post-conv grid = (4,4,4) -> 64 tokens
      - num_layers=6: audio_cond_projs has 6//2-1 = 2 entries
      - Audio injection fires at layer_i in {2, 3} (layer_i<=3 and >1)
      - audio_emb: [B, T_audio, 10752], T_audio=13 so (T_audio+3)/4=4, matching
        the latent T=4 — required for audio_cond_tmp and x to broadcast.
    """
    torch.manual_seed(seed)
    model = WanModel(
        dim=32,
        in_dim=32,           # C_x + C_y after concat
        ffn_dim=64,
        out_dim=16,
        text_dim=32,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=6,
        use_audio=True,
        audio_hidden_size=8,
        has_image_input=False,
    ).train()

    B, C_x, C_y, T, H, W = 1, 16, 16, 4, 8, 8
    T_audio = 13   # (T_audio + 3) / 4 == T — required for audio/x alignment
    x = torch.randn(B, C_x, T, H, W)
    y = torch.randn(B, C_y, T, H, W)
    timestep = torch.tensor([500.0])
    context = torch.randn(B, 77, 32)
    audio_emb = torch.randn(B, T_audio, 10752)
    return model, dict(x=x, timestep=timestep, context=context, y=y, audio_emb=audio_emb)


def _run_forward_backward(model, inputs, expand_scope: bool):
    """One deterministic forward + backward pass. Returns (output, dict-of-grads)."""
    # Re-seed at every call so any dropout/random-init ops are identical across toggles
    torch.manual_seed(0)
    model.zero_grad(set_to_none=True)

    out = model(
        **inputs,
        use_gradient_checkpointing=True,
        expand_audio_checkpoint_scope=expand_scope,
    )
    # Scalarize in a way that exercises every element of the output
    loss = (out * out).sum()
    loss.backward()

    grads = {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return out.detach().clone(), loss.detach().clone(), grads


def test_forward_output_equivalence():
    """Both toggle values produce numerically equal forward outputs."""
    model, inputs = _make_tiny_model_and_inputs()

    out_off, _, _ = _run_forward_backward(model, inputs, expand_scope=False)
    out_on, _, _ = _run_forward_backward(model, inputs, expand_scope=True)

    # Same ops in same order on same data — tolerance at fp32 round-off.
    assert out_off.shape == out_on.shape
    max_abs_diff = (out_off - out_on).abs().max().item()
    assert torch.allclose(out_off, out_on, atol=1e-5, rtol=1e-5), (
        f"forward outputs differ: max |diff| = {max_abs_diff}"
    )


def test_loss_scalar_equivalence():
    """The scalar loss is identical under both toggles (sanity check)."""
    model, inputs = _make_tiny_model_and_inputs()
    _, loss_off, _ = _run_forward_backward(model, inputs, expand_scope=False)
    _, loss_on, _ = _run_forward_backward(model, inputs, expand_scope=True)
    assert abs(loss_off.item() - loss_on.item()) < 1e-4


def test_gradient_equivalence():
    """Gradients on every trainable parameter must match to round-off."""
    model, inputs = _make_tiny_model_and_inputs()

    _, _, grads_off = _run_forward_backward(model, inputs, expand_scope=False)
    _, _, grads_on = _run_forward_backward(model, inputs, expand_scope=True)

    assert set(grads_off.keys()) == set(grads_on.keys()), (
        f"grad-key mismatch: off-on = {set(grads_off) - set(grads_on)}, "
        f"on-off = {set(grads_on) - set(grads_off)}"
    )

    # Track worst case for error reporting
    worst = ("", 0.0)
    mismatched = []
    for name in grads_off:
        g1 = grads_off[name]
        g2 = grads_on[name]
        max_abs = (g1 - g2).abs().max().item()
        if max_abs > worst[1]:
            worst = (name, max_abs)
        if not torch.allclose(g1, g2, atol=1e-4, rtol=1e-4):
            mismatched.append((name, max_abs, g1.abs().max().item()))

    assert not mismatched, (
        f"{len(mismatched)} parameter(s) have mismatched gradients.\n"
        f"Worst: {worst[0]} max|diff|={worst[1]:.3e}\n"
        f"First 5: {mismatched[:5]}"
    )


def test_forward_without_audio_is_unaffected():
    """When use_audio=False, the flag is a no-op: no audio_cond to inject, both
    paths fall through to plain block(x, ctx, tm, freqs). Regression guard to
    make sure the new branch doesn't silently change behavior for non-audio
    models (the flag's dispatch is guarded by `audio_cond_tmp is not None`)."""
    torch.manual_seed(42)
    model = WanModel(
        dim=32, in_dim=32, ffn_dim=64, out_dim=16,
        text_dim=32, freq_dim=32, eps=1e-6,
        patch_size=(1, 2, 2), num_heads=4,
        num_layers=4,
        use_audio=False,
        audio_hidden_size=8,
    ).train()

    B = 1
    x = torch.randn(B, 16, 4, 8, 8)
    y = torch.randn(B, 16, 4, 8, 8)
    timestep = torch.tensor([500.0])
    context = torch.randn(B, 77, 32)
    inputs = dict(x=x, timestep=timestep, context=context, y=y, audio_emb=None)

    out_off, _, grads_off = _run_forward_backward(model, inputs, expand_scope=False)
    out_on, _, grads_on = _run_forward_backward(model, inputs, expand_scope=True)

    assert torch.allclose(out_off, out_on, atol=1e-5, rtol=1e-5)
    for name in grads_off:
        assert torch.allclose(grads_off[name], grads_on[name], atol=1e-4, rtol=1e-4), (
            f"use_audio=False gradient mismatch on {name}"
        )


def test_forward_without_checkpointing_is_unaffected():
    """When use_gradient_checkpointing=False, the flag is a no-op: both toggles
    take the plain non-checkpointed path. Regression guard for eval-mode /
    inference behavior."""
    model, inputs = _make_tiny_model_and_inputs()

    # Run with use_gradient_checkpointing=False under both toggles (flag should
    # not matter in this regime)
    torch.manual_seed(0)
    model.zero_grad(set_to_none=True)
    out_off = model(**inputs, use_gradient_checkpointing=False, expand_audio_checkpoint_scope=False)

    torch.manual_seed(0)
    model.zero_grad(set_to_none=True)
    out_on = model(**inputs, use_gradient_checkpointing=False, expand_audio_checkpoint_scope=True)

    assert torch.allclose(out_off, out_on, atol=1e-6, rtol=1e-6)
