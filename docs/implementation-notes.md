# OmniAvatar Self-Forcing Integration — Implementation Notes & Bug Tracker

This document tracks bugs, hiccups, and important discoveries during implementation.
Kept separate from CLAUDE.md to avoid bloating context.

---

## Phase 0: Verification Samples

### Data Format Notes
- `vae_latents_mask_all.pt`: dict with keys `input_latents` [16,21,64,64], `masked_latents` [16,21,64,64]
- `audio_emb_omniavatar.pt`: dict with keys `audio_emb` [N,10752] (N varies, slice to 81), `metadata`
- `text_emb.pt`: tensor [1,512,4096]
- `ref_latents.pt`: dict with keys `ref_sequence_latents` [16,21,64,64], `metadata`
- `path.pth`: tensor [4,16,21,64,64] — pre-existing ODE pairs (NOT used, we generate our own)
- LatentSync mask: `/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png`

### GPU Usage
- GPU 2 (~82GB free) used for all verification testing
- 1.3B model: ~3GB weights + activations, fits easily

---

## Bugs & Issues Log

### Bug 001: sinusoidal_embedding_1d dtype mismatch
**Phase**: 1A (wan_model.py)
**Symptom**: `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`
**Cause**: `sinusoidal_embedding_1d()` returns float32 tensors but `time_embedding` Linear layers are bf16
**Fix**: Cast sinusoidal output: `.to(dtype=x.dtype)` before passing to time_embedding
**File**: `fastgen/networks/OmniAvatar/wan_model.py` line 338
**Note**: Original OmniAvatar code has the same issue but it works because they either use autocast
or the model stays in float32 during training (only inputs are bf16 via Accelerate mixed precision)

### Note 001: LoRA merge causes ~0.19 max_diff in bf16
**Phase**: 1A verification
**Observation**: Ported WanModel is BIT-IDENTICAL to original model when given same merged weights.
But merging LoRA (W_merged = W_base + alpha/rank * B@A) in bf16 causes max_diff ~0.19 vs
running LoRA live during forward. This is expected bf16 precision loss and is acceptable.
**Decision**: Accept for now. For exact reproduction, keep LoRA active. For inference/distillation,
merged weights are fine (mean_diff ~7.5e-3 over output range of ±1.5).

