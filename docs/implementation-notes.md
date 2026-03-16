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

### Bug 002: _build_y not slicing V2V conditioning for chunks
**Phase**: 1D (network_causal.py)
**Symptom**: `RuntimeError: Sizes of tensors must match except in dimension 1` when running AR mode
**Cause**: `_build_y` used full-length `masked_video` (21 frames) but `ref_repeated` was only chunk-length (3 frames)
**Fix**: Added `start_frame` param to `_build_y`, slice masked_video/ref_sequence to `[start_frame:start_frame+T]`
**File**: `fastgen/networks/OmniAvatar/network_causal.py`

### Bug 003: KV cache indexing in CausalSelfAttention AR mode
**Phase**: 1D (network_causal.py)
**Symptom**: `RuntimeError: expanded size of tensor (0) must match existing size (3072)`
**Cause**: Cache write used `end_index` (ever-incrementing) instead of `current_start` (position-based).
Self-Forcing calls student twice at same cur_start_frame (denoise then cache), so end_index doubled.
**Fix**: Use `current_start` for cache write position (idempotent). High-water mark via max().

### Bug 004: Flash attention backward fails after KV cache in-place write
**Phase**: Self-Forcing training step
**Symptom**: `RuntimeError: variable needed for gradient has been modified by inplace operation`
**Cause**: store_kv=True call writes to cache positions that flash attention's backward saved refs to.
**Fix**: `.clone()` KV cache contents before passing to flash attention in AR mode.
**File**: `network_causal.py` line 399-400

### Bug 005: Feature extraction return type incompatible with DMD2
**Phase**: Compatibility audit
**Symptom**: `ValueError: not enough values to unpack (expected 2, got 1)` in DMD2's discriminator training
**Cause**: DMD2 expects `teacher_x0, fake_feat = self.teacher(..., feature_indices=...)` (tuple of 2).
Our OmniAvatarWan returned just the x0 tensor, not `[x0, features]`.
**Fix**: Return `[out, []]` when `feature_indices` is non-empty. Empty features means
GAN loss is effectively disabled. Full feature extraction requires adding hooks to WanModel
(future work).
**Files**: `network.py`, `network_causal.py`

### Note 003: frame_offset for CausVid extrapolation
CausVid passes `frame_offset` for long-form generation. Our implementation absorbs it
into `**fwd_kwargs` but doesn't use it. Not a blocker for Self-Forcing training (which
uses `cur_start_frame` instead), but would need fixing for CausVid-style inference.

### Note 002: Memory for Self-Forcing with 3x 1.3B models
- 3 models loaded: 8.5 GB
- Rollout with gradient (start_gradient_frame=15, last 2 chunks): 40.5 GB peak
- After backward: 41.3 GB
- Fake score DSM update OOMs at ~83 GB limit on GPU 2 — needs separate GPU or cleanup
- With 14B teacher, expect ~70 GB additional → need multi-GPU or FSDP

