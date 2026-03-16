# OmniAvatar Self-Forcing DMD: Complete Changes Summary

All changes made to adapt FastGen's Self-Forcing distillation framework for OmniAvatar's
V2V audio-driven lip sync model (14B teacher → 1.3B student).

**Total**: 5,634 lines added across 18 files, 13 commits.

---

## New Files Created

### Network Layer (`fastgen/networks/OmniAvatar/`)

| File | Lines | Purpose |
|------|-------|---------|
| `wan_model.py` | 415 | Standalone OmniAvatar DiT (WanModel), ported from OmniAvatar repo with global `args` singleton removed. All config via constructor params. Includes AudioPack integration, per-layer audio injection, 3D RoPE, feature extraction hooks. Verified **BIT-IDENTICAL** to original. |
| `audio_pack.py` | 39 | AudioPack module: 10752-dim Wav2Vec2 features → 32-dim with [4,1,1] patching + LayerNorm. Direct port. |
| `network.py` | 739 | `OmniAvatarWan(FastGenNetwork)` — bidirectional wrapper for teacher and fake_score. Handles: weight loading (base Wan 2.1 safetensors + OmniAvatar LoRA/audio checkpoint + patch embedding expansion), V2V condition dict → y-tensor assembly, prediction type conversion, feature extraction for DMD2 discriminator. |
| `network_causal.py` | 1720 | `CausalOmniAvatarWan(CausalFastGenNetwork)` — causal wrapper for student. Contains full causal DiT reimplementation with: FlexAttention block masks, KV cache management (FastGen detach/cat pattern), dynamic RoPE (cache raw K, apply window-local RoPE at attention time), local attention window with rolling eviction, attention sink, gradient checkpointing with frozen cache metadata. |
| `__init__.py` | 4 | Package exports. |

### Method Layer (`fastgen/methods/`)

| File | Lines | Purpose |
|------|-------|---------|
| `omniavatar_self_forcing.py` | 86 | `OmniAvatarSelfForcingModel(SelfForcingModel)` — overrides `_prepare_training_data()` to build OmniAvatar condition dicts (text, audio, ref_latent, mask, masked_video, ref_sequence) and negative condition (null audio + negative text). |
| `omniavatar_kd.py` | 127 | `OmniAvatarKDModel(CausalKDModel)` — overrides `single_train_step()` for causal KD with OmniAvatar conditions and inhomogeneous timesteps. |
| `omniavatar_diffusion_forcing.py` | 113 | `OmniAvatarDiffusionForcingModel(KDModel)` — alternative Stage 1: diffusion forcing on real data with inhomogeneous block-wise timesteps. No ODE trajectory generation needed. |

### Dataset Layer (`fastgen/datasets/`)

| File | Lines | Purpose |
|------|-------|---------|
| `omniavatar_dataloader.py` | 205 | PyTorch Dataset reading OmniAvatar's precomputed .pt files (vae_latents, audio_emb, text_emb, ref_latents, spatial mask). Returns all tensors in bf16. 29,044 training samples. |

### Config Layer (`fastgen/configs/`)

| File | Lines | Purpose |
|------|-------|---------|
| `experiments/OmniAvatar/config_sf.py` | 127 | Self-Forcing experiment config: 14B teacher, 1.3B causal student, 1.3B bidirectional fake_score, discriminator, all hyperparams. |
| `experiments/OmniAvatar/config_kd.py` | 66 | Causal KD experiment config for Stage 1 ODE initialization. |
| `experiments/OmniAvatar/config_df.py` | 65 | Diffusion Forcing experiment config for Stage 1 (alternative to ODE KD). |
| `methods/config_omniavatar_sf.py` | 51 | Self-Forcing method config (model class registration). |
| `methods/config_omniavatar_df.py` | 48 | Diffusion Forcing method config (model class registration). |
| `methods/config_omniavatar_kd.py` | 43 | Causal KD method config. |

### Scripts (`scripts/`)

| File | Lines | Purpose |
|------|-------|---------|
| `generate_omniavatar_ode_pairs.py` | 584 | ODE trajectory extraction from teacher. Deterministic 50-step ODE solve with audio conditioning, saves `ode_path.pt` [4,16,21,64,64] per sample. Supports distributed processing via torchrun. |
| `create_verification_samples.py` | 422 | Generates reference inputs/outputs from original OmniAvatar code for numerical verification. |

### Documentation (`docs/`)

| File | Lines | Purpose |
|------|-------|---------|
| `omniavatar-self-forcing-plan.md` | 861 | Full implementation plan with architecture, design decisions, data flow, memory budget, risk register. |
| `implementation-notes.md` | 85 | Bug tracker and implementation notes (5 bugs fixed, 3 notes). |
| `omniavatar-changes-summary.md` | — | This file. |

---

## Key Design Decisions

1. **Custom WanModel (not diffusers)**: OmniAvatar's DiT is a custom implementation with audio conditioning baked in. We ported it directly rather than modifying diffusers' WanTransformer3DModel.

2. **Fake score = 1.3B bidirectional**: Same `OmniAvatarWan` wrapper as teacher but smaller. Saves ~25GB vs using 14B. Learns student's output distribution via DSM loss.

3. **All networks use V2V 65ch mode**: Teacher, student, and fake_score all receive the same condition dict (text, audio, ref, mask, masked_video, ref_sequence).

4. **Only student is causal**: Teacher and fake_score are bidirectional (single forward pass). Student uses CausalOmniAvatarWan with KV cache for autoregressive rollout.

5. **Audio identical everywhere**: Same AudioPack + per-layer injection in all three networks and during ODE extraction. No modified audio paths.

6. **Gradient-safe KV cache**: FastGen pattern — `.detach()` on cache writes, `cat [detached_past | live_current]` for attention. Gradients only flow through current chunk's Q/K/V.

---

## Key Architectural Features

### CausalSelfAttention (network_causal.py)

**Two attention modes**:
- **Full-sequence** (kv_cache=None): FlexAttention with blockwise causal mask
- **AR** (kv_cache provided): Flash attention with accumulated KV cache

**Two RoPE modes** (switchable via `use_dynamic_rope`):
- **Original** (`False`): Pre-rotate Q/K before caching. Standard approach.
- **Dynamic** (`True`): Cache raw K, apply window-local RoPE at attention time. Better positional generalization for long contexts.

**Local attention window** (`local_attn_size`):
- `-1`: Global attention (attend everything)
- `>0`: Rolling window of N frames. Cache sized to N × frame_seqlen.

**Attention sink** (`sink_size`):
- First N frames never evicted from cache, always in attention window.

**Gradient checkpointing**:
- Cache metadata frozen via `cache_local_end_override` (computed from immutable `current_start`), following FastGen's `proper_cache_len` pattern.

### Feature Extraction (wan_model.py + network.py)

- Block outputs collected at requested `feature_indices` (e.g., {15, 22, 29})
- Unpatchified to video shape: `[B, dim/4, T, H, W]` (384 for 1.3B, 1280 for 14B)
- Compatible with DMD2's `Discriminator_VideoDiT`
- `return_features_early=True` exits before head for efficiency
- Output unchanged when features not requested (verified 0.00 diff)

### Weight Loading (network.py)

Two-stage loading:
1. Base Wan 2.1 safetensor weights → WanModel (smart_load_weights handles shape mismatch)
2. OmniAvatar checkpoint → LoRA + audio modules + patch_embedding (with expansion 16→33→49→65ch)

Optional LoRA merge: `W_merged = W_base + (alpha/rank) × B @ A` in bf16.

---

## Bugs Fixed During Implementation

| # | Bug | Fix |
|---|-----|-----|
| 001 | `sinusoidal_embedding_1d` returns float32 vs bf16 model | Cast to `x.dtype` before time_embedding |
| 002 | `_build_y` not slicing V2V conditioning for chunks | Added `start_frame` param, slice masked_video/ref_sequence |
| 003 | KV cache indexing used ever-incrementing counter | Use `current_start` for position-based idempotent writes |
| 004 | Flash attention backward fails after in-place cache write | Replaced with detach/cat pattern (no in-place writes in grad path) |
| 005 | Feature extraction returned wrong type for DMD2 | Return `[x0, features]` tuple, not just x0 |

---

## Verification Results

| Test | Method | Result |
|------|--------|--------|
| WanModel forward pass | Compare ported vs original OmniAvatar output | **BIT-IDENTICAL** (0.00 max diff with merged weights) |
| OmniAvatarWan wrapper | Condition dict → _build_y → model → output | **BIT-IDENTICAL** |
| CausalOmniAvatarWan full-seq | FlexAttention causal forward | Correct shapes |
| CausalOmniAvatarWan AR | 7 chunks × 3 frames with KV cache | All chunks correct |
| Self-Forcing training step | Rollout → VSD loss → backward → optimizer | 703 params with gradients |
| Feature extraction | Blocks {15,22,29} → unpatchify → discriminator | [B,384,21,64,64] per feature |
| Gradient checkpointing | All 7 chunks with grad | 12.1 GB (vs 87+ GB without) |
| Dynamic RoPE | AR mode with cache raw K | Correct output |
| Local window + sink | 12-frame window, 2-frame sink | No eviction during 21-frame training |

---

## Memory Profile (1.3B stand-in, 512×512, 21 frames)

| Configuration | Peak Memory |
|--------------|-------------|
| 3× 1.3B models loaded | 8.5 GB |
| Rollout, no grad ckpt, all 7 chunks grad | >87 GB (OOM) |
| Rollout, WITH grad ckpt, all 7 chunks grad | **12.1 GB** |
| Rollout, WITH grad ckpt, last 2 chunks grad | 13.0 GB |
| **Estimated production (14B teacher + 1.3B student + 1.3B FS)** | **~70-85 GB** |

---

## Remaining Work

- [ ] Train 1.3B student with `--use_ref_sequence` (65ch) in OmniAvatar native training
- [ ] Generate ODE pairs with 14B teacher (script ready, needs GPU time)
- [ ] Run KD pre-training (Stage 1) — config ready
- [ ] Run Self-Forcing training (Stage 2) — config ready
- [ ] Inference script for distilled model
- [ ] `frame_offset` handling for CausVid-style long-form generation (not needed for SF)
