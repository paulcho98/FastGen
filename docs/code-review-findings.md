# Code Review Findings: FastGen OmniAvatar Adaptation

**Date**: 2026-03-23
**Scope**: Exhaustive line-level comparison of Adapted FastGen vs Original FastGen
**Method**: 5 parallel domain-specific comparison agents + adversarial verification agent

## Summary

The adaptation is architecturally clean — all core FastGen files (losses, EMA, gradient
clipping, checkpointing, noise schedule, trainer loop) are **byte-identical** between the
original and adapted repos. OmniAvatar integration uses proper inheritance and composition.

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 5 | Would crash or fundamentally break training |
| HIGH | 8 | Could subtly affect training quality |
| MEDIUM | 11 | Design choices worth documenting |
| LOW | 14+ | Informational, confirmed safe |

---

## Critical Issues Found

### CRIT-1: Dataloader not wired into experiment configs

**Files**: `config_sf.py`, `config_kd.py`
**Problem**: Neither config assigned an OmniAvatar dataloader. Inheritance chain defaulted to
`CIFAR10_Loader_Config` from `BaseConfig`. Training would crash immediately.

### CRIT-2: Fake_score instantiated as 14B instead of intended 1.3B

**Files**: `config_sf.py`, `dmd2.py:48`
**Problem**: `OmniAvatar_V2V_1_3B_FakeScore` was defined but never assigned. `DMD2Model.build_model()`
instantiates fake_score from `self.teacher_config` (14B). Would OOM on target hardware.

### CRIT-3: KD training crashes — `data["path"]` key missing

**Files**: `omniavatar_kd.py:94`, `omniavatar_dataloader.py`
**Problem**: KD `single_train_step()` accesses `data["path"]` for ODE trajectories but
`OmniAvatarDataset` didn't produce this key. Would crash with `KeyError`.

### CRIT-4: Discriminator inner_dim mismatch with 14B teacher

**Files**: `config_sf.py:100`, `discriminator.py`
**Problem**: Config used `Discriminator_Wan_1_3B_Config` (inner_dim=384) but features are
extracted from 14B teacher (dim=5120, unpatchified to 1280). Shape mismatch at runtime.

### CRIT-5: QwenImage configs reference non-existent loader class

**Files**: QwenImage `config_dmd2.py`, `config_sft.py`
**Problem**: Imported `HiDreamG5_JourneyDB_Loader_Config` which doesn't exist. ImportError.

---

## High-Risk Issues Found

### H-1: `mask[0]` batch assumption in `_prepare_training_data`

**File**: `omniavatar_self_forcing.py:62`
**Detail**: Uses `mask[0]` assuming all masks in batch are identical. True for current dataset
(single fixed LatentSync mask) but fragile if per-sample masks are added later.
**Decision**: Keep as-is — we use fixed mask.

### H-2: KD model used wrong `is_ar` default

**File**: `network_causal.py`
**Detail**: `CausalOmniAvatarWan.forward()` defaulted to `is_ar=True`, routing KD to AR mode
(chunk-by-chunk with KV cache). Original CausalWan defaults to `is_ar=False` (full-sequence
with FlexAttention causal mask). KD training should use full-sequence mode.
**Decision**: Fixed — changed default to `is_ar=False`.

### H-3: Per-frame timestep not supported in `_forward_full_sequence`

**File**: `network_causal.py`
**Detail**: `_forward_full_sequence` assumed scalar timestep `[B]`. CausalKD uses `sample_t_inhom`
which returns per-frame timesteps `[B, num_frames]` (constant within chunks). The time embedding
code would break with 2D input.
**Decision**: Fixed — added per-frame timestep embedding support.

### H-4: Causal mask was per-frame instead of per-chunk

**File**: `network_causal.py`, `_build_block_mask()`
**Detail**: OmniAvatar used per-frame causality (each frame only sees prior frames). Original
CausalWan uses per-chunk causality (frames within a chunk attend bidirectionally). Per-chunk
is consistent with `sample_t_inhom` which assigns the same timestep to all frames in a chunk.
**Decision**: Fixed — changed to per-chunk matching CausalWan.

### H-5: Noise schedule shift=5.0 default vs teacher's shift=3.0

**File**: `config_sf.py`, `config_kd.py`
**Detail**: OmniAvatar teacher trained with shift=3.0 (FlowMatchScheduler). FastGen's default
`SampleTConfig.shift` is 5.0. Using mismatched shift during distillation means student trains
on a different timestep distribution than what the teacher was optimized for.
**Decision**: Fixed — set `shift=3.0` explicitly in both configs.

### H-6: KD config inherited from BaseConfig, missing CausalKD fields

**File**: `config_omniavatar_kd.py`
**Detail**: Inherited from `BaseConfig` instead of `config_kd_causal.Config`, missing the
`context_noise` field. Code used `getattr(..., 0)` fallback so didn't crash, but was
inconsistent with the CausalKD design.
**Decision**: Fixed — inherits from `CausalKDConfig`.

### H-7: WanI2V `_replace_first_frame` return value discarded (shared file bug)

**File**: `WanI2V/network_causal.py:527`
**Detail**: Calls `_replace_first_frame()` but discards return value — the function creates a
new tensor, doesn't modify in-place. First frame replacement is a no-op.
**Decision**: Not OmniAvatar-specific. Skip for now.

### H-8: VAE encode `mode="argmax"` dropped in inference script

**File**: `video_model_inference.py`
**Detail**: Changes VAE encoding from deterministic to stochastic. Not applicable to OmniAvatar
training (uses precomputed latents).
**Decision**: Not applicable — skip.

---

## Verified Byte-Identical Files

All core training infrastructure is unchanged:

- `fastgen/methods/model.py` — FastGenModel base
- `fastgen/methods/common_loss.py` — VSD, DSM, GAN losses
- `fastgen/methods/distribution_matching/dmd2.py` — DMD2
- `fastgen/methods/distribution_matching/self_forcing.py` — Self-Forcing rollout
- `fastgen/methods/distribution_matching/causvid.py` — CausVid
- `fastgen/methods/knowledge_distillation/KD.py` — KD / CausalKD
- `fastgen/networks/noise_schedule.py` — Noise schedules
- `fastgen/callbacks/ema.py` — EMA updates
- `fastgen/callbacks/grad_clip.py` — Gradient clipping
- `fastgen/callbacks/callback.py` — Callback system
- `fastgen/utils/checkpointer.py` — Checkpoint save/load
- `fastgen/utils/lr_scheduler.py` — LR scheduling
- `fastgen/utils/basic_utils.py` — Utility functions
- `fastgen/trainer.py` — Training loop (6-line debug log removed, no functional change)

---

## Config Comparison

| Parameter | OmniAvatar SF | OmniAvatar KD | WanT2V SF | OmniAvatar-Train |
|-----------|:---:|:---:|:---:|:---:|
| net LR | 5e-6 | 1e-4 (default) | 5e-6 | 1e-4 |
| fake_score LR | 5e-6 | N/A | 5e-6 | N/A |
| disc LR | 5e-6 | N/A | 5e-6 | N/A |
| batch_size | 1 | 1 | 1 | 1 |
| max_iter | 5000 | 5000 | 5000 | epoch-based |
| t_list | [.999,.937,.833,.624,0] | same | same | N/A |
| student_update_freq | 5 | N/A | 5 | N/A |
| gan_loss_weight_gen | 0.003 | N/A | 0.003 | N/A |
| guidance_scale | 4.5 | None | 5.0 | 4.5 |
| shift | 3.0 | 3.0 | 5.0 (default) | 3.0 |
| in_dim | 65 | 65 | 16 | 33/49/65 |
| precision | bf16 | bf16 | bf16 | bf16 |
| context_noise | 0.0 | 0.0 | 0.0 | N/A |
| chunk_size | 3 | 3 | 3 | N/A |
| disc config | Wan_14B | N/A | Wan_1_3B | N/A |

---

## Adversarial Review Results

The adversarial reviewer confirmed 6 of 8 findings, challenged 2 (downgrading severity),
and found 1 new critical issue (CRIT-4: discriminator inner_dim mismatch).

- Gradient flow through self-forcing rollout: **verified correct**
- No import shadowing between OmniAvatar and FastGen modules: **confirmed**
- All loss computations identical: **confirmed**
