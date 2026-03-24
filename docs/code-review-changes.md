# Changes Made: Code Review Fixes (2026-03-23)

All changes fix issues found during the exhaustive code review comparing the
adapted FastGen against the original FastGen repo.

---

## 1. Dataloader wired into experiment configs

**Files**: `fastgen/configs/experiments/OmniAvatar/config_sf.py`, `config_kd.py`

Added `create_omniavatar_dataloader` import and `L(create_omniavatar_dataloader)(...)`
assignment to `config.dataloader_train` in both SF and KD experiment configs.

Previously both configs inherited the `CIFAR10_Loader_Config` default and only
overrode `batch_size=1`.

---

## 2. Fake_score uses 1.3B (not 14B teacher)

**Files**: `config_sf.py`, `config_omniavatar_sf.py`, `omniavatar_self_forcing.py`

- Added `config.model.fake_score = OmniAvatar_V2V_1_3B_FakeScore` to `config_sf.py`
- Added `OmniAvatarModelConfig` with `fake_score: Optional[DictConfig]` field to
  `config_omniavatar_sf.py`
- Added `build_model()` override in `OmniAvatarSelfForcingModel` that re-instantiates
  `self.fake_score` from `config.fake_score` when set

The base `DMD2Model.build_model()` always creates fake_score from `self.teacher_config`
(14B). The override replaces it with the 1.3B bidirectional model.

---

## 3. Discriminator uses 14B-compatible config

**File**: `config_sf.py`

Changed from `Discriminator_Wan_1_3B_Config` (inner_dim=384) to
`Discriminator_Wan_14B_Config` (inner_dim=1280). Features are extracted from the
14B teacher, so the discriminator input dimension must match.

---

## 4. KD dataset supports ODE trajectory loading

**File**: `fastgen/datasets/omniavatar_dataloader.py`

Added `load_ode_path: bool = False` parameter. When enabled, loads `ode_path.pt`
from each sample directory and returns it under the `"path"` key, which
`OmniAvatarKDModel.single_train_step()` expects.

---

## 5. QwenImage configs reverted to ImageLoaderConfig

**Files**: `fastgen/configs/experiments/QwenImage/config_dmd2.py`, `config_sft.py`

Reverted `HiDreamG5_JourneyDB_Loader_Config` (non-existent) back to `ImageLoaderConfig`.

---

## 6. Noise schedule shift=3.0

**Files**: `config_sf.py`, `config_kd.py`

Added `config.model.sample_t_cfg.shift = 3.0` to both configs. The OmniAvatar teacher
was trained with shift=3.0 (FlowMatchScheduler default). FastGen's default is 5.0.

---

## 7. CausalOmniAvatarWan `is_ar` default changed to False

**File**: `fastgen/networks/OmniAvatar/network_causal.py`

Changed `forward()` signature from `is_ar: bool = True` to `is_ar: bool = False`,
matching `CausalWan.forward()`. This ensures KD training routes to
`_forward_full_sequence` (full-sequence with FlexAttention causal mask) instead of
`_forward_ar` (chunk-by-chunk with KV cache).

Self-forcing is unaffected — `rollout_with_gradient()` passes `is_ar=True` explicitly.

---

## 8. Per-frame timestep support in `_forward_full_sequence`

**File**: `fastgen/networks/OmniAvatar/network_causal.py`

`CausalKDModel.single_train_step()` calls `gen_data_from_net()` with per-frame
timesteps `t_inhom` of shape `[B, num_frames]` (from `sample_t_inhom`, which assigns
one timestep per chunk, repeated across frames in each chunk).

Changes:
- `_forward_full_sequence`: Added `if timestep.ndim == 2:` branch that flattens
  per-frame timesteps to `[B*num_frames]`, embeds each independently, then reshapes
  back to `[B, num_frames, dim]` / `[B, num_frames, 6, dim]`
- `forward()`: Added `t_converted = t[:, None, :, None, None] if t.ndim == 2 else t`
  before `convert_model_output`, matching CausalWan's handling
- Head call: Changed from `t_emb.unsqueeze(1).unsqueeze(2).expand(...)` to
  `t_emb.unsqueeze(2)` since `t_emb` is now always `[B, f, dim]`

---

## 9. Chunk-wise causal mask (matching CausalWan)

**File**: `fastgen/networks/OmniAvatar/network_causal.py`

Changed `_build_block_mask()` from per-frame causality to per-chunk causality,
matching `CausalWan._prepare_blockwise_causal_attn_mask()`.

**Before** (per-frame): Each frame only attends to itself and prior frames.
**After** (per-chunk): Frames within the same chunk attend bidirectionally.
Tokens can attend to all previous chunks.

With chunk_size=3 and 21 frames: 7 chunks of 3 frames each. Consistent with
`sample_t_inhom` which assigns the same noise level to all frames in a chunk.

Added `chunk_size` parameter to `_build_block_mask()`.

---

## 10. KD config inherits from CausalKDConfig

**File**: `fastgen/configs/methods/config_omniavatar_kd.py`

Changed from `class Config(BaseConfig)` to `class Config(CausalKDConfig)`,
inheriting from `fastgen.configs.methods.config_kd_causal.Config`. This provides:
- `ModelConfig.context_noise` field (defaults to 0.0)
- `student_sample_steps = 4` at method level
- Proper CausalKDModel configuration structure
