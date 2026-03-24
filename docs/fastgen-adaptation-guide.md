# FastGen Self-Forcing Adaptation Guide

How to port a new talking-face model into FastGen's Self-Forcing distillation
framework. Based on the OmniAvatar adaptation as reference implementation.

**Reference repos:**
- Original FastGen: `original_FastGen/FastGen/` (Nvidia's reference, ground truth)
- OmniAvatar adaptation: `reference_FastGen_OmniAvatar/FastGen/` (working example)
- OmniAvatar baseline: `reference_FastGen_OmniAvatar/OmniAvatar-Train/` (source model)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [What FastGen Provides (DO NOT MODIFY)](#2-what-fastgen-provides)
3. [What You Must Implement (7 Components)](#3-what-you-must-implement)
4. [Component 1: Standalone DiT Model](#4-component-1-standalone-dit-model)
5. [Component 2: Bidirectional Network Wrapper](#5-component-2-bidirectional-network-wrapper)
6. [Component 3: Causal Network Wrapper](#6-component-3-causal-network-wrapper)
7. [Component 4: Dataset Adapter](#7-component-4-dataset-adapter)
8. [Component 5: Method Subclasses](#8-component-5-method-subclasses)
9. [Component 6: Config Files](#9-component-6-config-files)
10. [Component 7: ODE Trajectory Generation Script](#10-component-7-ode-trajectory-generation-script)
11. [Training Pipeline Stages](#11-training-pipeline-stages)
12. [Critical Gotchas (Lessons Learned)](#12-critical-gotchas)
13. [File Inventory](#13-file-inventory)

---

## 1. Architecture Overview

FastGen's Self-Forcing distillation trains a fast few-step student from a slow
multi-step teacher. The pipeline has two stages:

### Stage 1: Causal KD (ODE Initialization)
```
Pre-generated ODE trajectories (from teacher) → CausalKD training
    Student: causal network, full-sequence mode (is_ar=False)
    FlexAttention chunk-wise causal mask
    Per-chunk timesteps via sample_t_inhom
    Loss: L2(student_output, clean_data)
    Output: ode_init.pt checkpoint
```

### Stage 2: Self-Forcing DMD
```
Raw training data → Self-Forcing with DMD/VSD loss
    Student: causal network, AR mode (is_ar=True) with KV cache
    Teacher: bidirectional network (frozen, for CFG + VSD)
    Fake score: bidirectional network (trained, for VSD loss)
    Discriminator: GAN loss on teacher features
    Loss: VSD + GAN (student steps), DSM + GAN (fake_score/disc steps)
    Alternating: 1 student update per 5 iterations (student_update_freq=5)
```

### The 4 network roles

| Role | Architecture | Mode | Trainable | Size (OmniAvatar) |
|------|-------------|------|-----------|-------------------|
| Teacher | Bidirectional | Single forward pass | Frozen | 14B |
| Student | Causal | AR (SF) or Full-seq (KD) | Yes | 1.3B |
| Fake Score | Bidirectional | Single forward pass | Yes (DSM loss) | 1.3B |
| Discriminator | MLP on features | - | Yes (GAN loss) | Small |

The teacher and fake_score can be different sizes from the student. This
requires `config.model.fake_score` config support (see gotchas).

---

## 2. What FastGen Provides (DO NOT MODIFY)

These files are the core training infrastructure. They must remain byte-identical
to the original. Your adaptation works by subclassing and configuring, never patching.

### Training Loop
- `fastgen/trainer.py` — Main training loop, gradient accumulation, checkpointing
- `fastgen/methods/model.py` — `FastGenModel` base class
- `fastgen/methods/distribution_matching/dmd2.py` — DMD2 training (VSD + GAN)
- `fastgen/methods/distribution_matching/self_forcing.py` — Self-Forcing rollout
- `fastgen/methods/distribution_matching/causvid.py` — CausVid (parent of SF)
- `fastgen/methods/knowledge_distillation/KD.py` — KD + CausalKD
- `fastgen/methods/common_loss.py` — VSD, DSM, GAN loss functions

### Infrastructure
- `fastgen/networks/noise_schedule.py` — Noise schedules (RF, EDM, etc.)
- `fastgen/networks/network.py` — `FastGenNetwork` / `CausalFastGenNetwork` base classes
- `fastgen/callbacks/ema.py` — EMA updates
- `fastgen/callbacks/grad_clip.py` — Gradient clipping
- `fastgen/utils/checkpointer.py` — Checkpoint save/load
- `fastgen/configs/config.py` — Config classes (BaseConfig, BaseModelConfig, etc.)

### Key base class contracts

**`FastGenNetwork`** (for teacher + fake_score):
- Must implement `forward(x_t, t, condition, fwd_pred_type, feature_indices, return_features_early, ...)`
- Must expose `noise_scheduler` property
- Must expose `net_pred_type` property

**`CausalFastGenNetwork`** (for student):
- Same as above, plus `chunk_size`, `total_num_frames`
- `forward()` must support `is_ar` parameter (default `False`!)
- `is_ar=False` → full-sequence with FlexAttention causal mask (used by CausalKD)
- `is_ar=True` → chunk-by-chunk AR with KV cache (used by Self-Forcing)

---

## 3. What You Must Implement (7 Components)

```
fastgen/networks/<YourModel>/
    __init__.py                  # Exports
    <dit_model>.py               # Component 1: Standalone DiT
    network.py                   # Component 2: Bidirectional wrapper
    network_causal.py            # Component 3: Causal wrapper
fastgen/datasets/
    <your>_dataloader.py         # Component 4: Dataset adapter
fastgen/methods/
    <your>_self_forcing.py       # Component 5a: SF method subclass
    <your>_kd.py                 # Component 5b: KD method subclass
fastgen/configs/methods/
    config_<your>_sf.py          # Component 6a: SF method config
    config_<your>_kd.py          # Component 6b: KD method config
fastgen/configs/experiments/<YourModel>/
    config_sf.py                 # Component 6c: SF experiment config
    config_kd.py                 # Component 6d: KD experiment config
scripts/
    generate_<your>_ode_pairs.py # Component 7: ODE trajectory generation
```

---

## 4. Component 1: Standalone DiT Model

**OmniAvatar file:** `fastgen/networks/OmniAvatar/wan_model.py` (415 lines)
**Pattern file:** N/A (ported from source model repo)

This is a **clean copy** of your source model's DiT architecture, stripped of:
- Global `args` singletons (pass params explicitly)
- Sequence parallelism / context parallel logic (FastGen handles this via FSDP)
- Inference-only optimizations (TeaCache, gradient checkpointing offload)
- State dict converter classes (handled by the wrapper)

Added for FastGen:
- `feature_indices` and `return_features_early` parameters in `forward()` —
  needed for GAN discriminator feature extraction
- `_unpatchify_features()` method — converts patched features to spatial tensors

### What to port from your source model:
- All transformer blocks (DiT blocks, attention, FFN)
- RoPE / positional embedding computation
- Patch embedding and unpatchify
- Time embedding (sinusoidal → MLP → modulation)
- Any model-specific conditioning injection (e.g., audio additive residuals)
- Cross-attention for text conditioning

### Critical details:
- `sinusoidal_embedding_1d` may return float32 — cast to model dtype before
  passing to time_embedding MLP (Bug 001)
- Audio/conditioning injection must happen at the same layers as the source model
- The `forward()` signature should accept `**kwargs` for flexibility

---

## 5. Component 2: Bidirectional Network Wrapper

**OmniAvatar file:** `fastgen/networks/OmniAvatar/network.py` (739 lines)
**Pattern file:** `fastgen/networks/Wan/network.py`

Wraps the standalone DiT as a `FastGenNetwork` subclass. Used for teacher and fake_score.

### Required implementation:

```python
class YourModelWan(FastGenNetwork):
    def __init__(self,
        model_size: str,        # e.g. "14B", "1.3B"
        in_dim: int,            # Input channels (noise + conditioning)
        mode: str,              # e.g. "v2v", "i2v", "t2v"
        use_audio: bool,
        base_model_paths: str,  # Comma-separated safetensor paths
        your_ckpt_path: str,    # Fine-tuned checkpoint path
        net_pred_type: str,     # "flow", "x0", "eps", "v"
        schedule_type: str,     # "rf" for rectified flow
        ...
    ):
        super().__init__()
        # 1. Instantiate self.model = YourDiT(...)
        # 2. Instantiate self.noise_scheduler = RFNoiseSchedule(...)
        # 3. Call self._load_weights()
```

### Weight loading pipeline (3 stages):

**Stage 1: Base diffusers weights**
```python
def _load_weights(self):
    # Load base model (e.g., Wan2.1-T2V-1.3B from HuggingFace safetensors)
    base_sd = _load_state_dict(self.base_model_paths)
    # Convert key names: diffusers format → your DiT format
    converted_sd = _convert_diffusers_state_dict(base_sd)
    # Smart load: handle shape mismatches (e.g., patch_embedding expanded for V2V)
    _smart_load_weights(self.model, converted_sd)
```

**Stage 2: Fine-tuned checkpoint (LoRA or full)**
```python
    # Load your fine-tuned checkpoint
    ckpt_sd = torch.load(your_ckpt_path)
    if has_lora_keys(ckpt_sd):
        # Merge LoRA: W_merged = W_base + (alpha/rank) * B @ A
        _merge_lora_into_model(self.model, lora_sd, rank, alpha)
    else:
        # Direct load of non-LoRA weights (audio modules, expanded patch_emb)
        self.model.load_state_dict(non_lora_sd, strict=False)
```

**Stage 3: Patch embedding expansion (for V2V)**
```python
    # If in_dim > base in_dim, expand patch_embedding weight
    # e.g., base=16ch → V2V=65ch: zero-pad the extra channels
    if self.in_dim > base_in_dim:
        new_weight = torch.zeros(out_ch, self.in_dim, *kernel_size)
        new_weight[:, :base_in_dim] = old_weight
        self.model.patch_embedding.weight = nn.Parameter(new_weight)
```

### `_build_y()` — Condition tensor assembly

Assembles all non-noise conditioning into a single tensor `y` that gets
concatenated with the noisy latent in the DiT's forward pass.

```python
def _build_y(self, condition: dict, T: int, start_frame: int = 0) -> torch.Tensor:
    # For OmniAvatar V2V (65ch = 16 noise + 49 conditioning):
    #   y = cat([ref_repeated, mask, masked_video, ref_sequence], dim=1)
    #   shape: [B, 49, T, H, W]
    # The noise (16ch) is NOT included — it's concatenated in the DiT's forward()
```

### `forward()` — Main forward pass

```python
def forward(self, x_t, t, condition, fwd_pred_type, feature_indices, return_features_early, ...):
    # 1. Build y from condition dict
    y = self._build_y(condition, T=x_t.shape[2])
    # 2. Rescale timestep for the model
    timestep = self.noise_scheduler.rescale_t(t)
    # 3. Forward through DiT
    model_output = self.model(x=x_t, timestep=timestep, context=text_embeds, y=y, audio_emb=audio_emb,
                              feature_indices=feature_indices, return_features_early=return_features_early)
    # 4. Convert prediction type (flow → x0, etc.)
    out = self.noise_scheduler.convert_model_output(x_t, model_output, t, ...)
    return out  # or [out, features] if feature_indices set
```

---

## 6. Component 3: Causal Network Wrapper

**OmniAvatar file:** `fastgen/networks/OmniAvatar/network_causal.py` (1750 lines)
**Pattern file:** `fastgen/networks/Wan/network_causal.py`

This is the most complex component. It wraps the DiT as a `CausalFastGenNetwork`
with two forward modes.

### Architecture: Duplicate DiT components with causal modifications

The causal network contains its OWN copy of all DiT components (not shared with
the bidirectional version), with these modifications:

| Component | Bidirectional | Causal |
|-----------|--------------|--------|
| SelfAttention | Standard flash_attn | `CausalSelfAttention` with KV cache + FlexAttention |
| DiTBlock | Standard | `CausalDiTBlock` with per-frame modulation |
| Head | Standard | `CausalHead` with per-frame modulation |
| RoPE | Global position | Frame-offset position (`causal_rope_apply`) |

### Two forward modes:

**`_forward_full_sequence(is_ar=False)`** — Used by CausalKD
```
Input: Full noisy latent [B, 16, T, H, W] with per-frame timesteps [B, T]
Mask: FlexAttention chunk-wise causal block mask
Processing: Single forward pass, all frames at once
Per-frame timestep: Each frame embedded independently
    if timestep.ndim == 2:  # per-frame
        flat → embed → reshape to [B, num_frames, dim]
    else:  # scalar
        embed → expand to all frames
```

**`_forward_ar(is_ar=True)`** — Used by Self-Forcing
```
Input: Single chunk [B, 16, chunk_frames, H, W] with scalar timestep [B]
Cache: KV cache stores attention keys/values from previous chunks
Processing: Chunk sees cached past + computes current
Audio: Full audio processed globally, then sliced per chunk
```

### FlexAttention chunk-wise causal mask

```python
def _build_block_mask(self, device, num_frames, frame_seqlen, chunk_size):
    # With chunk_size=3 and 21 frames → 7 chunks
    # Chunk 0 (frames 0-2): tokens attend to frames 0-2
    # Chunk 1 (frames 3-5): tokens attend to frames 0-5
    # Chunk 2 (frames 6-8): tokens attend to frames 0-8
    # ... etc. Bidirectional within chunk, causal across chunks.
    # This matches sample_t_inhom which assigns same timestep per chunk.
```

### KV cache management

```python
def _init_caches(self, batch_size, total_tokens, frame_seqlen, device, dtype):
    # Pre-allocate: one KV cache per transformer block
    # shape: {k: [B, total_tokens, dim], v: [B, total_tokens, dim]}
    # Track: local_start, local_end per block

def clear_caches(self):
    # Reset all caches — call between samples
    self._kv_caches = None
    self._crossattn_caches = None
    self.block_mask = None
```

### Weight loading

Same 3-stage pipeline as the bidirectional wrapper. The causal model uses
identical parameter names to the bidirectional DiT, so weights transfer directly.

### CRITICAL: `is_ar` default must be `False`

```python
def forward(self, x_t, t, condition, is_ar=False, ...):
```

The default MUST be `False` to match `CausalWan.forward()`. Reason:
- `CausalKDModel.single_train_step()` calls `gen_data_from_net()` which calls
  `self.net(input, t, condition=cond, fwd_pred_type="x0")` — no `is_ar` passed
- With default `False`: routes to `_forward_full_sequence` (correct for KD)
- Self-Forcing explicitly passes `is_ar=True` in `rollout_with_gradient()`

---

## 7. Component 4: Dataset Adapter

**OmniAvatar file:** `fastgen/datasets/omniavatar_dataloader.py` (221 lines)
**Pattern file:** None (FastGen's built-in loaders use WebDataSet)

Custom `torch.utils.data.Dataset` that loads precomputed tensors.

### Required output dict keys:

| Key | Shape | Description | Used by |
|-----|-------|-------------|---------|
| `real` | `[C, T, H, W]` | Clean video latents | Both KD and SF |
| `condition-specific keys` | varies | Your model's conditioning | Both |
| `neg_*` keys | varies | Negative condition for CFG | SF only |
| `path` | `[num_steps, C, T, H, W]` | ODE trajectory | KD only |

For OmniAvatar:
- `real`: `[16, 21, 64, 64]` — VAE-encoded clean video
- `masked_video`: `[16, 21, 64, 64]` — mouth-masked video latents
- `audio_emb`: `[81, 10752]` — Wav2Vec2 all hidden states
- `text_embeds`: `[1, 512, 4096]` — T5 text embeddings
- `mask`: `[64, 64]` — spatial mask (shared, loaded once)
- `ref_sequence`: `[16, 21, 64, 64]` — reference sequence (optional)
- `neg_text_embeds`: `[1, 512, 4096]` — negative text for CFG
- `path`: `[4, 16, 21, 64, 64]` — ODE noisy states (KD only)

### Important patterns:
- All tensors cast to `bf16` in the dataset (matches model precision)
- Shared tensors (mask, neg_text_embeds) loaded once in `__init__`
- `collate_fn` filters `None` returns from failed loads
- `load_ode_path` parameter controls whether to load ODE trajectories
- Must wrap in `create_<your>_dataloader()` function for LazyCall configs

---

## 8. Component 5: Method Subclasses

### 5a: Self-Forcing method

**OmniAvatar file:** `fastgen/methods/omniavatar_self_forcing.py` (105 lines)
**Parent:** `SelfForcingModel` → `CausVidModel` → `DMD2Model` → `FastGenModel`

Overrides exactly TWO methods:

**`_prepare_training_data(data)`** — Map dataset output to (real_data, condition, neg_condition)
```python
def _prepare_training_data(self, data):
    real_data = data["real"]  # [B, C, T, H, W]
    condition = {
        "text_embeds": ...,
        "audio_emb": ...,
        # ... all your model's conditioning keys
    }
    neg_condition = {
        "text_embeds": data["neg_text_embeds"],
        "audio_emb": torch.zeros_like(...),  # null audio for CFG
        # ... same spatial conditioning as positive
    }
    return real_data, condition, neg_condition
```

**`build_model()`** — Re-instantiate fake_score if teacher≠fake_score architecture
```python
def build_model(self):
    super().build_model()  # Creates fake_score from teacher_config (wrong size)
    if getattr(self.config, "fake_score", None) is not None:
        self.fake_score = instantiate(self.config.fake_score)  # Correct size
```

### 5b: KD method

**OmniAvatar file:** `fastgen/methods/omniavatar_kd.py` (128 lines)
**Parent:** `CausalKDModel` → `KDModel` → `FastGenModel`

Overrides TWO methods:

**`_build_condition(data)`** — Build condition dict (same as SF's positive condition)
```python
def _build_condition(self, data):
    # Identical to SF's condition dict (without neg_condition)
    return condition
```

**`single_train_step(data, iteration)`** — Full override of parent
```python
def single_train_step(self, data, iteration):
    denoise_path = data["path"]      # [B, num_steps, C, T, H, W]
    denoised_data = data["real"]     # [B, C, T, H, W]
    condition = self._build_condition(data)

    # Sample per-chunk timesteps
    t_inhom, ids = self.net.noise_scheduler.sample_t_inhom(...)  # [B, T]

    # Gather noisy data from ODE path at sampled timestep indices
    noisy_data = torch.gather(denoise_path_all, 1, ids).squeeze(1)

    # Student forward (is_ar=False by default → full-sequence with causal mask)
    gen_data = self.gen_data_from_net(noisy_data, t_inhom, condition=condition)

    # L2 loss
    loss = 0.5 * F.mse_loss(gen_data, denoised_data)
    return loss_map, outputs
```

---

## 9. Component 6: Config Files

### 6a: SF method config

**OmniAvatar file:** `fastgen/configs/methods/config_omniavatar_sf.py`
**Parent:** `config_self_forcing.Config` → `config_dmd2.Config` → `BaseConfig`

```python
# Extend ModelConfig to add fake_score field (not in base DMD2)
class YourModelConfig(SFModelConfig):
    fake_score: Optional[DictConfig] = None

class Config(SFConfig):
    model: YourModelConfig = attrs.field(factory=YourModelConfig)
    model_class = L(YourSelfForcingModel)(config=None)
```

### 6b: KD method config

**OmniAvatar file:** `fastgen/configs/methods/config_omniavatar_kd.py`
**Parent:** `config_kd_causal.Config` (NOT BaseConfig!)

```python
class Config(CausalKDConfig):  # Provides context_noise field
    model_class = L(YourKDModel)(config=None)
```

### 6c: SF experiment config

**OmniAvatar file:** `fastgen/configs/experiments/OmniAvatar/config_sf.py`

Defines 3 network configs + all hyperparameters:
```python
Teacher_Config = L(YourBidirectionalNetwork)(model_size="14B", ...)
FakeScore_Config = L(YourBidirectionalNetwork)(model_size="1.3B", ...)
Student_Config = L(YourCausalNetwork)(model_size="1.3B", chunk_size=3, ...)

def create_config():
    config.model.net = Student_Config
    config.model.teacher = Teacher_Config
    config.model.fake_score = FakeScore_Config

    # Discriminator must match TEACHER feature dim, not student
    config.model.discriminator = Discriminator_Wan_14B_Config  # inner_dim = teacher_dim // 4
    config.model.discriminator.feature_indices = [15, 22, 29]  # valid block indices for teacher

    # shift must match YOUR teacher's training distribution
    config.model.sample_t_cfg.shift = 3.0  # OmniAvatar used 3.0, Wan2.1 base uses 5.0

    # Wire YOUR dataloader
    config.dataloader_train = L(create_your_dataloader)(...)
```

### 6d: KD experiment config

**OmniAvatar file:** `fastgen/configs/experiments/OmniAvatar/config_kd.py`

Only the student network (no teacher, no fake_score, no discriminator):
```python
Student_Config = L(YourCausalNetwork)(model_size="1.3B", ...)

def create_config():
    config.model.net = Student_Config
    config.dataloader_train = L(create_your_dataloader)(load_ode_path=True, ...)
```

---

## 10. Component 7: ODE Trajectory Generation Script

**OmniAvatar file:** `scripts/generate_omniavatar_ode_pairs.py` (584 lines)

Standalone script that generates ODE trajectories from the teacher for KD Stage 1.

### Pipeline:
```
For each training sample:
    1. Load precomputed data (VAE latents, audio, text, mask, etc.)
    2. Build condition dict (same as training)
    3. Sample noise
    4. Run deterministic multi-step ODE solve with teacher + CFG:
        for step in range(num_steps):
            x0_cond = teacher(x_t, t, condition)
            x0_uncond = teacher(x_t, t, neg_condition)
            x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
            eps = noise_scheduler.x0_to_eps(x_t, x0, t)
            x_t_next = noise_scheduler.forward_process(x0, eps, t_next)
            trajectory.append(x_t)
    5. Subsample trajectory at target t_list indices
    6. Save: ode_path.pt [num_steps, C, T, H, W]
```

### Key details:
- Uses FastGen's noise schedule (NOT the source model's scheduler)
- CFG applied during ODE solve
- Supports distributed processing via torchrun
- `t_list` must match the KD training config exactly

---

## 11. Training Pipeline Stages

### Full pipeline:

```
Step 1: Precompute data (offline)
    - VAE encode all training videos → latent .pt files
    - Extract audio features → .pt files
    - Extract text embeddings → .pt files
    - Generate spatial masks

Step 2: Generate ODE trajectories
    $ torchrun --nproc_per_node=4 scripts/generate_your_ode_pairs.py \
        --data_list /path/to/video_list.txt \
        --output_dir /path/to/data/ \
        --guidance_scale 4.5

Step 3: KD training (Stage 1)
    $ torchrun --nproc_per_node=4 train.py \
        --config fastgen/configs/experiments/YourModel/config_kd.py

Step 4: Self-Forcing training (Stage 2)
    # Uncomment pretrained_student_net_path in config_sf.py pointing to KD output
    $ torchrun --nproc_per_node=4 train.py \
        --config fastgen/configs/experiments/YourModel/config_sf.py
```

---

## 12. Critical Gotchas (Lessons Learned)

### Config gotchas

1. **Dataloader MUST be explicitly assigned.** The default is `CIFAR10_Loader_Config`.
   If you only set `config.dataloader_train.batch_size = 1`, you're still using CIFAR-10.

2. **Fake score defaults to teacher architecture.** `DMD2Model.build_model()` creates
   fake_score from `self.teacher_config`. If teacher is 14B and you want 1.3B fake_score,
   you must override `build_model()` and add a `config.model.fake_score` field.

3. **Discriminator inner_dim must match TEACHER, not student.** Features are extracted
   from the teacher. If teacher is 14B (dim=5120), use `Discriminator_Wan_14B_Config`
   (inner_dim=1280), not `Discriminator_Wan_1_3B_Config` (inner_dim=384).

4. **shift parameter must match your teacher's training distribution.** FastGen defaults
   to shift=5.0. If your teacher was trained with shift=3.0, set it explicitly.

5. **KD method config must inherit from `config_kd_causal`**, not `BaseConfig`. This
   provides the `context_noise` field and proper CausalKDModel defaults.

### Network gotchas

6. **`is_ar` default MUST be `False`.** CausalKD calls `gen_data_from_net()` without
   passing `is_ar`. Default `False` routes to full-sequence mode (correct). Default
   `True` would route to AR mode (wrong for KD — no chunk-by-chunk, no KV cache during KD).

7. **Per-frame timestep support in `_forward_full_sequence`.** CausalKD uses
   `sample_t_inhom` which returns `[B, num_frames]`. Your full-sequence forward must
   handle this by embedding each frame's timestep independently (flatten, embed, reshape).

8. **Chunk-wise causal mask, NOT per-frame.** The FlexAttention mask must match
   `sample_t_inhom` semantics: frames in the same chunk share a timestep and attend
   bidirectionally. The original CausalWan uses `_prepare_blockwise_causal_attn_mask`
   which groups frames into chunks.

9. **`sinusoidal_embedding_1d` returns float32.** Cast to model dtype before passing to
   time_embedding MLP. Otherwise you get dtype mismatch in bf16 training.

10. **`convert_model_output` needs per-frame t handling.** When t is `[B, num_frames]`:
    ```python
    t_converted = t[:, None, :, None, None] if t.ndim == 2 else t
    ```

### Data gotchas

11. **KD dataset must provide `data["path"]` key.** This is the ODE trajectory tensor.
    Use a flag like `load_ode_path=True` to conditionally load it.

12. **Condition dict keys must be identical** between SF's `_prepare_training_data()`,
    KD's `_build_condition()`, and the ODE generation script. Any mismatch means the
    networks see different conditioning, silently corrupting training.

---

## 13. File Inventory

### Files created (OmniAvatar adaptation): 15 files, ~5000 lines

| File | Lines | Purpose |
|------|-------|---------|
| `networks/OmniAvatar/__init__.py` | 4 | Package exports |
| `networks/OmniAvatar/wan_model.py` | 415 | Standalone DiT (from OmniAvatar) |
| `networks/OmniAvatar/audio_pack.py` | 39 | Audio conditioning module |
| `networks/OmniAvatar/network.py` | 739 | Bidirectional FastGenNetwork wrapper |
| `networks/OmniAvatar/network_causal.py` | 1750 | Causal CausalFastGenNetwork wrapper |
| `methods/omniavatar_self_forcing.py` | 105 | SF method (overrides _prepare_training_data + build_model) |
| `methods/omniavatar_kd.py` | 128 | KD method (overrides _build_condition + single_train_step) |
| `datasets/omniavatar_dataloader.py` | 221 | Dataset + DataLoader factory |
| `configs/methods/config_omniavatar_sf.py` | 61 | SF method config (adds fake_score field) |
| `configs/methods/config_omniavatar_kd.py` | 52 | KD method config (inherits CausalKDConfig) |
| `configs/experiments/OmniAvatar/__init__.py` | 0 | Package |
| `configs/experiments/OmniAvatar/config_sf.py` | 143 | SF experiment (3 networks + hyperparams) |
| `configs/experiments/OmniAvatar/config_kd.py` | 84 | KD experiment (1 network + hyperparams) |
| `scripts/generate_omniavatar_ode_pairs.py` | 584 | ODE trajectory generation |
| `scripts/create_verification_samples.py` | 422 | Dev/testing utility |

### Files NOT modified from original FastGen: ~140+

All core training logic, loss functions, callbacks, utilities, and noise schedules
remain byte-identical. The adaptation is purely additive.

### Original FastGen reference configs (for comparison)

| Original | OmniAvatar Equivalent | Key Differences |
|----------|----------------------|-----------------|
| `WanT2V/config_sf.py` | `OmniAvatar/config_sf.py` | Different networks, fake_score config, shift=3.0, custom dataloader |
| `WanT2V/config_kd.py` | `OmniAvatar/config_kd.py` | Uses CausalKD (not KD), custom dataloader with ODE paths |
| `config_self_forcing.py` | `config_omniavatar_sf.py` | Adds OmniAvatarModelConfig with fake_score field |
| `config_kd.py` | `config_omniavatar_kd.py` | Inherits CausalKDConfig instead of BaseConfig |
| `Wan/network.py` | `OmniAvatar/network.py` | Custom DiT, weight loading, V2V conditioning |
| `Wan/network_causal.py` | `OmniAvatar/network_causal.py` | Custom causal DiT, per-frame timestep, chunk mask |
