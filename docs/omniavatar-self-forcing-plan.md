# Self-Forcing Distillation Plan: OmniAvatar 14B → 1.3B

**Date**: 2026-03-15 (updated with user feedback)
**Goal**: Distill 14B OmniAvatar V2V lip sync teacher into 1.3B student using FastGen Self-Forcing
**Status**: Plan revised, implementation not started

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
3. [Architecture Deep Dive](#3-architecture-deep-dive)
4. [Design Decisions](#4-design-decisions)
5. [Implementation Plan](#5-implementation-plan)
6. [Data Flow](#6-data-flow)
7. [Memory Budget](#7-memory-budget)
8. [Risk Register](#8-risk-register)
9. [Validation Criteria](#9-validation-criteria)

---

## 1. Executive Summary

**What**: Adapt FastGen's Self-Forcing framework to distill OmniAvatar's 14B V2V audio-driven lip sync model into a 1.3B student for faster inference.

**Why**: The 14B model produces high-quality lip sync but is too slow for production. Self-Forcing enables few-step autoregressive generation from the distilled 1.3B student.

**How**: Two-stage training:
- **Stage 1 (ODE/KD init)**: Pre-train causal 1.3B student on ODE trajectories extracted from the bidirectional 14B teacher
- **Stage 2 (Self-Forcing)**: Fine-tune with gradient-enabled autoregressive rollout + VSD loss + GAN discriminator

**Key Challenge**: OmniAvatar uses a CUSTOM DiT implementation (not diffusers' WanTransformer3DModel) with audio conditioning, custom patch embeddings (49ch/65ch), and a global `args` singleton. FastGen uses diffusers' WanTransformer3DModel. We must bridge these fundamentally different implementations.

---

## 2. Current State Assessment

### What EXISTS in FastGen (we build on this):
```
fastgen/
├── methods/distribution_matching/
│   ├── self_forcing.py          # SelfForcingModel (inherits CausVid → DMD2)
│   ├── dmd2.py                  # DMD2Model (VSD + fake_score + discriminator)
│   ├── causvid.py               # CausVidModel (chunked autoregressive)
│   └── ...
├── networks/
│   ├── network.py               # FastGenNetwork / CausalFastGenNetwork (abstract)
│   ├── Wan/network.py           # Wan(FastGenNetwork) - uses diffusers WanTransformer3DModel
│   ├── Wan/network_causal.py    # CausalWan(CausalFastGenNetwork)
│   ├── VaceWan/network.py       # VACEWan - V2V with depth conditioning
│   ├── VaceWan/network_causal.py # CausalVACEWan
│   ├── noise_schedule.py        # Noise schedules including "rf" (rectified flow)
│   └── discriminators.py        # Discriminator architectures
├── configs/
│   ├── net.py                   # Network configs (LazyCall pattern)
│   ├── experiments/WanV2V/config_sf.py  # VACE V2V Self-Forcing config (reference)
│   └── experiments/WanT2V/config_sf.py  # T2V Self-Forcing config (reference)
├── datasets/                    # WebDataset-based dataloaders
├── trainer.py                   # Main training loop
└── train.py                     # Entry point
```

### What EXISTS in OmniAvatar (we port from this):
```
OmniAvatar/
├── OmniAvatar/models/
│   ├── wan_video_dit.py         # CUSTOM WanModel DiT with audio conditioning
│   ├── audio_pack.py            # AudioPack module (10752→32 dim)
│   ├── wav2vec.py               # Wav2Vec2 audio encoder
│   ├── wan_video_vae.py         # Video VAE
│   └── model_manager.py         # Weight loading + smart_load_weights
├── scripts/
│   ├── train_v2v.py             # V2V training (49ch/65ch, aux losses)
│   └── inference_v2v.py         # V2V inference
└── configs/                     # YAML configs
```

### Reference causal implementation (adapt patterns from this):
```
Self-Forcing-OmniAvatar/Self-Forcing/wan/modules/
├── causal_model.py              # CausalWanModel with FlexAttention + KV cache
├── audio_mixin.py               # OmniAvatarAudioMixin (shared audio logic)
└── model.py                     # Bidirectional WanModel (also uses audio_mixin)
```
Key patterns: `OmniAvatarAudioMixin` for shared audio processing, `causal_rope_apply` for
chunked RoPE, audio slicing at line 988-1005, `CausalWanSelfAttention` with KV cache.

### What DOES NOT exist (we must build):
```
fastgen/
├── networks/OmniAvatar/         # ENTIRE DIRECTORY - new
│   ├── __init__.py
│   ├── wan_model.py             # OmniAvatar DiT (adapted, no args singleton)
│   ├── audio_pack.py            # AudioPack (ported from OmniAvatar)
│   ├── network.py               # OmniAvatarWan(FastGenNetwork) - bidirectional wrapper
│   └── network_causal.py        # CausalOmniAvatarWan(CausalFastGenNetwork)
├── methods/
│   ├── omniavatar_self_forcing.py  # OmniAvatarSelfForcing(SelfForcingModel)
│   └── omniavatar_kd.py         # OmniAvatarKD(CausalKDModel)
├── datasets/
│   └── omniavatar_dataloader.py # Custom dataset for precomputed .pt files
├── configs/
│   ├── experiments/OmniAvatar/
│   │   ├── config_sf.py         # Self-Forcing experiment config
│   │   └── config_kd.py         # KD experiment config
│   └── methods/
│       ├── config_omniavatar_sf.py
│       └── config_omniavatar_kd.py
└── scripts/
    ├── generate_omniavatar_ode_pairs.py  # ODE trajectory extraction
    └── inference/omniavatar_inference.py  # Distilled model inference
```

---

## 3. Architecture Deep Dive

### 3.1 Self-Forcing in FastGen

**Class hierarchy**: `FastGenModel → DMD2Model → CausVidModel → SelfForcingModel`

**Stage 2 training loop** (alternating updates):
1. **Student update** (every `student_update_freq` iterations):
   - Roll out student autoregressively with gradient at stochastically-chosen exit steps
   - Perturb generated data: `x_t = forward_process(gen_data, eps, t)`
   - Get teacher prediction: `teacher_x0 = teacher(x_t, t, condition)`
   - Get fake_score prediction: `fake_x0 = fake_score(x_t, t, condition)`
   - VSD loss: `loss = VSD(gen_data, teacher_x0, fake_x0)`
   - Optional GAN loss from discriminator
   - Backprop through student (teacher/fake_score/discriminator frozen)

2. **Fake score + discriminator update** (other iterations):
   - Roll out student (no grad)
   - Train fake_score with DSM loss on generated data
   - Train discriminator on real vs fake features

**Key Self-Forcing innovation**: `rollout_with_gradient()` maintains autograd graph at exit steps, enabling direct optimization of autoregressive generation.

### 3.2 OmniAvatar DiT Architecture

**Forward signature** (from `wan_video_dit.py`):
```python
def forward(x, timestep, context, clip_feature=None, y=None,
            use_gradient_checkpointing=False, audio_emb=None, ...)
```

**Input assembly** (V2V 65ch):
- `x`: [B, 16, 21, 64, 64] — noisy latents
- `y`: [B, 49, 21, 64, 64] — ref_repeated(16) + mask(1) + masked_video(16) + ref_sequence(16)
- Concatenated: `torch.cat([x, y], dim=1)` → [B, 65, 21, 64, 64]
- Patch embedding: Conv3d(65, dim, [1,2,2])

**Audio pipeline**:
- Wav2Vec2: raw 16kHz → [B, 81, 10752] (13 layers × 768-dim)
- AudioPack: [B, 10752, T, 1, 1] → [B, 32, T_lat, 1, 1] with patch_size [4,1,1]
- Per-layer Linear projections: 32 → dim (19 for 14B, 14 for 1.3B)
- Additive residuals at layers 2 through num_layers//2

**Key difference from diffusers Wan**: OmniAvatar's WanModel is CUSTOM — different class, different forward signature, different attention implementation, different positional embeddings. Cannot use diffusers' WanTransformer3DModel.

### 3.3 Model Specifications

| | 14B Teacher | 1.3B Student |
|---|---|---|
| dim | 5120 | 1536 |
| num_layers | 40 | 30 |
| num_heads | 40 | 12 |
| ffn_dim | 13824 | 8960 |
| audio_cond_projs | 19 (layers 2-20) | 14 (layers 2-15) |
| in_dim (V2V + refseq) | 65 | 65 |
| Trainable (LoRA r=128) | ~335M | ~95M |

---

## 4. Design Decisions

### D1: Use OmniAvatar's custom WanModel, NOT diffusers' WanTransformer3DModel
**Rationale**: We need exact forward-pass compatibility with trained OmniAvatar checkpoints. The audio conditioning, custom attention, and patch embedding are deeply integrated into the custom WanModel. Trying to add these to diffusers' model would be error-prone and harder to verify.
**Implementation**: Port OmniAvatar's `wan_video_dit.py` into `fastgen/networks/OmniAvatar/wan_model.py`, removing the `args` singleton dependency and making all config explicit via constructor parameters.

### D2: Fake score is 1.3B BIDIRECTIONAL (not causal, not 14B)
**Rationale**: Memory constraint — 14B fake_score would add ~28GB VRAM. The fake_score learns the student's output distribution, so matching student architecture is sufficient. The fake_score does NOT need to be causal — it performs standard bidirectional forward passes only (no autoregressive rollout, no KV cache). It's architecturally the same wrapper as the teacher but using 1.3B weights.
**Implementation**: Same `OmniAvatarWan(FastGenNetwork)` wrapper as teacher, just with `model_size="1.3B"`.
**Trade-off**: May converge slower but saves critical VRAM.

### D3: Student standardized to 65ch (with ref_sequence) to match teacher
**Rationale**: Teacher was trained with `--use_ref_sequence`. Student must match conditioning to ensure training distribution consistency.
**Prerequisite**: 1.3B student needs refseq training before distillation (already planned).

### D4: Only the student is causal; teacher and fake_score are bidirectional
**Rationale**: Self-Forcing requires the student to generate autoregressively (causal with KV cache). The teacher and fake_score are only used for scoring (single forward pass, no autoregression needed). Both stay bidirectional. Reference causal implementation: `/home/work/.local/Self-Forcing-OmniAvatar/Self-Forcing/wan/modules/causal_model.py` with `OmniAvatarAudioMixin` from `audio_mixin.py`.

### D5: Audio conditioning is identical everywhere — no exceptions
**Rationale**: User-confirmed. The OmniAvatar audio pipeline (Wav2Vec2 → AudioPack → per-layer injection) is used identically in all contexts: teacher forward, student forward (causal and bidirectional), fake_score forward, ODE trajectory extraction, and inference. No modified audio paths. Use the `OmniAvatarAudioMixin` pattern from the reference implementation to ensure consistency.

### D6: Custom dataset loader (not WebDataset)
**Rationale**: OmniAvatar's precomputed data is directory-based (.pt files per sample). Converting to WebDataset TAR format adds complexity with no benefit. Create a simple PyTorch Dataset that reads the existing structure. FastGen's WebDataset advantages (S3, streaming, large-scale sharding) aren't relevant for local precomputed data.

### D7: All three networks operate in V2V mode (65ch)
**Rationale**: The VSD loss requires teacher and fake_score to score the student's output, which includes V2V conditioning. Using I2V mode (33ch) would create a distribution mismatch — the teacher would ignore masked_video and ref_sequence information that the student was conditioned on.
**Implementation**: All three networks (student, teacher, fake_score) use the same `_build_y()` logic with the full V2V condition dict.

### D8: Use GPU 3 for verification testing
**Rationale**: GPUs 0-2 are in use for other training. GPU 3 has available memory. All verification tests (forward pass comparison, weight loading, audio alignment) should run on `CUDA_VISIBLE_DEVICES=3` with real data samples at small scale to verify numerical correctness.

### D9: Reference causal implementation — audio slicing only
**Source**: `/home/work/.local/Self-Forcing-OmniAvatar/Self-Forcing/wan/modules/causal_model.py`
**What to reuse**: Only the audio slicing logic (lines 988-1005) is confirmed correct. The mixin
pattern (`OmniAvatarAudioMixin`) is NOT required — the teacher (from OmniAvatar repo) doesn't use it.
We can implement the causal version without the mixin as long as audio processing is identical.
**Key patterns for reference**:
- `causal_rope_apply()` with `start_frame` offset for RoPE in causal chunks
- Audio slicing: process full audio through AudioPack → [B, layers, 21, 1, 1, dim] → slice per-chunk
- `CausalWanSelfAttention` with KV cache and FlexAttention block mask
- `CausalWanModel.forward()` with `current_start`, `kv_cache`, `cache_start` parameters

### D10: Teacher checkpoint
**Path**: `/home/work/output_omniavatar_v2v_maskall_refseq_new_data_loss_weights_mouth_weights/step-1500.pt`
**Size**: ~1.2GB (trainable params only: LoRA + audio modules + patch_embedding)
**Format**: Slim checkpoint (not full Accelerate checkpoint)

### D11: 1.3B refseq training is pending
The 1.3B student has NOT been trained with `--use_ref_sequence` (65ch) yet. Our implementation
should assume it will be done and accept any 65ch checkpoint. We can swap in the trained checkpoint
once available. For testing, we can use the existing 49ch 1.3B checkpoint with patch_embedding
expansion (49→65ch, new channels zero-initialized).

### D12: ODE trajectory generation needs thorough verification
There is NO existing OmniAvatar-specific ODE generation code. The FastGen repo has generic KD/ODE
code that was created from scratch (not official). We must:
1. Thoroughly verify the existing FastGen ODE generation code for correctness
2. Then adapt it for OmniAvatar-style processing (audio conditioning, V2V inputs)
3. Test with real data samples on GPU 3

---

## 5. Implementation Plan

### Phase 0: Prerequisites (before any FastGen changes)

#### 0A: Train 1.3B student with refseq (65ch) [PENDING — not blocking implementation]
- **What**: Continue 1.3B V2V training with `--use_ref_sequence` flag using OmniAvatar's native `train_v2v.py`
- **From**: Current 1.3B maskall checkpoint (49ch)
- **To**: 1.3B refseq checkpoint (65ch, patch_embedding expanded 49→65ch)
- **Script**: Based on `scripts/train_v2v_1.3B_all_masked.sh` + `--use_ref_sequence`
- **Validation**: Run inference_v2v.py on validation sets, verify acceptable lip sync quality
- **Status**: NOT STARTED. Implementation proceeds with 49ch checkpoint + expansion for testing.
  Trained 65ch checkpoint will be swapped in when available.

#### 0B: Precompute ref_sequence latents
- **What**: Ensure every training sample has `ref_latents.pt` ([16, 21, 64, 64])
- **Script**: `scripts/preprocess_v2v_integrated.py` (already exists)
- **Validation**: Spot-check shapes: `torch.load("ref_latents.pt").shape == [16, 21, 64, 64]`

#### 0C: Verify teacher checkpoint [DONE]
- **Path**: `/home/work/output_omniavatar_v2v_maskall_refseq_new_data_loss_weights_mouth_weights/step-1500.pt`
- **Size**: 1.2GB (slim checkpoint: LoRA + audio modules + patch_embedding)
- **Config**: `config.json` in same directory has training config (in_dim=65, use_ref_sequence=True)
- **Validation**: Run inference_v2v.py with this checkpoint on validation data (if not already done)

---

### Phase 1: Network Wrappers [COMPLEXITY: HIGH, ~3-5 days]

This is the largest and most critical phase — creating FastGen-compatible wrappers around OmniAvatar's custom DiT.

#### 1A: Port OmniAvatar DiT to standalone module
**File**: `fastgen/networks/OmniAvatar/wan_model.py`
**Source**: `/home/work/.local/OmniAvatar/OmniAvatar/models/wan_video_dit.py`

**Changes from original**:
1. Remove ALL references to global `args` singleton
2. Make all config explicit via constructor: `dim, in_dim, ffn_dim, out_dim, text_dim, freq_dim, patch_size, num_heads, num_layers, audio_hidden_size, use_audio`
3. Remove TeaCache logic (not needed for training)
4. Keep: patch_embedding, text_embedding, time_embedding, transformer blocks, audio modules, RoPE, output head
5. Keep exact same forward pass logic (concatenate x+y, patchify, audio injection at layers 2 to num_layers//2)

**Key method**: `forward(x, timestep, context, y=None, audio_emb=None, use_gradient_checkpointing=False)`

**Validation**: Load OmniAvatar checkpoint, run forward pass, verify output matches OmniAvatar inference output numerically (< 1e-5 difference).

#### 1B: Port AudioPack module
**File**: `fastgen/networks/OmniAvatar/audio_pack.py`
**Source**: `/home/work/.local/OmniAvatar/OmniAvatar/models/audio_pack.py`
**Changes**: None significant — it's already self-contained.

#### 1C: Create bidirectional wrapper (teacher / fake_score)
**File**: `fastgen/networks/OmniAvatar/network.py`
**Class**: `OmniAvatarWan(FastGenNetwork)`

**Must implement**:
```python
class OmniAvatarWan(FastGenNetwork):
    def __init__(self,
                 model_size="14B",        # or "1.3B"
                 in_dim=65,               # 33 (I2V), 49 (V2V), 65 (V2V+refseq)
                 mode="v2v",              # "i2v" or "v2v"
                 use_audio=True,
                 base_model_paths=None,   # Wan 2.1 safetensor paths
                 omniavatar_ckpt_path=None,  # OmniAvatar trained weights
                 merge_lora=True,         # Merge LoRA into base for inference
                 net_pred_type="flow",
                 schedule_type="rf",
                 **kwargs):
        ...

    def forward(self, x_t, t, condition=None, r=None,
                return_features_early=False, feature_indices=None,
                return_logvar=False, fwd_pred_type=None, **fwd_kwargs):
        # 1. Unpack condition dict: text_embeds, audio_emb, ref_latent, mask, masked_video, ref_sequence
        # 2. Build y tensor (ref_repeated + mask_ch + masked_video [+ ref_sequence])
        # 3. Call self.model.forward(x_t, t, context, y=y, audio_emb=audio_emb)
        # 4. Convert prediction to requested type (flow → x0 if needed)
        # 5. If return_features_early: extract intermediate features for discriminator
        ...

    def _build_y(self, condition, num_frames):
        # Assemble V2V conditioning tensor
        ...

    def _load_weights(self):
        # 1. Create WanModel with correct dimensions
        # 2. Load Wan 2.1 base weights via smart_load_weights
        # 3. Load OmniAvatar LoRA + audio modules
        # 4. Optionally merge LoRA into base weights
        # 5. Handle patch_embedding expansion (33→49→65ch)
        ...
```

**Critical detail**: The `condition` dict is the bridge between FastGen's method layer and OmniAvatar's model. It carries:
```python
condition = {
    "text_embeds": [B, 512, 4096],     # T5 text encoding
    "audio_emb": [B, 81, 10752],        # Wav2Vec2 features
    "ref_latent": [B, 16, 1, 64, 64],   # Reference frame latent
    "mask": [64, 64],                    # Spatial mask (1=keep, 0=generate)
    "masked_video": [B, 16, 21, 64, 64], # Mouth-masked video latents
    "ref_sequence": [B, 16, 21, 64, 64], # Reference sequence latents (65ch only)
}
```

#### 1D: Create causal wrapper (student)
**File**: `fastgen/networks/OmniAvatar/network_causal.py`
**Class**: `CausalOmniAvatarWan(CausalFastGenNetwork)`
**Reference**: `/home/work/.local/Self-Forcing-OmniAvatar/Self-Forcing/wan/modules/causal_model.py`

The causal model wraps the same DiT architecture but adds:
- Causal self-attention with FlexAttention block masks
- KV cache management for chunk-by-chunk generation
- RoPE with `start_frame` offset (`causal_rope_apply`)
- Audio slicing per chunk (AFTER full AudioPack processing)

**Must implement** (in addition to 1C):
```python
class CausalOmniAvatarWan(CausalFastGenNetwork):
    def __init__(self, chunk_size=3, total_num_frames=21, **kwargs):
        ...

    def forward(self, x_t, t, condition=None, ...,
                cur_start_frame=0, store_kv=False, is_ar=True, **kwargs):
        # 1. Process FULL audio through AudioPack → [B, layers, 21, 1, 1, dim]
        # 2. Slice audio for current chunk:
        #    frame_seq_length = h * w  (spatial tokens per frame)
        #    current_frame_start = current_start // frame_seq_length
        #    processed_audio[:, :, current_frame_start:current_frame_start+chunk_frames]
        # 3. Build y for current chunk (slice ref_repeated, mask, masked_video, ref_sequence)
        # 4. Forward through model with KV cache and causal attention mask
        # 5. Handle store_kv flag for cache updates
        ...

    def clear_caches(self):
        # Clear all KV caches in transformer blocks
        ...
```

**Audio slicing approach** (from reference `causal_model.py:988-1005`):
- Process full audio ONCE through `_process_audio_embeddings()` → [B, layers, 21, 1, 1, dim]
- For each chunk: `current_frame_start = current_start // frame_seq_length`
- Slice: `processed_audio[:, :, current_frame_start:current_frame_end, :, :, :]`
- This means audio is in LATENT frame space (21 frames) after AudioPack, NOT video frame space
- The AudioPack's [4,1,1] patch size handles the 81→21 temporal compression

**Validation** (run on GPU 3 with real data):
1. Single-chunk forward matches bidirectional forward for that chunk (numerical comparison)
2. Full autoregressive generation (7 chunks × 3 frames) produces coherent video
3. Audio alignment: mouth movement correlates with audio across chunk boundaries
4. KV cache shapes are correct across all chunks

#### 1E: Weight loading infrastructure
**Key function**: `_smart_load_weights(model, checkpoint_path)`

Must handle:
1. Base Wan 2.1 weights (safetensors) → model base parameters
2. OmniAvatar checkpoint (.pt) → LoRA weights + audio modules + patch_embedding
3. Patch embedding expansion: 16ch (base) → 33ch (I2V OmniAvatar) → 49ch (V2V) → 65ch (V2V+refseq)
4. LoRA key mapping: `lora_A.weight` → `lora_A.default.weight` (PEFT convention)
5. Optionally merge LoRA into base weights for inference (teacher/fake_score)

---

### Phase 2: Dataset Adapter [COMPLEXITY: MEDIUM, ~1-2 days]

#### 2A: Create OmniAvatar dataset
**File**: `fastgen/datasets/omniavatar_dataloader.py`

```python
class OmniAvatarDataset(torch.utils.data.Dataset):
    def __init__(self, data_list_path, latentsync_mask_path,
                 use_ref_sequence=True, mask_all_frames=True,
                 neg_text_emb_path=None):
        # Read video_square_path.txt
        # Load spatial mask from mask.png
        # Load negative text embedding (for CFG)
        ...

    def __getitem__(self, idx):
        video_dir = self.dirs[idx]
        return {
            "real": load("vae_latents_mask_all.pt")["input_latents"],  # [16, 21, 64, 64]
            "masked_video": load("vae_latents_mask_all.pt")["masked_latents"],
            "audio_emb": load("audio_emb_omniavatar.pt")["audio_emb"][:81],  # [81, 10752]
            "text_embeds": load("text_emb.pt"),  # [1, 512, 4096]
            "ref_sequence": load("ref_latents.pt"),  # [16, 21, 64, 64]
            "neg_text_embeds": self.neg_text_emb,  # preloaded
        }
```

**Validation**: Load 10 samples, verify all shapes match expected dimensions.

#### 2B: Create dataloader config
Adapt `VideoLoaderConfig` pattern for our dataset class.

---

### Phase 3: Method Subclasses [COMPLEXITY: MEDIUM-HIGH, ~2-3 days]

#### 3A: OmniAvatar Self-Forcing model
**File**: `fastgen/methods/omniavatar_self_forcing.py`
**Class**: `OmniAvatarSelfForcingModel(SelfForcingModel)`

**Key overrides**:
```python
class OmniAvatarSelfForcingModel(SelfForcingModel):
    def _prepare_training_data(self, data):
        """Build condition and neg_condition dicts from dataset output."""
        ref_latent = data["real"][:, :, :1]  # First frame as reference
        mask = self.latentsync_mask  # Preloaded [64, 64]

        condition = {
            "text_embeds": data["text_embeds"],
            "audio_emb": data["audio_emb"],
            "ref_latent": ref_latent,
            "mask": mask,
            "masked_video": data["masked_video"],
            "ref_sequence": data["ref_sequence"],
        }

        neg_condition = {
            "text_embeds": data["neg_text_embeds"],
            "audio_emb": torch.zeros_like(data["audio_emb"]),
            "ref_latent": ref_latent,
            "mask": mask,
            "masked_video": data["masked_video"],
            "ref_sequence": data["ref_sequence"],
        }

        return data["real"], condition, neg_condition
```

**Also override** (if needed):
- `build_model()`: Handle OmniAvatar-specific initialization
- `_generate_noise_and_time()`: Ensure audio_emb is sliced correctly for inhomogeneous timesteps

#### 3B: OmniAvatar KD model (Stage 1)
**File**: `fastgen/methods/omniavatar_kd.py`
**Class**: `OmniAvatarKDModel(CausalKDModel)`

Similar condition dict assembly. Must also handle ODE pair loading.

---

### Phase 4: Config System [COMPLEXITY: SIMPLE, ~1 day]

#### 4A: Add OmniAvatar network configs to `configs/net.py`
```python
OmniAvatar_V2V_14B_Config: DictConfig = L(OmniAvatarWan)(
    model_size="14B", in_dim=65, mode="v2v", use_audio=True,
    base_model_paths="pretrained_models/Wan2.1-T2V-14B/...",
    omniavatar_ckpt_path="path/to/14B_v2v_refseq.pt",
    merge_lora=True,
    net_pred_type="flow", schedule_type="rf",
)

OmniAvatar_V2V_1_3B_Config: DictConfig = L(OmniAvatarWan)(
    model_size="1.3B", in_dim=65, mode="v2v", use_audio=True,
    base_model_paths="pretrained_models/Wan2.1-T2V-1.3B/...",
    omniavatar_ckpt_path="path/to/1.3B_v2v_refseq.pt",
    merge_lora=True,
    net_pred_type="flow", schedule_type="rf",
)

CausalOmniAvatar_V2V_1_3B_Config: DictConfig = L(CausalOmniAvatarWan)(
    model_size="1.3B", in_dim=65, mode="v2v", use_audio=True,
    chunk_size=3, total_num_frames=21,
    base_model_paths="pretrained_models/Wan2.1-T2V-1.3B/...",
    omniavatar_ckpt_path="path/to/1.3B_v2v_refseq.pt",
    net_pred_type="flow", schedule_type="rf",
)
```

#### 4B: Create experiment configs
**Files**:
- `fastgen/configs/experiments/OmniAvatar/config_kd.py` (Stage 1)
- `fastgen/configs/experiments/OmniAvatar/config_sf.py` (Stage 2)

Reference: `fastgen/configs/experiments/WanV2V/config_sf.py`

Key parameters:
```python
# Stage 2 (Self-Forcing)
config.model.net = CausalOmniAvatar_V2V_1_3B_Config
config.model.teacher = OmniAvatar_V2V_14B_Config
config.model.fake_score_net = OmniAvatar_V2V_1_3B_Config  # Same arch as student, bidirectional
config.model.input_shape = [16, 21, 64, 64]  # 512x512 @ 81 frames
config.model.guidance_scale = 4.5
config.model.student_sample_steps = 4
config.model.student_update_freq = 5
config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]
config.model.enable_gradient_in_rollout = True
config.model.gan_loss_weight_gen = 0.003
config.dataloader_train.batch_size = 1
```

---

### Phase 5: ODE Trajectory Extraction & Stage 1 [COMPLEXITY: HIGH, ~3-5 days code + training time]

#### 5A: VERIFY existing FastGen ODE/KD code [DONE — verified correct]
**Location**: `scripts/generate_ode_trajectories.py` (534 lines)
**KD Method**: `fastgen/methods/knowledge_distillation/KD.py` (KDModel + CausalKDModel)

**Verified correct**:
- Noise schedule: uses teacher's scheduler, linear t_list, proper alpha/sigma
- Deterministic ODE: no stochastic noise, pure algebraic epsilon computation
- Trajectory storage: WebDataset format, `path.pth` [num_steps-1, C, T, H, W] + `latent.pth` [C, T, H, W]
- Teacher forward: standard CFG formula, proper x0 prediction mode
- Timestep selection: nearest-neighbor matching to target t_list [0.999, 0.937, 0.833, 0.624, 0.0]

**Minor issues to address during adaptation**:
1. Consider bfloat16 instead of float16 storage for better precision
2. Document the 50-step ODE → 4-step path subsampling relationship clearly
3. For OmniAvatar: teacher is BIDIRECTIONAL (processes all 21 frames at once) — simpler than
   the agent's suggestion of causal chunked processing

#### 5B: Adapt ODE pair generation for OmniAvatar
**File**: `scripts/generate_omniavatar_ode_pairs.py`
**Based on**: `scripts/generate_ode_trajectories.py` (verified correct)

**Key differences from standard T2V ODE generation**:
1. **Teacher is bidirectional 14B OmniAvatar** (not diffusers WanTransformer3DModel)
   - Uses our `OmniAvatarWan` wrapper with `forward(x_t, t, condition)` interface
   - Processes all 21 latent frames at once (no chunking needed for ODE)
2. **Condition dict includes audio + V2V inputs**:
   ```python
   condition = {
       "text_embeds": text_emb,          # [1, 512, 4096]
       "audio_emb": audio_emb,            # [1, 81, 10752]
       "ref_latent": ref_latent,           # [1, 16, 1, 64, 64]
       "mask": mask,                       # [64, 64]
       "masked_video": masked_latents,     # [1, 16, 21, 64, 64]
       "ref_sequence": ref_seq_latents,    # [1, 16, 21, 64, 64]
   }
   ```
3. **Data source**: Reads from OmniAvatar's precomputed .pt files (not WebDataset)
4. **Storage**: Save `ode_path.pt` alongside existing precomputed files in each sample dir
   - `ode_path.pt`: [4, 16, 21, 64, 64] (noisy states at 4 timesteps, bfloat16)
   - Clean target is already available as `vae_latents_mask_all.pt["input_latents"]`

**Step-by-step for each sample**:
1. Load precomputed: vae_latents, audio_emb, text_emb, ref_latents, mask
2. Build condition dict with all V2V inputs + audio
3. Start from noise: `x_T ~ N(0, I)` → scale by sigma(t_max)
4. Run 50-step deterministic ODE with teacher (bidirectional, full sequence):
   ```python
   for step in range(50):
       x0_cond = teacher(x_t, t_cur, condition, fwd_pred_type="x0")
       x0_uncond = teacher(x_t, t_cur, neg_condition, fwd_pred_type="x0")
       x0_pred = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
       eps = noise_scheduler.x0_to_eps(x_t, x0_pred, t_cur)
       x_t = noise_scheduler.forward_process(x0_pred, eps, t_next)
   ```
5. Subsample trajectory at t_list=[0.999, 0.937, 0.833, 0.624] (4 noisy states)
6. Save as `ode_path.pt` in the sample directory

**Critical**: Audio MUST be included in teacher's forward pass during ODE extraction — lip sync
depends on it. The ODE trajectories encode audio-visual correspondence.

**Validation** (on GPU 3 with real data):
- Generate ODE pairs for 5-10 samples
- Decode final x_0 via VAE and visually inspect: does the mouth move with the audio?
- Compare trajectory shapes: [4, 16, 21, 64, 64]
- Verify determinism: same noise seed → same trajectory

#### 5C: Run KD pre-training (Stage 1)
**Command**:
```bash
torchrun --nproc_per_node=2 train.py \
  --config=fastgen/configs/experiments/OmniAvatar/config_kd.py
```

**Duration**: ~1-2 days on 2x H200
**Output**: `ode_init.pt` checkpoint for Stage 2

---

### Phase 6: Self-Forcing Training (Stage 2) [COMPLEXITY: HIGH, ~3-5 days training time]

#### 6A: Memory verification
Before full training, run single iteration with memory profiling:
```bash
torchrun --nproc_per_node=1 train.py \
  --config=fastgen/configs/experiments/OmniAvatar/config_sf.py \
  - trainer.max_iter=2
```

Verify peak VRAM < 140GB on H200.

#### 6B: Full Self-Forcing training
```bash
torchrun --nproc_per_node=2 train.py \
  --config=fastgen/configs/experiments/OmniAvatar/config_sf.py \
  - model.pretrained_student_net_path=path/to/ode_init.pt
```

**Key hyperparameters to tune**:
- `guidance_scale`: 4.0-5.0 (teacher CFG for VSD target)
- `student_update_freq`: 5 (1:4 student:fake_score update ratio)
- `context_noise`: 0.0-0.1 (noise on denoised frames before KV cache update)
- `start_gradient_frame`: 0 (increase if OOM during rollout)
- `learning_rate`: 5e-6 for all optimizers (student, fake_score, discriminator)

**Monitoring**:
- `vsd_loss` should decrease steadily
- `fake_score_loss` should be stable
- `discriminator_loss` should oscillate around a stable value
- Wandb video samples every 500 iterations

---

### Phase 7: Inference & Evaluation [COMPLEXITY: MEDIUM, ~1-2 days]

#### 7A: Create inference script
**File**: `scripts/inference/omniavatar_inference.py`

Autoregressive sampling with distilled 1.3B causal student:
1. Load CausalOmniAvatarWan from checkpoint
2. Build condition dict from input video + audio
3. Run autoregressive generation (7 chunks × 3 latent frames × 4 denoising steps)
4. Decode via VAE → output video
5. Optional: composite onto original-resolution video

#### 7B: Evaluation pipeline
- Run on validation sets: hdtf (33), hallo3 (30), hallo3_mixed (12)
- Metrics: FID, SSIM, FVD, CSIM, Sync-C, Sync-D, LMD
- Compare against: (a) 14B teacher, (b) 1.3B undistilled, (c) LatentSync baseline
- Measure inference speed (wall-clock time per video)

---

## 6. Data Flow

### Training Data Flow (Stage 2 - Self-Forcing)

```
┌─ OmniAvatarDataset ─────────────────────────────────────────────────────┐
│  Per sample dir:                                                         │
│    vae_latents_mask_all.pt → real [16,21,64,64], masked [16,21,64,64]   │
│    audio_emb_omniavatar.pt → audio [81, 10752]                          │
│    text_emb.pt → text [1, 512, 4096]                                    │
│    ref_latents.pt → ref_seq [16, 21, 64, 64]                           │
│    mask.png → spatial [64, 64]                                           │
└──────────────────────────────────────────┬──────────────────────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │  _prepare_training_data()        │
                          │  Build condition + neg_condition  │
                          └────────────────┬────────────────┘
                                           │
         ┌─────────────────────────────────▼─────────────────────────────┐
         │                    single_train_step()                         │
         │                                                                │
         │  IF student_update:                                            │
         │  ┌──────────────────────────────────────────────────────────┐  │
         │  │ rollout_with_gradient()                                   │  │
         │  │   For 7 chunks (3 latent frames each):                   │  │
         │  │     Sample exit_step ∈ {0,1,2,3}                         │  │
         │  │     For step=0..exit_step:                                │  │
         │  │       student(chunk, t, condition) [±grad at exit]       │  │
         │  │     Update KV cache (no grad)                            │  │
         │  │   → gen_data [B, 16, 21, 64, 64]                        │  │
         │  └──────────────────────────────────────────────────────────┘  │
         │                           │                                    │
         │         ┌─────────────────▼─────────────────┐                 │
         │         │ perturb: x_t = add_noise(gen, t)  │                 │
         │         └─────────────────┬─────────────────┘                 │
         │                           │                                    │
         │    ┌──────────────────────┼──────────────────────┐            │
         │    ▼                      ▼                      ▼            │
         │  teacher(x_t, t)    fake_score(x_t, t)    discriminator()    │
         │  14B bidirectional   1.3B bidirectional     Conv3D head      │
         │  → teacher_x0       → fake_x0              → gan_loss        │
         │    [no grad]          [no grad]              [no grad]        │
         │                           │                                    │
         │         ┌─────────────────▼─────────────────┐                 │
         │         │ VSD_loss + GAN_loss → backward     │                 │
         │         │ → update student only              │                 │
         │         └───────────────────────────────────┘                 │
         │                                                                │
         │  ELSE (fake_score + discriminator update):                     │
         │    rollout (no grad) → perturb → DSM loss → update fake_score │
         │    + discriminator adversarial loss → update discriminator     │
         └────────────────────────────────────────────────────────────────┘
```

---

## 7. Memory Budget

### 7A: Stage 2 (Self-Forcing) on H200 (150GB)

| Component | Size (GB) | Notes |
|-----------|-----------|-------|
| 14B Teacher (bf16) | ~28 | Frozen, eval mode |
| 1.3B Student (bf16) | ~2.6 | Trainable |
| 1.3B Fake score (bf16) | ~2.6 | Trainable on DSM |
| Discriminator | ~0.1 | Small conv3d head |
| Student optimizer (AdamW) | ~10.4 | 2× model + 2× momentum |
| Fake score optimizer | ~10.4 | 2× model + 2× momentum |
| Discriminator optimizer | ~0.1 | Tiny |
| Student rollout activations | ~30-50 | 7 chunks, grad at exit steps |
| Teacher forward activations | ~15-20 | Single pass, no grad |
| Fake score forward activations | ~5-8 | Single pass |
| KV caches (student) | ~2-4 | 30 layers × cache per layer |
| Precomputed data tensors | ~0.1 | Audio, text, latents |
| PyTorch overhead | ~2-3 | CUDA allocator, etc. |
| **TOTAL ESTIMATE** | **~110-135 GB** | **Fits H200 (tight)** |

### 7B: Risk Mitigation for OOM
1. `start_gradient_frame > 0`: Skip gradient on early chunks (saves ~10-15GB per skipped chunk)
2. Teacher FSDP: Shard teacher across 2 GPUs
3. Gradient checkpointing: Already used in DiT blocks
4. Reduce `student_sample_steps`: 4→2 (reduces rollout memory proportionally)
5. CPU offload teacher: Move to CPU between forward passes (~5s overhead per step)

### 7C: Multi-GPU Strategy
- **2x H200 with DDP**: Each GPU runs full pipeline (~65-70GB each) — comfortable
- **4x H200 with FSDP**: Shard teacher across GPUs, each handles own student — best throughput
- **Recommendation**: Start with 2x H200 DDP, scale to 4x if stable

---

## 8. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | **Audio threading through causal chunks** — misaligned audio→video mapping in chunk-by-chunk generation | High | Critical (broken lip sync) | Implement explicit latent_frame→video_frame mapping in CausalOmniAvatarWan; test with known audio-video pairs |
| 2 | **OOM during Self-Forcing rollout** — 7 chunks × 4 steps with gradient graph | High | Blocks training | Start with `start_gradient_frame=3` (gradient only on last 4 chunks); use 2-GPU DDP |
| 3 | **Weight loading I2V→V2V patch embedding expansion** — 33ch→65ch expansion may fail or produce bad init | Medium | Delays start | Test expansion path explicitly; verify with forward pass comparison |
| 4 | **Noise schedule mismatch** between OmniAvatar's RF and FastGen's RF | Medium | Poor distillation | Extract exact sigma schedule from both; add unit test comparing 100 timesteps |
| 5 | **Forward pass numerical mismatch** between ported WanModel and original | Medium | Silent quality degradation | Run bitwise comparison test before training; allow < 1e-4 tolerance for bf16 |
| 6 | **args singleton residual references** in ported code | Medium | Runtime crashes | Grep for all `args` references; test instantiation without global args |
| 7 | **KV cache state corruption** across chunks | Medium | Incoherent generation | Test multi-chunk generation with visualization; compare to full-sequence bidirectional |
| 8 | **Fake score (1.3B) insufficient capacity** to match 14B teacher distribution | Low-Med | Slow convergence | Monitor fake_score_loss; consider EMA; fallback to 14B fake_score with FSDP |
| 9 | **VSD loss doesn't converge** — teacher/student distribution gap too large | Low-Med | Wasted compute | Verify KD pre-training (Stage 1) quality first; tune guidance_scale |
| 10 | **Causal attention divergence** from bidirectional — student quality degrades | Medium | Lower output quality | Compare causal vs bidirectional on small eval set; accept some quality tradeoff |

---

## 9. Validation Criteria

### Phase 1 (Network Wrappers) — Go/No-Go Gate
- [ ] OmniAvatarWan loads 14B checkpoint and produces correct output (< 1e-4 vs original)
- [ ] OmniAvatarWan loads 1.3B checkpoint and produces correct output
- [ ] CausalOmniAvatarWan single-chunk output matches bidirectional for that chunk
- [ ] CausalOmniAvatarWan 7-chunk autoregressive generation produces coherent video
- [ ] Audio alignment verified: mouth movement correlates with audio across chunks
- [ ] Memory: single forward pass < 40GB on H200

### Phase 5 (Stage 1 KD) — Quality Gate
- [ ] KD loss converges to stable value within 5K iterations
- [ ] Generated samples show recognizable face + lip movement
- [ ] Inference quality on validation set within 20% of teacher's metrics

### Phase 6 (Stage 2 Self-Forcing) — Final Gate
- [ ] VSD loss decreases over training
- [ ] Fake score loss is stable
- [ ] Lip sync quality (Sync-C/D) approaches teacher within 15%
- [ ] Visual quality (FID, CSIM) within 20% of teacher
- [ ] 4-step inference produces usable output (subjective quality check)
- [ ] Inference speedup > 6x over teacher's 25-step generation

---

## Execution Timeline

```
Week 1:  Phase 0 (prerequisites) + Phase 1 (network wrappers)
         ├─ Day 1-2: Port WanModel, AudioPack, remove args singleton
         ├─ Day 3-4: Bidirectional + causal wrappers, weight loading
         └─ Day 5: Validation tests, go/no-go decision

Week 2:  Phase 2 (dataset) + Phase 3 (methods) + Phase 4 (configs)
         ├─ Day 1: Dataset adapter
         ├─ Day 2-3: Method subclasses (KD + SF)
         └─ Day 4-5: Configs, integration testing

Week 3:  Phase 5 (ODE extraction + Stage 1 KD training)
         ├─ Day 1-2: ODE pair generation (GPU-intensive)
         └─ Day 3-5: KD training + checkpoint extraction

Week 4-5: Phase 6 (Stage 2 Self-Forcing training)
          ├─ Day 1: Memory verification + smoke test
          └─ Day 2-10: Full training with monitoring

Week 6:  Phase 7 (inference + evaluation)
         └─ Inference script, metrics, comparison
```

**Total estimated time**: 4-6 weeks (including training compute)

---

## Appendix: File Creation Checklist

New files to create (14 files):
- [ ] `fastgen/networks/OmniAvatar/__init__.py`
- [ ] `fastgen/networks/OmniAvatar/wan_model.py`
- [ ] `fastgen/networks/OmniAvatar/audio_pack.py`
- [ ] `fastgen/networks/OmniAvatar/network.py`
- [ ] `fastgen/networks/OmniAvatar/network_causal.py`
- [ ] `fastgen/methods/omniavatar_self_forcing.py`
- [ ] `fastgen/methods/omniavatar_kd.py`
- [ ] `fastgen/datasets/omniavatar_dataloader.py`
- [ ] `fastgen/configs/experiments/OmniAvatar/__init__.py`
- [ ] `fastgen/configs/experiments/OmniAvatar/config_sf.py`
- [ ] `fastgen/configs/experiments/OmniAvatar/config_kd.py`
- [ ] `fastgen/configs/methods/config_omniavatar_sf.py`
- [ ] `fastgen/configs/methods/config_omniavatar_kd.py`
- [ ] `scripts/generate_omniavatar_ode_pairs.py`

Files to modify (2 files):
- [ ] `fastgen/configs/net.py` (add OmniAvatar configs)
- [ ] `fastgen/networks/__init__.py` (register OmniAvatar)

Existing files to use as reference (NOT modify):
- `fastgen/networks/Wan/network.py` — pattern for bidirectional wrapper
- `fastgen/networks/Wan/network_causal.py` — pattern for causal wrapper
- `fastgen/networks/VaceWan/network.py` — pattern for V2V conditioning
- `fastgen/configs/experiments/WanV2V/config_sf.py` — pattern for SF config
- `fastgen/methods/distribution_matching/self_forcing.py` — base class
- `OmniAvatar/OmniAvatar/models/wan_video_dit.py` — source for porting
