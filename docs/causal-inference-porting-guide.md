# Causal Inference Porting Guide

Reference document for creating block-wise AR inference scripts for FastGen-based
causal models (Self-Forcing / Diffusion Forcing trained students).

**Created from:** OmniAvatar causal inference implementation (2026-03-30)
**Script:** `scripts/inference/inference_causal.py`
**Purpose:** Guide porting to other baselines (InfiniteTalk, etc.)

---

## 1. Architecture Overview

The inference pipeline has 5 stages:

```
Load models → Preprocess inputs → Block-wise AR loop → VAE decode → Save with audio
```

The core insight: the training code's `rollout_with_gradient()` in `self_forcing.py`
performs block-wise autoregressive generation. The inference script mirrors this loop
but without gradients, random exit steps, or the VSD loss machinery.

### What FastGen's training code provides (reuse directly)
- `CausalFastGenNetwork.forward()` — dispatches to `_forward_ar()` with KV cache
- `_build_y()` — constructs V2V conditioning per chunk
- `_process_audio_embeddings()` — caches and slices audio per chunk
- `noise_scheduler.forward_process()` — adds noise for multi-step denoising
- KV cache management — allocation, eviction, rolling window (all internal)

### What FastGen's training code does NOT provide
- No standalone inference entry point
- No raw video/audio preprocessing
- No VAE encode/decode orchestration
- No variable-length generation support (training uses fixed `total_num_frames`)
- No video saving or audio muxing

---

## 2. Critical Gotchas (Bugs We Hit)

### 2.1 `_core.` Prefix Mismatch — SILENT COMPLETE FAILURE

**Severity: Critical. Will silently produce garbage output.**

The `CausalOmniAvatarWan` (and similar causal models) wraps all submodules in
`self._core = nn.Module()` for FSDP2 compatibility. This means:

- **Model state dict keys:** `_core.patch_embedding.weight`, `_core.blocks.0.attn.qkv.weight`, etc.
- **FSDP checkpoint keys:** `patch_embedding.weight`, `blocks.0.attn.qkv.weight` (no `_core.`)

Loading with `strict=False` silently matches **zero keys** — the model runs on random/base
weights with no error. The only symptom is bad output quality.

**Fix:**
```python
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if len(missing) > len(state_dict) * 0.5:
    prefixed = {"_core." + k: v for k, v in state_dict.items()}
    missing2, _ = model.load_state_dict(prefixed, strict=False)
    if len(missing2) < len(missing):
        # Use prefixed version
```

**Check for this in every new baseline.** The `_core` wrapper may or may not exist
depending on the model class. Print `list(model.state_dict().keys())[:3]` to verify.

### 2.2 FSDP Checkpoint Nesting

FastGen's `FSDPCheckpointer` saves as:
```python
{
    "model": {"net": OrderedDict(...)},  # student weights
    "optimizer": ...,
    "scheduler": ...,
    "iteration": int,
}
```

Not just `{"net": ...}`. Handle all variants:
```python
if "model" in ckpt and "net" in ckpt["model"]:
    sd = ckpt["model"]["net"]
elif "net" in ckpt:
    sd = ckpt["net"]
else:
    sd = ckpt  # bare state dict
```

### 2.3 VAE Requires float32

`WanVideoVAE` uses Conv3d layers in float32. Passing bf16 tensors causes:
```
RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same
```

**Fix:** Always cast to float32 before VAE encode/decode, then cast output back to model dtype:
```python
latents = vae.encode([video.to(torch.float32)], device=device)
latents = latents.to(dtype=model_dtype)  # back to bf16
```

### 2.4 Wav2Vec2 SDPA Incompatibility

OmniAvatar's `Wav2VecModel.forward()` sets `self.config.output_attentions = True`,
which is incompatible with the default SDPA attention in newer transformers versions.

**Fix:** Load with `attn_implementation="eager"`:
```python
model = Wav2VecModel.from_pretrained(path, attn_implementation="eager")
```

### 2.5 No ffprobe/ffmpeg on System PATH

The environment may only have ffmpeg via `imageio_ffmpeg`. Use:
```python
import shutil
ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
```

For audio duration, use `librosa.get_duration(path=audio_path)` instead of ffprobe.

### 2.6 Audio Frame Count = VIDEO Frames, Not Latent Frames

The model's audio conditioning expects `[B, num_VIDEO_frames, 10752]`:
- For 21 latent frames: audio should be `[B, 81, 10752]` (81 = 1 + 20*4)
- The `AudioPack` module handles temporal downsampling internally (kernel [4,1,1])

Training slices audio to exactly `num_video_frames`. Get this wrong and audio-lip
sync breaks silently.

### 2.7 Video Frame Count to Latent Frame Count

VAE temporal compression: first frame is 1:1, then groups of 4.
```python
num_latent = 1 + (num_video - 1) // 4
num_video = 1 + (num_latent - 1) * 4
```

When deriving from audio duration, round **down** to nearest multiple of `chunk_size`:
```python
num_video_raw = int(audio_duration * fps)  # floor, not ceil
num_latent_raw = 1 + (num_video_raw - 1) // 4
num_latent = (num_latent_raw // chunk_size) * chunk_size
```

---

## 3. The Conditioning Dict

### 3.1 Required Keys

The model's `forward()` expects a `condition` dict:

| Key | Shape | Description |
|-----|-------|-------------|
| `text_embeds` | `[B, 512, 4096]` | T5-XXL text embeddings |
| `audio_emb` | `[B, num_VIDEO_frames, 10752]` | Wav2Vec2 all hidden states concatenated |
| `ref_latent` | `[B, 16, 1, H_lat, W_lat]` | First frame of reference video (VAE latent) |
| `mask` | `[H_lat, W_lat]` | LatentSync spatial mask (1=keep, 0=generate) |
| `masked_video` | `[B, 16, T_lat, H_lat, W_lat]` | Mouth-masked reference latents |
| `ref_sequence` | `[B, 16, T_lat, H_lat, W_lat]` | Full reference sequence latents |

**For OmniAvatar (in_dim=65):** All 6 keys used → 16+1+16+16+16 = 65 input channels.
**For other models:** Check `in_dim` and which keys `_build_y()` reads.

### 3.2 Training vs Inference Conditioning

Training uses **pre-computed .pt files** from a preprocessing pipeline:
- `vae_latents_mask_all.pt` → `input_latents` (unmasked) + `masked_latents` (mouth-masked)
- `ref_latents.pt` → `ref_sequence_latents` (from a different video segment)
- `audio_emb_omniavatar.pt` → `audio_emb` (sliced to 81 frames)
- `text_emb.pt` → T5 embeddings

Inference can either:
1. **Load pre-computed tensors** (`--precomputed_dir`) — exact match to training, best for debugging
2. **Encode from raw files** — VAE encode reference video, Wav2Vec encode audio, T5 encode text

We verified that fresh VAE encoding matches pre-computed latents within max_diff=0.046
(VAE is deterministic). So raw encoding is fine for production use.

### 3.3 Mask Convention

- **LatentSync mask.png:** 1=keep (upper face), 0=mask (mouth region)
- **Model internally inverts:** `inverted_mask = 1.0 - mask` → 0=keep, 1=generate
- **Masked video:** `video * mask` in pixel space (mouth region zeroed to 0.0 in [-1,1])
- **mask_all_frames=True:** All frames including frame 0 get spatial mask applied

---

## 4. The AR Inference Loop

### 4.1 Structure (mirrors `rollout_with_gradient`)

```python
model.total_num_frames = num_latent_frames  # update for cache allocation
model.clear_caches()

for block_idx in range(num_blocks):
    cur_start_frame = block_idx * chunk_size
    noisy_input = noise[:, :, cur_start_frame:cur_start_frame + chunk_size]

    # Multi-step denoising (e.g., 4 steps from t_list)
    for step_idx in range(len(t_list) - 1):
        x0_pred = model(noisy_input, t_cur, condition=condition,
                        cur_start_frame=cur_start_frame,
                        store_kv=False, is_ar=True, fwd_pred_type="x0")
        if t_next > 0:
            noisy_input = noise_scheduler.forward_process(x0_pred, fresh_noise, t_next)

    output[:, :, cur_start_frame:...] = x0_pred

    # Cache update (context for next block)
    model(x0_pred, t_cache, condition=condition,
          cur_start_frame=cur_start_frame,
          store_kv=True, is_ar=True, fwd_pred_type="x0")

model.clear_caches()
```

### 4.2 Key Parameters

- **`store_kv=False`** during denoising steps (don't pollute cache with intermediate noise)
- **`store_kv=True`** only for the cache update after each block completes
- **`is_ar=True`** always (triggers `_forward_ar` with KV cache, not `_forward_full_sequence`)
- **`cur_start_frame`** in latent frame units — model converts to token offset internally
- **`fwd_pred_type="x0"`** — predict clean output directly
- **`context_noise`** typically 0 — cache update uses clean denoised output

### 4.3 Variable Length

Set `model.total_num_frames = N` before `clear_caches()`. The cache is allocated on
first forward call based on this value. For `local_attn_size > 0`, cache size is bounded
by `local_attn_size * frame_seqlen` regardless of total length.

### 4.4 Denoising Schedule

Default `t_list = [0.999, 0.900, 0.750, 0.500, 0.0]` → 4 denoising steps per block.
This matches the training config's `sample_t_cfg.t_list`. The last value (0.0) is the
final step producing clean output.

---

## 5. Post-Processing

1. **VAE decode** in float32: `vae.decode([latent.float()], device=device)`
2. **Normalize:** `[-1, 1]` → `[0, 255]` uint8
3. **Save silent video** via `imageio.v3.imwrite()` with libx264
4. **Mux audio** via ffmpeg subprocess, clip to video duration with `-t`

---

## 6. Porting Checklist for New Baselines

When adapting this script for a new model (e.g., InfiniteTalk):

### 6.1 Model Construction
- [ ] Identify the causal model class (equivalent of `CausalOmniAvatarWan`)
- [ ] Check constructor parameters: `in_dim`, `chunk_size`, `total_num_frames`, etc.
- [ ] Check if model uses `_core` wrapper — affects state dict key prefix
- [ ] Check `model_size` and weight paths

### 6.2 Checkpoint Loading
- [ ] Identify checkpoint format (FSDP nesting structure)
- [ ] Print first 5 state dict keys from both model and checkpoint
- [ ] Verify key prefix matches (e.g., `_core.` vs bare)
- [ ] After loading, verify `0 missing, 0 unexpected`

### 6.3 Conditioning
- [ ] Read the training dataloader to understand exact condition dict keys and shapes
- [ ] Check `in_dim` to understand what channels the model expects
- [ ] Check if model uses `_build_y()` and what it reads from the condition dict
- [ ] Verify audio frame count convention (video frames vs latent frames)
- [ ] Check mask convention and `mask_all_frames` flag
- [ ] Check if ref_sequence is used (depends on in_dim)

### 6.4 Audio Encoding
- [ ] Same Wav2Vec2 model? Same hidden state concatenation (14 states = 10752 dim)?
- [ ] Different audio encoder? (InfiniteTalk might use different audio features)
- [ ] Audio temporal alignment: how many audio frames per video frame?

### 6.5 Inference Loop
- [ ] Same `rollout_with_gradient` structure? Check the training method's rollout code
- [ ] Same `forward()` dispatch signature? (`cur_start_frame`, `store_kv`, `is_ar`)
- [ ] Same noise scheduler? (`forward_process`, `get_t_list`)
- [ ] Same `t_list` values? (check config)
- [ ] Same `context_noise` default? (check config)

### 6.6 Environment
- [ ] ffmpeg available? (use imageio_ffmpeg fallback)
- [ ] Required pip packages installed?
- [ ] CUDA memory sufficient? (1.3B model ≈ 3GB, 14B ≈ 28GB in bf16)

---

## 7. InfiniteTalk-Specific Notes

Based on previous analysis (`project_infinitetalk_implementation.md`):

- **Model:** 14B Wan2.1-I2V based (not 1.3B like OmniAvatar student — unless DF-distilled)
- **DF checkpoint:** At `.../InfiniteTalk/weights/` — already DF-trained
- **SF status:** Analysis complete, 5 bugs fixed, but never tested end-to-end
- **Audio:** Uses same Wav2Vec2 pattern but might have different `audio_hidden_size`
- **Resolution:** 480P (480x832), not 512x512 — different latent spatial dims
- **Temporal:** 93 video frames / 24 latent frames (vs OmniAvatar's 81/21)
- **Frame rate:** May differ from 25fps
- **Mask:** May use different spatial mask or no mask at all (I2V vs V2V)
- **in_dim:** Check if it uses ref_sequence (65ch) or not (49ch or 33ch)
- **Key file:** `fastgen/methods/infinitetalk_self_forcing.py` — has 3-call CFG
- **Config:** `fastgen/configs/experiments/InfiniteTalk/config_sf.py` — shift=7 (vs OmniAvatar shift=3)

The DF-trained checkpoint can be used with this inference script pattern directly —
just need to adjust model class, config, and conditioning construction.

---

## 8. File Reference

| What | Where |
|------|-------|
| OmniAvatar inference script | `FastGen/scripts/inference/inference_causal.py` |
| Design spec | `FastGen/docs/superpowers/specs/2026-03-30-causal-omnivatar-inference-design.md` |
| Implementation plan | `FastGen/docs/superpowers/plans/2026-03-30-causal-omnivatar-inference.md` |
| Training rollout | `FastGen/fastgen/methods/distribution_matching/self_forcing.py` (`rollout_with_gradient`) |
| Causal model | `FastGen/fastgen/networks/OmniAvatar/network_causal.py` |
| Training dataloader | `FastGen/fastgen/datasets/omniavatar_dataloader.py` |
| Training condition assembly | `FastGen/fastgen/methods/omniavatar_self_forcing.py` |
| Self-Forcing inference ref | `Self-Forcing/pipeline/causal_inference.py` |
| OmniAvatar V2V inference ref | `OmniAvatar-Train/scripts/inference_v2v.py` |
| Prior analysis | `analysis/self-forcing-comparison/` (5 files) |
