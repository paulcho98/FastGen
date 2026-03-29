# InfiniteTalk: Comprehensive Pipeline Analysis for Self-Forcing Adaptation

**Purpose:** This document provides a complete technical analysis of the InfiniteTalk inference pipeline — from raw inputs through the denoising forward pass — to inform the design of a FastGen Self-Forcing adaptation. All relevant conditioning mechanisms, tensor shapes, and architectural details are documented.

**Key Difference from OmniAvatar:** OmniAvatar is a **V2V** model (video-to-video, 65ch input). InfiniteTalk is fundamentally an **I2V** model (image-to-video, 36ch input) with audio conditioning injected via cross-attention, not channel concatenation.

---

## 1. High-Level Architecture

| Component | Details |
|---|---|
| Base model | Wan2.1-I2V-14B (40 layers, dim=5120, 40 heads) |
| Model type | `i2v` — requires CLIP features + VAE-encoded condition frame |
| `in_dim` | **36** = 16 (noise latent) + 4 (mask) + 16 (VAE condition) |
| `out_dim` | 16 (VAE latent channels) |
| Text encoder | UMT5-XXL (bf16), text_len=512 |
| Image encoder | CLIP ViT-H/14 (fp16), produces 257 tokens × 1280 dim |
| Audio encoder | wav2vec2 (chinese-wav2vec2-base), 12 hidden layers × 768 dim |
| VAE | Wan2.1 VAE, stride (4, 8, 8) — temporal 4×, spatial 8× |
| Patch size | (1, 2, 2) |
| Noise schedule | Flow matching, 1000 timesteps, shift=7 (480p) or shift=11 (720p) |
| Sampling | Euler ODE, 40 steps default |
| Resolution | 480p: aspect-ratio bucketing around 640×640; 720p: around 960×960 |
| Frame count | 81 frames per clip (= 20 latent frames + 1 first frame → 21 latent) |
| FPS | 25 fps (output video) |
| InfiniteTalk weights | Separate `.safetensors` merged on top of base Wan I2V weights |

---

## 2. Input Preprocessing Pipeline

### 2.1 Reference Image/Video

**Input:** Either a single image (`.png`) or a video (`.mp4`) path via `cond_video` in JSON.

**Processing:**
1. Extract frame 0 from the video (or use the image directly): `extract_specific_frames(cond_file_path, 0)`
2. Aspect-ratio bucketing: find closest bucket in `ASPECT_RATIO_627` (480p) or `ASPECT_RATIO_960` (720p)
3. Resize + center crop to `(target_h, target_w)`
4. Normalize: `cond_image = (cond_image / 255 - 0.5) * 2` → range [-1, 1]
5. Shape becomes `[1, 3, 1, H, W]` (batch, channels, 1 frame, height, width)

### 2.2 Audio Processing

**Input:** `.wav` file(s) at 16kHz. Supports 1 or 2 speakers.

**Step 1 — Raw audio loading:**
```
librosa.load(audio_path, sr=16000)  →  numpy array
loudness_norm(array, sr=16000, lufs=-23)  →  normalized numpy array
```

**Step 2 — wav2vec2 feature extraction:**
```python
# wav2vec_feature_extractor (HuggingFace Wav2Vec2FeatureExtractor)
audio_feature = wav2vec_feature_extractor(speech_array, sampling_rate=16000).input_values
audio_feature = torch.from_numpy(audio_feature).float().unsqueeze(0)  # [1, num_samples]

# wav2vec2 encoder with linear interpolation to video frame count
video_length = audio_duration * 25  # frames at 25fps
embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

# Stack all 12 hidden layers (skip layer 0)
audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
# Shape: [12, num_video_frames, 768]  →  rearrange → [num_video_frames, 12, 768]
audio_emb = rearrange(audio_emb, "b s d -> s b d")
# Final shape: [num_video_frames, 12, 768]
```

**Important:** The wav2vec2 model performs `linear_interpolation` internally to match audio features to video frame count at 25fps. Each output frame gets 12 hidden layers × 768 dim.

**Step 3 — Audio windowing (in inference loop):**
```python
# Per-frame audio window: 5 frames centered on current frame
indices = (torch.arange(2 * 2 + 1) - 2) * 1  # = [-2, -1, 0, 1, 2]
center_indices = torch.arange(audio_start_idx, audio_end_idx).unsqueeze(1) + indices.unsqueeze(0)
center_indices = torch.clamp(center_indices, min=0, max=total_frames-1)
audio_emb = full_audio_emb[center_indices]
# Shape: [1, clip_length, 5, 12, 768]  (per-human)
```

For multi-speaker: stack along batch dim → `[HUMAN_NUMBER, clip_length, 5, 12, 768]` → cast to bf16.

### 2.3 Text Processing

**Input:** Prompt string + negative prompt (from config).

```python
context, context_null = text_encoder([prompt, neg_prompt], device)
# Each is a list of 1 tensor: [text_len, 4096] (UMT5-XXL output dim)
```

### 2.4 CLIP Image Features

```python
clip_context = clip.visual(cond_image[:, :, -1:, :, :])  # last frame of cond_image
# Shape: [1, 257, 1280]  →  cast to bf16
```

The 257 tokens = 1 CLS token + 256 patch tokens from ViT-H/14.

---

## 3. Condition Assembly (Channel Concatenation)

This is the core difference from a pure I2V model. InfiniteTalk uses the **standard Wan I2V conditioning** approach:

### 3.1 VAE Encoding of Condition

```python
# Zero-pad condition image to full video length
video_frames = torch.zeros(1, 3, frame_num - 1, H, W)  # frame_num-1 zero frames
padding = torch.cat([cond_image, video_frames], dim=2)   # [1, 3, frame_num, H, W]
y = vae.encode(padding)  # [1, 16, T_lat, H_lat, W_lat]  where T_lat = (frame_num-1)//4 + 1
```

### 3.2 Mask Construction

```python
msk = torch.ones(1, frame_num, lat_h, lat_w)   # all ones initially
msk[:, 1:] = 0                                   # only first frame is 1 (condition)
# Reshape to latent temporal dim:
# Repeat first frame 4× to match VAE temporal stride, then group by 4
msk = torch.concat([
    torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
    msk[:, 1:]
], dim=1)
msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
msk = msk.transpose(1, 2)  # [1, 4, T_lat, H_lat, W_lat]
```

### 3.3 Final Input Assembly

```python
y = torch.cat([msk, y], dim=1)  # [1, 4+16, T_lat, H_lat, W_lat] = [1, 20, T_lat, H_lat, W_lat]
```

The model's `in_dim = 36` breaks down as:
- **16 channels**: noise latent `x` (the denoised target)
- **20 channels**: condition `y` = 4 (mask) + 16 (VAE-encoded reference + zeros)

These are concatenated **channel-wise** in the model's forward:
```python
x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
# x[0] shape: [36, T_lat, H_lat, W_lat]
```

Then passed through `patch_embedding` (Conv3d with kernel (1,2,2)):
```python
x = self.patch_embedding(x.unsqueeze(0))  # [1, 5120, T_lat, H_lat//2, W_lat//2]
```

---

## 4. Model Forward Pass (DiT)

### 4.1 Input Processing

```python
# 1. Patch embedding: Conv3d(36 → 5120, kernel=(1,2,2), stride=(1,2,2))
x = patch_embedding(x)  # [B, 5120, T, H/2, W/2]

# 2. Flatten to sequence: [B, T*H/2*W/2, 5120]

# 3. Time embedding: sinusoidal → MLP → projection to 6*dim for modulation
e = time_embedding(sinusoidal_embedding_1d(256, t))  # [B, 5120]
e0 = time_projection(e)  # [B, 6, 5120]  (6 modulation signals)

# 4. Text embedding: Linear(4096→5120) + GELU + Linear(5120→5120)
context = text_embedding(context)  # [B, text_len, 5120]

# 5. CLIP embedding: MLPProj(1280→5120)
clip_tokens = img_emb(clip_fea)  # [B, 257, 5120]
context = concat([clip_tokens, context], dim=1)  # [B, 257+text_len, 5120]
```

### 4.2 Audio Projection (AudioProjModel)

The audio goes through a dedicated projection module before being fed to cross-attention:

```python
# Audio input shape: [HUMAN_NUMBER, clip_length, window=5, blocks=12, channels=768]

# Split into first-frame audio and latter-frame audio:
first_frame_audio = audio[:, :1, ...]     # [H, 1, 5, 12, 768]
latter_frame_audio = audio[:, 1:, ...]    # [H, clip_len-1, 5, 12, 768]

# Latter frames are grouped by VAE temporal stride (4):
latter_frame_audio = rearrange(latter, "b (n_t n) w s c -> b n_t n w s c", n=4)

# Complex windowing: first/middle/last frames within each group get different windows
# First frame in group: window[:middle_index+1]  → (1, 3, 12, 768) → flattened
# Middle frames: window[middle_index:middle_index+1]  → per-frame
# Last frame: window[middle_index:]  → (1, 3, 12, 768) → flattened

# Projection:
# proj1:     Linear(5*12*768=46080 → 512)  for first frame
# proj1_vf:  Linear(8*12*768=73728 → 512)  for variable-length latter frames
# proj2:     Linear(512 → 512)
# proj3:     Linear(512 → context_tokens*output_dim = 32*768 = 24576)

# Output shape: [B, video_length, context_tokens=32, output_dim=768]
# For multi-speaker: concat along context_tokens → [1, video_length, 32*HUMAN_NUM, 768]
```

### 4.3 Human Mask / Reference-Target Masks

For multi-speaker scenarios, spatial masks identify which person occupies which region:
```python
# ref_target_masks: [num_classes, H, W]  →  resize to [num_classes, H_lat, W_lat]
# num_classes = HUMAN_NUMBER + 1 (background)
# Single speaker: all ones (no spatial separation needed)
# Multi speaker: bounding-box or left/right half masks
```

These masks are used in the self-attention block to compute per-region attention maps, which then route the correct audio to the correct person.

### 4.4 Attention Block (WanAttentionBlock)

Each of the 40 transformer blocks contains:

```
1. AdaLN modulation (6 signals from timestep)
2. Self-Attention (WanSelfAttention)
   - QKV projection → RoPE → flash_attention
   - Also computes ref_attn_map: attention map from visual tokens to reference frame tokens
   - Used to determine which person each spatial token belongs to
3. Cross-Attention to Text+CLIP (WanI2VCrossAttention)
   - Separate K,V projections for CLIP tokens (first 257) vs text tokens
   - img_x = attn(Q, K_clip, V_clip)
   - text_x = attn(Q, K_text, V_text)
   - output = img_x + text_x
4. Audio Cross-Attention (SingleStreamMutiAttention)
   - For single speaker: standard cross-attention to audio tokens
   - For multi speaker:
     * Uses ref_attn_map to assign spatial tokens to speakers
     * 1D RoPE positional encoding based on speaker assignment
     * Routes speaker-specific audio to correct spatial regions
5. FFN with AdaLN modulation
```

### 4.5 CFG Strategy (Triple-Condition)

InfiniteTalk uses a **three-way CFG** with separate text and audio guidance:

```python
# When text_guide_scale != 1.0 (default):
noise_pred_cond      = model(x, t, text=prompt,  audio=audio)       # full condition
noise_pred_drop_text = model(x, t, text=neg_prompt, audio=audio)    # drop text
noise_pred_uncond    = model(x, t, text=neg_prompt, audio=zeros)    # drop both

noise_pred = noise_pred_uncond
    + text_guide_scale * (noise_pred_cond - noise_pred_drop_text)
    + audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)

# When text_guide_scale == 1.0:
noise_pred_cond       = model(x, t, text=prompt, audio=audio)
noise_pred_drop_audio = model(x, t, text=prompt, audio=zeros)

noise_pred = noise_pred_drop_audio + audio_guide_scale * (noise_pred_cond - noise_pred_drop_audio)
```

**Important for Self-Forcing:** The model is called **3 times per step** (cond, drop_text, uncond). This means the teacher computation is 3× more expensive than OmniAvatar's 2-call CFG.

The `noise_pred` is then negated (`noise_pred = -noise_pred`) before the Euler step.

### 4.6 Euler ODE Sampling

```python
dt = (timesteps[i] - timesteps[i+1]) / num_timesteps
latent = latent + noise_pred * dt
```

Standard flow matching Euler integration with timestep transform (shift).

---

## 5. Streaming / Long Video Generation

InfiniteTalk supports infinite-length video by iteratively generating clips with overlapping **motion frames**:

1. **First clip:** Condition = reference image (1 frame). Generate 81 frames.
2. **Subsequent clips:**
   - Take last `motion_frame` (default 9) frames from previous clip as condition
   - VAE-encode them as `latent_motion_frames`
   - At each denoising step: inject noised motion frame latents into first portion of noise:
     ```python
     add_latent = add_noise(latent_motion_frames, random_noise, timestep)
     latent[:, :T_motion] = add_latent
     ```
   - After denoising: keep only `latent[:, :T_motion] = latent_motion_frames` (clean)
   - Also extract a new reference frame from the *original* video at the current position for the CLIP+mask condition
3. **Audio sliding:** `audio_start_idx += (frame_num - motion_frame)` each iteration

**Key insight for Self-Forcing:** The streaming mechanism already implements a form of autoregressive generation. Self-Forcing could operate at the clip level rather than the frame level.

---

## 6. Comparison: InfiniteTalk vs OmniAvatar

| Aspect | OmniAvatar | InfiniteTalk |
|---|---|---|
| **Model type** | V2V (video-to-video) | I2V (image-to-video) |
| **Base model** | Wan2.1-T2V-14B | Wan2.1-I2V-14B |
| **in_dim** | 65 = 16+16+1+16+16 | 36 = 16+4+16 |
| **Conditioning** | All via channel concat (mask, ref, video, ref_seq) | Channel concat (mask + VAE ref) + CLIP cross-attn |
| **Audio injection** | Separate audio cross-attn (LatentSync-style) | Separate audio cross-attn (wav2vec2-based AudioProjModel) |
| **Audio encoder** | HuBERT / wav2vec2 (details in LatentSync) | chinese-wav2vec2-base, 12 layers × 768 dim |
| **Audio proj** | Linear projection per frame | AudioProjModel: windowed multi-scale → 32 tokens/frame |
| **CFG** | 2-call (cond, uncond) | 3-call (cond, drop_text, uncond) or 2-call (cond, drop_audio) |
| **Reference frame** | VAE-encoded, concat as 16ch | VAE-encoded (16ch) + CLIP visual (257 tokens cross-attn) |
| **Mask** | 1ch binary latentsync mask | 4ch temporal mask (which frames are conditioned) |
| **ref_sequence** | 16ch additional reference | N/A (uses CLIP instead) |
| **Multi-speaker** | Not supported | Supported (spatial attention routing) |
| **Noise shift** | 3.0 | 7.0 (480p), 11.0 (720p) |
| **Student model** | Separate 1.3B causal model | No student yet (adaptation needed) |
| **Streaming** | Not implemented | Built-in motion frame injection |
| **Output resolution** | Fixed (from OmniAvatar) | Aspect-ratio bucketing (640×640 default) |

---

## 7. Key Considerations for Self-Forcing Adaptation

### 7.1 Network Wrapper (FastGenNetwork equivalent)

Need to create `InfiniteTalkWan(FastGenNetwork)` that wraps InfiniteTalk's `WanModel`:
- `forward()` must accept `(x, timestep, condition, neg_condition)` matching FastGen's interface
- `condition` dict needs: `context` (text), `clip_fea` (CLIP), `y` (mask+VAE ref), `audio` (audio embs), `ref_target_masks`
- `neg_condition` dict: same but with `context_null` and `audio=zeros`
- The wrapper handles the channel concat `x = [cat(noise, y)]` internally

### 7.2 Causal Student Model

For the causal 1.3B student, need `CausalInfiniteTalkWan(CausalFastGenNetwork)`:
- **Challenge:** The base 1.3B Wan model is T2V (in_dim=16), not I2V (in_dim=36). Need to either:
  1. Modify patch_embedding to accept 36 channels, or
  2. Use a different 1.3B checkpoint that already has I2V architecture
- The Wan2.1-I2V-14B config.json shows in_dim=36. There may not be a 1.3B I2V variant.
- **Alternative:** Initialize a 1.3B model with in_dim=36 (random init for the extra channels in patch_embedding)

### 7.3 Audio Conditioning in Student

The `AudioProjModel` and `SingleStreamMutiAttention` are InfiniteTalk-specific additions:
- These weights come from `infinitetalk.safetensors`, not the base Wan checkpoint
- The student must include these audio modules
- For single-speaker Self-Forcing, can simplify to `SingleStreamAttention` (no RoPE routing)

### 7.4 Fake Score Network

Like OmniAvatar, the fake score net should be a **1.3B bidirectional** model (not causal), with the same in_dim=36 and audio modules.

### 7.5 CFG in Training

OmniAvatar uses 2-call CFG. InfiniteTalk uses 3-call. For Self-Forcing:
- Teacher must run 3 forward passes per step (significantly more expensive)
- Could simplify to 2-call for training (fix text_guide_scale=1.0, only audio CFG)
- This halves teacher compute and simplifies the VSD loss

### 7.6 Dataloader

Training data needs:
- VAE-encoded reference frames (first frame latent + mask)
- CLIP features of reference frame
- wav2vec2 audio embeddings (pre-computed, shape `[num_frames, 12, 768]`)
- Text embeddings (T5)
- For ODE KD: ODE trajectories from 14B teacher

### 7.7 Noise Schedule

InfiniteTalk uses shift=7 (480p) vs OmniAvatar's shift=3. The `timestep_transform` function is identical to FastGen's — just needs correct shift parameter.

### 7.8 Motion Frame Injection

The streaming mechanism injects noised previous-clip frames at each denoising step. For Self-Forcing:
- This is conceptually similar to the autoregressive rollout in Self-Forcing
- During Self-Forcing training, the student generates frames autoregressively and these become the "motion frames" for subsequent clips
- The noise injection on motion frames acts as a form of SDEdit bridge

---

## 8. File Inventory

### Core Pipeline Files
| File | Purpose |
|---|---|
| `wan/multitalk.py` | `InfiniteTalkPipeline` — full inference pipeline |
| `wan/modules/multitalk_model.py` | `WanModel` — DiT with audio cross-attention |
| `wan/modules/attention.py` | `flash_attention`, `SingleStreamMutiAttention` |
| `wan/configs/wan_multitalk_14B.py` | Model config (dim=5120, 40 layers, 40 heads) |
| `wan/configs/shared_config.py` | Shared config (T5 dtype, text_len, timesteps) |
| `wan/modules/model.py` | Base Wan model (for comparison) |

### Audio Processing
| File | Purpose |
|---|---|
| `src/audio_analysis/wav2vec2.py` | Modified Wav2Vec2Model with `linear_interpolation` |
| `src/audio_analysis/torch_utils.py` | `linear_interpolation` helper |
| `generate_infinitetalk.py` | Audio preprocessing (`get_embedding`, `audio_prepare_single`) |

### Utilities
| File | Purpose |
|---|---|
| `wan/utils/multitalk_utils.py` | Spatial attention routing, RoPE1D, video save, aspect ratio buckets |
| `wan/utils/utils.py` | Video codec utils, frame extraction |
| `wan/modules/vae.py` | Wan VAE encoder/decoder |
| `wan/modules/clip.py` | CLIP visual encoder |
| `wan/modules/t5.py` | T5 text encoder |

### Weights
| File | Purpose |
|---|---|
| `weights/Wan2.1-I2V-14B-480P/` | Base 14B I2V model (7 safetensor shards) |
| `weights/InfiniteTalk/single/infinitetalk.safetensors` | Single-speaker audio modules |
| `weights/InfiniteTalk/multi/infinitetalk.safetensors` | Multi-speaker audio modules |
| `weights/chinese-wav2vec2-base/` | wav2vec2 audio encoder |

---

## 9. Tensor Shape Summary

| Tensor | Shape | Notes |
|---|---|---|
| Raw audio | `[num_samples]` | 16kHz mono float32 |
| wav2vec2 output | `[num_video_frames, 12, 768]` | 12 hidden layers, 768 dim each |
| Audio window | `[HUMAN_NUM, clip_len, 5, 12, 768]` | 5-frame window around each frame |
| Audio projected | `[1, video_len, 32*HUMAN_NUM, 768]` | 32 context tokens per frame per human |
| Cond image | `[1, 3, 1, H, W]` | Single reference frame, [-1,1] |
| CLIP features | `[1, 257, 1280]` | CLS + 256 patch tokens |
| VAE latent (ref) | `[1, 16, T_lat, H_lat, W_lat]` | T_lat = (frame_num-1)//4+1 = 21 |
| Mask | `[1, 4, T_lat, H_lat, W_lat]` | 4ch mask in latent space |
| Condition y | `[1, 20, T_lat, H_lat, W_lat]` | mask(4) + vae_ref(16) |
| Noise latent x | `[16, T_lat, H_lat, W_lat]` | Target to denoise |
| Model input | `[36, T_lat, H_lat, W_lat]` | x(16) + y(20) channel-concat |
| Text context | `[1, 512, 4096]` → `[1, 512, 5120]` | After text_embedding |
| Full context | `[1, 257+512, 5120]` | CLIP + text concatenated |
| ref_target_masks | `[num_classes, H_lat, W_lat]` | Human region masks |
| Timestep | `[1]` scalar | Single scalar timestep |

---

## 10. Summary: What We Need for FastGen Adaptation

1. **Network wrapper** (`InfiniteTalkWan`): Wrap multitalk WanModel, handle condition dict → forward args
2. **Causal variant** (`CausalInfiniteTalkWan`): 1.3B with chunk-wise causal attention + KV cache, in_dim=36
3. **Audio modules**: Port `AudioProjModel` + `SingleStreamMutiAttention` to both teacher and student
4. **Dataloader**: Load pre-computed VAE latents, CLIP features, wav2vec2 embeddings, T5 embeddings
5. **Config**: Set shift=7, in_dim=36, configure triple or dual CFG
6. **ODE trajectory generation**: Adapt script for InfiniteTalk's 3-call CFG
7. **VSD loss adaptation**: Account for 36ch input and audio conditioning
