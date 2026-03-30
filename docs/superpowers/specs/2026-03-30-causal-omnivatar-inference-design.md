# Causal OmniAvatar Inference Pipeline — Design Spec

**Date:** 2026-03-30
**Status:** Draft
**Location:** `FastGen/scripts/inference/inference_causal.py`

## Goal

Create a standalone inference script for the 1.3B CausalOmniAvatarWan student model that supports:
- Fixed-length generation (e.g., 21 latent frames, as during training)
- Variable-length generation driven by audio clip duration
- Block-wise autoregressive inference with KV cache (Self-Forcing style)
- OmniAvatar-style V2V + audio conditioning
- Rolling window attention via `local_attn_size` for long sequences

## Non-Goals (for now)

- LatentSync compositing (detect → crop → affine → generate → paste back). The generation pipeline assumes 512x512 pre-processed inputs. Compositing wraps this later.
- Teacher/bidirectional inference. Student-only, few-step.
- Batch inference over multiple videos. Designed so a batch loop can be added trivially (model loaded once, per-sample functions take explicit arguments).

## Architecture

```
inference_causal.py
├── parse_args()
├── load_models()              — CausalOmniAvatarWan (1.3B), VAE, Wav2Vec, T5
├── preprocess_inputs()        — Per-sample preprocessing
│   ├── extract_audio()        — ffmpeg from video if no separate audio
│   ├── compute_num_latent_frames() — from audio duration or CLI override
│   ├── adjust_video_length()  — ping-pong extend or clip to match audio
│   ├── encode_video()         — VAE encode → latents
│   ├── encode_audio()         — Wav2Vec → 10752-dim (same as original scripts)
│   └── encode_text()          — T5 encode or load pre-computed .pt
├── build_condition_dict()     — ref_latent, mask, masked_video, ref_sequence, audio_emb, text_embeds
├── run_inference()            — Block-wise AR loop
├── decode_and_save()          — VAE decode → silent video → ffmpeg mux audio
└── main()
```

All per-sample functions take explicit arguments (not global state), so wrapping in a batch loop later requires no structural changes — just an input iterator around `preprocess → infer → save` with cache clearing between samples.

## Input Handling

### CLI Arguments

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `--video_path` | Yes | — | Reference video (must be 512x512). Error if not. |
| `--audio_path` | No | Extract from `--video_path` | Separate audio source (WAV/MP4) |
| `--output_path` | Yes | — | Output video path |
| `--ckpt_path` | Yes | — | Student model checkpoint |
| `--num_latent_frames` | No | From audio duration | Override; must be multiple of `chunk_size` (3) |
| `--prompt` | No | `"a person talking"` | Text prompt for T5 |
| `--text_embeds_path` | No | — | Pre-computed T5 embeddings (.pt), skips T5 load |
| `--local_attn_size` | No | `-1` | Rolling window in frames (-1 = global attention) |
| `--config` | No | OmniAvatar SF config | Model config path/name |
| `--seed` | No | `42` | Random seed |
| `--mask_path` | No | LatentSync default | Path to spatial mask image (LatentSync mask.png) |
| `--device` | No | `cuda` | Device |

### Preprocessing Flow

1. **Audio extraction**: If `--audio_path` not provided, extract audio from `--video_path` via ffmpeg to temp WAV file.

2. **Length calculation**:
   - Get audio duration in seconds
   - Compute number of video frames: `num_video_frames = floor(audio_duration * 25)`
   - Compute number of latent frames: `num_latent_frames = 1 + (num_video_frames - 1) // 4`
   - Round **down** to nearest multiple of `chunk_size` (3)
   - If `--num_latent_frames` provided, use that instead (must be multiple of 3, must not exceed audio-derived length)
   - Note: VAE temporal compression is `1 + (N-1)*4` — first frame has no temporal compression

3. **Video length adjustment**:
   - Compute required video frames: `num_video_frames = 1 + (num_latent_frames - 1) * 4`
   - If video has fewer frames: extend by appending reversed copies (ping-pong) until length matches
   - If video has more frames: clip to `num_video_frames`

4. **Input validation**: Video must be 512x512. Error/warn if not (no auto-resize).

5. **VAE encode**: Reference video frames → VAE encoder → latents `[1, 16, num_latent_frames, H_lat, W_lat]`

6. **Wav2Vec encode**: Audio → Wav2Vec2 feature extraction → `[1, num_video_frames, 10752]` → interpolate to match video frame count. Follows the exact same process as OmniAvatar's original scripts (`Wav2VecModel` with all 14 hidden states concatenated, `linear_interpolation` to frame count).

7. **T5 encode**: Either load from `--text_embeds_path` or encode `--prompt` via T5 at runtime → `[1, 512, 4096]`

8. **Build conditioning dict**: Assemble `ref_latent`, `masked_video`, `ref_sequence`, `mask`, `audio_emb`, `text_embeds` following `_build_y` conventions. Mask uses LatentSync convention (1 = keep region from reference, 0 = generate). Use the same LatentSync mask from training (`OmniAvatar-Train/OmniAvatar/utils/latentsync/mask.png`) — the model was trained with this spatial pattern and expects it at inference. `masked_video = video * (1 - mask)` zeros out the mouth region; the model generates that region.

## Block-wise AR Inference Loop

### Setup

1. Load 1.3B CausalOmniAvatarWan from checkpoint
2. Clear KV caches: `model.clear_caches()`
3. Generate noise: `[1, 16, num_latent_frames, H_lat, W_lat]`
4. Get denoising schedule from config: `t_list = [0.999, 0.900, 0.750, 0.500, 0.0]` (4 denoising steps)
5. Compute `num_blocks = num_latent_frames // chunk_size` (chunk_size = 3)

### Main Loop

```
output = zeros([1, 16, num_latent_frames, H, W])

For block_idx in range(num_blocks):
    cur_start_frame = block_idx * chunk_size

    # Slice noise for this chunk
    noisy_input = noise[:, :, cur_start_frame : cur_start_frame + chunk_size]

    # Build y conditioning for this chunk via _build_y
    y_chunk = model._build_y(condition, T=chunk_size, start_frame=cur_start_frame)

    # Multi-step denoising (4 steps)
    for step_idx, t_cur in enumerate(t_list[:-1]):
        x0_pred = model._forward_ar(
            noisy_input, t_cur, context, y=y_chunk,
            audio_emb=audio_emb,
            current_start=cur_start_frame * frame_seqlen,
            store_kv=False
        )
        t_next = t_list[step_idx + 1]
        if t_next > 0:
            noisy_input = noise_scheduler.forward_process(x0_pred, randn_like(...), t_next)
        else:
            noisy_input = x0_pred  # Final step → clean

    # Store denoised chunk
    output[:, :, cur_start_frame : cur_start_frame + chunk_size] = x0_pred

    # Update KV cache with denoised output (context for next block)
    with torch.no_grad():
        model._forward_ar(
            x0_pred, t=0, context, y=y_chunk,
            audio_emb=audio_emb,
            current_start=cur_start_frame * frame_seqlen,
            store_kv=True
        )

model.clear_caches()
```

### Rolling Window (when `local_attn_size > 0`)

KV cache eviction is handled internally by `CausalSelfAttention`:
- When cache exceeds `local_attn_size * frame_seqlen` tokens, oldest non-sink tokens are evicted
- Sink tokens (first N frames, controlled by `sink_size`) are preserved
- No extra logic needed in the inference loop — just set `local_attn_size` in model config
- For short sequences (21 frames), leave at `-1` (global, no eviction)

### Verification Checklist

After implementation, verify with a short dummy run (2-3 blocks):
1. **KV cache indices**: After each `store_kv=True` call, `global_end_index` and `local_end_index` advance by `chunk_size * frame_seqlen`
2. **`current_start` units**: Confirm `cur_start_frame * frame_seqlen` matches post-patchification token offset (frame_seqlen = h_patches * w_patches)
3. **Rolling eviction**: With `local_attn_size=3`, generate 4+ blocks and verify cache doesn't exceed `3 * frame_seqlen` tokens
4. **Timestep=0 cache update**: Verify produces same cache state as training's context update
5. **Audio slicing**: Verify `self._cached_audio` is sliced correctly per chunk via `current_start // frame_seqlen`

## Post-processing & Output

1. **VAE Decode**: `video = vae.decode(output)` → `[1, C, T_video, H, W]` where `T_video = 1 + (num_latent_frames - 1) * 4`
2. **Normalize**: `[-1, 1]` → `[0, 255]` uint8
3. **Save silent video**: Write to temp file via `imageio` (libx264), FPS = 25
4. **Audio mux**: ffmpeg mux silent video + audio → `--output_path`
   ```
   ffmpeg -y -i silent.mp4 -i audio.wav -map 0:v:0 -map 1:a:0
          -c:v libx264 -crf 18 -c:a aac -t <video_duration> output.mp4
   ```
5. **Cleanup**: Remove temp files (extracted audio WAV, silent video)

## Key Implementation References

| Component | Source File | What to Reuse |
|-----------|-----------|---------------|
| `_forward_ar` | `FastGen/fastgen/networks/OmniAvatar/network_causal.py` | AR forward pass with KV cache |
| `_build_y` | Same file | V2V conditioning construction |
| `_process_audio_embeddings` | Same file | Audio feature projection (called internally) |
| `rollout_with_gradient` | `FastGen/fastgen/methods/distribution_matching/self_forcing.py` | Reference for block loop structure |
| `CausalInferencePipeline.inference` | `Self-Forcing/pipeline/causal_inference.py` | Reference for variable-length AR inference pattern |
| `Wav2VecModel` | `OmniAvatar-Train/OmniAvatar/models/wav2vec.py` | Audio feature extraction |
| `inference_v2v.py` | `OmniAvatar-Train/scripts/inference_v2v.py` | Audio extraction, muxing, overall pipeline |
| `save_video` / `save_media` | `FastGen/fastgen/utils/basic_utils.py` | Video saving utilities |
| `NoiseSchedulerFlowMatching` | `FastGen/fastgen/networks/OmniAvatar/noise_schedule.py` | `forward_process`, `get_t_list` |

## Future Extensions

- **Batch inference**: Wrap per-sample functions in a loop over `--input_dir`. Model loaded once, caches cleared between samples. Skip completed outputs for resume.
- **LatentSync compositing**: Add detection → crop → affine before generation, paste-back after. Generation pipeline unchanged — just receives different 512x512 crops.
- **Pre-computed tensors**: Accept `--latents_path`, `--audio_emb_path` to skip VAE/Wav2Vec encoding.
