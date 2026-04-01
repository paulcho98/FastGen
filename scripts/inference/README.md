# Causal OmniAvatar Inference

Block-wise autoregressive inference for the 1.3B CausalOmniAvatarWan student model
(trained via Diffusion Forcing / Self-Forcing). Generates lip-synced video from
a reference video and audio.

## Prerequisites

**Models** (all under `OmniAvatar-Train/pretrained_models/`):

| Model | Path | Notes |
|-------|------|-------|
| Base Wan 2.1 (1.3B) | `Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors` | Base T2V weights |
| OmniAvatar LoRA+audio | `OmniAvatar-1.3B/pytorch_model.pt` | LoRA + audio modules |
| DF/SF checkpoint | `1.3B-causal-step-XXXX.pth` | Trained student weights |
| VAE | `Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | Shared between 1.3B/14B |
| Wav2Vec2 | `wav2vec2-base-960h/` | Audio encoder |
| LatentSync mask | `OmniAvatar/utils/latentsync/mask.png` | Spatial mouth mask |

**Text embeddings:** Either a pre-computed `.pt` file (`--text_embeds_path`) or a T5
encoder checkpoint (`--text_encoder_path`). Pre-computed is recommended to avoid loading T5.

**For LatentSync compositing:** `pip install insightface onnxruntime-gpu kornia`

## Quick Start

Set common paths (adjust for your environment):

```bash
cd /path/to/FastGen
PRETRAINED="../OmniAvatar-Train/pretrained_models"
MASK="../OmniAvatar-Train/OmniAvatar/utils/latentsync/mask.png"
CKPT="$PRETRAINED/1.3B-causal-step-0002500.pth"
BASE="$PRETRAINED/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
LORA="$PRETRAINED/OmniAvatar-1.3B/pytorch_model.pt"
VAE="$PRETRAINED/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
WAV2VEC="$PRETRAINED/wav2vec2-base-960h"
TEXT_EMB="/path/to/text_emb.pt"  # pre-computed T5 embedding of "a person is talking"
```

---

## Mode 1: Pre-processed 512x512 Input (Single Video)

For inputs that are already cropped/aligned to 512x512 (e.g., from LatentSync
preprocessing or training data).

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference/inference_causal.py \
    --video_path /path/to/512x512_video.mp4 \
    --audio_path /path/to/audio.wav \
    --output_path /path/to/output.mp4 \
    --ckpt_path "$CKPT" \
    --base_model_paths "$BASE" \
    --omniavatar_ckpt_path "$LORA" \
    --vae_path "$VAE" \
    --wav2vec_path "$WAV2VEC" \
    --mask_path "$MASK" \
    --text_embeds_path "$TEXT_EMB" \
    --num_latent_frames 21 \
    --seed 42
```

**With pre-computed tensors** (exact training-style conditioning, fastest):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference/inference_causal.py \
    --video_path /path/to/sample_dir/sub_clip.mp4 \
    --audio_path /path/to/sample_dir/audio.wav \
    --output_path /path/to/output.mp4 \
    --ckpt_path "$CKPT" \
    --base_model_paths "$BASE" \
    --omniavatar_ckpt_path "$LORA" \
    --vae_path "$VAE" \
    --wav2vec_path "$WAV2VEC" \
    --mask_path "$MASK" \
    --precomputed_dir /path/to/sample_dir \
    --num_latent_frames 21 \
    --seed 42
```

The `--precomputed_dir` should contain: `vae_latents_mask_all.pt`, `audio_emb_omniavatar.pt`,
`text_emb.pt`, and optionally `ref_latents.pt`. This bypasses VAE/Wav2Vec encoding.

---

## Mode 2: LatentSync Compositing (Arbitrary Resolution)

For arbitrary-resolution input videos. The script detects faces, aligns to 512x512,
generates, and composites back onto the original resolution.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference/inference_causal.py \
    --video_path /path/to/original_video.mp4 \
    --output_path /path/to/output.mp4 \
    --ckpt_path "$CKPT" \
    --base_model_paths "$BASE" \
    --omniavatar_ckpt_path "$LORA" \
    --vae_path "$VAE" \
    --wav2vec_path "$WAV2VEC" \
    --mask_path "$MASK" \
    --text_embeds_path "$TEXT_EMB" \
    --latentsync \
    --face_cache_dir /path/to/face_cache \
    --num_latent_frames 21 \
    --seed 42
```

**Outputs:**
- `output.mp4` — composited at original resolution with audio
- `output_aligned.mp4` — aligned 512x512 generation with audio (for metrics)

**Face caches** are saved to `--face_cache_dir` as `{video_stem}_face_cache.pt`.
On re-runs with the same cache dir, face detection is skipped (loaded from cache).

---

## Mode 3: Batch Inference

Process a directory of samples. Each subdirectory should contain `sub_clip.mp4`
and `audio.wav`. If pre-computed `.pt` files are present, they are used automatically.

**Batch with pre-processed 512x512 inputs:**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference/inference_causal.py \
    --input_dir /path/to/dataset \
    --output_dir /path/to/results \
    --ckpt_path "$CKPT" \
    --base_model_paths "$BASE" \
    --omniavatar_ckpt_path "$LORA" \
    --vae_path "$VAE" \
    --wav2vec_path "$WAV2VEC" \
    --mask_path "$MASK" \
    --text_embeds_path "$TEXT_EMB" \
    --num_latent_frames 21 \
    --skip_existing \
    --seed 42
```

**Batch with LatentSync compositing:**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference/inference_causal.py \
    --input_dir /path/to/dataset \
    --output_dir /path/to/results \
    --ckpt_path "$CKPT" \
    --base_model_paths "$BASE" \
    --omniavatar_ckpt_path "$LORA" \
    --vae_path "$VAE" \
    --wav2vec_path "$WAV2VEC" \
    --mask_path "$MASK" \
    --text_embeds_path "$TEXT_EMB" \
    --latentsync \
    --face_cache_dir /path/to/face_cache \
    --num_latent_frames 21 \
    --skip_existing \
    --seed 42
```

**Expected input directory structure:**
```
dataset/
├── sample_001/
│   ├── sub_clip.mp4          # reference video
│   ��── audio.wav             # audio
│   ├── text_emb.pt           # (optional) pre-computed T5 embeddings
│   ├── audio_emb_omniavatar.pt  # (optional) pre-computed Wav2Vec features
│   ├── vae_latents_mask_all.pt  # (optional) pre-computed VAE latents
│   └── ref_latents.pt        # (optional) pre-computed reference sequence
├── sample_002/
│   └── ...
```

**Features:**
- `--skip_existing` skips samples whose output file already exists (for resuming)
- Failed samples are logged and skipped (other samples continue)
- Summary printed at end: succeeded / failed / skipped

---

## Variable-Length Generation

By default, generation length is derived from audio duration:

```
num_video_frames = floor(audio_duration_s * 25)
num_latent_frames = 1 + (num_video_frames - 1) // 4   (VAE temporal compression)
num_latent_frames rounded DOWN to multiple of chunk_size (3)
```

Override with `--num_latent_frames N` (must be a multiple of 3 and not exceed
audio-derived length).

For long sequences, use `--local_attn_size N` to enable rolling window attention
(e.g., `--local_attn_size 7` limits each frame to attending to the last 7 frames).

---

## Key Arguments Reference

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--ckpt_path` | Yes | — | DF/SF trained student checkpoint |
| `--base_model_paths` | Yes* | — | Base Wan 2.1 1.3B safetensors |
| `--omniavatar_ckpt_path` | Yes* | — | OmniAvatar LoRA+audio checkpoint |
| `--vae_path` | Yes | — | Wan 2.1 VAE checkpoint |
| `--wav2vec_path` | Yes | — | Wav2Vec2 model directory |
| `--mask_path` | Yes | — | LatentSync spatial mask PNG |
| `--video_path` | Mode 1/2 | — | Single input video |
| `--output_path` | Mode 1/2 | — | Single output video |
| `--input_dir` | Mode 3 | — | Batch input directory |
| `--output_dir` | Mode 3 | — | Batch output directory |
| `--text_embeds_path` | Recommended | — | Pre-computed T5 embeddings |
| `--precomputed_dir` | Optional | — | Directory with pre-computed .pt tensors |
| `--latentsync` | Optional | False | Enable face detection + compositing |
| `--face_cache_dir` | With `--latentsync` | — | Face detection cache directory |
| `--num_latent_frames` | Optional | Auto | Generation length (multiple of 3) |
| `--skip_existing` | Optional | False | Skip completed samples in batch |
| `--local_attn_size` | Optional | -1 | Rolling window (-1 = full attention) |
| `--seed` | Optional | 42 | Random seed |
| `--fps` | Optional | 25 | Output video framerate |

*Required for model construction. The `--ckpt_path` loads trained weights on top.

---

## Troubleshooting

**"857 missing keys" when loading checkpoint:**
The model uses a `_core.` prefix in state dict keys (FSDP2 wrapper). The script
auto-detects this and adds the prefix. If you see `0 missing, 0 unexpected` in
the log, loading is correct. If you see many missing keys, check checkpoint format.

**VAE dtype error (BFloat16 vs float):**
The VAE requires float32 input. The script handles this automatically — VAE
encode/decode always uses float32 internally.

**Wav2Vec SDPA error:**
The script loads Wav2Vec with `attn_implementation="eager"` to avoid SDPA
incompatibility with `output_attentions`.

**Face detection fails (LatentSync):**
Ensure `onnxruntime-gpu` is installed (not `onnxruntime`). InsightFace models
are downloaded automatically on first run to `checkpoints/auxiliary/`.
