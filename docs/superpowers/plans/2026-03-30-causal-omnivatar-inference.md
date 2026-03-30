# Causal OmniAvatar Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone inference script for the 1.3B CausalOmniAvatarWan student model that supports variable-length block-wise AR generation with audio conditioning.

**Architecture:** Single script that loads models once, preprocesses raw video/audio inputs, runs block-wise AR inference with KV cache (adapted from Self-Forcing's pattern using OmniAvatar's `_forward_ar`), and saves output video with audio. All per-sample functions take explicit parameters for easy batch loop extension.

**Tech Stack:** PyTorch, FastGen (CausalOmniAvatarWan, RFNoiseSchedule), OmniAvatar (WanVideoVAE, Wav2VecModel, T5), ffmpeg, imageio, librosa

**Spec:** `docs/superpowers/specs/2026-03-30-causal-omnivatar-inference-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `FastGen/scripts/inference/inference_causal.py` | **Create.** Main inference script — CLI args, model loading, preprocessing, AR inference loop, post-processing. Single self-contained file. |

---

### Task 1: Script Skeleton — CLI Args, Imports, Main

**Files:**
- Create: `FastGen/scripts/inference/inference_causal.py`

- [ ] **Step 1: Create the script with imports and sys.path setup**

```python
#!/usr/bin/env python3
"""Causal OmniAvatar inference — block-wise AR generation with audio conditioning.

Generates lip-synced video from a reference video and audio using the 1.3B
CausalOmniAvatarWan student model trained via Self-Forcing.

Usage:
    python scripts/inference/inference_causal.py \
        --video_path /path/to/reference.mp4 \
        --output_path /path/to/output.mp4 \
        --ckpt_path /path/to/sf_trained_student.pth \
        --vae_path /path/to/Wan2.1_VAE.pth \
        --wav2vec_path /path/to/wav2vec2-base-960h \
        --mask_path /path/to/mask.png
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F

# Add FastGen and OmniAvatar to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTGEN_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, FASTGEN_ROOT)

OMNIAVATAR_ROOT = os.getenv(
    "OMNIAVATAR_ROOT",
    os.path.abspath(os.path.join(FASTGEN_ROOT, "..", "reference_FastGen_OmniAvatar", "OmniAvatar-Train")),
)
sys.path.insert(0, OMNIAVATAR_ROOT)
```

- [ ] **Step 2: Add CLI argument parser**

```python
def parse_args():
    p = argparse.ArgumentParser(description="Causal OmniAvatar inference")

    # Required
    p.add_argument("--video_path", required=True, help="Reference video (must be 512x512)")
    p.add_argument("--output_path", required=True, help="Output video path")
    p.add_argument("--ckpt_path", required=True, help="SF-trained student checkpoint (.pth)")
    p.add_argument("--vae_path", required=True, help="Path to Wan2.1_VAE.pth")
    p.add_argument("--wav2vec_path", required=True, help="Path to wav2vec2-base-960h directory")
    p.add_argument("--mask_path", required=True, help="Path to LatentSync mask.png")

    # Model construction
    p.add_argument("--base_model_paths", default=None,
                   help="Comma-separated safetensor paths for base Wan 2.1 T2V 1.3B weights. "
                        "Required for model construction.")
    p.add_argument("--omniavatar_ckpt_path", default=None,
                   help="OmniAvatar LoRA+audio checkpoint (.pt) for model construction.")

    # Audio / length control
    p.add_argument("--audio_path", default=None, help="Separate audio source (WAV/MP4)")
    p.add_argument("--num_latent_frames", type=int, default=None,
                   help="Override generation length (must be multiple of chunk_size=3). "
                        "If omitted, derived from audio duration.")

    # Text conditioning
    p.add_argument("--prompt", default="a person talking", help="Text prompt for T5")
    p.add_argument("--text_embeds_path", default=None,
                   help="Pre-computed T5 embeddings (.pt), skips T5 encoding")
    p.add_argument("--text_encoder_path", default=None,
                   help="Path to T5 text encoder .pth (for runtime encoding)")

    # Inference config
    p.add_argument("--t_list", type=float, nargs="+",
                   default=[0.999, 0.900, 0.750, 0.500, 0.0],
                   help="Denoising timestep schedule")
    p.add_argument("--local_attn_size", type=int, default=-1,
                   help="Rolling window size in frames (-1 = global attention)")
    p.add_argument("--chunk_size", type=int, default=3,
                   help="Frames per AR block")
    p.add_argument("--context_noise", type=float, default=0.0,
                   help="Noise level for context cache updates (0 = clean)")

    # Runtime
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--fps", type=int, default=25, help="Output video FPS")

    return p.parse_args()
```

- [ ] **Step 3: Add main function skeleton**

```python
def main():
    args = parse_args()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    torch.manual_seed(args.seed)

    # --- 1. Load models ---
    print("[1/5] Loading models...")
    model = load_diffusion_model(args, device, dtype)
    vae = load_vae(args.vae_path, device)
    wav2vec_model, wav2vec_extractor = load_wav2vec(args.wav2vec_path, device)
    text_embeds = load_or_encode_text(args, device, dtype)

    # --- 2. Preprocess inputs ---
    print("[2/5] Preprocessing inputs...")
    audio_path, tmp_audio = resolve_audio(args)
    num_latent_frames, num_video_frames = compute_generation_length(
        audio_path, args.num_latent_frames, args.chunk_size, args.fps,
    )
    print(f"  Generating {num_latent_frames} latent frames ({num_video_frames} video frames)")

    video_frames = load_and_adjust_video(args.video_path, num_video_frames)
    condition = build_condition(
        vae, wav2vec_model, wav2vec_extractor,
        video_frames, audio_path, text_embeds,
        args.mask_path, num_video_frames, num_latent_frames,
        device, dtype,
    )

    # --- 3. Run inference ---
    print("[3/5] Running AR inference...")
    output_latents = run_inference(
        model, condition, num_latent_frames,
        args.t_list, args.chunk_size, args.context_noise,
        args.seed, device, dtype,
    )

    # --- 4. Decode and save ---
    print("[4/5] VAE decoding and saving...")
    decode_and_save(vae, output_latents, audio_path, args.output_path, args.fps, device)

    # --- 5. Cleanup ---
    if tmp_audio and os.path.exists(tmp_audio):
        os.remove(tmp_audio)
    print(f"[5/5] Done! Output saved to {args.output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify script parses args without errors**

Run: `cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/FastGen && python scripts/inference/inference_causal.py --help`

Expected: Help text printed, no import errors (model loading functions don't exist yet, but `--help` exits before calling them).

- [ ] **Step 5: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat: add inference_causal.py skeleton with CLI args"
```

---

### Task 2: Model Loading Functions

**Files:**
- Modify: `FastGen/scripts/inference/inference_causal.py`

- [ ] **Step 1: Implement load_diffusion_model**

Add after the imports section:

```python
def load_diffusion_model(args, device, dtype):
    """Construct CausalOmniAvatarWan and load SF-trained checkpoint."""
    from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

    print("  Loading 1.3B CausalOmniAvatarWan...")
    model = CausalOmniAvatarWan(
        model_size="1.3B",
        in_dim=65,  # 16 noise + 16 ref + 1 mask + 16 masked_video + 16 ref_sequence
        mode="v2v",
        use_audio=True,
        audio_hidden_size=32,
        chunk_size=args.chunk_size,
        total_num_frames=21,  # Will be updated per-video in run_inference
        base_model_paths=args.base_model_paths,
        omniavatar_ckpt_path=args.omniavatar_ckpt_path,
        merge_lora=True,
        lora_rank=128,
        lora_alpha=64,
        net_pred_type="flow",
        schedule_type="rf",
        mask_all_frames=True,
        dtype=args.dtype,
        local_attn_size=args.local_attn_size,
    )

    # Load SF-trained checkpoint on top of base + LoRA weights
    print(f"  Loading SF checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("net", ckpt)  # Handle both {"net": sd} and bare sd formats
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (expected for partial checkpoints)")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model
```

- [ ] **Step 2: Implement load_vae**

```python
def load_vae(vae_path, device):
    """Load WanVideoVAE from checkpoint."""
    from OmniAvatar.models.wan_video_vae import WanVideoVAE

    print(f"  Loading VAE from {vae_path}")
    vae = WanVideoVAE(z_dim=16)
    vae_sd = torch.load(vae_path, map_location="cpu", weights_only=False)
    # Handle both nested and flat state dict formats
    if any(k.startswith("model.") for k in vae_sd.keys()):
        vae.load_state_dict(vae_sd, strict=True)
    else:
        vae.model.load_state_dict(vae_sd, strict=True)
    vae = vae.to(device)
    vae.eval()
    return vae
```

- [ ] **Step 3: Implement load_wav2vec**

```python
def load_wav2vec(wav2vec_path, device):
    """Load Wav2Vec2 model and feature extractor."""
    from transformers import Wav2Vec2FeatureExtractor
    from OmniAvatar.models.wav2vec import Wav2VecModel

    print(f"  Loading Wav2Vec from {wav2vec_path}")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path)
    model = Wav2VecModel.from_pretrained(wav2vec_path, local_files_only=True).to(device)
    model.feature_extractor._freeze_parameters()
    model.eval()
    return model, extractor
```

- [ ] **Step 4: Implement load_or_encode_text**

```python
def load_or_encode_text(args, device, dtype):
    """Load pre-computed text embeddings or encode prompt via T5."""
    if args.text_embeds_path is not None:
        print(f"  Loading text embeddings from {args.text_embeds_path}")
        text_embeds = torch.load(args.text_embeds_path, map_location="cpu", weights_only=False)
        if isinstance(text_embeds, dict):
            text_embeds = next(v for v in text_embeds.values() if isinstance(v, torch.Tensor))
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0)
        return text_embeds.to(device=device, dtype=dtype)

    if args.text_encoder_path is not None:
        print(f"  Loading T5 encoder from {args.text_encoder_path}")
        from OmniAvatar.models.wan_video_text_encoder import WanTextEncoder
        text_encoder = WanTextEncoder(
            dtype=dtype, device=device, model_path=args.text_encoder_path
        )
        text_embeds = text_encoder([args.prompt])
        del text_encoder
        torch.cuda.empty_cache()
        return text_embeds.to(device=device, dtype=dtype)

    raise ValueError(
        "Must provide either --text_embeds_path (pre-computed T5 embeddings) "
        "or --text_encoder_path (T5 model path for runtime encoding)"
    )
```

- [ ] **Step 5: Verify model loading compiles**

Run: `cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/FastGen && python -c "import scripts.inference.inference_causal"`

Expected: No import errors. (Actual model loading requires checkpoints, tested in Task 6.)

- [ ] **Step 6: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat: add model loading functions for inference_causal.py"
```

---

### Task 3: Input Preprocessing Functions

**Files:**
- Modify: `FastGen/scripts/inference/inference_causal.py`

- [ ] **Step 1: Implement audio extraction and length calculation**

```python
def resolve_audio(args):
    """Return (audio_path, tmp_path_or_None). Extract from video if needed."""
    if args.audio_path is not None:
        return args.audio_path, None
    # Extract audio from video to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-i", args.video_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")
    return tmp.name, tmp.name


def get_audio_duration(audio_path):
    """Get audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def compute_generation_length(audio_path, override_frames, chunk_size, fps):
    """Compute (num_latent_frames, num_video_frames) from audio duration.

    VAE temporal compression: first frame 1:1, then groups of 4.
    num_video_frames = 1 + (num_latent_frames - 1) * 4
    num_latent_frames = 1 + (num_video_frames - 1) // 4
    """
    audio_duration = get_audio_duration(audio_path)
    num_video_frames_raw = int(audio_duration * fps)  # floor
    num_latent_frames_raw = 1 + (num_video_frames_raw - 1) // 4

    if override_frames is not None:
        num_latent_frames = override_frames
        assert num_latent_frames % chunk_size == 0, (
            f"--num_latent_frames ({num_latent_frames}) must be a multiple of chunk_size ({chunk_size})"
        )
        assert num_latent_frames <= num_latent_frames_raw, (
            f"--num_latent_frames ({num_latent_frames}) exceeds audio-derived max ({num_latent_frames_raw})"
        )
    else:
        # Round DOWN to nearest multiple of chunk_size
        num_latent_frames = (num_latent_frames_raw // chunk_size) * chunk_size

    assert num_latent_frames >= chunk_size, (
        f"Audio too short: only {num_latent_frames} latent frames (need >= {chunk_size})"
    )

    num_video_frames = 1 + (num_latent_frames - 1) * 4
    return num_latent_frames, num_video_frames
```

- [ ] **Step 2: Implement video loading and length adjustment**

```python
def load_video_frames(video_path, max_frames=None):
    """Load video frames as [N, H, W, 3] uint8 numpy array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from {video_path}")

    frames_np = np.stack(frames, axis=0)
    H, W = frames_np.shape[1], frames_np.shape[2]
    if H != 512 or W != 512:
        raise ValueError(
            f"Video must be 512x512, got {W}x{H}. "
            "Resize or use LatentSync compositing pipeline."
        )
    return frames_np


def adjust_video_length(frames_np, target_num_frames):
    """Extend (ping-pong) or clip video to target_num_frames.

    Args:
        frames_np: [N, H, W, 3] uint8 numpy array
        target_num_frames: desired number of video frames

    Returns:
        [target_num_frames, H, W, 3] uint8 numpy array
    """
    n = len(frames_np)
    if n >= target_num_frames:
        return frames_np[:target_num_frames]

    # Ping-pong extend: forward, reverse, forward, ...
    extended = list(frames_np)
    forward = True
    idx = n - 1
    while len(extended) < target_num_frames:
        if forward:
            idx -= 1
            if idx < 0:
                idx = 1
                forward = False
                # If only 1 frame, just repeat it
                if n == 1:
                    idx = 0
        else:
            idx += 1
            if idx >= n:
                idx = n - 2
                forward = True
                if n == 1:
                    idx = 0
        extended.append(frames_np[idx])

    return np.stack(extended[:target_num_frames], axis=0)


def load_and_adjust_video(video_path, num_video_frames):
    """Load video and adjust to exact frame count."""
    frames = load_video_frames(video_path)
    frames = adjust_video_length(frames, num_video_frames)
    return frames
```

- [ ] **Step 3: Implement reference video encoding and condition building**

```python
def frames_to_tensor(frames_np):
    """Convert [N, H, W, 3] uint8 numpy → [1, 3, N, H, W] float [-1, 1] tensor."""
    t = torch.from_numpy(frames_np).float() / 255.0  # [N, H, W, 3] in [0, 1]
    t = t.permute(0, 3, 1, 2)  # [N, 3, H, W]
    t = t * 2.0 - 1.0  # [-1, 1]
    return t.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, N, H, W]


def load_latentsync_mask(mask_path, latent_h, latent_w):
    """Load LatentSync mask → [H_lat, W_lat] float tensor (1=keep, 0=generate)."""
    from PIL import Image
    mask_img = Image.open(mask_path).convert("L")
    mask_arr = np.array(mask_img).astype(np.float32) / 255.0
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    mask_resized = F.interpolate(
        mask_t, size=(latent_h, latent_w), mode="bilinear", align_corners=False
    )
    return (mask_resized > 0.5).float().squeeze(0).squeeze(0)  # [H_lat, W_lat]


def apply_spatial_mask(video_tensor, mask_np, mask_all_frames=True):
    """Apply LatentSync spatial mask in [-1,1] space. Masked region → 0.0.

    Args:
        video_tensor: [1, 3, N, H, W] float in [-1, 1]
        mask_np: [H, W] float, 1=keep, 0=mask
    Returns:
        [1, 3, N, H, W] float in [-1, 1] with mouth region zeroed
    """
    mask_t = torch.from_numpy(mask_np).float()[None, None, None, :, :]  # [1, 1, 1, H, W]
    masked = video_tensor.clone()
    if mask_all_frames:
        masked *= mask_t
    else:
        masked[:, :, 1:, :, :] *= mask_t
    return masked


@torch.no_grad()
def encode_reference_video(vae, video_frames_np, mask_path, device, dtype):
    """VAE-encode reference video → (ref_latent, masked_latents, ref_sequence, mask).

    Returns:
        ref_latent: [1, 16, 1, H_lat, W_lat] — first frame latent
        masked_latents: [1, 16, T_lat, H_lat, W_lat] — mouth-masked video latents
        ref_sequence: [1, 16, T_lat, H_lat, W_lat] — full video latents
        mask: [H_lat, W_lat] — LatentSync mask at latent resolution
    """
    video_tensor = frames_to_tensor(video_frames_np)  # [1, 3, N, H, W]

    # Load pixel-space mask for masking before VAE encode
    H, W = video_frames_np.shape[1], video_frames_np.shape[2]
    from PIL import Image
    mask_img = Image.open(mask_path).convert("L")
    mask_pixel = np.array(mask_img.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
    mask_pixel_binary = (mask_pixel > 0.5).astype(np.float32)

    masked_video_tensor = apply_spatial_mask(video_tensor, mask_pixel_binary, mask_all_frames=True)

    # VAE encode both versions — vae.encode expects list of [C, T, H, W] tensors
    video_for_vae = video_tensor[0].to(device=device, dtype=dtype)  # [3, N, H, W]
    masked_for_vae = masked_video_tensor[0].to(device=device, dtype=dtype)

    ref_sequence = vae.encode([video_for_vae], device=device)  # [1, 16, T_lat, H_lat, W_lat]
    masked_latents = vae.encode([masked_for_vae], device=device)

    ref_latent = ref_sequence[:, :, :1, :, :]  # [1, 16, 1, H_lat, W_lat]

    # Latent-space mask
    H_lat, W_lat = ref_sequence.shape[3], ref_sequence.shape[4]
    mask = load_latentsync_mask(mask_path, H_lat, W_lat).to(device)

    return (
        ref_latent.to(dtype=dtype),
        masked_latents.to(dtype=dtype),
        ref_sequence.to(dtype=dtype),
        mask.to(dtype=dtype),
    )


@torch.no_grad()
def encode_audio(wav2vec_model, wav2vec_extractor, audio_path, num_video_frames, device):
    """Encode audio → [1, num_video_frames, 10752] float tensor.

    Follows OmniAvatar's exact encoding pipeline:
    1. Load at 16kHz
    2. Encode at full audio length
    3. Slice to num_video_frames
    """
    WAV2VEC_SR = 16000
    FPS = 25

    audio, sr = librosa.load(audio_path, sr=WAV2VEC_SR)
    input_values = np.squeeze(
        wav2vec_extractor(audio, sampling_rate=WAV2VEC_SR).input_values
    )
    input_values = torch.from_numpy(input_values).float().to(device).unsqueeze(0)

    # Compute natural frame count for full audio
    samples_per_frame = WAV2VEC_SR // FPS  # 640
    total_audio_frames = math.ceil(input_values.shape[1] / samples_per_frame)
    total_audio_frames = max(total_audio_frames, num_video_frames)

    # Pad to align
    target_samples = total_audio_frames * samples_per_frame
    if input_values.shape[1] < target_samples:
        input_values = F.pad(input_values, (0, target_samples - input_values.shape[1]))

    # Encode — concatenate all 14 hidden states → 768 * 14 = 10752 dim
    hidden_states = wav2vec_model(
        input_values, seq_len=total_audio_frames, output_hidden_states=True
    )
    audio_emb = hidden_states.last_hidden_state
    for hs in hidden_states.hidden_states:
        audio_emb = torch.cat((audio_emb, hs), dim=-1)
    # audio_emb: [1, total_audio_frames, 10752]

    # Slice to target frame count
    audio_emb = audio_emb[:, :num_video_frames, :]
    return audio_emb  # [1, num_video_frames, 10752]


def build_condition(
    vae, wav2vec_model, wav2vec_extractor,
    video_frames_np, audio_path, text_embeds,
    mask_path, num_video_frames, num_latent_frames,
    device, dtype,
):
    """Build the full conditioning dict for the model.

    Returns dict with keys:
        text_embeds: [1, 512, 4096]
        audio_emb: [1, num_video_frames, 10752]
        ref_latent: [1, 16, 1, H_lat, W_lat]
        mask: [H_lat, W_lat]
        masked_video: [1, 16, T_lat, H_lat, W_lat]
        ref_sequence: [1, 16, T_lat, H_lat, W_lat]
    """
    ref_latent, masked_latents, ref_sequence, mask = encode_reference_video(
        vae, video_frames_np, mask_path, device, dtype,
    )
    audio_emb = encode_audio(
        wav2vec_model, wav2vec_extractor, audio_path, num_video_frames, device,
    )

    return {
        "text_embeds": text_embeds,
        "audio_emb": audio_emb.to(dtype=dtype),
        "ref_latent": ref_latent,
        "mask": mask,
        "masked_video": masked_latents,
        "ref_sequence": ref_sequence,
    }
```

- [ ] **Step 4: Verify preprocessing functions**

Run a quick test of the utility functions that don't need GPU:

```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/FastGen
python -c "
from scripts.inference.inference_causal import (
    compute_generation_length, adjust_video_length
)
import numpy as np

# Test length computation (mock audio duration)
# 3.24s at 25fps = 81 frames -> latent = 1 + 80//4 = 21 -> round to 21 (21%3==0)
# We need to test with a real file or mock get_audio_duration

# Test video extension
frames = np.zeros((10, 512, 512, 3), dtype=np.uint8)
extended = adjust_video_length(frames, 25)
assert extended.shape == (25, 512, 512, 3), f'Got {extended.shape}'
print('adjust_video_length: OK')

clipped = adjust_video_length(frames, 5)
assert clipped.shape == (5, 512, 512, 3), f'Got {clipped.shape}'
print('clip: OK')

print('All preprocessing tests passed!')
"
```

Expected: All assertions pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat: add input preprocessing for inference_causal.py"
```

---

### Task 4: Core AR Inference Loop

**Files:**
- Modify: `FastGen/scripts/inference/inference_causal.py`

- [ ] **Step 1: Implement run_inference**

```python
@torch.no_grad()
def run_inference(
    model, condition, num_latent_frames, t_list,
    chunk_size, context_noise, seed, device, dtype,
):
    """Block-wise AR inference loop.

    Adapted from Self-Forcing's rollout_with_gradient but inference-only:
    - No gradients, no random exit steps
    - Full denoising per block (all steps in t_list)
    - KV cache updated after each block with denoised output
    - Rolling window eviction handled internally by CausalSelfAttention

    Args:
        model: CausalOmniAvatarWan (1.3B student)
        condition: dict with text_embeds, audio_emb, ref_latent, mask, etc.
        num_latent_frames: total latent frames to generate
        t_list: denoising timestep schedule (e.g. [0.999, 0.9, 0.75, 0.5, 0.0])
        chunk_size: frames per AR block (3)
        context_noise: noise level for cache updates (0 = clean)
        seed: random seed
        device: torch device
        dtype: torch dtype

    Returns:
        output: [1, 16, num_latent_frames, H_lat, W_lat] denoised latents
    """
    # Update model's total_num_frames for correct cache allocation
    model.total_num_frames = num_latent_frames
    model.clear_caches()

    # Determine spatial dims from ref_latent
    ref_latent = condition["ref_latent"]  # [1, 16, 1, H_lat, W_lat]
    B = ref_latent.shape[0]
    C = 16
    H_lat, W_lat = ref_latent.shape[3], ref_latent.shape[4]

    num_blocks = num_latent_frames // chunk_size
    assert num_latent_frames % chunk_size == 0, (
        f"num_latent_frames ({num_latent_frames}) must be divisible by chunk_size ({chunk_size})"
    )

    # Generate noise
    torch.manual_seed(seed)
    noise = torch.randn(B, C, num_latent_frames, H_lat, W_lat, device=device, dtype=dtype)

    # Convert t_list to tensor
    t_list_t = torch.tensor(t_list, device=device, dtype=torch.float64)

    # Output accumulator
    output = torch.zeros_like(noise)

    print(f"  {num_blocks} blocks x {len(t_list) - 1} denoising steps")
    for block_idx in range(num_blocks):
        cur_start_frame = block_idx * chunk_size

        # Slice noise for this chunk
        noisy_input = noise[:, :, cur_start_frame:cur_start_frame + chunk_size]

        # Multi-step denoising
        for step_idx in range(len(t_list_t) - 1):
            t_cur = t_list_t[step_idx]
            t_next = t_list_t[step_idx + 1]

            # Forward pass — model.forward() handles _build_y, rescale_t, _forward_ar
            x0_pred = model(
                noisy_input,
                t_cur.float().expand(B),
                condition=condition,
                cur_start_frame=cur_start_frame,
                store_kv=False,
                is_ar=True,
                fwd_pred_type="x0",
                use_gradient_checkpointing=False,
            )

            if t_next > 0:
                # Add noise for next step (SDE: fresh random noise)
                eps = torch.randn_like(x0_pred)
                noisy_input = model.noise_scheduler.forward_process(
                    x0_pred, eps, t_next.float().expand(B),
                )
            else:
                # Final step — clean output
                noisy_input = x0_pred

        # Store denoised chunk
        output[:, :, cur_start_frame:cur_start_frame + chunk_size] = x0_pred

        # Update KV cache with denoised output (context for next block)
        cache_input = x0_pred
        t_cache = torch.full((B,), context_noise, device=device, dtype=dtype)
        if context_noise > 0:
            cache_eps = torch.randn_like(x0_pred)
            cache_input = model.noise_scheduler.forward_process(
                x0_pred, cache_eps,
                torch.tensor(context_noise, device=device, dtype=torch.float64).expand(B),
            )

        model(
            cache_input,
            t_cache,
            condition=condition,
            cur_start_frame=cur_start_frame,
            store_kv=True,
            is_ar=True,
            fwd_pred_type="x0",
            use_gradient_checkpointing=False,
        )

        if (block_idx + 1) % 10 == 0 or block_idx == num_blocks - 1:
            print(f"  Block {block_idx + 1}/{num_blocks} done")

    model.clear_caches()
    return output
```

- [ ] **Step 2: Add KV cache verification helper (for debugging)**

```python
def verify_kv_cache(model, chunk_size, frame_seqlen, block_idx):
    """Debug helper — verify KV cache state after a store_kv=True call."""
    if model._kv_caches is None:
        print("  [WARN] KV caches are None!")
        return
    cache_0 = model._kv_caches[0]
    expected_global = (block_idx + 1) * chunk_size * frame_seqlen
    actual_global = cache_0["global_end_index"].item()
    actual_local = cache_0["local_end_index"].item()
    match = "OK" if actual_global == expected_global else "MISMATCH"
    print(
        f"  [Cache] block={block_idx} global_end={actual_global} "
        f"(expected={expected_global}) local_end={actual_local} [{match}]"
    )
```

- [ ] **Step 3: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat: add AR inference loop for inference_causal.py"
```

---

### Task 5: Post-processing — VAE Decode, Save, Audio Mux

**Files:**
- Modify: `FastGen/scripts/inference/inference_causal.py`

- [ ] **Step 1: Implement decode_and_save**

```python
@torch.no_grad()
def decode_and_save(vae, output_latents, audio_path, output_path, fps, device):
    """VAE decode latents → save silent video → mux with audio.

    Args:
        vae: WanVideoVAE
        output_latents: [1, 16, T_lat, H_lat, W_lat]
        audio_path: path to audio file
        output_path: final output video path
        fps: video framerate (25)
        device: torch device
    """
    import imageio.v3 as iio

    # VAE decode — expects list of [C, T_lat, H_lat, W_lat] tensors
    latent_for_vae = output_latents[0]  # [16, T_lat, H_lat, W_lat]
    video_tensor = vae.decode([latent_for_vae], device=device)  # [1, 3, T_video, H, W]
    video_tensor = video_tensor.clamp(-1, 1)

    # Convert to uint8 frames: [T, H, W, 3]
    video_np = video_tensor[0]  # [3, T, H, W]
    video_np = video_np.permute(1, 2, 3, 0)  # [T, H, W, 3]
    video_np = ((video_np.float() + 1) * 127.5).clamp(0, 255).cpu().to(torch.uint8).numpy()

    # Save silent video to temp file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_silent = output_path + ".silent.mp4"
    iio.imwrite(
        tmp_silent,
        video_np,
        fps=fps,
        codec="libx264",
        output_params=["-loglevel", "quiet", "-crf", "18"],
    )
    print(f"  Silent video: {video_np.shape[0]} frames at {fps}fps")

    # Mux with audio
    video_duration = video_np.shape[0] / fps
    mux_video_with_audio(tmp_silent, audio_path, output_path, duration_s=video_duration)

    # Cleanup
    if os.path.exists(tmp_silent):
        os.remove(tmp_silent)


def mux_video_with_audio(video_path, audio_path, output_path, duration_s=None):
    """Mux silent video with audio via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-i", video_path, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "aac", "-q:v", "0", "-q:a", "0",
    ]
    if duration_s is not None:
        cmd.extend(["-t", f"{duration_s:.4f}"])
    cmd.append(output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {result.stderr}")
```

- [ ] **Step 2: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat: add VAE decode and video saving for inference_causal.py"
```

---

### Task 6: Integration Verification

**Files:**
- Modify: `FastGen/scripts/inference/inference_causal.py` (minor debug additions if needed)

- [ ] **Step 1: Verify end-to-end execution with test data**

Run with actual model weights and a test video. This requires setting up paths for your environment:

```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/FastGen

# Set paths (adjust for your environment)
OMNIAVATAR_ROOT="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_OmniAvatar/OmniAvatar-Train"
PRETRAINED="$OMNIAVATAR_ROOT/pretrained_models"
MASK_PATH="$OMNIAVATAR_ROOT/OmniAvatar/utils/latentsync/mask.png"
BASE_WEIGHTS="$PRETRAINED/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
STUDENT_CKPT="$PRETRAINED/step-1000.pt"
VAE_PATH="$PRETRAINED/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
WAV2VEC="$PRETRAINED/wav2vec2-base-960h"
# Use a precomputed text embedding (avoids loading T5)
TEXT_EMB="/path/to/common_prompt.pt"

# Pick a test video from the datasets
TEST_VIDEO="/data/karlo-research_715/workspace/kinemaar/datasets/sample_hallo3_latentsync/some_video/sub_clip.mp4"

python scripts/inference/inference_causal.py \
    --video_path "$TEST_VIDEO" \
    --output_path /tmp/test_inference_causal.mp4 \
    --ckpt_path "$STUDENT_CKPT" \
    --vae_path "$VAE_PATH" \
    --wav2vec_path "$WAV2VEC" \
    --mask_path "$MASK_PATH" \
    --base_model_paths "$BASE_WEIGHTS" \
    --omniavatar_ckpt_path "$STUDENT_CKPT" \
    --text_embeds_path "$TEXT_EMB" \
    --num_latent_frames 21 \
    --seed 42
```

Expected: Script runs to completion, generates `/tmp/test_inference_causal.mp4` with audio.

- [ ] **Step 2: Verify KV cache behavior**

Add `verify_kv_cache` calls inside `run_inference` (after each `store_kv=True` call) for the first 3 blocks, then remove. Check that:

1. `global_end_index` advances by `chunk_size * frame_seqlen` (= 3 * 1024 = 3072) per block
2. `local_end_index` matches `global_end_index` when `local_attn_size=-1`
3. Audio is cached once (`model._cached_audio is not None` after first block)

- [ ] **Step 3: Verify variable-length generation**

Run without `--num_latent_frames` to let the script derive length from audio:

```bash
python scripts/inference/inference_causal.py \
    --video_path "$TEST_VIDEO" \
    --output_path /tmp/test_variable_length.mp4 \
    --ckpt_path "$STUDENT_CKPT" \
    --vae_path "$VAE_PATH" \
    --wav2vec_path "$WAV2VEC" \
    --mask_path "$MASK_PATH" \
    --base_model_paths "$BASE_WEIGHTS" \
    --omniavatar_ckpt_path "$STUDENT_CKPT" \
    --text_embeds_path "$TEXT_EMB"
```

Verify the output video duration matches the input audio duration (within 1 frame).

- [ ] **Step 4: Verify rolling window (if local_attn_size > 0)**

```bash
python scripts/inference/inference_causal.py \
    --video_path "$TEST_VIDEO" \
    --output_path /tmp/test_rolling.mp4 \
    --ckpt_path "$STUDENT_CKPT" \
    --vae_path "$VAE_PATH" \
    --wav2vec_path "$WAV2VEC" \
    --mask_path "$MASK_PATH" \
    --base_model_paths "$BASE_WEIGHTS" \
    --omniavatar_ckpt_path "$STUDENT_CKPT" \
    --text_embeds_path "$TEXT_EMB" \
    --local_attn_size 7 \
    --num_latent_frames 21
```

Verify: no OOM, KV cache size stays bounded at `7 * 1024 = 7168` tokens per block.

- [ ] **Step 5: Fix any issues found during verification and commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat: verified inference_causal.py end-to-end"
```

---

## Verification Checklist Summary

| Check | How | Expected |
|-------|-----|----------|
| KV cache indices advance correctly | `verify_kv_cache` after each `store_kv=True` | global_end increments by 3072/block |
| `current_start` units correct | Print `cur_start_frame * frame_seqlen` vs model internals | Matches post-patchification token offset |
| Rolling eviction works | Run with `--local_attn_size 7`, check `local_end_index` stays bounded | Never exceeds `7 * 1024` |
| Audio cached and sliced | Check `model._cached_audio is not None`, verify shape per chunk | Correct temporal slice per block |
| Context cache at t=0 | Compare cache state after t=0 update vs training's pattern | Should produce same cache as training |
| Output video has audio | Play output file | Audio synced with lip movement |
| Variable length matches audio | Compare output duration vs input audio duration | Within 1 frame |
