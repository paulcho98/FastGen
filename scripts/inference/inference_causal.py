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
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTGEN_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, FASTGEN_ROOT)

OMNIAVATAR_ROOT = os.getenv(
    "OMNIAVATAR_ROOT",
    os.path.abspath(os.path.join(FASTGEN_ROOT, "..", "OmniAvatar-Train")),
)
sys.path.insert(0, OMNIAVATAR_ROOT)


def _get_ffmpeg():
    """Return path to ffmpeg binary (system or imageio_ffmpeg fallback)."""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or pip install imageio-ffmpeg.")


# ===========================================================================
# CLI argument parsing
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Causal OmniAvatar inference (block-wise AR with audio conditioning)"
    )

    # --- Required ---
    parser.add_argument("--video_path", type=str, required=True,
                        help="Reference video path (must be 512x512)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output video path")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="SF-trained student checkpoint (.pth)")
    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path to Wan2.1_VAE.pth")
    parser.add_argument("--wav2vec_path", type=str, required=True,
                        help="Path to wav2vec2-base-960h directory")
    parser.add_argument("--mask_path", type=str, required=True,
                        help="Path to LatentSync mask.png")

    # --- Optional model paths ---
    parser.add_argument("--base_model_paths", type=str, default=None,
                        help="Comma-separated safetensor paths for base Wan 2.1 T2V 1.3B")
    parser.add_argument("--omniavatar_ckpt_path", type=str, default=None,
                        help="OmniAvatar LoRA+audio checkpoint")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Separate audio source (extracted from video if not provided)")

    # --- Generation parameters ---
    parser.add_argument("--num_latent_frames", type=int, default=None,
                        help="Override generation length (must be multiple of chunk_size)")
    parser.add_argument("--prompt", type=str, default="a person talking",
                        help="Text prompt")
    parser.add_argument("--text_embeds_path", type=str, default=None,
                        help="Pre-computed T5 embeddings .pt file")
    parser.add_argument("--text_encoder_path", type=str, default=None,
                        help="T5 model path for runtime encoding")
    parser.add_argument("--t_list", type=float, nargs="+",
                        default=[0.999, 0.900, 0.750, 0.500, 0.0],
                        help="Noise schedule timestep list for AR generation")
    parser.add_argument("--local_attn_size", type=int, default=-1,
                        help="Rolling local attention window in frames (-1 = full)")
    parser.add_argument("--chunk_size", type=int, default=3,
                        help="Number of latent frames per AR chunk")
    parser.add_argument("--context_noise", type=float, default=0.0,
                        help="Noise added to context frames during AR generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Model dtype")
    parser.add_argument("--fps", type=int, default=25,
                        help="Output video FPS")

    return parser.parse_args()


# ===========================================================================
# Model loading functions
# ===========================================================================

def load_diffusion_model(args, device, dtype):
    """Load the CausalOmniAvatarWan student model.

    Constructs the model, loads base Wan + OmniAvatar weights (if paths given),
    then overlays the Self-Forcing trained checkpoint on top.

    Returns:
        CausalOmniAvatarWan model in eval mode on *device*.
    """
    from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

    model = CausalOmniAvatarWan(
        model_size="1.3B",
        in_dim=65,
        mode="v2v",
        use_audio=True,
        audio_hidden_size=32,
        chunk_size=args.chunk_size,
        total_num_frames=21,
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

    # Load Self-Forcing checkpoint on top
    print(f"Loading SF checkpoint from {args.ckpt_path} ...")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "net" in ckpt:
        ckpt = ckpt["net"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"  SF checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def load_vae(vae_path, device):
    """Load the Wan 2.1 Video VAE.

    Returns:
        WanVideoVAE instance in eval mode on *device*.
    """
    from OmniAvatar.models.wan_video_vae import WanVideoVAE

    vae = WanVideoVAE(z_dim=16)

    print(f"Loading VAE from {vae_path} ...")
    state_dict = torch.load(vae_path, map_location="cpu", weights_only=False)

    # Handle both 'model.xxx' prefixed and flat key formats
    if any(k.startswith("model.") for k in state_dict):
        # Already has model. prefix — load directly into WanVideoVAE
        vae.load_state_dict(state_dict, strict=True)
    elif "model_state" in state_dict:
        # CivitAI format: state_dict['model_state'] with flat keys
        converter = WanVideoVAE.state_dict_converter()
        converted = converter.from_civitai(state_dict)
        vae.load_state_dict(converted, strict=True)
    else:
        # Flat keys — add 'model.' prefix
        prefixed = {"model." + k: v for k, v in state_dict.items()}
        vae.load_state_dict(prefixed, strict=True)

    vae = vae.to(device)
    vae.eval()
    return vae


def load_wav2vec(wav2vec_path, device):
    """Load wav2vec2-base-960h model and feature extractor.

    Returns:
        (wav2vec_model, wav2vec_extractor) — model in eval/float32 on *device*.
    """
    from transformers import Wav2Vec2FeatureExtractor
    from OmniAvatar.models.wav2vec import Wav2VecModel

    print(f"Loading Wav2Vec2 from {wav2vec_path} ...")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path)
    model = Wav2VecModel.from_pretrained(wav2vec_path, attn_implementation="eager")

    # Freeze feature extractor (CNN) — must stay float32
    model.feature_extractor.requires_grad_(False)
    model = model.to(device).float()
    model.eval()
    return model, extractor


def load_or_encode_text(args, device, dtype):
    """Get text embeddings — either from file or by encoding the prompt.

    Returns:
        text_embeds: [1, 512, 4096] tensor on *device* in *dtype*.
    """
    if args.text_embeds_path is not None:
        print(f"Loading text embeddings from {args.text_embeds_path} ...")
        data = torch.load(args.text_embeds_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            # Handle dict formats: {'context': tensor} or {'text_emb': tensor}
            for key in ("context", "text_emb", "prompt_emb"):
                if key in data:
                    text_embeds = data[key]
                    break
            else:
                # Take first tensor value
                text_embeds = next(iter(data.values()))
        else:
            text_embeds = data

        # Ensure shape [1, 512, 4096]
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0)
        assert text_embeds.shape == (1, 512, 4096), (
            f"Expected text_embeds shape [1, 512, 4096], got {text_embeds.shape}"
        )
        return text_embeds.to(device=device, dtype=dtype)

    elif args.text_encoder_path is not None:
        print(f"Loading T5 text encoder from {args.text_encoder_path} ...")
        from OmniAvatar.models.wan_video_text_encoder import WanTextEncoder
        from OmniAvatar.prompters.wan_prompter import WanPrompter

        # Load text encoder
        text_encoder = WanTextEncoder()
        te_state = torch.load(args.text_encoder_path, map_location="cpu", weights_only=False)
        converter = WanTextEncoder.state_dict_converter()
        te_state = converter.from_civitai(te_state)
        text_encoder.load_state_dict(te_state, strict=True)
        text_encoder = text_encoder.to(device).eval()

        # Set up prompter with tokenizer from same directory
        tokenizer_path = os.path.dirname(args.text_encoder_path)
        prompter = WanPrompter(tokenizer_path=tokenizer_path, text_len=512)
        prompter.fetch_models(text_encoder=text_encoder)

        # Encode
        with torch.no_grad():
            text_embeds = prompter.encode_prompt(
                args.prompt, positive=True, device=device
            )

        # Ensure shape [1, 512, 4096]
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0)

        # Cleanup to free VRAM
        del text_encoder, prompter
        torch.cuda.empty_cache()

        return text_embeds.to(dtype=dtype)

    else:
        raise ValueError(
            "Must provide either --text_embeds_path or --text_encoder_path "
            "to obtain text embeddings."
        )


# ===========================================================================
# Input preprocessing functions
# ===========================================================================

def resolve_audio(args):
    """Determine the audio source path.

    If --audio_path is provided, use it directly.  Otherwise extract audio
    from the reference video using ffmpeg.

    Returns:
        (audio_path, tmp_path_or_None) — tmp_path is set when a temp file
        was created and must be cleaned up later.
    """
    if args.audio_path is not None:
        return args.audio_path, None

    # Extract audio from video
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    cmd = [
        _get_ffmpeg(), "-y", "-loglevel", "error", "-nostdin",
        "-i", args.video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed:\n{result.stderr}"
        )
    print(f"Extracted audio to {tmp_path}")
    return tmp_path, tmp_path


def get_audio_duration(audio_path):
    """Get audio duration in seconds using ffprobe.

    Returns:
        float — duration in seconds.
    """
    # Use librosa instead of ffprobe (ffprobe may not be installed)
    duration = librosa.get_duration(filename=audio_path)
    return duration


def compute_generation_length(audio_path, override_frames, chunk_size, fps):
    """Compute generation length in both latent and video frames.

    The VAE temporal compression is: num_latent = 1 + (num_video - 1) // 4.
    We round DOWN num_latent to the nearest multiple of chunk_size so the AR
    loop produces complete chunks.

    Args:
        audio_path: path to audio file (for duration)
        override_frames: explicit num_latent_frames (or None)
        chunk_size: AR chunk size in latent frames
        fps: video frames per second

    Returns:
        (num_latent_frames, num_video_frames)
    """
    duration = get_audio_duration(audio_path)
    num_video_raw = int(duration * fps)  # floor
    num_latent_raw = 1 + (num_video_raw - 1) // 4

    if override_frames is not None:
        num_latent = override_frames
        if num_latent % chunk_size != 0:
            raise ValueError(
                f"--num_latent_frames ({num_latent}) must be a multiple of "
                f"chunk_size ({chunk_size})"
            )
        if num_latent > num_latent_raw:
            raise ValueError(
                f"--num_latent_frames ({num_latent}) exceeds audio-derived max "
                f"({num_latent_raw}). Audio is {duration:.2f}s."
            )
    else:
        # Round DOWN to multiple of chunk_size
        num_latent = (num_latent_raw // chunk_size) * chunk_size
        num_latent = max(num_latent, chunk_size)  # at least one chunk

    # Inverse: num_video = 1 + (num_latent - 1) * 4
    num_video = 1 + (num_latent - 1) * 4

    print(f"Generation length: {num_latent} latent frames, {num_video} video frames")
    return num_latent, num_video


def load_video_frames(video_path, max_frames=None):
    """Load video frames as [N, H, W, 3] uint8 numpy array.

    Validates that frames are 512x512.

    Args:
        video_path: path to video file
        max_frames: if set, read at most this many frames

    Returns:
        frames: [N, H, W, 3] uint8 numpy array
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    checked_size = False
    while True:
        if max_frames is not None and len(frames) >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not checked_size:
            h, w = frame.shape[:2]
            if h != 512 or w != 512:
                cap.release()
                raise ValueError(
                    f"Video must be 512x512, got {w}x{h}. "
                    "Resize or use LatentSync compositing pipeline."
                )
            checked_size = True
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Could not read any frames from {video_path}")

    return np.stack(frames, axis=0)  # [N, H, W, 3] uint8


def adjust_video_length(frames_np, target):
    """Adjust video to exactly *target* frames via ping-pong extension or clipping.

    Args:
        frames_np: [N, H, W, 3] uint8
        target: desired number of frames

    Returns:
        [target, H, W, 3] uint8
    """
    n = len(frames_np)
    if n >= target:
        return frames_np[:target]

    # Ping-pong extend: 0,1,...,n-1,n-2,...,1,0,1,...,n-1,...
    if n == 1:
        # Single frame: just repeat
        indices = [0] * target
    else:
        cycle = list(range(n)) + list(range(n - 2, 0, -1))
        indices = []
        while len(indices) < target:
            indices.extend(cycle)
        indices = indices[:target]

    return frames_np[indices]


def load_and_adjust_video(video_path, num_video_frames):
    """Load video and adjust to exactly *num_video_frames* frames.

    Returns:
        [num_video_frames, H, W, 3] uint8 numpy array.
    """
    frames = load_video_frames(video_path)
    return adjust_video_length(frames, num_video_frames)


def frames_to_tensor(frames_np):
    """Convert [N, H, W, 3] uint8 numpy → [1, 3, N, H, W] float tensor in [-1, 1].

    Matches OmniAvatar's frames_to_video_tensor exactly.
    """
    t = torch.from_numpy(frames_np).float() / 255.0  # [N, H, W, 3] in [0, 1]
    t = t.permute(0, 3, 1, 2)  # [N, 3, H, W]
    t = t * 2.0 - 1.0  # [-1, 1]
    t = t.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, N, H, W]
    return t


def load_latentsync_mask(mask_path, latent_h, latent_w):
    """Load LatentSync mask and resize to latent resolution.

    Returns:
        [H_lat, W_lat] float tensor.  1=keep, 0=mask (LatentSync convention).
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_arr = np.array(mask_img).astype(np.float32) / 255.0  # 1=keep, 0=mask
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    mask_resized = F.interpolate(
        mask_t, size=(latent_h, latent_w), mode="bilinear", align_corners=False
    )
    return (mask_resized > 0.5).float().squeeze(0).squeeze(0)  # [H_lat, W_lat]


def apply_spatial_mask(video_tensor, mask_np, mask_all_frames=True):
    """Apply LatentSync spatial mask to a normalized video tensor.

    Matches training convention: normalize first (already done), then mask.
    Masked region becomes 0.0 in [-1,1] space (mid-gray).

    Args:
        video_tensor: [1, 3, N, H, W] float in [-1, 1]
        mask_np: [H, W] float32, 1=keep, 0=mask (LatentSync convention)
        mask_all_frames: if True, mask ALL frames including frame 0

    Returns:
        masked_tensor: [1, 3, N, H, W] float in [-1, 1]
    """
    mask_t = torch.from_numpy(mask_np).float()  # [H, W]
    mask_t = mask_t[None, None, None, :, :]  # [1, 1, 1, H, W]
    masked = video_tensor.clone()
    if mask_all_frames:
        masked *= mask_t
    else:
        masked[:, :, 1:, :, :] *= mask_t
    return masked


def encode_reference_video(vae, video_frames_np, mask_path, device, dtype):
    """Encode reference video through VAE (both unmasked and masked).

    Args:
        vae: WanVideoVAE instance
        video_frames_np: [N, H, W, 3] uint8
        mask_path: path to LatentSync mask
        device: torch device
        dtype: torch dtype

    Returns:
        (ref_latent, masked_latents, ref_sequence, latent_mask) where:
        - ref_latent: [1, 16, 1, H_lat, W_lat] — first frame latent
        - masked_latents: [1, 16, T_lat, H_lat, W_lat] — spatially masked
        - ref_sequence: [1, 16, T_lat, H_lat, W_lat] — unmasked full video
        - latent_mask: [H_lat, W_lat] float (LatentSync convention)
    """
    H, W = 512, 512
    latent_h, latent_w = H // 8, W // 8

    # Convert to tensor
    video_tensor = frames_to_tensor(video_frames_np)  # [1, 3, N, H, W]

    # Load pixel-level mask
    mask_img = Image.open(mask_path).convert("L")
    mask_pixel = np.array(mask_img.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
    mask_pixel_binary = (mask_pixel > 0.5).astype(np.float32)

    # Apply spatial mask (all frames)
    masked_video_tensor = apply_spatial_mask(video_tensor, mask_pixel_binary, mask_all_frames=True)

    # VAE encode both unmasked and masked (VAE requires float32)
    with torch.no_grad():
        source_latents = vae.encode(
            [video_tensor[0].to(dtype=torch.float32)], device=device
        )  # [1, 16, T_lat, H_lat, W_lat]

        masked_latents = vae.encode(
            [masked_video_tensor[0].to(dtype=torch.float32)], device=device
        )  # [1, 16, T_lat, H_lat, W_lat]

    ref_latent = source_latents[:, :, :1].to(dtype=dtype)  # [1, 16, 1, H_lat, W_lat]
    ref_sequence = source_latents.to(dtype=dtype)  # [1, 16, T_lat, H_lat, W_lat]
    masked_latents = masked_latents.to(dtype=dtype)
    H_lat, W_lat = ref_latent.shape[3], ref_latent.shape[4]
    latent_mask = load_latentsync_mask(mask_path, H_lat, W_lat).to(device=device, dtype=dtype)

    return ref_latent, masked_latents, ref_sequence, latent_mask


def encode_audio(wav2vec_model, wav2vec_extractor, audio_path, num_video_frames, device):
    """Encode audio to wav2vec2 features matching OmniAvatar's encode_audio.

    Encodes at the FULL audio's natural frame count, then slices to
    num_video_frames. This preserves the temporal grid that the model was
    trained on.

    Args:
        wav2vec_model: Wav2VecModel instance (on device, float32)
        wav2vec_extractor: Wav2Vec2FeatureExtractor
        audio_path: path to audio file
        num_video_frames: number of video frames to produce embeddings for
        device: torch device

    Returns:
        audio_emb: [1, num_video_frames, 10752] float tensor
    """
    wav2vec_sr = 16000  # Wav2Vec2 native sample rate
    fps = 25  # OmniAvatar default

    audio, sr = librosa.load(audio_path, sr=wav2vec_sr)
    input_values = np.squeeze(
        wav2vec_extractor(audio, sampling_rate=wav2vec_sr).input_values
    )
    input_values = torch.from_numpy(input_values).float().to(device=device)
    input_values = input_values.unsqueeze(0)

    # Compute the full audio's natural frame count (matches precompute script)
    samples_per_frame = wav2vec_sr // fps  # 640 at 16kHz/25fps
    total_audio_frames = math.ceil(input_values.shape[1] / samples_per_frame)
    total_audio_frames = max(total_audio_frames, num_video_frames)  # at least num_frames

    # Pad to align with total_audio_frames
    target_samples = total_audio_frames * samples_per_frame
    if input_values.shape[1] < target_samples:
        input_values = F.pad(input_values, (0, target_samples - input_values.shape[1]))

    # Encode at full length, then slice — matches training precompute + slice pattern
    with torch.no_grad():
        hidden_states = wav2vec_model(
            input_values, seq_len=total_audio_frames, output_hidden_states=True
        )
    audio_emb = hidden_states.last_hidden_state
    for hs in hidden_states.hidden_states:
        audio_emb = torch.cat((audio_emb, hs), -1)
    # audio_emb: [1, total_audio_frames, 10752]

    # Slice to num_video_frames (matches training: full_emb[:num_training_frames])
    audio_emb = audio_emb[:, :num_video_frames, :]
    return audio_emb  # [1, num_video_frames, 10752]


def build_condition(vae, wav2vec_model, wav2vec_extractor, video_frames_np,
                    audio_path, text_embeds, mask_path, num_video_frames,
                    num_latent_frames, device, dtype):
    """Build the full conditioning dictionary for the causal model.

    Args:
        vae: WanVideoVAE
        wav2vec_model: Wav2VecModel
        wav2vec_extractor: Wav2Vec2FeatureExtractor
        video_frames_np: [N, H, W, 3] uint8
        audio_path: path to audio file
        text_embeds: [1, 512, 4096] tensor
        mask_path: path to LatentSync mask
        num_video_frames: total video frames
        num_latent_frames: total latent frames
        device: torch device
        dtype: torch dtype

    Returns:
        dict with keys: text_embeds, audio_emb, ref_latent, mask,
        masked_video, ref_sequence
    """
    print("Encoding reference video ...")
    ref_latent, masked_latents, ref_sequence, latent_mask = encode_reference_video(
        vae, video_frames_np, mask_path, device, dtype
    )

    print("Encoding audio ...")
    audio_emb = encode_audio(
        wav2vec_model, wav2vec_extractor, audio_path, num_video_frames, device
    )
    audio_emb = audio_emb.to(dtype=dtype)

    return {
        "text_embeds": text_embeds,
        "audio_emb": audio_emb,
        "ref_latent": ref_latent.to(device=device, dtype=dtype),
        "mask": latent_mask.to(device=device),
        "masked_video": masked_latents.to(device=device, dtype=dtype),
        "ref_sequence": ref_sequence.to(device=device, dtype=dtype),
    }


# ===========================================================================
# Inference & post-processing (Tasks 4 & 5)
# ===========================================================================

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
    assert num_latent_frames % chunk_size == 0

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

            # Forward pass — model.forward() handles _build_y, rescale_t, _forward_ar internally
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


@torch.no_grad()
def decode_and_save(vae, output_latents, audio_path, output_path, fps, device):
    """VAE decode latents -> save silent video -> mux with audio."""
    import imageio.v3 as iio

    # VAE decode — expects list of [C, T_lat, H_lat, W_lat] in float32
    latent_for_vae = output_latents[0].to(torch.float32)  # [16, T_lat, H_lat, W_lat]
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
        _get_ffmpeg(), "-y", "-loglevel", "error", "-nostdin",
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


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    # --- Resolve dtype ---
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    # --- Seed ---
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Resolve audio ---
    audio_path, tmp_audio = resolve_audio(args)

    try:
        # --- Compute generation length ---
        num_latent_frames, num_video_frames = compute_generation_length(
            audio_path, args.num_latent_frames, args.chunk_size, args.fps
        )

        # --- Load models ---
        print("Loading VAE ...")
        vae = load_vae(args.vae_path, device)

        print("Loading Wav2Vec2 ...")
        wav2vec_model, wav2vec_extractor = load_wav2vec(args.wav2vec_path, device)

        print("Loading text embeddings ...")
        text_embeds = load_or_encode_text(args, device, dtype)

        print("Loading diffusion model ...")
        model = load_diffusion_model(args, device, dtype)

        # --- Preprocess inputs ---
        print("Loading and adjusting reference video ...")
        video_frames_np = load_and_adjust_video(args.video_path, num_video_frames)

        print("Building conditioning ...")
        condition = build_condition(
            vae, wav2vec_model, wav2vec_extractor, video_frames_np,
            audio_path, text_embeds, args.mask_path,
            num_video_frames, num_latent_frames, device, dtype,
        )

        # --- Free audio encoder VRAM ---
        del wav2vec_model
        torch.cuda.empty_cache()

        # --- Run inference ---
        print("Running inference ...")
        output_latents = run_inference(
            model, condition, num_latent_frames, args.t_list,
            args.chunk_size, args.context_noise, args.seed, device, dtype,
        )

        # --- Decode and save ---
        print("Decoding and saving ...")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        decode_and_save(vae, output_latents, audio_path, args.output_path, args.fps, device)

        print(f"Done! Output saved to {args.output_path}")

    finally:
        # --- Cleanup temp audio ---
        if tmp_audio is not None and os.path.exists(tmp_audio):
            os.remove(tmp_audio)
            print(f"Cleaned up temp audio: {tmp_audio}")


if __name__ == "__main__":
    main()
