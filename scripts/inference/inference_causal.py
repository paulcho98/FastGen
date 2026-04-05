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

    # --- Single-sample mode ---
    parser.add_argument("--video_path", type=str, default=None,
                        help="Reference video path (must be 512x512)")
    parser.add_argument("--output_path", type=str, default=None,
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
    parser.add_argument("--precomputed_dir", type=str, default=None,
                        help="Directory with precomputed .pt files (vae_latents_mask_all.pt, "
                             "ref_latents.pt, audio_emb_omniavatar.pt, text_emb.pt). "
                             "Bypasses VAE/Wav2Vec encoding — uses exact training-style tensors.")

    # --- Batch inference ---
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of sample subdirs (each with sub_clip.mp4, audio.wav). "
                             "Mutually exclusive with --video_path.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for batch mode")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples whose output already exists (for resume)")

    # --- LatentSync compositing ---
    parser.add_argument("--latentsync", action="store_true",
                        help="Enable face detection + 512x512 alignment + compositing")
    parser.add_argument("--face_cache_dir", type=str, default=None,
                        help="Directory for face detection caches (required with --latentsync)")
    parser.add_argument("--use_mouth_only", action="store_true", default=True,
                        help="Blend only mouth region, keep original upper face (default: True)")
    parser.add_argument("--no_mouth_only", action="store_false", dest="use_mouth_only",
                        help="Composite entire generated face (disable mouth-only blending)")

    parser.add_argument("--t_list", type=float, nargs="+",
                        default=[0.999, 0.900, 0.750, 0.500, 0.0],
                        help="Noise schedule timestep list for AR generation")
    parser.add_argument("--local_attn_size", type=int, default=-1,
                        help="Rolling local attention window in frames (-1 = full)")
    parser.add_argument("--sink_size", type=int, default=0,
                        help="Number of initial frames always kept in attention window")
    parser.add_argument("--use_dynamic_rope", action="store_true", default=False,
                        help="Use window-local dynamic RoPE (recommended for sliding window)")
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


def validate_args(args):
    if args.input_dir is not None and args.video_path is not None:
        raise ValueError("--input_dir and --video_path are mutually exclusive")
    if args.input_dir is None and args.video_path is None:
        raise ValueError("Must provide either --input_dir or --video_path")
    if args.input_dir is not None and args.output_dir is None:
        raise ValueError("--input_dir requires --output_dir")
    if args.latentsync and args.face_cache_dir is None:
        raise ValueError("--latentsync requires --face_cache_dir")
    if args.input_dir is None and args.output_path is None:
        raise ValueError("--video_path mode requires --output_path")


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
        sink_size=args.sink_size,
        use_dynamic_rope=args.use_dynamic_rope,
    )

    # Load Self-Forcing checkpoint on top
    # Handle FSDP checkpoint format: ckpt["model"]["net"] or ckpt["net"] or bare state_dict
    print(f"Loading SF checkpoint from {args.ckpt_path} ...")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict) and "net" in ckpt["model"]:
            state_dict = ckpt["model"]["net"]
        elif "net" in ckpt:
            state_dict = ckpt["net"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Keys in checkpoint use plain names (e.g. "patch_embedding.weight")
    # Model wraps everything under _core, so keys are "_core.xxx"
    # Try loading directly first, if too many missing, try adding _core prefix
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > len(state_dict) * 0.5 and not any(k.startswith("_core.") for k in state_dict):
        # Try with _core prefix
        prefixed_sd = {"_core." + k: v for k, v in state_dict.items()}
        missing2, unexpected2 = model.load_state_dict(prefixed_sd, strict=False)
        if len(missing2) < len(missing):
            missing, unexpected = missing2, unexpected2
            print(f"  Applied _core. prefix for key matching")

    print(f"  SF checkpoint: {len(state_dict)} params, {len(missing)} missing, {len(unexpected)} unexpected")

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

def resolve_audio(audio_path=None, video_path=None, args=None):
    """Determine the audio source path.

    Accepts explicit *audio_path* / *video_path* for batch mode, or falls
    back to reading from *args* for single-sample backward-compatibility.

    Returns:
        (audio_path, tmp_path_or_None) — tmp_path is set when a temp file
        was created and must be cleaned up later.
    """
    if audio_path is None and args is not None:
        audio_path = getattr(args, "audio_path", None)
    if video_path is None and args is not None:
        video_path = getattr(args, "video_path", None)

    if audio_path is not None:
        return audio_path, None

    if video_path is None:
        raise ValueError("resolve_audio: need either audio_path or video_path")

    # Extract audio from video
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    cmd = [
        _get_ffmpeg(), "-y", "-loglevel", "error", "-nostdin",
        "-i", video_path,
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


def build_condition_from_precomputed(precomputed_dir, mask_path, num_latent_frames, device, dtype):
    """Build conditioning dict from pre-computed .pt files (exact training format).

    This bypasses VAE/Wav2Vec encoding and uses the same tensors the model was
    trained on, enabling direct comparison.
    """
    print(f"Loading precomputed tensors from {precomputed_dir} ...")

    # VAE latents (input + masked)
    vae_data = torch.load(
        os.path.join(precomputed_dir, "vae_latents_mask_all.pt"),
        map_location="cpu", weights_only=False,
    )
    input_latents = vae_data["input_latents"].to(dtype=dtype)  # [16, T, H, W]
    masked_latents = vae_data["masked_latents"].to(dtype=dtype)

    # ref_latent = first frame of input video
    ref_latent = input_latents[:, :1].unsqueeze(0)  # [1, 16, 1, H, W]

    # Slice to num_latent_frames
    input_latents = input_latents[:, :num_latent_frames].unsqueeze(0)  # [1, 16, T, H, W]
    masked_latents = masked_latents[:, :num_latent_frames].unsqueeze(0)

    # ref_sequence (from separate file)
    ref_path = os.path.join(precomputed_dir, "ref_latents.pt")
    if os.path.exists(ref_path):
        ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
        ref_seq_key = "ref_sequence_latents" if "ref_sequence_latents" in ref_data else list(ref_data.keys())[0]
        ref_sequence = ref_data[ref_seq_key].to(dtype=dtype)[:, :num_latent_frames].unsqueeze(0)
    else:
        print("  Warning: ref_latents.pt not found, using input_latents as ref_sequence")
        ref_sequence = input_latents

    # Audio (video-frame-rate)
    audio_data = torch.load(
        os.path.join(precomputed_dir, "audio_emb_omniavatar.pt"),
        map_location="cpu", weights_only=False,
    )
    audio_emb = audio_data["audio_emb"] if isinstance(audio_data, dict) else audio_data
    # Training slices to num_video_frames = 1 + (num_latent - 1) * 4
    num_video_frames = 1 + (num_latent_frames - 1) * 4
    audio_emb = audio_emb[:num_video_frames].unsqueeze(0).to(dtype=dtype)  # [1, V, 10752]
    print(f"  audio_emb: {audio_emb.shape} (sliced to {num_video_frames} video frames)")

    # Text
    text_data = torch.load(
        os.path.join(precomputed_dir, "text_emb.pt"),
        map_location="cpu", weights_only=False,
    )
    if isinstance(text_data, dict):
        text_embeds = next(v for v in text_data.values() if isinstance(v, torch.Tensor))
    else:
        text_embeds = text_data
    if text_embeds.dim() == 2:
        text_embeds = text_embeds.unsqueeze(0)
    text_embeds = text_embeds.to(dtype=dtype)

    # Mask
    from PIL import Image
    H_lat, W_lat = ref_latent.shape[3], ref_latent.shape[4]
    latent_mask = load_latentsync_mask(mask_path, H_lat, W_lat)

    print(f"  ref_latent: {ref_latent.shape}, masked_video: {masked_latents.shape}")
    print(f"  ref_sequence: {ref_sequence.shape}, mask: {latent_mask.shape}")

    return {
        "text_embeds": text_embeds.to(device),
        "audio_emb": audio_emb.to(device),
        "ref_latent": ref_latent.to(device),
        "mask": latent_mask.to(device=device, dtype=dtype),
        "masked_video": masked_latents.to(device),
        "ref_sequence": ref_sequence.to(device),
    }


# ===========================================================================
# LatentSync preprocessing / compositing (copied from OmniAvatar-Train)
# ===========================================================================

def load_image_processor(mask_path, device):
    """Load LatentSync ImageProcessor for face detection and alignment.

    Matches the reference inference_v2v.py initialization exactly:
    - mask_image loaded via load_fixed_mask (uses caller-specified path)
    - insightface_root defaults to "checkpoints/auxiliary" (same as reference)
    - device passed as string for InsightFace compatibility
    """
    import os as _os
    _os.environ.setdefault("ORT_DISABLE_THREAD_AFFINITY", "1")
    from OmniAvatar.utils.latentsync.image_processor import ImageProcessor, load_fixed_mask
    print("Loading LatentSync ImageProcessor ...")
    device_str = str(device) if isinstance(device, torch.device) else device
    mask_tensor = load_fixed_mask(512, mask_image_path=mask_path) if mask_path else None
    processor = ImageProcessor(
        resolution=512,
        device=device_str,
        mask_image=mask_tensor,
        insightface_root="checkpoints/auxiliary",  # match reference default
    )
    return processor


def preprocess_with_latentsync(video_path, image_processor, face_detection_cache_dir, num_frames=81):
    """Detect faces, align to 512x512 via affine transform, with caching."""
    if not os.path.exists(video_path):
        print(f"[LatentSync] WARNING: Video not found: {video_path}")
        return None

    try:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        if video_basename in ("sub_clip", "video"):
            video_stem = os.path.basename(os.path.dirname(video_path))
        else:
            video_stem = video_basename
        face_cache_path = os.path.join(face_detection_cache_dir, f"{video_stem}_face_cache.pt")

        face_cache_loaded = False
        original_frames = None
        if os.path.isfile(face_cache_path):
            try:
                face_cache = torch.load(face_cache_path, weights_only=False)
                if face_cache.get("resolution") == image_processor.resolution:
                    boxes = face_cache["boxes"]
                    affine_matrices = face_cache["affine_matrices"]
                    aligned_faces = face_cache["aligned_faces"]
                    detection_failures = []
                    face_cache_loaded = True
                    print(f"[LatentSync] Loaded face cache: {face_cache_path}")
                else:
                    print(f"[LatentSync] Cache stale, recomputing...")
            except Exception as e:
                print(f"[LatentSync] Cache corrupt ({e}), recomputing...")
                os.remove(face_cache_path)

        if not face_cache_loaded:
            cap = cv2.VideoCapture(video_path)
            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

            if len(frames) < 5:
                print(f"[LatentSync] Too few frames ({len(frames)}) in {video_path}")
                return None

            while len(frames) < num_frames:
                frames.append(frames[-1].copy())

            original_frames = np.stack(frames, axis=0)
            boxes = []
            affine_matrices = []
            aligned_faces = []
            detection_failures = []

            # Reset temporal smoothing bias for new video
            image_processor.restorer.p_bias = None

            for i, frame in enumerate(frames):
                try:
                    face, box, affine_matrix = image_processor.affine_transform(frame)
                    boxes.append(box)
                    affine_matrices.append(affine_matrix)
                    aligned_faces.append(face)
                except RuntimeError as e:
                    print(f"[LatentSync] Face detection failed for frame {i}: {e}")
                    boxes.append(None)
                    affine_matrices.append(None)
                    detection_failures.append(i)

            if detection_failures:
                print(f"[LatentSync] Face detection failed for {len(detection_failures)} frames, skipping")
                return None

            os.makedirs(face_detection_cache_dir, exist_ok=True)
            torch.save({
                "aligned_faces": aligned_faces,
                "boxes": boxes,
                "affine_matrices": affine_matrices,
                "resolution": image_processor.resolution,
                "num_frames": len(original_frames),
            }, face_cache_path)
            print(f"[LatentSync] Saved face cache: {face_cache_path}")

        return {
            "video_path": video_path,
            "original_frames": original_frames,
            "num_frames": num_frames,
            "aligned_faces": aligned_faces,
            "boxes": boxes,
            "affine_matrices": affine_matrices,
            "detection_failures": detection_failures if not face_cache_loaded else [],
        }

    except Exception as e:
        print(f"[LatentSync] Preprocessing failed for {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def composite_with_latentsync_float(generated_float, latentsync_metadata, image_processor,
                                     use_mouth_only_compositing=False):
    """Composite generated faces back onto original video, staying in float space.

    Unlike composite_with_latentsync (which takes uint8 numpy), this function accepts the
    model output as a float tensor and avoids uint8 quantization before compositing.
    This matches LatentSync-train's data flow for maximum precision.

    Args:
        generated_float: [T, C, H, W] float tensor in [0, 1]
    """
    import torchvision.transforms.functional as TF_v

    original_frames = latentsync_metadata["original_frames"]
    if original_frames is None:
        video_path = latentsync_metadata["video_path"]
        num_frames = latentsync_metadata.get("num_frames", 81)
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
        original_frames = np.stack(frames, axis=0)

    boxes = latentsync_metadata["boxes"]
    affine_matrices = latentsync_metadata["affine_matrices"]
    detection_failures = latentsync_metadata.get("detection_failures", [])
    aligned_faces = latentsync_metadata.get("aligned_faces", None)

    composite_frames = []

    for i in range(generated_float.shape[0]):
        if i in detection_failures or boxes[i] is None:
            composite_frames.append(original_frames[i])
            continue

        face = generated_float[i]  # [C, H, W] float [0,1]

        # Mouth-only compositing in float space (no uint8 quantization)
        if use_mouth_only_compositing and aligned_faces is not None:
            mouth_mask = image_processor.mask_image.float()  # [C, H, W] float32
            original_aligned_float = aligned_faces[i].float() / 255.0  # uint8 → [0,1]
            face = face * (1 - mouth_mask) + original_aligned_float * mouth_mask

        # Resize in float space
        x1, y1, x2, y2 = boxes[i]
        height = int(y2 - y1)
        width = int(x2 - x1)
        face_resized = TF_v.resize(
            face, size=[height, width],
            interpolation=TF_v.InterpolationMode.BICUBIC, antialias=True,
        )

        # Convert [0,1] → [-1,1] for restore_img (NO uint8 round-trip)
        face_resized = face_resized * 2.0 - 1.0

        try:
            restored_frame = image_processor.restorer.restore_img(
                original_frames[i], face_resized, affine_matrices[i]
            )
            composite_frames.append(restored_frame)
        except Exception as e:
            print(f"[LatentSync] Restoration failed for frame {i}: {e}")
            composite_frames.append(original_frames[i])

    return np.stack(composite_frames)


def save_frames_as_video(frames_np, output_path, fps=25):
    """Save [N, H, W, 3] uint8 numpy array as mp4 video.

    Uses CRF 13 + macro_block_size=None to match LatentSync-train's write_video().
    """
    import imageio
    writer = imageio.get_writer(
        output_path, fps=fps, codec='libx264',
        macro_block_size=None,
        ffmpeg_params=["-crf", "13"],
        ffmpeg_log_level="error",
    )
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()


# ===========================================================================
# Batch enumeration
# ===========================================================================

def enumerate_samples(args):
    if args.input_dir is not None:
        for entry in sorted(os.listdir(args.input_dir)):
            sample_dir = os.path.join(args.input_dir, entry)
            if not os.path.isdir(sample_dir):
                continue
            video_path = os.path.join(sample_dir, "sub_clip.mp4")
            if not os.path.isfile(video_path):
                continue
            audio_path = os.path.join(sample_dir, "audio.wav")
            if not os.path.isfile(audio_path):
                print(f"[Skip] No audio.wav in {sample_dir}")
                continue
            precomputed = sample_dir if os.path.isfile(
                os.path.join(sample_dir, "vae_latents_mask_all.pt")
            ) else None
            yield entry, video_path, audio_path, precomputed
    else:
        name = os.path.splitext(os.path.basename(args.video_path))[0]
        yield name, args.video_path, args.audio_path, args.precomputed_dir


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
    validate_args(args)

    # --- Resolve dtype ---
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    # --- Seed ---
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ===================================================================
    # Load models once (expensive — minutes for 14B/1.3B weights)
    # ===================================================================
    print("Loading diffusion model ...")
    model = load_diffusion_model(args, device, dtype)

    print("Loading VAE ...")
    vae = load_vae(args.vae_path, device)

    # Wav2Vec + text only needed when NOT using precomputed tensors for every sample
    wav2vec_model = wav2vec_extractor = None
    text_embeds = None

    # Optional LatentSync ImageProcessor
    image_processor = None
    if args.latentsync:
        image_processor = load_image_processor(args.mask_path, device)

    # ===================================================================
    # Loop over samples
    # ===================================================================
    samples = list(enumerate_samples(args))
    succeeded, failed, skipped = [], [], []

    for sample_idx, (name, video_path, audio_path_sample, precomputed_dir) in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"[{sample_idx+1}/{len(samples)}] {name}")
        print(f"{'='*60}")

        # --- Determine output path ---
        if args.input_dir is not None:
            output_path = os.path.join(args.output_dir, f"{name}.mp4")
        else:
            output_path = args.output_path

        # --- Skip existing ---
        if args.skip_existing and os.path.isfile(output_path):
            print(f"  [Skip] Output exists: {output_path}")
            skipped.append(name)
            continue

        tmp_audio = None
        try:
            # --- Resolve audio ---
            audio_path, tmp_audio = resolve_audio(
                audio_path=audio_path_sample, video_path=video_path,
            )

            # --- Compute generation length ---
            num_latent_frames, num_video_frames = compute_generation_length(
                audio_path, args.num_latent_frames, args.chunk_size, args.fps,
            )

            # --- Optional LatentSync preprocessing ---
            latentsync_metadata = None
            if args.latentsync:
                print("Running LatentSync face detection ...")
                latentsync_metadata = preprocess_with_latentsync(
                    video_path, image_processor, args.face_cache_dir,
                    num_frames=num_video_frames,
                )
                if latentsync_metadata is None:
                    print(f"  [FAIL] LatentSync preprocessing failed, skipping {name}")
                    failed.append(name)
                    continue

            # --- Build conditioning ---
            if precomputed_dir is not None:
                condition = build_condition_from_precomputed(
                    precomputed_dir, args.mask_path,
                    num_latent_frames, device, dtype,
                )
            else:
                # Lazy-load Wav2Vec + text on first non-precomputed sample
                if wav2vec_model is None:
                    print("Loading Wav2Vec2 ...")
                    wav2vec_model, wav2vec_extractor = load_wav2vec(
                        args.wav2vec_path, device,
                    )
                if text_embeds is None:
                    print("Loading text embeddings ...")
                    text_embeds = load_or_encode_text(args, device, dtype)

                # Reference frames: aligned faces from LatentSync or raw video
                if args.latentsync and latentsync_metadata is not None:
                    aligned_faces = latentsync_metadata["aligned_faces"]
                    ref_frames_np = np.stack([
                        f.permute(1, 2, 0).numpy() if isinstance(f, torch.Tensor) else f
                        for f in aligned_faces[:num_video_frames]
                    ], axis=0)
                else:
                    ref_frames_np = load_and_adjust_video(video_path, num_video_frames)

                print("Building conditioning ...")
                condition = build_condition(
                    vae, wav2vec_model, wav2vec_extractor, ref_frames_np,
                    audio_path, text_embeds, args.mask_path,
                    num_video_frames, num_latent_frames, device, dtype,
                )

            # --- Run inference ---
            print("Running inference ...")
            output_latents = run_inference(
                model, condition, num_latent_frames, args.t_list,
                args.chunk_size, args.context_noise, args.seed, device, dtype,
            )

            # --- Post-processing: decode + save ---
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            if args.latentsync and latentsync_metadata is not None:
                # LatentSync compositing path — float-space decode + composite
                print("VAE decoding (float) ...")
                latent_for_vae = output_latents[0].to(torch.float32)
                video_decoded = vae.decode([latent_for_vae], device=device)
                video_decoded = video_decoded.clamp(-1, 1)
                # [1, 3, T_video, H, W] -> [T, 3, H, W] in [0, 1]
                generated_float = video_decoded[0].permute(1, 0, 2, 3)  # [3,T,H,W] -> [T,3,H,W]
                generated_float = ((generated_float + 1) / 2).clamp(0, 1)  # [-1,1] -> [0,1]

                # Composite onto original frames
                print("Compositing ...")
                composited_np = composite_with_latentsync_float(
                    generated_float.cpu(), latentsync_metadata, image_processor,
                    use_mouth_only_compositing=args.use_mouth_only,
                )

                # Save composited video (original resolution) with audio
                composited_path = output_path
                save_frames_as_video(composited_np, composited_path, fps=args.fps)
                video_duration = composited_np.shape[0] / args.fps
                tmp_composited = composited_path + ".tmp.mp4"
                os.rename(composited_path, tmp_composited)
                mux_video_with_audio(tmp_composited, audio_path, composited_path,
                                     duration_s=video_duration)
                if os.path.exists(tmp_composited):
                    os.remove(tmp_composited)

                # Also save aligned (512x512) video with audio
                aligned_path = output_path.replace(".mp4", "_aligned.mp4")
                aligned_np = ((generated_float.permute(0, 2, 3, 1).cpu().float()) * 255
                              ).clamp(0, 255).to(torch.uint8).numpy()
                save_frames_as_video(aligned_np, aligned_path, fps=args.fps)
                tmp_aligned = aligned_path + ".tmp.mp4"
                os.rename(aligned_path, tmp_aligned)
                mux_video_with_audio(tmp_aligned, audio_path, aligned_path,
                                     duration_s=video_duration)
                if os.path.exists(tmp_aligned):
                    os.remove(tmp_aligned)

                print(f"  Saved composited: {composited_path}")
                print(f"  Saved aligned:    {aligned_path}")
            else:
                # Standard decode + save (no LatentSync)
                print("Decoding and saving ...")
                decode_and_save(vae, output_latents, audio_path, output_path,
                                args.fps, device)

            succeeded.append(name)
            print(f"  Done: {output_path}")

        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

        finally:
            # Cleanup per-sample temp audio
            if tmp_audio is not None and os.path.exists(tmp_audio):
                os.remove(tmp_audio)

            # Free per-sample GPU memory
            torch.cuda.empty_cache()

    # ===================================================================
    # Summary
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"Summary: {len(succeeded)} succeeded, {len(failed)} failed, {len(skipped)} skipped "
          f"(out of {len(samples)} total)")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
