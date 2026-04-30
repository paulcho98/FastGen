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
import csv
import math
import os
import subprocess
import sys
import tempfile
import time

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Timing instrumentation (opt-in via --timing).
# Uses torch.cuda.synchronize() + time.perf_counter() so GPU timings reflect
# actual kernel execution, not just launch time.
# ---------------------------------------------------------------------------
_TIMING_ENABLED = False
_TIMING_CURRENT: dict = {}
_TIMING_ROWS: list = []
_a2d_t0 = None
_enc_to_dec_t0 = None
_e2d_t0 = None
_TIMING_STAGE_ORDER = [
    "audio_extract", "face_detect_align", "vae_encode", "pure_vae_encode", "audio_encode",
    "denoise", "pure_vae_decode", "first_frame_latency", "composite", "save_mux",
    "whole_generation", "pure_encode_to_decode", "audio_to_decode", "encode_to_decode", "total_post_load",
    "ablation_1", "ablation_2", "ablation_3", "ablation_4",
    "ff_full_pipeline", "ff_generation", "ff_pure_generation",
    "ff_enc_gen_dec", "ff_enc_audio_gen_dec", "ff_pure_enc_to_dec",
    # Three-definition first-frame latencies (matching the full a2d/e2d/pure)
    "ff_audio_to_decode", "ff_encode_to_decode", "ff_pure_encode_to_decode",
    "streaming_total",
    "ff_ablation_1", "ff_ablation_2", "ff_ablation_3", "ff_ablation_4",
]


def _gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _Stage:
    """Context manager that records elapsed seconds for `name` into _TIMING_CURRENT."""
    def __init__(self, name: str, use_gpu: bool = True):
        self.name = name
        self.use_gpu = use_gpu
        self.t0 = 0.0

    def __enter__(self):
        if _TIMING_ENABLED and self.use_gpu:
            _gpu_sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if _TIMING_ENABLED:
            if self.use_gpu:
                _gpu_sync()
            _TIMING_CURRENT[self.name] = time.perf_counter() - self.t0

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
    parser.add_argument("--taehv_ckpt", type=str, default=None,
                        help="Optional path to TAEHV taew2_1.pth. If set, uses the TAEHV "
                             "tiny decoder for latent->pixel decoding (full Wan VAE is still "
                             "used for encoding driving video unless --taehv_encode is set).")
    parser.add_argument("--taehv_encode", action="store_true",
                        help="Also use TAEHV for encoding the driving video (requires --taehv_ckpt). "
                             "Default: full Wan VAE encoder.")
    parser.add_argument("--taehv_streaming", action="store_true",
                        help="Use StreamingTAEHV for decoding (feeds latents one at a time, "
                             "records first_frame_latency). Requires --taehv_ckpt.")
    parser.add_argument("--streaming_pipeline", type=str, default=None,
                        help="DEPRECATED: use inference_streaming.py instead. "
                             "'sequential': per-chunk encode→denoise→decode→composite in series. "
                             "'pipelined': per-chunk encode→denoise→decode on GPU, composite "
                             "overlapped on CPU background thread (GPU+CPU parallel). "
                             "Requires --taehv_ckpt and --latentsync.")
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
    parser.add_argument("--min_latent_frames", type=int, default=0,
                        help="Floor on num_latent; if audio is shorter, pad via zero-audio + ping-pong "
                             "video. 0 disables. 21 matches FastGen's 81-frame training length.")
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
    parser.add_argument("--timing", action="store_true",
                        help="Enable per-stage timing; excludes model load. "
                             "Writes timing CSV to --timing_csv.")
    parser.add_argument("--timing_csv", type=str, default=None,
                        help="CSV path for timing results (default: <output_path>.timing.csv)")

    # --- torch.compile ---
    parser.add_argument("--compile", action="store_true",
                        help="Wrap diffusion model + Wan VAE encoder/decoder + "
                             "TAEHV (when present) with torch.compile. First "
                             "warmup clip absorbs the compile time; subsequent "
                             "clips run on the compiled graphs.")

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
    # Supports: regular .pt/.pth, FSDP distcp directory, or .pth + adjacent distcp dir
    print(f"Loading SF checkpoint from {args.ckpt_path} ...")

    state_dict = None

    # Check for FSDP distributed checkpoint: look for .net_model/ directory
    ckpt_stem = args.ckpt_path.replace(".pth", "")
    fsdp_net_dir = ckpt_stem + ".net_model"
    if os.path.isdir(fsdp_net_dir):
        # FSDP2 distributed checkpoint — load via torch.distributed.checkpoint
        print(f"  Loading FSDP distributed checkpoint from {fsdp_net_dir} ...")
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

        reader = FileSystemReader(fsdp_net_dir)
        md = reader.read_metadata()
        state_dict = {}
        for key, meta in md.state_dict_metadata.items():
            if hasattr(meta, "size"):
                state_dict[key] = torch.empty(meta.size)
        dcp_load(state_dict, storage_reader=reader, no_dist=True)
        print(f"  Loaded {len(state_dict)} tensors from FSDP distcp")
    elif os.path.isdir(args.ckpt_path):
        # Direct distcp directory path
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

        reader = FileSystemReader(args.ckpt_path)
        md = reader.read_metadata()
        state_dict = {}
        for key, meta in md.state_dict_metadata.items():
            if hasattr(meta, "size"):
                state_dict[key] = torch.empty(meta.size)
        dcp_load(state_dict, storage_reader=reader, no_dist=True)
        print(f"  Loaded {len(state_dict)} tensors from distcp directory")
    else:
        # Regular .pt/.pth checkpoint
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


class TAEHVDecoderWrapper:
    """Drop-in decode-only replacement that mimics WanVideoVAE.decode().

    TAEHV convention:
      - Input:  diffusion-space latents (same space the denoiser works in).
                No mean/std scaling is applied here — TAEHV was distilled to
                consume these directly.
      - Output: pixels in [0, 1], NTCHW layout.
    WanVideoVAE convention:
      - decode() returns pixels in [-1, 1], shape [1, 3, T_video, H, W] (NCTHW).
    This wrapper converts TAEHV output to Wan's range/layout so downstream
    code (decode_and_save, LatentSync path) works unchanged.
    """
    def __init__(self, checkpoint_path, device):
        from taehv import TAEHV
        self.device = device
        self.taehv = TAEHV(checkpoint_path=checkpoint_path).to(device, torch.float16).eval()

    @torch.no_grad()
    def decode(self, latents_list, device=None):
        # latents_list: list of one [C=16, T_lat, H, W] tensor (matches WanVideoVAE.decode signature)
        target_device = device if device is not None else self.device
        lat = latents_list[0].to(target_device, dtype=torch.float16)      # [16, T, H, W]
        lat = lat.permute(1, 0, 2, 3).unsqueeze(0)                        # [1, T, 16, H, W]  NTCHW
        vid = self.taehv.decode_video(lat, parallel=False)                # [1, T*4, 3, H', W']  in [0, 1]
        # diagdistill's taehv.py disables the front-trim (line 233), so match Wan's
        # temporal length convention: num_video = 1 + (num_latent - 1) * 4 = T_lat*4 - frames_to_trim.
        vid = vid[:, self.taehv.frames_to_trim:]                          # [1, T_lat*4 - 3, 3, H', W']
        vid = vid.mul(2).sub(1)                                           # -> [-1, 1] (match Wan)
        return vid.permute(0, 2, 1, 3, 4).float()                         # [1, 3, T_video, H', W']  NCTHW

    @torch.no_grad()
    def encode(self, videos_list, device=None):
        """Drop-in replacement for WanVideoVAE.encode().

        Wan convention: input list of [3, T, H, W] in [-1, 1]; returns [N, 16, T_lat, H_lat, W_lat]
        with T_lat = 1 + (T-1)//4 = ⌈T/4⌉.
        TAEHV: wants NTCHW in [0, 1], its temporal compression is floor(T/4). We pad the
        INPUT video to the next multiple of 4 so floor(T_pad/4) = ⌈T/4⌉, matching Wan's T_lat
        naturally — no latent-side duplication needed.
        """
        target_device = device if device is not None else self.device
        outs = []
        for vid in videos_list:
            T = vid.shape[1]
            T_pad = ((T + 3) // 4) * 4                                    # round up to multiple of 4
            if T_pad > T:
                # PREPEND copies of the first frame so TAEHV's latent 0 pools [f0,f0,f0,f0]
                # = encoding of the static starting frame. This matches Wan's convention where
                # latent 0 encodes frame 0 alone; latents i>0 encode groups of 4 consecutive frames.
                pad = vid[:, :1].expand(-1, T_pad - T, -1, -1).contiguous()
                vid = torch.cat([pad, vid], dim=1)
            x = vid.to(target_device, dtype=torch.float16)
            x = x.add(1).div(2)                                           # [-1,1] -> [0,1]
            x = x.permute(1, 0, 2, 3).unsqueeze(0)                        # [1, T_pad, 3, H, W]  NTCHW
            lat = self.taehv.encode_video(x, parallel=False, show_progress_bar=False)  # [1, T_pad/4, 16, H', W']
            lat = lat.permute(0, 2, 1, 3, 4).float()                      # [1, 16, T_pad/4, H', W']
            outs.append(lat.squeeze(0))                                   # [16, T_pad/4, H', W']
        return torch.stack(outs)                                          # [N, 16, T_pad/4, H', W']


class StreamingTAEHVDecoderWrapper:
    """Drop-in decoder using StreamingTAEHV — feeds latents one at a time and
    collects pixel frames as they emerge.

    Same signature as TAEHVDecoderWrapper.decode() so downstream code works
    unchanged. Additionally records first_frame_latency in _TIMING_CURRENT.
    """
    def __init__(self, checkpoint_path, device):
        from taehv import TAEHV, StreamingTAEHV
        self.device = device
        taehv_model = TAEHV(checkpoint_path=checkpoint_path).to(device, torch.float16).eval()
        self.streaming = StreamingTAEHV(taehv_model)

    @torch.no_grad()
    def decode(self, latents_list, device=None):
        target_device = device if device is not None else self.device
        self.streaming.reset()
        lat = latents_list[0].to(target_device, dtype=torch.float16)      # [16, T_lat, H, W]
        lat = lat.permute(1, 0, 2, 3).unsqueeze(0)                        # [1, T_lat, 16, H, W]  NTCHW

        frames = []
        first_frame_time = None
        if _TIMING_ENABLED:
            _gpu_sync()
        t0 = time.perf_counter()

        for t in range(lat.shape[1]):
            latent_t = lat[:, t:t+1]                                       # [1, 1, 16, H, W]
            frame = self.streaming.decode(latent_t)
            while frame is not None:
                if first_frame_time is None:
                    if _TIMING_ENABLED:
                        _gpu_sync()
                    first_frame_time = time.perf_counter() - t0
                frames.append(frame)
                frame = self.streaming.decode()

        for frame in self.streaming.flush_decoder():
            if first_frame_time is None:
                if _TIMING_ENABLED:
                    _gpu_sync()
                first_frame_time = time.perf_counter() - t0
            frames.append(frame)

        if _TIMING_ENABLED and first_frame_time is not None:
            _TIMING_CURRENT["first_frame_latency"] = first_frame_time

        # Stack [N1CHW, ...] → [1, T, C, H, W] NTCHW, convert to NCTHW [-1, 1]
        vid = torch.cat(frames, dim=1)                                     # [1, T, 3, H', W']
        vid = vid.mul(2).sub(1)                                            # [0,1] → [-1,1]
        return vid.permute(0, 2, 1, 3, 4).float()                         # [1, 3, T, H', W']


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

    vae = vae.to(device=device)
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


def compute_generation_length(audio_path, override_frames, chunk_size, fps,
                              min_latent_frames=0):
    """Compute generation length in both latent and video frames.

    The VAE temporal compression is: num_latent = 1 + (num_video - 1) // 4.
    We round DOWN num_latent to the nearest multiple of chunk_size so the AR
    loop produces complete chunks.

    If ``min_latent_frames`` > 0 and the audio-derived num_latent is shorter,
    we pad up to ``min_latent_frames`` (FastGen-style): audio zero-pads via
    wav2vec; video frames are ping-pong extended in adjust_video_length.

    Args:
        audio_path: path to audio file (for duration)
        override_frames: explicit num_latent_frames (or None)
        chunk_size: AR chunk size in latent frames
        fps: video frames per second
        min_latent_frames: floor on num_latent; 0 disables padding.

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
    else:
        # Round DOWN to multiple of chunk_size
        num_latent = (num_latent_raw // chunk_size) * chunk_size
        num_latent = max(num_latent, chunk_size)  # at least one chunk

    if min_latent_frames and num_latent < min_latent_frames:
        print(f"  Audio too short ({duration:.2f}s → {num_latent} latent frames), "
              f"padding to {min_latent_frames}")
        num_latent = min_latent_frames

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

    # Load pixel-level mask — use cv2 bilinear resize to match precomputation
    # (preprocess_v2v_integrated.py uses cv2.INTER_LINEAR)
    mask_img = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    if mask_np.shape[0] != H or mask_np.shape[1] != W:
        mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_pixel_binary = (mask_np > 0.5).astype(np.float32)

    # Apply spatial mask (all frames)
    masked_video_tensor = apply_spatial_mask(video_tensor, mask_pixel_binary, mask_all_frames=True)

    # VAE encode. Wan VAE runs in bf16 (cast temporarily); TAEHV runs in fp16 natively.
    # pure_vae_encode times ONLY the encode calls; pure_encode_to_decode starts here too.
    global _e2d_t0
    if _TIMING_ENABLED:
        _gpu_sync()
    _e2d_t0 = time.perf_counter()
    is_taehv = isinstance(vae, TAEHVDecoderWrapper)

    # For long videos, encoding the full sequence in one shot OOMs.
    # Temporal-chunk in groups of 81 video frames (21 latent frames).
    # Each chunk's first latent represents that chunk's first video frame; we
    # concat naively, accepting that boundary latents are computed from chunk-local
    # context rather than the full causal history (small accuracy impact, large
    # memory win for >300-frame videos).
    N = video_tensor.shape[2]
    CHUNK = 81  # video frames per encode chunk

    def _chunked_encode(vt):
        if N <= CHUNK:
            return vae.encode([vt[0]], device=device)
        out = []
        for s in range(0, N, CHUNK):
            e = min(s + CHUNK, N)
            chunk = vt[:, :, s:e]
            out.append(vae.encode([chunk[0]], device=device))
        return torch.cat(out, dim=2)

    if is_taehv:
        with _Stage("pure_vae_encode", use_gpu=True):
            with torch.no_grad():
                source_latents = _chunked_encode(video_tensor)
                masked_latents = _chunked_encode(masked_video_tensor)
    else:
        original_dtype = next(vae.parameters()).dtype
        vae.to(dtype=torch.bfloat16)
        video_tensor = video_tensor.to(dtype=torch.bfloat16)
        masked_video_tensor = masked_video_tensor.to(dtype=torch.bfloat16)
        with _Stage("pure_vae_encode", use_gpu=True):
            with torch.no_grad():
                source_latents = _chunked_encode(video_tensor)
                masked_latents = _chunked_encode(masked_video_tensor)
        vae.to(dtype=original_dtype)

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
    # ============================================================
    # Audio encode FIRST (matches LatentSync/X-Dub order)
    # ============================================================
    # _a2d_t0: start of audio_to_decode span (Def 1)
    global _a2d_t0
    if _TIMING_ENABLED:
        _gpu_sync()
    _a2d_t0 = time.perf_counter()

    print("Encoding audio ...")
    with _Stage("audio_encode", use_gpu=True):
        audio_emb = encode_audio(
            wav2vec_model, wav2vec_extractor, audio_path, num_video_frames, device
        )
    audio_emb = audio_emb.to(dtype=dtype)

    # ============================================================
    # VAE encode reference video
    # ============================================================
    # _enc_to_dec_t0: start of encode_to_decode span (broad: vae_encode wrapper → vae_decode)
    # _e2d_t0: start of pure_encode_to_decode span (narrow: first vae.encode call → vae_decode),
    #         set inside encode_reference_video right before the first vae.encode call.
    global _enc_to_dec_t0
    if _TIMING_ENABLED:
        _gpu_sync()
    _enc_to_dec_t0 = time.perf_counter()

    print("Encoding reference video ...")
    with _Stage("vae_encode", use_gpu=True):
        ref_latent, masked_latents, ref_sequence, latent_mask = encode_reference_video(
            vae, video_frames_np, mask_path, device, dtype
        )

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
    # All three timing markers point to the same instant since no real encode happens
    global _a2d_t0, _e2d_t0, _enc_to_dec_t0
    if _TIMING_ENABLED:
        _gpu_sync()
    _a2d_t0 = time.perf_counter()
    _enc_to_dec_t0 = _a2d_t0
    _e2d_t0 = _a2d_t0
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
                                     use_mouth_only_compositing=False, frame_offset=0):
    """Composite generated faces back onto original video, staying in float space.

    Unlike composite_with_latentsync (which takes uint8 numpy), this function accepts the
    model output as a float tensor and avoids uint8 quantization before compositing.
    This matches LatentSync-train's data flow for maximum precision.

    Args:
        generated_float: [T, C, H, W] float tensor in [0, 1]
        frame_offset: offset into the metadata arrays (for per-chunk streaming)
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
        gi = i + frame_offset
        if gi >= len(original_frames):
            break
        if gi in detection_failures or boxes[gi] is None:
            composite_frames.append(original_frames[gi])
            continue

        face = generated_float[i]  # [C, H, W] float [0,1]

        if use_mouth_only_compositing and aligned_faces is not None:
            mouth_mask = image_processor.mask_image.float()
            original_aligned_float = aligned_faces[gi].float() / 255.0
            face = face * (1 - mouth_mask) + original_aligned_float * mouth_mask

        x1, y1, x2, y2 = boxes[gi]
        height = int(y2 - y1)
        width = int(x2 - x1)
        face_resized = TF_v.resize(
            face, size=[height, width],
            interpolation=TF_v.InterpolationMode.BICUBIC, antialias=True,
        )

        face_resized = face_resized * 2.0 - 1.0

        try:
            restored_frame = image_processor.restorer.restore_img(
                original_frames[gi], face_resized, affine_matrices[gi]
            )
            composite_frames.append(restored_frame)
        except Exception as e:
            print(f"[LatentSync] Restoration failed for frame {gi}: {e}")
            composite_frames.append(original_frames[gi])

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
            # Keep timesteps in float64 to match CausVidModel._student_sample_loop precision
            x0_pred = model(
                noisy_input,
                t_cur.expand(B),
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
                    x0_pred, eps, t_next.expand(B),
                )
            else:
                # Final step — clean output
                noisy_input = x0_pred

        # Store denoised chunk
        output[:, :, cur_start_frame:cur_start_frame + chunk_size] = x0_pred

        # Update KV cache with denoised output (context for next block)
        cache_input = x0_pred
        t_cache = torch.full((B,), context_noise, device=device, dtype=torch.float64)
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
def run_inference_streaming(
    model, condition, num_latent_frames, t_list,
    chunk_size, context_noise, seed, device, dtype,
):
    """Streaming version of run_inference — yields denoised latents per chunk.

    Same AR generation logic, but instead of accumulating all chunks and returning
    at the end, yields each chunk's denoised latents immediately. The caller can
    decode + composite each chunk as it arrives, reducing first-frame latency.

    Yields:
        chunk_latents: [1, 16, chunk_size, H_lat, W_lat] per chunk
    """
    model.total_num_frames = num_latent_frames
    model.clear_caches()

    ref_latent = condition["ref_latent"]
    B = ref_latent.shape[0]
    C = 16
    H_lat, W_lat = ref_latent.shape[3], ref_latent.shape[4]

    num_blocks = num_latent_frames // chunk_size
    assert num_latent_frames % chunk_size == 0

    torch.manual_seed(seed)
    noise = torch.randn(B, C, num_latent_frames, H_lat, W_lat, device=device, dtype=dtype)
    t_list_t = torch.tensor(t_list, device=device, dtype=torch.float64)

    for block_idx in range(num_blocks):
        cur_start_frame = block_idx * chunk_size
        noisy_input = noise[:, :, cur_start_frame:cur_start_frame + chunk_size]

        for step_idx in range(len(t_list_t) - 1):
            t_cur = t_list_t[step_idx]
            t_next = t_list_t[step_idx + 1]

            x0_pred = model(
                noisy_input, t_cur.expand(B),
                condition=condition,
                cur_start_frame=cur_start_frame,
                store_kv=False, is_ar=True,
                fwd_pred_type="x0",
                use_gradient_checkpointing=False,
            )

            if t_next > 0:
                eps = torch.randn_like(x0_pred)
                noisy_input = model.noise_scheduler.forward_process(
                    x0_pred, eps, t_next.expand(B),
                )
            else:
                noisy_input = x0_pred

        # Yield this chunk's denoised latents immediately
        yield x0_pred

        # Update KV cache for next block
        cache_input = x0_pred
        t_cache = torch.full((B,), context_noise, device=device, dtype=torch.float64)
        if context_noise > 0:
            cache_eps = torch.randn_like(x0_pred)
            cache_input = model.noise_scheduler.forward_process(
                x0_pred, cache_eps,
                torch.tensor(context_noise, device=device, dtype=torch.float64).expand(B),
            )

        model(
            cache_input, t_cache,
            condition=condition,
            cur_start_frame=cur_start_frame,
            store_kv=True, is_ar=True,
            fwd_pred_type="x0",
            use_gradient_checkpointing=False,
        )

    model.clear_caches()


def encode_reference_video_chunk(vae, video_frames_np, mask_path, chunk_start_frame,
                                  chunk_size_video, device, dtype, latent_mask=None):
    """Encode a chunk of reference video frames through VAE.

    Used by the streaming pipeline to encode frames on demand per AR chunk,
    instead of encoding the full video upfront.

    Args:
        vae: WanVideoVAE or TAEHVDecoderWrapper
        video_frames_np: [N, H, W, 3] uint8 — FULL video (sliced internally)
        mask_path: path to LatentSync mask
        chunk_start_frame: first VIDEO frame index for this chunk
        chunk_size_video: number of VIDEO frames to encode
        device, dtype: target device/dtype
        latent_mask: precomputed [H_lat, W_lat] mask (pass to avoid reloading)

    Returns:
        (chunk_source_latents, chunk_masked_latents, latent_mask)
    """
    H, W = 512, 512
    latent_h, latent_w = H // 8, W // 8

    # Slice video frames for this chunk
    end_frame = min(chunk_start_frame + chunk_size_video, len(video_frames_np))
    chunk_frames = video_frames_np[chunk_start_frame:end_frame]

    video_tensor = frames_to_tensor(chunk_frames)

    # Load mask (reuse if provided)
    mask_img = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    if mask_np.shape[0] != H or mask_np.shape[1] != W:
        mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_pixel_binary = (mask_np > 0.5).astype(np.float32)

    masked_video_tensor = apply_spatial_mask(video_tensor, mask_pixel_binary, mask_all_frames=True)

    is_taehv = isinstance(vae, TAEHVDecoderWrapper)
    if is_taehv:
        with torch.no_grad():
            source_latents = vae.encode([video_tensor[0]], device=device)
            masked_latents = vae.encode([masked_video_tensor[0]], device=device)
    else:
        original_dtype = next(vae.parameters()).dtype
        vae.to(dtype=torch.bfloat16)
        with torch.no_grad():
            source_latents = vae.encode(
                [video_tensor[0].to(dtype=torch.bfloat16)], device=device
            )
            masked_latents = vae.encode(
                [masked_video_tensor[0].to(dtype=torch.bfloat16)], device=device
            )
        vae.to(dtype=original_dtype)

    source_latents = source_latents.to(dtype=dtype)
    masked_latents = masked_latents.to(dtype=dtype)

    if latent_mask is None:
        latent_mask = load_latentsync_mask(mask_path, latent_h, latent_w).to(device=device, dtype=dtype)

    return source_latents, masked_latents, latent_mask


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
    global _TIMING_ENABLED, _TIMING_CURRENT, _TIMING_ROWS
    args = parse_args()
    validate_args(args)
    _TIMING_ENABLED = bool(args.timing)

    # Activate per-function torch.compile decorators in network_causal.py
    # BEFORE the model class is imported (which happens inside
    # load_diffusion_model below).
    if args.compile:
        os.environ["FASTGEN_COMPILE"] = "true"

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

    # Decoder selection: full Wan VAE stays loaded for encoding the driving video;
    # decoding swaps to TAEHV tiny decoder if --taehv_ckpt is provided.
    if args.taehv_streaming:
        if not args.taehv_ckpt:
            raise ValueError("--taehv_streaming requires --taehv_ckpt")
        print(f"Loading StreamingTAEHV decoder from {args.taehv_ckpt} ...")
        decoder_vae = StreamingTAEHVDecoderWrapper(args.taehv_ckpt, device)
    elif args.taehv_ckpt:
        print(f"Loading TAEHV tiny decoder from {args.taehv_ckpt} ...")
        decoder_vae = TAEHVDecoderWrapper(args.taehv_ckpt, device)
    else:
        decoder_vae = vae

    # Encoder selection: default to full Wan VAE. If --taehv_encode is set,
    # reuse the same TAEHV model (it implements both encode and decode).
    if args.taehv_encode:
        if not args.taehv_ckpt:
            raise ValueError("--taehv_encode requires --taehv_ckpt")
        print("Using TAEHV tiny encoder for driving video encoding.")
        encoder_vae = decoder_vae
    else:
        encoder_vae = vae

    # Eagerly load Wav2Vec + text to avoid warmup artifacts in timing
    wav2vec_model = wav2vec_extractor = None
    text_embeds = None
    if args.wav2vec_path:
        print("Loading Wav2Vec2 (eager) ...")
        wav2vec_model, wav2vec_extractor = load_wav2vec(args.wav2vec_path, device)
        # Warmup forward pass to compile CUDA kernels.
        # OmniAvatar Wav2VecModel requires seq_len + output_hidden_states.
        _dummy_audio = np.zeros(16000, dtype=np.float32)  # 1s @ 16kHz → 25 video-frames
        _dummy_input = wav2vec_extractor(_dummy_audio, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            wav2vec_model(
                _dummy_input.input_values.to(device),
                seq_len=25, output_hidden_states=True,
            )
        print("Wav2Vec2 warmed up.")
    if args.text_embeds_path or args.prompt:
        print("Loading text embeddings (eager) ...")
        text_embeds = load_or_encode_text(args, device, dtype)

    # Optional LatentSync ImageProcessor
    image_processor = None
    if args.latentsync:
        image_processor = load_image_processor(args.mask_path, device)

    # ===================================================================
    # Optional torch.compile wrapping (compile time absorbed by warmup)
    # ===================================================================
    # NOTE: torch.compile is now applied via @conditional_compile decorators
    # on hot functions inside network_causal.py (rope_apply, _forward_ar).
    # Activation is via the FASTGEN_COMPILE env var, which must be set BEFORE
    # the model class is imported. We handle this at the top of main(), so
    # by the time we reach this point the model is already compile-decorated
    # if --compile was passed. Nothing more to do here.
    if args.compile:
        print("[--compile] Hot functions decorated with @conditional_compile. "
              "Warmup clip will trigger Dynamo trace.")

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
        _TIMING_CURRENT = {"name": name}
        try:
          with _Stage("total_post_load", use_gpu=True):
            # --- Resolve audio ---
            with _Stage("audio_extract", use_gpu=False):
                audio_path, tmp_audio = resolve_audio(
                    audio_path=audio_path_sample, video_path=video_path,
                )

            # --- Compute generation length ---
            num_latent_frames, num_video_frames = compute_generation_length(
                audio_path, args.num_latent_frames, args.chunk_size, args.fps,
                min_latent_frames=args.min_latent_frames,
            )
            _TIMING_CURRENT["num_video_frames"] = num_video_frames

            # --- Optional LatentSync preprocessing ---
            latentsync_metadata = None
            if args.latentsync:
                print("Running LatentSync face detection ...")
                if _TIMING_ENABLED:
                    _gpu_sync()
                _wg_t0 = time.perf_counter()
                with _Stage("face_detect_align", use_gpu=False):
                    latentsync_metadata = preprocess_with_latentsync(
                        video_path, image_processor, args.face_cache_dir,
                        num_frames=num_video_frames,
                    )
                if latentsync_metadata is None:
                    print(f"  [FAIL] LatentSync preprocessing failed, skipping {name}")
                    failed.append(name)
                    continue

            # --- Build conditioning ---
            # _a2d_t0, _enc_to_dec_t0, _e2d_t0 are all set INSIDE build_condition:
            #   _a2d_t0       = before audio_encode (Def 1)
            #   _enc_to_dec_t0 = before vae_encode wrapper (Def 2 broad)
            #   _e2d_t0       = before first vae.encode() call (Def 2 narrow)
            if precomputed_dir is not None:
                condition = build_condition_from_precomputed(
                    precomputed_dir, args.mask_path,
                    num_latent_frames, device, dtype,
                )
            else:
                # Wav2Vec + text already loaded eagerly before the loop

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
                # vae_encode / audio_encode timed inside build_condition().
                condition = build_condition(
                    encoder_vae, wav2vec_model, wav2vec_extractor, ref_frames_np,
                    audio_path, text_embeds, args.mask_path,
                    num_video_frames, num_latent_frames, device, dtype,
                )

            # --- Run inference ---
            print("Running inference ...")
            with _Stage("denoise", use_gpu=True):
                output_latents = run_inference(
                    model, condition, num_latent_frames, args.t_list,
                    args.chunk_size, args.context_noise, args.seed, device, dtype,
                )

            # --- Post-processing: decode + save ---
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            if args.latentsync and latentsync_metadata is not None:
                # LatentSync compositing path — float-space decode + composite
                print("VAE decoding (float) ...")
                with _Stage("pure_vae_decode", use_gpu=True):
                    latent_for_vae = output_latents[0].to(torch.float32)
                    video_decoded = decoder_vae.decode([latent_for_vae], device=device)
                    video_decoded = video_decoded.clamp(-1, 1)

                if _TIMING_ENABLED:
                    _gpu_sync()
                    _now = time.perf_counter()
                    _TIMING_CURRENT["pure_encode_to_decode"] = _now - _e2d_t0
                    _TIMING_CURRENT["audio_to_decode"] = _now - _a2d_t0
                    _TIMING_CURRENT["encode_to_decode"] = _now - _enc_to_dec_t0

                # [1, 3, T_video, H, W] -> [T, 3, H, W] in [0, 1]
                generated_float = video_decoded[0].permute(1, 0, 2, 3)  # [3,T,H,W] -> [T,3,H,W]
                generated_float = ((generated_float + 1) / 2).clamp(0, 1)  # [-1,1] -> [0,1]

                # Composite onto original frames
                print("Compositing ...")
                with _Stage("composite", use_gpu=False):
                    composited_np = composite_with_latentsync_float(
                        generated_float.cpu(), latentsync_metadata, image_processor,
                        use_mouth_only_compositing=args.use_mouth_only,
                    )

                if _TIMING_ENABLED:
                    _TIMING_CURRENT["whole_generation"] = time.perf_counter() - _wg_t0

                with _Stage("save_mux", use_gpu=False):
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
                # Standard decode + save (no LatentSync) — timed as one block
                print("Decoding and saving ...")
                with _Stage("pure_vae_decode", use_gpu=True):
                    decode_and_save(decoder_vae, output_latents, audio_path, output_path,
                                    args.fps, device)
                if _TIMING_ENABLED:
                    _gpu_sync()
                    _now = time.perf_counter()
                    _TIMING_CURRENT["pure_encode_to_decode"] = _now - _e2d_t0
                    _TIMING_CURRENT["audio_to_decode"] = _now - _a2d_t0
                    _TIMING_CURRENT["encode_to_decode"] = _now - _enc_to_dec_t0

            succeeded.append(name)
            print(f"  Done: {output_path}")
          # _Stage("total_post_load") has now exited, so its timing is populated.
          if _TIMING_ENABLED:
              ve = _TIMING_CURRENT.get("pure_vae_encode", 0.0)
              dn = _TIMING_CURRENT.get("denoise", 0.0)
              vd = _TIMING_CURRENT.get("pure_vae_decode", 0.0)
              ae = _TIMING_CURRENT.get("audio_encode", 0.0)
              _TIMING_CURRENT["ablation_1"] = ve + dn + vd
              _TIMING_CURRENT["ablation_2"] = _TIMING_CURRENT.get("pure_encode_to_decode", ve + dn + vd)
              _TIMING_CURRENT["ablation_3"] = _TIMING_CURRENT.get("whole_generation", 0.0)
              _TIMING_CURRENT["ablation_4"] = ae + ve + dn + vd
              _TIMING_ROWS.append(dict(_TIMING_CURRENT))
              parts = [f"{k}={_TIMING_CURRENT[k]:.3f}s" for k in _TIMING_STAGE_ORDER if k in _TIMING_CURRENT]
              print(f"  [Timing] {', '.join(parts)}")

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

    # ===================================================================
    # Timing CSV + averages
    # ===================================================================
    if _TIMING_ENABLED and _TIMING_ROWS:
        if args.timing_csv:
            csv_path = args.timing_csv
        elif args.output_dir:
            csv_path = os.path.join(args.output_dir, "timing.csv")
        elif args.output_path:
            csv_path = args.output_path + ".timing.csv"
        else:
            csv_path = "timing.csv"

        peak_alloc = peak_reserved = 0.0
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / 1e9
            peak_reserved = torch.cuda.max_memory_reserved() / 1e9

        fieldnames = (["name", "num_video_frames"] + _TIMING_STAGE_ORDER
                      + ["peak_alloc_gb", "peak_reserved_gb"])
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in _TIMING_ROWS:
                out_row = {}
                for k in fieldnames:
                    v = row.get(k)
                    if isinstance(v, float):
                        out_row[k] = f"{v:.6f}"
                    elif isinstance(v, int):
                        out_row[k] = str(v)
                    else:
                        out_row[k] = v if v is not None else ""
                # Same lifetime peak on every row.
                out_row["peak_alloc_gb"] = f"{peak_alloc:.3f}"
                out_row["peak_reserved_gb"] = f"{peak_reserved:.3f}"
                writer.writerow(out_row)
            avg = {"name": "AVERAGE"}
            # mean num_video_frames across clips (so sweep can compute fps from CSV alone)
            nvf = [r["num_video_frames"] for r in _TIMING_ROWS if isinstance(r.get("num_video_frames"), int)]
            if nvf:
                avg["num_video_frames"] = f"{sum(nvf)/len(nvf):.2f}"
            for k in _TIMING_STAGE_ORDER:
                vals = [r[k] for r in _TIMING_ROWS if isinstance(r.get(k), float)]
                if vals:
                    avg[k] = f"{sum(vals)/len(vals):.6f}"
            avg["peak_alloc_gb"] = f"{peak_alloc:.3f}"
            avg["peak_reserved_gb"] = f"{peak_reserved:.3f}"
            writer.writerow(avg)
        print(f"\n[Timing] wrote {len(_TIMING_ROWS)} rows + average → {csv_path}")
        print(f"[Timing] per-stage average (s) | FPS (num_video_frames / stage_time):")
        nvf = [r["num_video_frames"] for r in _TIMING_ROWS if isinstance(r.get("num_video_frames"), int)]
        mean_nvf = (sum(nvf) / len(nvf)) if nvf else 0.0
        for k in _TIMING_STAGE_ORDER:
            vals = [r[k] for r in _TIMING_ROWS if isinstance(r.get(k), float)]
            if vals:
                mean_t = sum(vals) / len(vals)
                fps = (mean_nvf / mean_t) if mean_t > 0 else 0.0
                print(f"  {k:20s} {mean_t:7.4f} s   {fps:7.2f} fps")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        print(f"[VRAM] peak_allocated={peak_gb:.2f} GB peak_reserved={reserved_gb:.2f} GB")


if __name__ == "__main__":
    main()
