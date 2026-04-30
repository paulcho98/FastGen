#!/usr/bin/env python3
"""Run inference_causal_taehv on a long video by slicing into 81-frame chunks.

The full streaming pipeline OOMs on >300-frame videos due to cross-attention
to the entire ref_sequence. This wrapper:
  1. Slices the input video into N consecutive 81-frame sub-clips with ffmpeg
  2. Loads the model once
  3. Runs single-clip inference per slice, accumulating timings
  4. Reports summed audio_to_decode / encode_to_decode / pure_encode_to_decode

Same CLI as inference_causal_taehv, plus the timing CSV holds one summed row.
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
import shutil

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))

import inference_causal_taehv as base
import torch  # noqa: E402

# Frames per slice — matches our model's 21-latent x 4 + 1 = 81 video frames.
SLICE_FRAMES = 81
FPS = 25.0


def ffmpeg_slice(video_path: str, slice_dir: str) -> list[tuple[str, str]]:
    """Slice video into 81-frame mp4 + matching wav using ffmpeg.

    Returns list of (sub_clip.mp4, audio.wav) paths.
    """
    os.makedirs(slice_dir, exist_ok=True)
    # Probe duration
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path]
    ).decode().strip()
    total_dur = float(out)
    slice_dur = SLICE_FRAMES / FPS  # 3.24s
    n_slices = max(1, int(total_dur // slice_dur))

    pairs = []
    for i in range(n_slices):
        sub_dir = os.path.join(slice_dir, f"slice_{i:03d}")
        os.makedirs(sub_dir, exist_ok=True)
        sub_mp4 = os.path.join(sub_dir, "sub_clip.mp4")
        sub_wav = os.path.join(sub_dir, "audio.wav")
        start = i * slice_dur
        # Video slice (re-encode for exact frame count)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", f"{start:.4f}", "-i", video_path,
             "-frames:v", str(SLICE_FRAMES), "-r", "25",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-an", sub_mp4],
            check=True,
        )
        # Audio slice (16kHz mono pcm wav)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", f"{start:.4f}", "-i", video_path,
             "-t", f"{slice_dur:.4f}",
             "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", sub_wav],
            check=True,
        )
        pairs.append((sub_mp4, sub_wav))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--wav2vec_path", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--base_model_paths", default=None)
    parser.add_argument("--omniavatar_ckpt_path", default=None)
    parser.add_argument("--text_embeds_path", default=None)
    parser.add_argument("--text_encoder_path", default=None)
    parser.add_argument("--prompt", default="a person talking")
    parser.add_argument("--taehv_ckpt", default=None)
    parser.add_argument("--use_taehv", action="store_true")
    parser.add_argument("--t_list", nargs="+", type=float, default=[0.999, 0.833, 0.0])
    parser.add_argument("--chunk_size", type=int, default=3)
    parser.add_argument("--local_attn_size", type=int, default=7)
    parser.add_argument("--sink_size", type=int, default=1)
    parser.add_argument("--use_dynamic_rope", action="store_true")
    parser.add_argument("--latentsync", action="store_true")
    parser.add_argument("--face_cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context_noise", type=float, default=0.0)
    parser.add_argument("--timing_csv", required=True)
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.video_path))[0]
    print(f"[60s-chunked] {name}: slicing into {SLICE_FRAMES}-frame sub-clips ...")

    with tempfile.TemporaryDirectory(prefix=f"sliced_{name}_") as slice_root:
        pairs = ffmpeg_slice(args.video_path, slice_root)
        n_slices = len(pairs)
        print(f"[60s-chunked] {name}: {n_slices} slices.")

        # Build a synthetic args namespace for base.main()'s loop body, but call
        # the inference flow manually so we control per-slice CSV rows.
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map[args.dtype]
        device = torch.device(args.device)

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        base._TIMING_ENABLED = True
        base._TIMING_ROWS = []

        # Prepend args needed by base.main()-style loaders
        args.num_latent_frames = 21
        args.min_latent_frames = 21
        args.fps = FPS
        args.audio_path = None
        args.input_dir = None
        args.output_dir = None
        args.precomputed_dir = None
        args.taehv_streaming = False
        args.taehv_encode = False
        args.streaming_pipeline = None
        args.no_streaming_taehv = False
        args.use_mouth_only = False
        args.skip_existing = False
        args.timing = True
        args.neg_prompt = None
        args.neg_text_emb_path = None

        print("Loading diffusion model ...")
        model = base.load_diffusion_model(args, device, dtype)
        print("Loading VAE ...")
        vae = base.load_vae(args.vae_path, device)

        if args.use_taehv:
            print("Loading TAEHV ...")
            decoder_vae = base.TAEHVDecoderWrapper(args.taehv_ckpt, device)
        else:
            decoder_vae = vae
        encoder_vae = vae

        print("Loading Wav2Vec2 ...")
        wav2vec_model, wav2vec_extractor = base.load_wav2vec(args.wav2vec_path, device)
        # Warmup
        _dummy = np.zeros(16000, dtype=np.float32)
        _di = wav2vec_extractor(_dummy, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            wav2vec_model(_di.input_values.to(device), seq_len=25, output_hidden_states=True)

        text_embeds = base.load_or_encode_text(args, device, dtype) if (args.text_embeds_path or args.prompt) else None

        image_processor = None
        if args.latentsync:
            image_processor = base.load_image_processor(args.mask_path, device)

        # Sums for the three measurements
        sums = {"audio_to_decode": 0.0, "encode_to_decode": 0.0, "pure_encode_to_decode": 0.0,
                "num_video_frames": 0}

        for i, (sub_mp4, sub_wav) in enumerate(pairs):
            stem = f"{name}_slice{i:03d}"
            print(f"\n[Slice {i+1}/{n_slices}] {stem}")
            # Per-slice timing dict
            base._TIMING_CURRENT = {"name": stem}

            num_latent_frames, num_video_frames = base.compute_generation_length(
                sub_wav, 21, args.chunk_size, FPS, min_latent_frames=21,
            )
            base._TIMING_CURRENT["num_video_frames"] = num_video_frames

            latentsync_metadata = None
            if args.latentsync:
                with base._Stage("face_detect_align", use_gpu=False):
                    latentsync_metadata = base.preprocess_with_latentsync(
                        sub_mp4, image_processor, args.face_cache_dir, num_frames=num_video_frames,
                    )
                aligned = latentsync_metadata["aligned_faces"][:num_video_frames]
                video_frames_np = np.stack([
                    f.permute(1, 2, 0).numpy() if isinstance(f, torch.Tensor) else f
                    for f in aligned
                ], axis=0)
            else:
                video_frames_np = base.load_and_adjust_video(sub_mp4, num_video_frames)

            condition = base.build_condition(
                encoder_vae, wav2vec_model, wav2vec_extractor, video_frames_np,
                sub_wav, text_embeds, args.mask_path,
                num_video_frames, num_latent_frames, device, dtype,
            )

            with base._Stage("denoise", use_gpu=True):
                output_latents = base.run_inference(
                    model, condition, num_latent_frames, args.t_list,
                    args.chunk_size, args.context_noise, args.seed, device, dtype,
                )

            with base._Stage("pure_vae_decode", use_gpu=True):
                with torch.no_grad():
                    latent_for_dec = output_latents[0].to(torch.float32)
                    decoded = decoder_vae.decode([latent_for_dec], device=device)
            base._gpu_sync()
            _now = time.perf_counter()
            base._TIMING_CURRENT["pure_encode_to_decode"] = _now - base._e2d_t0
            base._TIMING_CURRENT["audio_to_decode"] = _now - base._a2d_t0
            base._TIMING_CURRENT["encode_to_decode"] = _now - base._enc_to_dec_t0

            # Accumulate
            for k in sums:
                v = base._TIMING_CURRENT.get(k, 0.0)
                sums[k] += v if k != "num_video_frames" else int(v)

            # Free per-slice tensors
            del condition, output_latents, decoded, latentsync_metadata
            torch.cuda.empty_cache()

        peak_alloc = peak_reserved = 0.0
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / 1e9
            peak_reserved = torch.cuda.max_memory_reserved() / 1e9

        # Write summed row
        os.makedirs(os.path.dirname(os.path.abspath(args.timing_csv)) or ".", exist_ok=True)
        with open(args.timing_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "num_video_frames", "audio_to_decode",
                        "encode_to_decode", "pure_encode_to_decode", "num_slices",
                        "peak_alloc_gb", "peak_reserved_gb"])
            w.writerow([
                name, sums["num_video_frames"],
                f"{sums['audio_to_decode']:.6f}",
                f"{sums['encode_to_decode']:.6f}",
                f"{sums['pure_encode_to_decode']:.6f}",
                n_slices,
                f"{peak_alloc:.3f}", f"{peak_reserved:.3f}",
            ])
        print(f"\n[60s-chunked] {name}: summed over {n_slices} slices "
              f"({sums['num_video_frames']} frames):")
        print(f"  audio_to_decode       = {sums['audio_to_decode']:.3f}s")
        print(f"  encode_to_decode      = {sums['encode_to_decode']:.3f}s")
        print(f"  pure_encode_to_decode = {sums['pure_encode_to_decode']:.3f}s")
        print(f"[VRAM] peak_allocated={peak_alloc:.2f} GB peak_reserved={peak_reserved:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
