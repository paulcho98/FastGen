#!/usr/bin/env python3
"""Streaming inference — per-chunk AR generation with decode-as-you-go.

Generates lip-synced video using the streaming pipeline: each AR chunk is
denoised, decoded, and composited before moving to the next. This enables
first-frame output before the full video is generated.

Supports three decoder modes:
  - StreamingTAEHV: temporal state across chunks (no boundary artifacts)
  - Batch TAEHV: independent per-chunk decode (faster, possible boundary seams)
  - Wan VAE: full VAE decode per chunk (highest quality, slowest)

Usage:
    python scripts/inference/inference_streaming.py \
        --video_path /path/to/reference.mp4 \
        --output_path /path/to/output.mp4 \
        --ckpt_path /path/to/sf_trained_student.pth \
        --vae_path /path/to/Wan2.1_VAE.pth \
        --wav2vec_path /path/to/wav2vec2-base-960h \
        --mask_path /path/to/mask.png \
        --streaming_decoder streaming_taehv \
        --taehv_ckpt /path/to/taew2_1.pth
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch

# Import shared utilities from the throughput script
from inference_causal_taehv import (
    _gpu_sync, _Stage, _get_ffmpeg,
    _TIMING_ENABLED, _TIMING_CURRENT, _TIMING_ROWS, _TIMING_STAGE_ORDER,
    _e2d_t0,
    load_diffusion_model, load_vae, load_wav2vec, load_or_encode_text,
    TAEHVDecoderWrapper, StreamingTAEHVDecoderWrapper,
    resolve_audio, compute_generation_length,
    load_image_processor, preprocess_with_latentsync,
    build_condition, build_condition_from_precomputed,
    composite_with_latentsync_float,
    save_frames_as_video, mux_video_with_audio,
    load_and_adjust_video,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Streaming inference with per-chunk decode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model paths ---
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--wav2vec_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--base_model_paths", type=str, default=None)
    parser.add_argument("--omniavatar_ckpt_path", type=str, default=None)
    parser.add_argument("--text_embeds_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--neg_prompt", type=str, default=None)
    parser.add_argument("--neg_text_emb_path", type=str, default=None)

    # --- TAEHV ---
    parser.add_argument("--taehv_ckpt", type=str, default=None)

    # --- Streaming decoder mode ---
    parser.add_argument("--streaming_decoder", type=str, default="streaming_taehv",
                        choices=["streaming_taehv", "batch_taehv", "wan_vae"],
                        help="Decoder mode for streaming pipeline.")

    # --- Input/output ---
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="output.mp4")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")

    # --- Generation params ---
    parser.add_argument("--t_list", nargs="+", type=float, default=[0.999, 0.833, 0.0])
    parser.add_argument("--chunk_size", type=int, default=3)
    parser.add_argument("--num_latent_frames", type=int, default=None)
    parser.add_argument("--min_latent_frames", type=int, default=None)
    parser.add_argument("--context_noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")

    # --- Attention ---
    parser.add_argument("--local_attn_size", type=int, default=None)
    parser.add_argument("--sink_size", type=int, default=0)
    parser.add_argument("--use_dynamic_rope", action="store_true")

    # --- LatentSync ---
    parser.add_argument("--latentsync", action="store_true")
    parser.add_argument("--face_cache_dir", type=str, default=None)
    parser.add_argument("--use_mouth_only", action="store_true")

    # --- Timing ---
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--timing_csv", type=str, default=None)

    return parser.parse_args()


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
                continue
            precomputed = sample_dir if os.path.isfile(
                os.path.join(sample_dir, "vae_latents_mask_all.pt")
            ) else None
            yield entry, video_path, audio_path, precomputed
    else:
        name = os.path.splitext(os.path.basename(args.video_path))[0]
        yield name, args.video_path, args.audio_path, None


def run_streaming_pipeline(
    model, decoder_vae, vae, condition, num_latent_frames, num_video_frames,
    args, latentsync_metadata, image_processor, audio_path, output_path,
    device, dtype,
):
    """Run the streaming pipeline: per-chunk denoise → decode → composite.

    Returns:
        composited_np: [N, H, W, 3] uint8 numpy array of composited frames.
    """
    global _TIMING_CURRENT, _e2d_t0

    # --- Decoder selection ---
    _use_streaming_dec = False
    if args.streaming_decoder == "streaming_taehv":
        try:
            from taehv import StreamingTAEHV
            if hasattr(decoder_vae, 'taehv') and decoder_vae.taehv is not None:
                streaming_dec = StreamingTAEHV(decoder_vae.taehv)
                _use_streaming_dec = True
                print("  Using StreamingTAEHV decoder (temporal state across chunks)")
        except Exception:
            pass
    if not _use_streaming_dec:
        if args.streaming_decoder == "wan_vae":
            print("  Using Wan VAE decoder per chunk")
        else:
            print("  Using batch TAEHV decoder per chunk")

    # --- Free encoder memory if decoder is different ---
    if decoder_vae is not vae and hasattr(vae, 'parameters'):
        vae.to("cpu")
    torch.cuda.empty_cache()

    # --- Prepare model ---
    model.total_num_frames = num_latent_frames
    model.clear_caches()
    B, C = 1, 16
    H_lat, W_lat = condition["ref_latent"].shape[3], condition["ref_latent"].shape[4]
    t_list_t = torch.tensor(args.t_list, device=device, dtype=torch.float64)

    # Pre-generate all noise at once (must match non-streaming pipeline)
    torch.manual_seed(args.seed)
    all_noise = torch.randn(B, C, num_latent_frames, H_lat, W_lat, device=device, dtype=dtype)

    num_blocks = num_latent_frames // args.chunk_size
    all_composited_frames = []
    video_frame_offset = 0

    if _TIMING_ENABLED:
        _gpu_sync()
    streaming_t0 = time.perf_counter()
    first_frame_done = False

    # Timing marks for first-frame latency
    ff_gen_t0 = None
    ff_pure_gen_t0 = None
    ff_enc_gen_dec_t0 = None

    for block_idx in range(num_blocks):
        cur_start_frame = block_idx * args.chunk_size

        # --- Timing marks for first chunk ---
        if block_idx == 0 and _TIMING_ENABLED:
            _gpu_sync()
            ff_gen_t0 = time.perf_counter()
            ff_enc_gen_dec_t0 = time.perf_counter()
            ff_pure_gen_t0 = time.perf_counter()

        # --- Per-chunk denoise ---
        noisy_input = all_noise[:, :, cur_start_frame:cur_start_frame + args.chunk_size]
        for step_idx in range(len(t_list_t) - 1):
            t_cur = t_list_t[step_idx]
            t_next = t_list_t[step_idx + 1]
            x0_pred = model(
                noisy_input, t_cur.expand(B),
                condition=condition,
                cur_start_frame=cur_start_frame,
                store_kv=False, is_ar=True,
                fwd_pred_type="x0", use_gradient_checkpointing=False,
            )
            if t_next > 0:
                eps = torch.randn_like(x0_pred)
                noisy_input = model.noise_scheduler.forward_process(
                    x0_pred, eps, t_next.expand(B))
            else:
                noisy_input = x0_pred

        # --- Per-chunk decode ---
        if _use_streaming_dec:
            chunk_latent = x0_pred[0].to(device, dtype=torch.float16)
            chunk_latent_ntchw = chunk_latent.permute(1, 0, 2, 3).unsqueeze(0)
            chunk_frames = []
            for t in range(chunk_latent_ntchw.shape[1]):
                latent_t = chunk_latent_ntchw[:, t:t+1]
                frame = streaming_dec.decode(latent_t)
                while frame is not None:
                    chunk_frames.append(frame)
                    frame = streaming_dec.decode()
            if chunk_frames:
                chunk_float = torch.cat(chunk_frames, dim=1).squeeze(0)
            else:
                chunk_float = None
        else:
            chunk_latent = x0_pred[0].to(torch.float32)
            chunk_decoded = decoder_vae.decode([chunk_latent], device=device)
            chunk_decoded = chunk_decoded.clamp(-1, 1)
            chunk_float = chunk_decoded[0].permute(1, 0, 2, 3)
            chunk_float = ((chunk_float + 1) / 2).clamp(0, 1)

        # --- First-frame timing ---
        if not first_frame_done and _TIMING_ENABLED and chunk_float is not None:
            _gpu_sync()
            ff_now = time.perf_counter()
            _TIMING_CURRENT["ff_full_pipeline"] = ff_now - streaming_t0
            if ff_pure_gen_t0:
                _TIMING_CURRENT["ff_pure_generation"] = ff_now - ff_pure_gen_t0
            if ff_enc_gen_dec_t0:
                _TIMING_CURRENT["ff_enc_gen_dec"] = ff_now - ff_enc_gen_dec_t0
            if _e2d_t0:
                _TIMING_CURRENT["ff_pure_enc_to_dec"] = ff_now - _e2d_t0
            first_frame_done = True

        # --- Per-chunk composite ---
        if chunk_float is not None:
            chunk_float_cpu = chunk_float.cpu()
            composited = composite_with_latentsync_float(
                chunk_float_cpu, latentsync_metadata, image_processor,
                use_mouth_only_compositing=args.use_mouth_only,
                frame_offset=video_frame_offset,
            )
            all_composited_frames.append(composited)
            video_frame_offset += composited.shape[0]

            if block_idx == 0 and _TIMING_ENABLED and ff_gen_t0 and "ff_generation" not in _TIMING_CURRENT:
                _TIMING_CURRENT["ff_generation"] = time.perf_counter() - ff_gen_t0

        # --- Update KV cache ---
        cache_input = x0_pred
        t_cache = torch.full((B,), args.context_noise, device=device, dtype=torch.float64)
        if args.context_noise > 0:
            cache_eps = torch.randn_like(x0_pred)
            cache_input = model.noise_scheduler.forward_process(
                x0_pred, cache_eps,
                torch.tensor(args.context_noise, device=device, dtype=torch.float64).expand(B))
        model(cache_input, t_cache, condition=condition,
              cur_start_frame=cur_start_frame, store_kv=True, is_ar=True,
              fwd_pred_type="x0", use_gradient_checkpointing=False)

        if (block_idx + 1) % 5 == 0 or block_idx == num_blocks - 1:
            print(f"  Streaming block {block_idx + 1}/{num_blocks} done")

    model.clear_caches()

    # Flush StreamingTAEHV decoder
    if _use_streaming_dec:
        flush_frames = streaming_dec.flush_decoder()
        if flush_frames:
            flush_float = torch.cat(flush_frames, dim=1).squeeze(0)
            flush_cpu = flush_float.cpu()
            composited = composite_with_latentsync_float(
                flush_cpu, latentsync_metadata, image_processor,
                use_mouth_only_compositing=args.use_mouth_only,
                frame_offset=video_frame_offset,
            )
            all_composited_frames.append(composited)
            video_frame_offset += composited.shape[0]

    if _TIMING_ENABLED:
        _gpu_sync()
        _TIMING_CURRENT["streaming_total"] = time.perf_counter() - streaming_t0
        audio_enc_time = _TIMING_CURRENT.get("audio_encode", 0.0)
        enc_gen_dec = _TIMING_CURRENT.get("ff_enc_gen_dec", 0.0)
        if audio_enc_time and enc_gen_dec:
            _TIMING_CURRENT["ff_enc_audio_gen_dec"] = audio_enc_time + enc_gen_dec

    composited_np = np.concatenate(all_composited_frames, axis=0)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_frames_as_video(composited_np, output_path, fps=args.fps)
    video_duration = composited_np.shape[0] / args.fps
    tmp_composited = output_path + ".tmp.mp4"
    os.rename(output_path, tmp_composited)
    mux_video_with_audio(tmp_composited, audio_path, output_path, duration_s=video_duration)
    if os.path.exists(tmp_composited):
        os.remove(tmp_composited)

    return composited_np


def main():
    global _TIMING_ENABLED, _TIMING_CURRENT, _TIMING_ROWS

    import inference_causal_taehv as _base
    args = parse_args()

    _TIMING_ENABLED = bool(args.timing)
    _base._TIMING_ENABLED = _TIMING_ENABLED

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Load models ---
    print("Loading diffusion model ...")
    model = load_diffusion_model(args, device, dtype)

    print("Loading VAE ...")
    vae = load_vae(args.vae_path, device)

    # Decoder selection based on streaming_decoder mode
    if args.streaming_decoder in ("streaming_taehv", "batch_taehv"):
        if not args.taehv_ckpt:
            raise ValueError(f"--streaming_decoder {args.streaming_decoder} requires --taehv_ckpt")
        print(f"Loading TAEHV decoder from {args.taehv_ckpt} ...")
        decoder_vae = TAEHVDecoderWrapper(args.taehv_ckpt, device)
    else:
        decoder_vae = vae

    encoder_vae = vae

    # Eagerly load Wav2Vec + text
    print("Loading Wav2Vec2 (eager) ...")
    wav2vec_model, wav2vec_extractor = load_wav2vec(args.wav2vec_path, device)
    _dummy_audio = np.zeros(16000, dtype=np.float32)
    _dummy_input = wav2vec_extractor(_dummy_audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        wav2vec_model(_dummy_input.input_values.to(device))
    print("Wav2Vec2 warmed up.")

    text_embeds = None
    if args.text_embeds_path or args.prompt:
        print("Loading text embeddings ...")
        text_embeds = load_or_encode_text(args, device, dtype)

    # LatentSync
    image_processor = None
    if args.latentsync:
        image_processor = load_image_processor(args.mask_path, device)

    # --- Loop over samples ---
    samples = list(enumerate_samples(args))
    succeeded, failed, skipped = [], [], []

    for sample_idx, (name, video_path, audio_path_sample, precomputed_dir) in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"[{sample_idx+1}/{len(samples)}] {name}")
        print(f"{'='*60}")

        if args.input_dir is not None:
            output_path = os.path.join(args.output_dir, f"{name}.mp4")
        else:
            output_path = args.output_path

        if args.skip_existing and os.path.isfile(output_path):
            print(f"  [Skip] {output_path}")
            skipped.append(name)
            continue

        tmp_audio = None
        _TIMING_CURRENT = {"name": name}
        _base._TIMING_CURRENT = _TIMING_CURRENT

        try:
          with _Stage("total_post_load", use_gpu=True):
            with _Stage("audio_extract", use_gpu=False):
                audio_path, tmp_audio = resolve_audio(
                    audio_path=audio_path_sample, video_path=video_path,
                )

            num_latent_frames, num_video_frames = compute_generation_length(
                audio_path, args.num_latent_frames, args.chunk_size, args.fps,
                min_latent_frames=args.min_latent_frames,
            )
            _TIMING_CURRENT["num_video_frames"] = num_video_frames

            # LatentSync preprocessing
            latentsync_metadata = None
            if args.latentsync:
                print("Running LatentSync face detection ...")
                with _Stage("face_detect", use_gpu=False):
                    latentsync_metadata = preprocess_with_latentsync(
                        video_path, image_processor, args.face_cache_dir,
                        num_frames=num_video_frames,
                    )
                if latentsync_metadata is None:
                    print(f"  [FAIL] LatentSync preprocessing failed")
                    failed.append(name)
                    continue

            if not args.latentsync or latentsync_metadata is None:
                raise ValueError("Streaming pipeline requires --latentsync")

            # Build conditioning
            if precomputed_dir is not None:
                condition = build_condition_from_precomputed(
                    precomputed_dir, args.mask_path,
                    num_latent_frames, device, dtype,
                )
            else:
                aligned_faces = latentsync_metadata["aligned_faces"]
                ref_frames_np = np.stack([
                    f.permute(1, 2, 0).numpy() if isinstance(f, torch.Tensor) else f
                    for f in aligned_faces[:num_video_frames]
                ], axis=0)

                print("Building conditioning ...")
                condition = build_condition(
                    encoder_vae, wav2vec_model, wav2vec_extractor, ref_frames_np,
                    audio_path, text_embeds, args.mask_path,
                    num_video_frames, num_latent_frames, device, dtype,
                )

            # Run streaming pipeline
            print(f"Running streaming pipeline ({args.streaming_decoder}) ...")
            run_streaming_pipeline(
                model, decoder_vae, vae, condition,
                num_latent_frames, num_video_frames,
                args, latentsync_metadata, image_processor,
                audio_path, output_path, device, dtype,
            )

            succeeded.append(name)
            if _TIMING_ENABLED:
                ve = _TIMING_CURRENT.get("vae_encode", 0.0)
                dn = _TIMING_CURRENT.get("denoise", 0.0)
                vd = _TIMING_CURRENT.get("vae_decode", 0.0)
                ae = _TIMING_CURRENT.get("audio_encode", 0.0)
                _TIMING_CURRENT["ablation_1"] = ve + dn + vd
                _TIMING_CURRENT["ablation_2"] = _TIMING_CURRENT.get("encode_to_decode", ve + dn + vd)
                _TIMING_CURRENT["ablation_3"] = _TIMING_CURRENT.get("streaming_total", 0.0)
                _TIMING_CURRENT["ablation_4"] = ae + ve + dn + vd
                ff_egd = _TIMING_CURRENT.get("ff_enc_gen_dec", 0.0)
                ff_pe2d = _TIMING_CURRENT.get("ff_pure_enc_to_dec", 0.0)
                ff_fp = _TIMING_CURRENT.get("ff_full_pipeline", 0.0)
                ff_eagd = _TIMING_CURRENT.get("ff_enc_audio_gen_dec", 0.0)
                _TIMING_CURRENT["ff_ablation_1"] = ff_egd
                _TIMING_CURRENT["ff_ablation_2"] = ff_pe2d
                _TIMING_CURRENT["ff_ablation_3"] = ff_fp
                _TIMING_CURRENT["ff_ablation_4"] = ff_eagd
                _TIMING_ROWS.append(dict(_TIMING_CURRENT))
                parts = [f"{k}={_TIMING_CURRENT[k]:.3f}s"
                         for k in _TIMING_STAGE_ORDER if k in _TIMING_CURRENT]
                print(f"  [Timing] {', '.join(parts)}")
            print(f"  Done: {output_path}")

        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)
        finally:
            if tmp_audio is not None and os.path.exists(tmp_audio):
                os.remove(tmp_audio)
            torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Summary: {len(succeeded)} succeeded, {len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print(f"  Failed: {failed}")

    # --- Timing CSV ---
    if _TIMING_ENABLED and _TIMING_ROWS:
        csv_path = args.timing_csv or (args.output_path + ".timing.csv")
        fieldnames = ["name", "num_video_frames"] + _TIMING_STAGE_ORDER
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
                writer.writerow(out_row)
        print(f"\n[Timing] wrote {len(_TIMING_ROWS)} rows → {csv_path}")


if __name__ == "__main__":
    main()
