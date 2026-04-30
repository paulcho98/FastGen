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
    encode_audio, frames_to_tensor, apply_spatial_mask, load_latentsync_mask,
)
from PIL import Image
import cv2


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

    # --- Streamwise encoding (truly interleaved encode/denoise/decode) ---
    parser.add_argument("--streamwise_encode", action="store_true",
                        help="Encode source video chunk-by-chunk inside the AR "
                             "loop (encoder feat_cache preserved across chunks). "
                             "Each AR block: encode 9 (block 0) or 12 frames -> "
                             "denoise 3 latents -> decode 3 latents.")

    # --- Deferred compositing (move lip-blend + affine warp out of AR loop) ---
    parser.add_argument("--defer_composite", action="store_true",
                        help="Skip per-block compositing inside the AR loop; "
                             "concat all decoded chunks and run "
                             "composite_with_latentsync_float once after the "
                             "loop ends. Improves throughput (no per-block "
                             ".cpu() sync) and makes timing comparable to the "
                             "non-streaming path which excludes composite. "
                             "Trades first-frame latency for total throughput.")

    # --- torch.compile ---
    parser.add_argument("--compile", action="store_true",
                        help="Wrap diffusion model + Wan VAE encoder/decoder + "
                             "TAEHV (when present) with torch.compile.")

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


def build_condition_streamwise(vae, wav2vec_model, wav2vec_extractor,
                                video_frames_np, audio_path, text_embeds,
                                mask_path, num_video_frames, num_latent_frames,
                                device, dtype):
    """Build a *minimal* condition dict for streamwise AR inference.

    Encodes only audio (full upfront) and the very first ref_latent (1 frame).
    Returns the condition with ref_sequence/masked_latents = None plus the
    pixel-space video tensors that the AR loop will encode incrementally.

    The three timing markers (_a2d_t0, _enc_to_dec_t0, _e2d_t0) are set on the
    base module so downstream timing math matches the non-streamwise path.
    """
    import inference_causal_taehv as _base

    # ============================================================
    # STAGE 2: Wav2Vec2 audio encode (full audio at once)
    # ============================================================
    if _base._TIMING_ENABLED:
        _base._gpu_sync()
    _base._a2d_t0 = time.perf_counter()
    print("Encoding audio (full) ...")
    with _Stage("audio_encode", use_gpu=True):
        audio_emb = encode_audio(
            wav2vec_model, wav2vec_extractor, audio_path, num_video_frames, device
        )
    audio_emb = audio_emb.to(dtype=dtype)

    # ============================================================
    # STAGE 3a (streamwise stub): pixel-space tensors only, no encode
    # ============================================================
    # The reference video stays in pixel space here. STAGE 3b inside the
    # AR loop encodes it chunk-by-chunk via streaming_encode_chunk.
    # _enc_to_dec_t0 is set in run_streaming_pipeline right before the AR
    # loop; _e2d_t0 is set inside the AR loop right before the very first
    # streaming_encode_chunk() call. We reset it to None here so the
    # marker captures only the actual first encode of THIS sample.
    _base._e2d_t0 = None

    H, W = 512, 512
    video_tensor = frames_to_tensor(video_frames_np)  # [1, 3, T, H, W] in [-1, 1]

    mask_img = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    if mask_np.shape[0] != H or mask_np.shape[1] != W:
        mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_pixel_binary = (mask_np > 0.5).astype(np.float32)
    masked_video_tensor = apply_spatial_mask(
        video_tensor, mask_pixel_binary, mask_all_frames=True
    )

    # No VAE encoding here. The AR loop in run_streaming_pipeline will encode
    # both the unmasked (ref) and masked streams chunk-by-chunk, in lockstep
    # with denoise + decode.
    H_lat = H // 8
    W_lat = W // 8
    latent_mask = load_latentsync_mask(mask_path, H_lat, W_lat).to(device=device, dtype=dtype)

    condition = {
        "text_embeds": text_embeds,
        "audio_emb": audio_emb,
        "ref_latent": None,       # set on block 0 from growing_ref_seq[..., :1]
        "mask": latent_mask.to(device=device),
        "masked_video": None,   # built incrementally in AR loop
        "ref_sequence": None,     # built incrementally in AR loop
    }
    return condition, video_tensor, masked_video_tensor


@torch.no_grad()
def run_streaming_pipeline(
    model, decoder_vae, vae, condition, num_latent_frames, num_video_frames,
    args, latentsync_metadata, image_processor, audio_path, output_path,
    device, dtype,
    video_tensor=None, masked_video_tensor=None,
):
    """Run the streaming pipeline: per-chunk denoise → decode → composite.

    Stage map (mirrors the labelling used for X-Dub / LatentSync):
        STAGE 1  : Face detect + 512x512 alignment    (handled before this
                   function, in main() via preprocess_with_latentsync)
        STAGE 2  : Wav2Vec2 audio encode               (handled in build_condition[_streamwise])
        STAGE 3a : Reference VAE encode (full or first-frame only)
        STAGE 3b : (streamwise only) per-block VAE encode of unmasked + masked
        STAGE 4  : Per-block 2-step DDIM denoise (CausalOmniAvatarWan)
        STAGE 5  : Per-block VAE decode (Wan VAE / TAEHV / StreamingTAEHV)
        STAGE 6  : Per-block compositing (paste lip region back into full-res frame)
        STAGE 7  : Per-block KV cache update (model forward with denoised x0 stored as cache)
        STAGE 8  : (streaming_taehv only) flush remaining buffered frames
        STAGE 9  : Save MP4 + ffmpeg audio mux         (handled by caller after return)

    Returns:
        composited_np: [N, H, W, 3] uint8 numpy array of composited frames.
    """
    global _TIMING_CURRENT, _e2d_t0

    # ============================================================
    # STAGE 5-prep: Decoder selection
    # ============================================================
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

    # ============================================================
    # VRAM management: offload encoder to CPU when only the decoder is needed
    # ============================================================
    # In streamwise_encode mode the encoder is needed inside the AR loop, so
    # keep it on GPU even when the decoder is a different module (e.g. TAEHV).
    if (decoder_vae is not vae and hasattr(vae, 'parameters')
            and not args.streamwise_encode):
        vae.to("cpu")
    torch.cuda.empty_cache()

    # ============================================================
    # STAGE 5-prep: Wan VAE streaming-decode cache reset
    # ============================================================
    # For wan_vae streaming, reset feat_cache once before AR loop. Subsequent
    # streaming_decode_chunk() calls preserve cache across chunks so output
    # at chunk boundaries is bit-identical to a single full-length decode.
    _use_wan_streaming = (args.streaming_decoder == "wan_vae"
                          and hasattr(decoder_vae, "streaming_decode_chunk"))
    if _use_wan_streaming:
        print("  Wan VAE: streaming-decode mode (cache continuity across chunks)")
        decoder_vae.reset_decode_cache()

    # ============================================================
    # STAGE 3b-prep: Streamwise encode setup (only if --streamwise_encode)
    # ============================================================
    _streamwise = args.streamwise_encode and video_tensor is not None
    if _streamwise:
        print("  Wan VAE: streamwise-encode mode (encoder in AR loop)")
        # Two independent encoder feat_cache streams: one for unmasked
        # (ref_sequence) and one for masked (masked_latents). We share a
        # single VAE instance and swap cache state between calls.
        vae.reset_encode_cache()
        unmasked_state = vae.save_encode_cache_state()  # both empty
        masked_state = vae.save_encode_cache_state()
        growing_ref_seq = None     # [1, 16, T_so_far, H_lat, W_lat]
        growing_masked = None
        original_vae_dtype = next(vae.parameters()).dtype
        vae.to(dtype=torch.bfloat16)

        def _frame_chunks_for_block(block_idx):
            """Return list of (start, end) frame indices for an AR block.
            Block 0: 1 + 4 + 4 = 9 frames (3 latents).
            Block i>=1: 4 + 4 + 4 = 12 frames (3 latents).
            """
            if block_idx == 0:
                return [(0, 1), (1, 5), (5, 9)]
            base = 9 + 12 * (block_idx - 1)
            return [(base, base + 4), (base + 4, base + 8), (base + 8, base + 12)]

        def _stream_encode_block(block_idx, source_video_tensor, prev_state):
            """Encode the next AR block's worth of frames into 3 latents."""
            import inference_causal_taehv as _base
            vae.load_encode_cache_state(prev_state)
            chunks = []
            for s, e in _frame_chunks_for_block(block_idx):
                chunk = source_video_tensor[0, :, s:e].to(
                    dtype=torch.bfloat16, device=device)
                # Set _e2d_t0 right before the very first encode call of
                # this sample (across all blocks and both streams).
                if _base._e2d_t0 is None:
                    if _TIMING_ENABLED:
                        _gpu_sync()
                    _base._e2d_t0 = time.perf_counter()
                chunks.append(vae.streaming_encode_chunk(chunk, device=device))
            new_state = vae.save_encode_cache_state()
            # chunks are [1, 16, 1, H, W]; concat along time
            new_lats = torch.cat([c.squeeze(0) for c in chunks], dim=1).unsqueeze(0)
            return new_lats.to(dtype=dtype), new_state

    # --- Prepare model ---
    model.total_num_frames = num_latent_frames
    model.clear_caches()
    B, C = 1, 16
    if condition.get("ref_latent") is not None:
        H_lat, W_lat = condition["ref_latent"].shape[3], condition["ref_latent"].shape[4]
    else:
        # streamwise mode: derive from pixel-space video tensor (8x spatial compression)
        H_lat = video_tensor.shape[-2] // 8
        W_lat = video_tensor.shape[-1] // 8
    t_list_t = torch.tensor(args.t_list, device=device, dtype=torch.float64)

    # Pre-generate all noise at once (must match non-streaming pipeline)
    torch.manual_seed(args.seed)
    all_noise = torch.randn(B, C, num_latent_frames, H_lat, W_lat, device=device, dtype=dtype)

    num_blocks = num_latent_frames // args.chunk_size
    all_composited_frames = []
    # When --defer_composite is on, we skip per-block compositing and stash
    # the raw decoded chunk_float tensors here, then composite once after
    # the AR loop ends.
    all_decoded_chunks_cpu = []
    video_frame_offset = 0

    if _TIMING_ENABLED:
        _gpu_sync()
    streaming_t0 = time.perf_counter()
    first_frame_done = False

    # Timing marks for first-frame latency
    ff_gen_t0 = None
    ff_pure_gen_t0 = None
    ff_enc_gen_dec_t0 = None

    # In streamwise mode, mark _enc_to_dec_t0 right before the AR loop —
    # this is when all VAE-encode work begins (analogous to the upfront
    # vae_encode wrapper in non-streamwise mode). _e2d_t0 fires inside
    # _stream_encode_block on the very first streaming_encode_chunk() call.
    if _streamwise:
        import inference_causal_taehv as _base
        if _TIMING_ENABLED:
            _gpu_sync()
        _base._enc_to_dec_t0 = time.perf_counter()

    # ============================================================
    # STAGE 3b/4/5/6/7: AR streaming loop (repeats num_blocks times)
    # ============================================================
    for block_idx in range(num_blocks):
        cur_start_frame = block_idx * args.chunk_size

        # --- First-chunk timing marks (first-frame latency stages) ---
        if block_idx == 0 and _TIMING_ENABLED:
            _gpu_sync()
            ff_gen_t0 = time.perf_counter()
            ff_enc_gen_dec_t0 = time.perf_counter()
            ff_pure_gen_t0 = time.perf_counter()

        # ----------------------------------------------------------
        # STAGE 3b: per-block VAE encode (streamwise mode only)
        # ----------------------------------------------------------
        # Block 0 ingests 9 video frames (1+4+4) -> 3 latents.
        # Subsequent blocks ingest 12 frames (4+4+4) -> 3 latents.
        # Two encoder feat_cache streams are swapped (unmasked vs masked) so
        # each stream maintains its own continuous temporal context across
        # all blocks.
        if _streamwise:
            new_unmasked, unmasked_state = _stream_encode_block(
                block_idx, video_tensor, unmasked_state)
            new_masked, masked_state = _stream_encode_block(
                block_idx, masked_video_tensor, masked_state)

            growing_ref_seq = (new_unmasked if growing_ref_seq is None
                               else torch.cat([growing_ref_seq, new_unmasked], dim=2))
            growing_masked = (new_masked if growing_masked is None
                              else torch.cat([growing_masked, new_masked], dim=2))

            condition["ref_sequence"] = growing_ref_seq
            condition["masked_video"] = growing_masked
            condition["ref_latent"] = growing_ref_seq[:, :, :1].contiguous()

        # ----------------------------------------------------------
        # STAGE 4: 2-step DDIM denoise on this block's 3 latents
        # ----------------------------------------------------------
        # t_list = [0.999, 0.833, 0]; len(t_list)-1 = 2 model forwards per
        # block. Self-attention is causal sliding-window (sink=1, window=7
        # AR chunks); cross-attention attends to audio_emb + ref_sequence.
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
                noisy_input = model.noise_scheduler. forward_process(
                    x0_pred, eps, t_next.expand(B))
            else:
                noisy_input = x0_pred

        # ----------------------------------------------------------
        # STAGE 5: per-block VAE decode (3 latents -> ~12 video frames)
        # ----------------------------------------------------------
        # Three decoder modes:
        #   - StreamingTAEHV: per-latent decode with MemBlock temporal state
        #     across chunks; first chunk emits fewer frames (buffering).
        #   - Wan VAE streaming: per-latent decode_chunk with feat_cache
        #     persistence; bit-identical to single full-length decode.
        #   - batch TAEHV / Wan VAE batch: 3 latents at once, no continuity.
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
        elif _use_wan_streaming:
            # Stream one latent at a time so feat_cache state advances
            # exactly as in the per-latent inner loop of VideoVAE_.decode.
            vae_dtype = next(decoder_vae.parameters()).dtype
            chunk_latent = x0_pred[0].to(vae_dtype)  # [16, 3, h_lat, w_lat]
            video_chunks = []
            for t in range(chunk_latent.shape[1]):
                latent_t = chunk_latent[:, t:t+1]  # [16, 1, h_lat, w_lat]
                v = decoder_vae.streaming_decode_chunk(latent_t, device=device)
                # v: [1, 3, t_video, H, W] in [-1, 1]; t_video = 1 on the very
                # first call, 4 thereafter (Wan VAE 4x temporal upsampling).
                video_chunks.append(v)
            chunk_decoded = torch.cat(video_chunks, dim=2)  # [1, 3, T, H, W]
            chunk_float = chunk_decoded[0].permute(1, 0, 2, 3)
            chunk_float = ((chunk_float + 1) / 2).clamp(0, 1)
        else:
            chunk_latent = x0_pred[0].to(torch.float32)
            chunk_decoded = decoder_vae.decode([chunk_latent], device=device)
            chunk_decoded = chunk_decoded.clamp(-1, 1)
            chunk_float = chunk_decoded[0].permute(1, 0, 2, 3)
            chunk_float = ((chunk_float + 1) / 2).clamp(0, 1)

        # --- First-frame timing ---
        # Captured right after the first chunk's last decode produces frames,
        # BEFORE the per-block composite (so it's measurable even with
        # --defer_composite). End point = "first frame of decoded output is
        # in chunk_float on CPU/GPU memory, ready to be composited."
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
            # Three-definition first-frame latencies, same end point as above.
            import inference_causal_taehv as _base
            if getattr(_base, "_a2d_t0", None):
                _TIMING_CURRENT["ff_audio_to_decode"]  = ff_now - _base._a2d_t0
            if getattr(_base, "_enc_to_dec_t0", None):
                _TIMING_CURRENT["ff_encode_to_decode"] = ff_now - _base._enc_to_dec_t0
            if getattr(_base, "_e2d_t0", None):
                _TIMING_CURRENT["ff_pure_encode_to_decode"] = ff_now - _base._e2d_t0
            first_frame_done = True

        # ----------------------------------------------------------
        # STAGE 6: per-block compositing (CPU)
        # ----------------------------------------------------------
        # Paste the generated 512x512 lip region back into the full-res
        # frame using the LatentSync affine matrices captured during
        # Stage 1. CPU-bound; runs after each chunk_float arrives.
        # When --defer_composite is on, skip this and stash the raw
        # decoded chunk for one batch composite after the AR loop.
        if chunk_float is not None:
            if args.defer_composite:
                # Move to CPU now (still cheaper than a per-block sync if
                # we need it eventually) but don't composite. Keep as a
                # tensor so we can torch.cat at the end.
                all_decoded_chunks_cpu.append(chunk_float.cpu())
                video_frame_offset += chunk_float.shape[0]
            else:
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

        # ----------------------------------------------------------
        # STAGE 7: KV cache update (extra model forward per block)
        # ----------------------------------------------------------
        # Re-run the model with this block's denoised x0_pred (or a
        # noised version when context_noise > 0) and store_kv=True so
        # subsequent blocks have valid sliding-window self-attention
        # context. This is the AR carry that makes the next block's
        # generation conditioned on the past.
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

    if _streamwise:
        vae.to(dtype=original_vae_dtype)

    # Capture the three throughput-definition end points BEFORE the flush so
    # that streaming_taehv's StreamingTAEHV buffer-drain work is excluded.
    # wan_vae / batch_taehv have no flush, so this gives all decoder modes a
    # common reference point: "end of last in-loop decode."
    if _TIMING_ENABLED:
        _gpu_sync()
        _t_post_loop = time.perf_counter()
        import inference_causal_taehv as _base
        if getattr(_base, "_a2d_t0", None):
            _TIMING_CURRENT["audio_to_decode"] = _t_post_loop - _base._a2d_t0
        if getattr(_base, "_enc_to_dec_t0", None):
            _TIMING_CURRENT["encode_to_decode"] = _t_post_loop - _base._enc_to_dec_t0
        if getattr(_base, "_e2d_t0", None):
            _TIMING_CURRENT["pure_encode_to_decode"] = _t_post_loop - _base._e2d_t0

    # ============================================================
    # STAGE 8: Flush remaining buffered frames (streaming_taehv only)
    # ============================================================
    # StreamingTAEHV needs future temporal context to emit frames, so
    # the very last latents stay buffered until we explicitly flush at
    # the end of the AR loop.
    if _use_streaming_dec:
        flush_frames = streaming_dec.flush_decoder()
        if flush_frames:
            flush_float = torch.cat(flush_frames, dim=1).squeeze(0)
            flush_cpu = flush_float.cpu()
            if args.defer_composite:
                all_decoded_chunks_cpu.append(flush_cpu)
                video_frame_offset += flush_cpu.shape[0]
            else:
                composited = composite_with_latentsync_float(
                    flush_cpu, latentsync_metadata, image_processor,
                    use_mouth_only_compositing=args.use_mouth_only,
                    frame_offset=video_frame_offset,
                )
                all_composited_frames.append(composited)
                video_frame_offset += composited.shape[0]

    # streaming_total brackets the whole pipeline incl. flush — it's the
    # honest wall-clock for "time the last frame is available to the user."
    if _TIMING_ENABLED:
        _gpu_sync()
        _now = time.perf_counter()
        _TIMING_CURRENT["streaming_total"] = _now - streaming_t0
        audio_enc_time = _TIMING_CURRENT.get("audio_encode", 0.0)
        enc_gen_dec = _TIMING_CURRENT.get("ff_enc_gen_dec", 0.0)
        if audio_enc_time and enc_gen_dec:
            _TIMING_CURRENT["ff_enc_audio_gen_dec"] = audio_enc_time + enc_gen_dec

    # ============================================================
    # STAGE 6 (deferred): one-shot compositing over all decoded frames
    # ============================================================
    # Only runs when --defer_composite is on. The lip blend + affine
    # warp are per-frame ops with no temporal coupling, so doing them
    # in one batch is identical to doing them per chunk -- but lets
    # the GPU run the whole AR loop without breaking pipelining on
    # per-block .cpu() syncs.
    if args.defer_composite and all_decoded_chunks_cpu:
        all_decoded = torch.cat(all_decoded_chunks_cpu, dim=0)
        composited_np = composite_with_latentsync_float(
            all_decoded, latentsync_metadata, image_processor,
            use_mouth_only_compositing=args.use_mouth_only,
            frame_offset=0,
        )
    else:
        composited_np = np.concatenate(all_composited_frames, axis=0)

    # ============================================================
    # STAGE 9: Save MP4 + ffmpeg audio mux (CPU)
    # ============================================================
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

    # Activate @conditional_compile decorators in network_causal.py BEFORE
    # the model class is imported (which happens later via load_diffusion_model).
    if args.compile:
        os.environ["FASTGEN_COMPILE"] = "true"

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
    # OmniAvatar Wav2VecModel requires seq_len + output_hidden_states.
    _dummy_audio = np.zeros(16000, dtype=np.float32)
    _dummy_input = wav2vec_extractor(_dummy_audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        wav2vec_model(
            _dummy_input.input_values.to(device),
            seq_len=25, output_hidden_states=True,
        )
    print("Wav2Vec2 warmed up.")

    text_embeds = None
    if args.text_embeds_path or args.prompt:
        print("Loading text embeddings ...")
        text_embeds = load_or_encode_text(args, device, dtype)

    # LatentSync
    image_processor = None
    if args.latentsync:
        image_processor = load_image_processor(args.mask_path, device)

    # ===================================================================
    # Optional torch.compile wrapping (compile time absorbed by warmup)
    # ===================================================================
    if args.compile:
        # Compile activated via @conditional_compile decorators on hot
        # functions (see fastgen/networks/OmniAvatar/inference_utils.py).
        # The env var that activates them was set at top of main() before
        # the model was imported.
        print("[--compile] Hot functions decorated with @conditional_compile.")

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

            # ============================================================
            # STAGE 1: Face detect + 512x512 affine alignment (CPU+GPU)
            # ============================================================
            # InsightFace (buffalo_l) bounding box detection followed by
            # LatentSync's affine_transform crop. Returns aligned 512x512
            # face crops + per-frame affine matrices for paste-back.
            latentsync_metadata = None
            if args.latentsync:
                print("Running LatentSync face detection ...")
                with _Stage("face_detect_align", use_gpu=False):
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

                # ============================================================
                # STAGE 2 + 3a: Audio encode + reference VAE encode (build condition)
                # ============================================================
                # Streamwise mode: only audio_emb is computed here; the
                # reference video is encoded inside the AR loop (STAGE 3b).
                # Default mode: full reference video VAE-encoded upfront.
                if args.streamwise_encode:
                    print("Building conditioning (streamwise) ...")
                    condition, video_tensor, masked_video_tensor = (
                        build_condition_streamwise(
                            encoder_vae, wav2vec_model, wav2vec_extractor,
                            ref_frames_np, audio_path, text_embeds, args.mask_path,
                            num_video_frames, num_latent_frames, device, dtype,
                        )
                    )
                else:
                    print("Building conditioning ...")
                    condition = build_condition(
                        encoder_vae, wav2vec_model, wav2vec_extractor, ref_frames_np,
                        audio_path, text_embeds, args.mask_path,
                        num_video_frames, num_latent_frames, device, dtype,
                    )
                    video_tensor = masked_video_tensor = None

            # Run streaming pipeline
            print(f"Running streaming pipeline ({args.streaming_decoder}) ...")
            run_streaming_pipeline(
                model, decoder_vae, vae, condition,
                num_latent_frames, num_video_frames,
                args, latentsync_metadata, image_processor,
                audio_path, output_path, device, dtype,
                video_tensor=video_tensor,
                masked_video_tensor=masked_video_tensor,
            )

            succeeded.append(name)
            if _TIMING_ENABLED:
                ve = _TIMING_CURRENT.get("pure_vae_encode", 0.0)
                dn = _TIMING_CURRENT.get("denoise", 0.0)
                vd = _TIMING_CURRENT.get("pure_vae_decode", 0.0)
                ae = _TIMING_CURRENT.get("audio_encode", 0.0)
                _TIMING_CURRENT["ablation_1"] = ve + dn + vd
                _TIMING_CURRENT["ablation_2"] = _TIMING_CURRENT.get("pure_encode_to_decode", ve + dn + vd)
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
        peak_alloc = peak_reserved = 0.0
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / 1e9
            peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        csv_path = args.timing_csv or (args.output_path + ".timing.csv")
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
                out_row["peak_alloc_gb"] = f"{peak_alloc:.3f}"
                out_row["peak_reserved_gb"] = f"{peak_reserved:.3f}"
                writer.writerow(out_row)
        print(f"\n[Timing] wrote {len(_TIMING_ROWS)} rows → {csv_path}")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        print(f"[VRAM] peak_allocated={peak_gb:.2f} GB peak_reserved={reserved_gb:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
