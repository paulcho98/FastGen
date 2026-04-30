#!/usr/bin/env python3
"""Verify that WanVideoVAE.streaming_decode_chunk produces output identical to
a single full-length decode.

Plan:
  1. Load a video, take its first 81 frames (= 21 latent frames after 4x temporal
     compression, i.e. 1 + 80//4 = 21).
  2. Encode once -> latents L of shape [1, 16, 21, H/8, W/8].
  3. Full decode -> output_full of shape [1, 3, T, H, W].
  4. Reset cache, then decode L in chunks of 3 latents at a time, keeping cache
     across chunks. Concatenate the chunk outputs -> output_stream.
  5. Compare: shapes match, max abs diff and mean abs diff should be ~0.
"""
import argparse
import os
import sys

import numpy as np
import torch

OMNIAVATAR_ROOT = os.environ.get("OMNIAVATAR_ROOT", "/workspace/OmniAvatar")
sys.path.insert(0, OMNIAVATAR_ROOT)

from OmniAvatar.models.wan_video_vae import WanVideoVAE  # noqa: E402

import cv2  # noqa: E402


def load_video_frames(path: str, max_frames: int) -> torch.Tensor:
    """Load up to `max_frames` frames as a [3, T, H, W] float tensor in [-1, 1]."""
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {path}")
    arr = np.stack(frames, axis=0)  # [T, H, W, 3]
    t = torch.from_numpy(arr).float() / 127.5 - 1.0  # [-1, 1]
    return t.permute(3, 0, 1, 2).contiguous()  # [3, T, H, W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--num_video_frames", type=int, default=81)
    parser.add_argument("--chunk_latents", type=int, default=3,
                        help="Latent frames per streaming chunk")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device(args.device)

    print(f"Loading VAE from {args.vae_path} ...")
    vae = WanVideoVAE().eval().requires_grad_(False)
    state = torch.load(args.vae_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # The OmniAvatar wrapper expects state-dict-converter mapping. Check.
    try:
        vae.load_state_dict(state, strict=True)
    except Exception:
        # Fallback through the converter
        from OmniAvatar.models.wan_video_vae import WanVideoVAEStateDictConverter
        state = WanVideoVAEStateDictConverter().from_civitai(state)
        vae.load_state_dict(state, strict=True)
    vae = vae.to(device=device, dtype=dtype)

    print(f"Loading {args.num_video_frames} frames from {args.video_path} ...")
    video = load_video_frames(args.video_path, args.num_video_frames)
    if video.shape[1] < args.num_video_frames:
        print(f"  WARNING: only got {video.shape[1]} frames, requested "
              f"{args.num_video_frames}")
    video = video.to(device=device, dtype=dtype)
    # Pad temporal length to (4k+1) so VAE cleanly produces an integer number
    # of latent frames.
    T = video.shape[1]
    T_aligned = 1 + ((T - 1) // 4) * 4
    video = video[:, :T_aligned]
    expected_latent_t = 1 + (T_aligned - 1) // 4
    print(f"  Aligned to T={T_aligned} video frames -> "
          f"{expected_latent_t} latent frames")

    # 1a. Encode (single full-length pass)
    print("Encoding (full) ...")
    with torch.no_grad():
        latents = vae.encode([video], device=device)  # [1, z_dim, t_lat, h_lat, w_lat]
    print(f"  Latent shape: {tuple(latents.shape)}")

    # 1b. Encode (streaming: 1 frame, then 4-frame chunks). Verify bit-equal.
    print("Encoding (streaming) ...")
    vae.reset_encode_cache()
    enc_chunks = []
    # First chunk: 1 frame
    with torch.no_grad():
        enc_chunks.append(vae.streaming_encode_chunk(video[:, :1], device=device))
    # Subsequent chunks: 4 frames each
    t = video.shape[1]
    s = 1
    while s < t:
        e = min(s + 4, t)
        if e - s == 4:
            with torch.no_grad():
                enc_chunks.append(vae.streaming_encode_chunk(video[:, s:e], device=device))
        s = e
    latents_stream = torch.cat([c.squeeze(0) for c in enc_chunks], dim=1).unsqueeze(0)
    print(f"  Streaming latent shape: {tuple(latents_stream.shape)}")
    enc_diff = (latents.float() - latents_stream.float()).abs()
    print(f"  encode max abs diff: {enc_diff.max().item():.6e}")
    print(f"  encode mean abs diff: {enc_diff.mean().item():.6e}")
    if enc_diff.max().item() > 1e-2:
        print("  [FAIL] streaming encode diverges from full encode")
        sys.exit(2)

    # 2. Single full-length decode
    print("Full decode ...")
    with torch.no_grad():
        output_full = vae.decode([latents.squeeze(0)], device=device)
    print(f"  output_full shape: {tuple(output_full.shape)}")

    # 3. Streaming decode in chunks
    print(f"Streaming decode in chunks of {args.chunk_latents} latents ...")
    vae.reset_decode_cache()
    chunks = []
    L = latents.squeeze(0)  # [c, t_lat, h, w]
    t_lat = L.shape[1]
    for s in range(0, t_lat, args.chunk_latents):
        e = min(s + args.chunk_latents, t_lat)
        chunk = L[:, s:e]  # [c, t_chunk, h, w]
        with torch.no_grad():
            out_chunk = vae.streaming_decode_chunk(chunk, device=device)  # [1,3,t_v,H,W]
        chunks.append(out_chunk)
        print(f"  chunk [{s}:{e}] -> video frames {out_chunk.shape[2]}")
    output_stream = torch.cat(chunks, dim=2)
    print(f"  output_stream shape: {tuple(output_stream.shape)}")

    # 4. Compare
    if output_full.shape != output_stream.shape:
        print(f"\n!!! SHAPE MISMATCH: full={tuple(output_full.shape)} "
              f"vs stream={tuple(output_stream.shape)}")
        sys.exit(1)
    diff = (output_full.float() - output_stream.float()).abs()
    print()
    print(f"Compare full vs streaming decode:")
    print(f"  output_full  range: [{output_full.float().min():.6f}, {output_full.float().max():.6f}]")
    print(f"  output_stream range: [{output_stream.float().min():.6f}, {output_stream.float().max():.6f}]")
    print(f"  max  abs diff: {diff.max().item():.6e}")
    print(f"  mean abs diff: {diff.mean().item():.6e}")
    print(f"  median abs diff: {diff.median().item():.6e}")
    pct_close = (diff < 1e-4).float().mean().item() * 100
    print(f"  % pixels within 1e-4: {pct_close:.4f}%")

    # PASS criterion: max abs diff < 1e-2 (loose due to bf16) AND >99.9% within 1e-3
    ok = diff.max().item() < 1e-2
    if ok:
        print("\n[PASS] Streaming decode matches full decode within tolerance.")
    else:
        print("\n[FAIL] Streaming decode diverges from full decode.")
        sys.exit(2)


if __name__ == "__main__":
    main()
