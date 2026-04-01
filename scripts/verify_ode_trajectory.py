#!/usr/bin/env python3
"""Verify ODE trajectory extraction by VAE-decoding the final x0 prediction.

Loads step_049_x0.pt (final denoised output) from each sample, decodes through
the Wan 2.1 VAE, and saves as an mp4 video for visual inspection.

Usage:
    python scripts/verify_ode_trajectory.py \
        --trajectory_dir /home/work/ode_full_trajectories/1.3B \
        --vae_path /home/work/.local/OmniAvatar/pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
        --output_dir /home/work/ode_full_trajectories/1.3B_verify
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append("/home/work/.local/OmniAvatar")

from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.wan_video import WanVideoPipeline


def save_video(frames, path, fps=25):
    """Save list of PIL images as mp4 using imageio."""
    import imageio
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Verify ODE trajectories by VAE decoding")
    parser.add_argument("--trajectory_dir", type=str, required=True,
                        help="Root dir with per-sample ODE trajectory subdirs")
    parser.add_argument("--vae_path", type=str,
                        default="/home/work/.local/OmniAvatar/pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save decoded videos (default: trajectory_dir/../verify)")
    parser.add_argument("--step", type=int, default=49,
                        help="Which step's x0 to decode (default: 49, the final prediction)")
    parser.add_argument("--also_decode_gt", action="store_true", default=True,
                        help="Also decode input_latents.pt as ground truth reference")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.output_dir is None:
        parent = os.path.dirname(args.trajectory_dir.rstrip("/"))
        name = os.path.basename(args.trajectory_dir.rstrip("/"))
        args.output_dir = os.path.join(parent, f"{name}_verify")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # Load VAE
    print(f"Loading VAE from {args.vae_path}...")
    model_manager = ModelManager(device="cpu", infer=True)
    model_manager.load_model(args.vae_path, device="cpu", torch_dtype=dtype)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=dtype, device="cpu")
    pipe.device = device
    pipe.vae.model.to(device)
    print("VAE loaded.")

    # Find sample dirs
    sample_dirs = sorted([
        d for d in os.listdir(args.trajectory_dir)
        if os.path.isdir(os.path.join(args.trajectory_dir, d))
    ])
    print(f"Found {len(sample_dirs)} samples")

    for sample_name in sample_dirs:
        sample_path = os.path.join(args.trajectory_dir, sample_name)
        x0_path = os.path.join(sample_path, f"step_{args.step:03d}_x0.pt")

        if not os.path.exists(x0_path):
            print(f"  Skipping {sample_name}: missing {x0_path}")
            continue

        print(f"  Decoding {sample_name}...", end=" ", flush=True)

        # Decode final x0 prediction
        x0 = torch.load(x0_path, map_location="cpu", weights_only=True)  # [16, 21, 64, 64]
        x0 = x0.unsqueeze(0).to(dtype=dtype)  # [1, 16, 21, 64, 64]

        with torch.no_grad():
            frames = pipe.decode_video(x0, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
        if frames.dim() == 5:
            frames = frames[0]  # [C, T, H, W]
        video_frames = pipe.tensor2video(frames)

        out_path = os.path.join(args.output_dir, f"{sample_name}_step{args.step:03d}_x0.mp4")
        save_video(video_frames, out_path)
        print(f"saved ({len(video_frames)} frames)")

        # Also decode ground truth
        if args.also_decode_gt:
            gt_path = os.path.join(sample_path, "input_latents.pt")
            if os.path.exists(gt_path):
                gt = torch.load(gt_path, map_location="cpu", weights_only=True)
                gt = gt.unsqueeze(0).to(dtype=dtype)
                with torch.no_grad():
                    gt_frames = pipe.decode_video(gt, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
                if gt_frames.dim() == 5:
                    gt_frames = gt_frames[0]
                gt_video = pipe.tensor2video(gt_frames)
                gt_out = os.path.join(args.output_dir, f"{sample_name}_gt.mp4")
                save_video(gt_video, gt_out)

        torch.cuda.empty_cache()

    print(f"\nDone. Videos saved to {args.output_dir}")


if __name__ == "__main__":
    main()
