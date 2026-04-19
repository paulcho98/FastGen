# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate FULL ODE trajectory from OmniAvatar teacher — all timesteps + denoised outputs.

For each ODE step i (0..num_steps-1), saves two files per sample:
  - step_{i:03d}_xt.pt    — noisy state x_t at timestep t_i  [16, 21, H, W]
  - step_{i:03d}_x0.pt    — denoised prediction x0_pred      [16, 21, H, W]

Also saves timestep metadata as ode_schedule.json per sample.

With 50 steps × 2 files × 10 samples = 1000 .pt files.
Each file ≈ 2.6 MB (bf16), total ≈ 2.6 GB.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_omniavatar_ode_pairs_full.py \
        --model_size 14B --in_dim 65 \
        --base_model_paths /path/to/14B/shards.safetensors \
        --omniavatar_ckpt_path /path/to/teacher.pt \
        --data_dir /home/work/stableavatar_data/v2v_validation_data/recon \
        --latentsync_mask_path /path/to/mask.png \
        --output_dir /path/to/output \
        --num_inference_steps 50 --guidance_scale 4.5 --shift 5.0

Usage (distributed, 4 GPUs):
    torchrun --nproc_per_node=4 scripts/generate_omniavatar_ode_pairs_full.py \
        --model_size 14B --in_dim 65 \
        --base_model_paths /path/to/14B/shards.safetensors \
        --omniavatar_ckpt_path /path/to/teacher.pt \
        --data_dir /home/work/stableavatar_data/v2v_validation_data/recon \
        --latentsync_mask_path /path/to/mask.png \
        --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastgen.networks.OmniAvatar.network import OmniAvatarWan
import fastgen.utils.logging_utils as logger


# ─────────────────────────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full ODE trajectory (all timesteps + denoised) from OmniAvatar teacher"
    )

    # Model
    parser.add_argument("--model_size", type=str, default="14B", choices=["14B", "1.3B"],
                        help="Teacher model size")
    parser.add_argument("--in_dim", type=int, default=65,
                        help="Input channels (49=V2V, 65=V2V+refseq)")
    parser.add_argument("--base_model_paths", type=str, required=True,
                        help="Comma-separated safetensor paths for base Wan 2.1 T2V weights")
    parser.add_argument("--omniavatar_ckpt_path", type=str, default=None,
                        help="Path to OmniAvatar LoRA+audio checkpoint (.pt)")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing sample subdirectories (each with vae_latents.pt, etc.)")
    parser.add_argument("--latentsync_mask_path", type=str, required=True,
                        help="Path to LatentSync spatial mask PNG")
    parser.add_argument("--neg_text_emb_path", type=str, default=None,
                        help="Path to precomputed negative text embedding .pt")
    parser.add_argument("--num_video_frames", type=int, default=81,
                        help="Number of video frames for audio slicing")
    parser.add_argument("--latent_h", type=int, default=64,
                        help="Latent height")
    parser.add_argument("--latent_w", type=int, default=64,
                        help="Latent width")

    # ODE parameters
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of ODE solver steps")
    parser.add_argument("--guidance_scale", type=float, default=4.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--shift", type=float, default=5.0,
                        help="Timestep shift (matches OmniAvatar inference schedule)")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory for saved trajectories")

    # Processing
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Limit number of samples")
    parser.add_argument("--skip_existing", action="store_true", default=False,
                        help="Skip samples that already have output subdirectory")

    # Ablation
    parser.add_argument("--zero_audio", action="store_true", default=False,
                        help="Zero out audio embeddings (for audio ablation analysis)")
    parser.add_argument("--cfg_drop_text", type=str, default="true",
                        choices=["true", "false"],
                        help="If true (default), negative branch uses negative text embedding. "
                             "If false, negative branch keeps positive text embedding (audio-only CFG).")
    parser.add_argument("--cfg_crossover", type=int, default=None,
                        help="If set to K, use guidance_scale=1.0 for steps 0..K-1 and "
                             "the provided --guidance_scale for steps K..num_steps-1. "
                             "Schedule CFG: noCFG early, CFG late.")

    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_mask(mask_path: str, latent_h: int = 64, latent_w: int = 64) -> torch.Tensor:
    """Load and resize LatentSync spatial mask.

    Returns:
        mask: [H_lat, W_lat] float32 tensor. 1=keep, 0=mask (LatentSync convention).
    """
    mask_img = Image.open(mask_path)
    mask_arr = np.array(mask_img, dtype=np.float32)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    mask_arr = mask_arr / 255.0
    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    mask_resized = F.interpolate(
        mask_tensor, size=(latent_h, latent_w), mode="bilinear", align_corners=False
    )
    return (mask_resized.squeeze() > 0.5).float()  # [H_lat, W_lat]


def load_sample(
    sample_dir: str,
    mask: torch.Tensor,
    neg_text_embeds: torch.Tensor,
    num_video_frames: int = 81,
    cfg_drop_text: bool = True,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Optional[Dict[str, torch.Tensor]]:
    """Load a single sample's precomputed tensors from vae_latents.pt format.

    Returns:
        Dict with keys: input_latents, condition, neg_condition
        or None if loading fails.
    """
    try:
        # VAE latents — prefer mask_all version (all frames masked including frame 0)
        vae_path = os.path.join(sample_dir, "vae_latents_mask_all.pt")
        if not os.path.exists(vae_path):
            vae_path = os.path.join(sample_dir, "vae_latents.pt")
            logger.warning(
                f"Using vae_latents.pt (frame 0 unmasked) for {sample_dir}. "
                f"Run precompute_vae_latents_masked.py first for correct mask_all_frames behavior."
            )
        vae_data = torch.load(
            vae_path,
            map_location="cpu", weights_only=False,
        )
        input_latents = vae_data["input_latents"].to(dtype)   # [16, 21, H, W]
        masked_latents = vae_data["masked_latents"].to(dtype)  # [16, 21, H, W]

        # Audio embedding
        audio_data = torch.load(
            os.path.join(sample_dir, "audio_emb_omniavatar.pt"),
            map_location="cpu", weights_only=False,
        )
        audio_emb = audio_data["audio_emb"][:num_video_frames].to(dtype)  # [81, 10752]

        # Text embedding
        text_emb = torch.load(
            os.path.join(sample_dir, "text_emb.pt"),
            map_location="cpu", weights_only=False,
        )
        if isinstance(text_emb, dict):
            text_emb = next(v for v in text_emb.values() if isinstance(v, torch.Tensor))
        text_emb = text_emb.to(dtype)
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(0)  # [1, 512, 4096]

        # Reference latent: first frame of input_latents
        ref_latent = input_latents[:, 0:1, :, :]  # [16, 1, H, W]

        # Reference sequence
        ref_path = os.path.join(sample_dir, "ref_latents.pt")
        ref_seq = None
        if os.path.exists(ref_path):
            ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
            ref_seq = ref_data["ref_sequence_latents"].to(dtype)  # [16, 21, H, W]

        # Build condition dict (add batch dim)
        condition = {
            "text_embeds": text_emb.unsqueeze(0).to(device),          # [1, 1, 512, 4096]
            "audio_emb": audio_emb.unsqueeze(0).to(device),           # [1, 81, 10752]
            "ref_latent": ref_latent.unsqueeze(0).to(device),         # [1, 16, 1, H, W]
            "mask": mask.to(device),                                   # [H, W]
            "masked_video": masked_latents.unsqueeze(0).to(device),   # [1, 16, 21, H, W]
        }

        # Fix text_embeds shape: should be [B, 512, 4096] not [B, 1, 512, 4096]
        if condition["text_embeds"].dim() == 4:
            condition["text_embeds"] = condition["text_embeds"].squeeze(1)

        # Reference sequence
        if ref_seq is not None:
            condition["ref_sequence"] = ref_seq.unsqueeze(0).to(device)  # [1, 16, 21, H, W]
        else:
            condition["ref_sequence"] = torch.zeros_like(condition["masked_video"])

        # Negative condition (for CFG)
        if cfg_drop_text:
            neg_text = neg_text_embeds.to(device=device, dtype=dtype)
            if neg_text.dim() == 2:
                neg_text = neg_text.unsqueeze(0)
        else:
            # audio-only CFG: keep positive text
            neg_text = condition["text_embeds"]
        neg_condition = {
            "text_embeds": neg_text,                                    # [1, 512, 4096]
            "audio_emb": torch.zeros_like(condition["audio_emb"]),      # [1, 81, 10752]
            "ref_latent": condition["ref_latent"],                      # same ref
            "mask": condition["mask"],                                  # same mask
            "masked_video": condition["masked_video"],                  # same masked video
            "ref_sequence": condition["ref_sequence"],                  # same ref_sequence
        }

        return {
            "input_latents": input_latents.to(device),   # [16, 21, H, W]
            "condition": condition,
            "neg_condition": neg_condition,
        }

    except Exception as e:
        logger.warning(f"Failed to load sample {sample_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ODE Schedule with Shift
# ─────────────────────────────────────────────────────────────────────────────

def get_shifted_t_list(
    noise_scheduler,
    num_steps: int,
    shift: float,
    device: torch.device,
) -> torch.Tensor:
    """Get ODE timestep schedule with shift applied (matching OmniAvatar inference).

    The shift formula: t_shifted = shift * t / (1 + (shift - 1) * t)
    This concentrates steps at higher noise levels for larger shift values.

    Args:
        noise_scheduler: RF noise scheduler (for max_t).
        num_steps: Number of ODE steps.
        shift: Timestep shift value (5.0 for OmniAvatar 14B inference).
        device: Target device.

    Returns:
        Tensor [num_steps+1] — shifted timesteps from max to 0.
    """
    # Uniform schedule: [max_t, ..., 0]
    max_t = noise_scheduler.max_t  # 0.999
    t_list = torch.linspace(max_t, 0, num_steps + 1, device=device, dtype=torch.float64)

    # Apply shift: t_shifted = shift * t / (1 + (shift - 1) * t)
    if shift != 1.0:
        t_list = shift * t_list / (1 + (shift - 1) * t_list)

    # Ensure endpoints are exact
    t_list[0] = max_t
    t_list[-1] = 0.0

    return t_list.float()


# ─────────────────────────────────────────────────────────────────────────────
# ODE Trajectory Extraction — Full
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_full_ode_trajectory(
    teacher: OmniAvatarWan,
    noise_scheduler,
    latent_shape: tuple,
    condition: Dict[str, torch.Tensor],
    neg_condition: Dict[str, torch.Tensor],
    num_steps: int = 50,
    guidance_scale: float = 4.5,
    shift: float = 5.0,
    output_dir: str = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
    cfg_crossover: Optional[int] = None,
) -> Dict[str, Any]:
    """Run teacher ODE solve and save (x_t, x0_pred) at every step.

    For step i (0..num_steps-1):
      - step_{i:03d}_xt.pt: noisy state x_t at timestep t_i
      - step_{i:03d}_x0.pt: teacher's denoised prediction before re-noising

    Also saves ode_schedule.json with the full timestep schedule + metadata.

    Returns:
        Dict with summary info (shapes, timing, etc.)
    """
    # Get shifted timestep schedule
    ode_t_list = get_shifted_t_list(noise_scheduler, num_steps, shift, device)

    # Save schedule metadata as JSON (human-readable)
    schedule_info = {
        "t_list": ode_t_list.cpu().tolist(),   # [num_steps+1] shifted timesteps
        "num_steps": num_steps,
        "shift": shift,
        "guidance_scale": guidance_scale,
        "cfg_crossover": cfg_crossover,
        "latent_shape": list(latent_shape),
    }
    with open(os.path.join(output_dir, "ode_schedule.json"), "w") as f:
        json.dump(schedule_info, f, indent=2)

    # Start from pure noise
    noise = torch.randn(1, *latent_shape, device=device, dtype=dtype)
    x_t = noise_scheduler.latents(noise=noise, t_init=ode_t_list[0])

    for step_idx in range(num_steps):
        t_cur = ode_t_list[step_idx].unsqueeze(0)  # [1]

        # Save x_t (noisy state at this timestep)
        torch.save(
            x_t.squeeze(0).to(torch.bfloat16).cpu(),
            os.path.join(output_dir, f"step_{step_idx:03d}_xt.pt"),
        )

        # Teacher prediction with CFG — support scheduled CFG via cfg_crossover
        if cfg_crossover is not None and step_idx < cfg_crossover:
            effective_cfg = 1.0
        else:
            effective_cfg = guidance_scale

        x0_cond = teacher(x_t, t_cur, condition=condition, fwd_pred_type="x0")

        if effective_cfg != 1.0:
            x0_uncond = teacher(x_t, t_cur, condition=neg_condition, fwd_pred_type="x0")
            x0_pred = x0_uncond + effective_cfg * (x0_cond - x0_uncond)
        else:
            x0_pred = x0_cond

        # Save x0_pred (denoised prediction)
        torch.save(
            x0_pred.squeeze(0).to(torch.bfloat16).cpu(),
            os.path.join(output_dir, f"step_{step_idx:03d}_x0.pt"),
        )

        # ODE step: x0 -> eps -> forward_process to next timestep
        t_next = ode_t_list[step_idx + 1]
        if t_next > 0:
            eps = noise_scheduler.x0_to_eps(xt=x_t, x0=x0_pred, t=t_cur)
            x_t = noise_scheduler.forward_process(x0_pred, eps, t_next.unsqueeze(0))
        else:
            x_t = x0_pred

        del x0_cond, x0_pred
        if effective_cfg != 1.0:
            del x0_uncond

    return {
        "num_steps": num_steps,
        "latent_shape": latent_shape,
        "t_range": (ode_t_list[0].item(), ode_t_list[-1].item()),
        "shift": shift,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Distributed setup
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        global_rank = 0
        world_size = 1

    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Seed for reproducibility
    torch.manual_seed(args.seed + global_rank)

    # ── Load teacher model ──
    if global_rank == 0:
        logger.info(f"Loading OmniAvatar teacher: model_size={args.model_size}, in_dim={args.in_dim}")

    if global_rank == 0:
        cfg_mode = "text+audio (original)" if args.cfg_drop_text == "true" else "audio-only"
        logger.info(f"CFG drop mode: {cfg_mode}")

    teacher = OmniAvatarWan(
        model_size=args.model_size,
        in_dim=args.in_dim,
        mode="v2v",
        use_audio=True,
        base_model_paths=args.base_model_paths,
        omniavatar_ckpt_path=args.omniavatar_ckpt_path,
        merge_lora=True,
        net_pred_type="flow",
        schedule_type="rf",
    ).to(device, dtype=dtype).eval()
    teacher.requires_grad_(False)

    noise_scheduler = teacher.noise_scheduler

    if global_rank == 0:
        logger.info(f"Teacher loaded. Noise schedule: RF, max_t={noise_scheduler.max_t}")

    # ── Log shifted schedule ──
    if global_rank == 0:
        ode_t_schedule = get_shifted_t_list(noise_scheduler, args.num_inference_steps, args.shift, "cpu")
        logger.info(f"ODE schedule (shift={args.shift}, {args.num_inference_steps} steps):")
        logger.info(f"  First 5 t: {ode_t_schedule[:5].tolist()}")
        logger.info(f"  Last 5 t:  {ode_t_schedule[-5:].tolist()}")
        logger.info(f"  Total states to save per sample: {args.num_inference_steps} × 2 = {args.num_inference_steps * 2}")

    # ── Load spatial mask ──
    mask = load_mask(args.latentsync_mask_path, args.latent_h, args.latent_w)
    if global_rank == 0:
        logger.info(f"Loaded mask: {mask.shape}, keep_ratio={mask.mean():.3f}")

    # ── Load negative text embedding ──
    if args.neg_text_emb_path is not None and os.path.exists(args.neg_text_emb_path):
        neg_text_embeds = torch.load(args.neg_text_emb_path, map_location="cpu", weights_only=False)
        if isinstance(neg_text_embeds, dict):
            neg_text_embeds = next(v for v in neg_text_embeds.values() if isinstance(v, torch.Tensor))
        neg_text_embeds = neg_text_embeds.to(dtype)
    else:
        neg_text_embeds = torch.zeros(1, 512, 4096, dtype=dtype)
        if global_rank == 0:
            logger.warning(
                "No --neg_text_emb_path provided, using zeros for negative text embedding. "
                "This WILL corrupt CFG guidance! Generate the proper embedding with: "
                "pipe.encode_prompt('', positive=False) using the T5 text encoder."
            )

    if global_rank == 0:
        logger.info(f"Negative text embedding shape: {neg_text_embeds.shape}")

    # ── Gather sample directories ──
    # List subdirectories in data_dir (exclude files like video_square_path.txt)
    all_entries = sorted(os.listdir(args.data_dir))
    all_dirs = [
        os.path.join(args.data_dir, d) for d in all_entries
        if os.path.isdir(os.path.join(args.data_dir, d))
    ]

    # Apply max_samples
    if args.max_samples is not None:
        all_dirs = all_dirs[:args.max_samples]

    # Filter: check required files exist
    required_files = ["vae_latents.pt", "audio_emb_omniavatar.pt", "text_emb.pt"]
    valid_dirs = []
    for d in all_dirs:
        missing = [fn for fn in required_files if not os.path.exists(os.path.join(d, fn))]
        if missing:
            if global_rank == 0:
                logger.warning(f"Skipping {d}: missing {missing}")
        else:
            valid_dirs.append(d)

    if global_rank == 0:
        logger.info(f"Processing {len(valid_dirs)} samples (max_samples={args.max_samples})")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Skip existing if requested
    if args.skip_existing:
        before = len(valid_dirs)
        valid_dirs = [
            d for d in valid_dirs
            if not os.path.exists(os.path.join(args.output_dir, os.path.basename(d), "ode_schedule.json"))
        ]
        if global_rank == 0:
            logger.info(f"Skipped {before - len(valid_dirs)} samples with existing output")

    # ── Process samples ──
    # Distribute across ranks
    rank_dirs = valid_dirs[global_rank::world_size]

    if global_rank == 0:
        logger.info(f"Rank {global_rank}: processing {len(rank_dirs)} samples")

    total_time = 0.0
    success_count = 0
    fail_count = 0

    pbar = tqdm(rank_dirs, disable=global_rank != 0, desc="Generating full ODE trajectories")
    for sample_dir in pbar:
        t_start = time.time()
        sample_name = os.path.basename(sample_dir)

        # Create per-sample output directory
        sample_output_dir = os.path.join(args.output_dir, sample_name)
        os.makedirs(sample_output_dir, exist_ok=True)

        # Load sample
        sample = load_sample(
            sample_dir=sample_dir,
            mask=mask,
            neg_text_embeds=neg_text_embeds,
            num_video_frames=args.num_video_frames,
            cfg_drop_text=(args.cfg_drop_text == "true"),
            device=device,
            dtype=dtype,
        )

        if sample is None:
            fail_count += 1
            continue

        input_latents = sample["input_latents"]   # [16, 21, H, W]
        condition = sample["condition"]
        neg_condition = sample["neg_condition"]

        # Audio ablation: zero out audio in both conditional and unconditional
        if args.zero_audio:
            condition["audio_emb"] = torch.zeros_like(condition["audio_emb"])
            neg_condition["audio_emb"] = torch.zeros_like(neg_condition["audio_emb"])

        latent_shape = tuple(input_latents.shape)  # (16, 21, H, W)

        try:
            # Extract and save full ODE trajectory
            info = extract_full_ode_trajectory(
                teacher=teacher,
                noise_scheduler=noise_scheduler,
                latent_shape=latent_shape,
                condition=condition,
                neg_condition=neg_condition,
                num_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                shift=args.shift,
                output_dir=sample_output_dir,
                device=device,
                dtype=dtype,
                cfg_crossover=args.cfg_crossover,
            )

            # Also save input_latents (ground truth) for convenience
            torch.save(
                input_latents.to(torch.bfloat16).cpu(),
                os.path.join(sample_output_dir, "input_latents.pt"),
            )

            t_elapsed = time.time() - t_start
            total_time += t_elapsed
            success_count += 1

            num_files = args.num_inference_steps * 2 + 2  # xt + x0 + schedule + input_latents
            pbar.set_postfix({
                "done": success_count,
                "fail": fail_count,
                "files": num_files,
                "time": f"{t_elapsed:.1f}s",
            })

        except Exception as e:
            logger.warning(f"Failed ODE extraction for {sample_dir}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            continue

        # Free memory
        del sample
        torch.cuda.empty_cache()

    # ── Summary ──
    if world_size > 1:
        dist.barrier()

    avg_time = total_time / max(success_count, 1)
    total_files = success_count * (args.num_inference_steps * 2 + 2)
    if global_rank == 0:
        logger.info(
            f"Full ODE trajectory generation complete. "
            f"Success: {success_count}, Failed: {fail_count}, "
            f"Total files: {total_files}, "
            f"Avg time: {avg_time:.1f}s/sample"
        )
        if success_count > 0:
            # Verify a saved file
            test_name = os.path.basename(rank_dirs[0]) if rank_dirs else os.path.basename(valid_dirs[0])
            test_dir = os.path.join(args.output_dir, test_name)
            test_xt = os.path.join(test_dir, "step_000_xt.pt")
            test_x0 = os.path.join(test_dir, "step_000_x0.pt")
            if os.path.exists(test_xt) and os.path.exists(test_x0):
                xt = torch.load(test_xt, map_location="cpu", weights_only=True)
                x0 = torch.load(test_x0, map_location="cpu", weights_only=True)
                logger.info(
                    f"Verification (step 0): "
                    f"xt shape={list(xt.shape)}, x0 shape={list(x0.shape)}, "
                    f"xt range=[{xt.float().min():.4f}, {xt.float().max():.4f}], "
                    f"x0 range=[{x0.float().min():.4f}, {x0.float().max():.4f}]"
                )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
