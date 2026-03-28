# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate ODE trajectory pairs from OmniAvatar teacher for KD pre-training (Stage 1).

Runs a multi-step deterministic ODE solve with classifier-free guidance (CFG) using
the OmniAvatar bidirectional teacher (14B or 1.3B), then saves subsampled noisy states
as `ode_path.pt` in each sample directory.

Input:  OmniAvatar precomputed .pt files (vae_latents_mask_all, audio_emb, text_emb, ref_latents)
Output: Per-sample `ode_path.pt` with shape [4, 16, 21, H, W] (4 noisy states, bf16)

The clean target is already available as `input_latents` in `vae_latents_mask_all.pt`.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=2 python scripts/generate_omniavatar_ode_pairs.py \
        --model_size 14B --in_dim 65 \
        --base_model_paths /path/to/14B/shards.safetensors \
        --omniavatar_ckpt_path /path/to/teacher.pt \
        --data_list_path /path/to/video_square_path.txt \
        --latentsync_mask_path /path/to/mask.png \
        --num_inference_steps 50 --guidance_scale 4.5

Usage (distributed, 4 GPUs):
    torchrun --nproc_per_node=4 scripts/generate_omniavatar_ode_pairs.py \
        --model_size 14B --in_dim 65 \
        --base_model_paths /path/to/14B/shards.safetensors \
        --omniavatar_ckpt_path /path/to/teacher.pt \
        --data_list_path /path/to/video_square_path.txt \
        --latentsync_mask_path /path/to/mask.png
"""

from __future__ import annotations

import argparse
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
        description="Generate ODE trajectory pairs from OmniAvatar teacher for KD"
    )

    # Model
    parser.add_argument("--model_size", type=str, default="14B", choices=["14B", "1.3B"],
                        help="Teacher model size")
    parser.add_argument("--in_dim", type=int, default=49,
                        help="Input channels (49=V2V, 65=V2V+refseq)")
    parser.add_argument("--base_model_paths", type=str, required=True,
                        help="Comma-separated safetensor paths for base Wan 2.1 T2V weights")
    parser.add_argument("--omniavatar_ckpt_path", type=str, default=None,
                        help="Path to OmniAvatar LoRA+audio checkpoint (.pt)")

    # Data
    parser.add_argument("--data_list_path", type=str, required=True,
                        help="Path to text file with one sample directory per line")
    parser.add_argument("--latentsync_mask_path", type=str, required=True,
                        help="Path to LatentSync spatial mask PNG")
    parser.add_argument("--neg_text_emb_path", type=str, default=None,
                        help="Path to precomputed negative text embedding .pt")
    parser.add_argument("--use_ref_sequence", action="store_true", default=False,
                        help="Include ref_sequence in conditioning (requires in_dim=65)")
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
    parser.add_argument("--t_list", type=float, nargs="+",
                        default=[0.999, 0.900, 0.750, 0.500, 0.0],  # shift=3.0
                        help="Target noise levels for trajectory subsampling")

    # Output
    parser.add_argument("--output_key", type=str, default="ode_path.pt",
                        help="Filename for saved ODE path tensors in each sample dir")

    # Processing range
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for distributed/partial processing")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index for distributed/partial processing")
    parser.add_argument("--skip_existing", action="store_true", default=False,
                        help="Skip samples that already have ode_path.pt")

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
    use_ref_sequence: bool = False,
    num_video_frames: int = 81,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Optional[Dict[str, torch.Tensor]]:
    """Load a single sample's precomputed tensors.

    Returns:
        Dict with keys: input_latents, condition, neg_condition
        or None if loading fails.
    """
    try:
        # VAE latents
        vae_data = torch.load(
            os.path.join(sample_dir, "vae_latents_mask_all.pt"),
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

        # Build condition dict (add batch dim)
        condition = {
            "text_embeds": text_emb.unsqueeze(0).to(device),          # [1, 1, 512, 4096] -> squeeze below
            "audio_emb": audio_emb.unsqueeze(0).to(device),           # [1, 81, 10752]
            "ref_latent": ref_latent.unsqueeze(0).to(device),         # [1, 16, 1, H, W]
            "mask": mask.to(device),                                   # [H, W]
            "masked_video": masked_latents.unsqueeze(0).to(device),   # [1, 16, 21, H, W]
        }

        # Fix text_embeds shape: should be [B, 512, 4096] not [B, 1, 512, 4096]
        if condition["text_embeds"].dim() == 4:
            condition["text_embeds"] = condition["text_embeds"].squeeze(1)

        # Reference sequence (optional)
        if use_ref_sequence:
            ref_path = os.path.join(sample_dir, "ref_latents.pt")
            if os.path.exists(ref_path):
                ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
                ref_seq = ref_data["ref_sequence_latents"].to(dtype)  # [16, 21, H, W]
                condition["ref_sequence"] = ref_seq.unsqueeze(0).to(device)  # [1, 16, 21, H, W]
            else:
                # Zero fallback
                condition["ref_sequence"] = torch.zeros_like(
                    condition["masked_video"]
                )

        # Negative condition (for CFG)
        neg_text = neg_text_embeds.to(device=device, dtype=dtype)
        if neg_text.dim() == 2:
            neg_text = neg_text.unsqueeze(0)
        neg_condition = {
            "text_embeds": neg_text,                                    # [1, 512, 4096]
            "audio_emb": torch.zeros_like(condition["audio_emb"]),      # [1, 81, 10752]
            "ref_latent": condition["ref_latent"],                      # same ref
            "mask": condition["mask"],                                  # same mask
            "masked_video": condition["masked_video"],                  # same masked video
        }
        if "ref_sequence" in condition:
            neg_condition["ref_sequence"] = condition["ref_sequence"]

        return {
            "input_latents": input_latents.to(device),   # [16, 21, H, W]
            "condition": condition,
            "neg_condition": neg_condition,
        }

    except Exception as e:
        logger.warning(f"Failed to load sample {sample_dir}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ODE Trajectory Extraction
# ─────────────────────────────────────────────────────────────────────────────

def compute_subsample_indices(
    ode_t_list: torch.Tensor,
    target_t_values: List[float],
) -> List[int]:
    """Compute which ODE trajectory indices best match the target noise levels.

    Args:
        ode_t_list: [num_steps+1] noise levels from the ODE schedule (descending).
        target_t_values: Target noise levels (e.g., [0.999, 0.937, 0.833, 0.624, 0.0]).

    Returns:
        List of trajectory indices. For t=0.0, uses index -1 (final clean state).
    """
    target_non_zero = [t for t in target_t_values if t > 0]

    indices = []
    for t_target in target_non_zero:
        diffs = (ode_t_list - t_target).abs()
        best_idx = diffs.argmin().item()
        indices.append(best_idx)

    # Always include the final clean state
    indices.append(-1)

    return indices


@torch.no_grad()
def extract_ode_trajectory(
    teacher: OmniAvatarWan,
    noise_scheduler,
    latent_shape: tuple,
    condition: Dict[str, torch.Tensor],
    neg_condition: Dict[str, torch.Tensor],
    num_steps: int = 50,
    guidance_scale: float = 4.5,
    target_t_list: List[float] = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run teacher ODE solve and return subsampled trajectory.

    Uses the same ODE discretization as generate_ode_trajectories.py but with
    OmniAvatar's audio+V2V conditioning.

    Args:
        teacher: OmniAvatarWan bidirectional teacher.
        noise_scheduler: RF noise scheduler from teacher.
        latent_shape: [C, T, H, W] shape of the latent.
        condition: OmniAvatar condition dict.
        neg_condition: Negative condition dict for CFG.
        num_steps: Number of ODE steps.
        guidance_scale: CFG scale.
        target_t_list: Target noise levels for subsampling.
        device: CUDA device.
        dtype: Compute dtype.

    Returns:
        Tensor [num_subsample, C, T, H, W] — trajectory from noisiest to clean.
        The clean state (last element) is excluded in the final saved output.
    """
    if target_t_list is None:
        target_t_list = [0.999, 0.900, 0.750, 0.500, 0.0]  # shift=3.0

    # Get evenly-spaced timestep schedule: [max_t, ..., 0]
    ode_t_list = noise_scheduler.get_t_list(sample_steps=num_steps, device=device)
    # Ensure t_list is on the correct device (noise_scheduler may keep sigmas on CPU)
    ode_t_list = ode_t_list.to(device)
    subsample_indices = compute_subsample_indices(ode_t_list, target_t_list)

    # Start from pure noise
    noise = torch.randn(1, *latent_shape, device=device, dtype=dtype)
    x_t = noise_scheduler.latents(noise=noise, t_init=ode_t_list[0])

    # Collect trajectory states
    trajectory = [x_t.clone()]

    for step_idx in range(len(ode_t_list) - 1):
        t_cur = ode_t_list[step_idx].unsqueeze(0)  # [1]

        # Teacher prediction with CFG
        x0_cond = teacher(x_t, t_cur, condition=condition, fwd_pred_type="x0")

        if guidance_scale != 1.0:
            x0_uncond = teacher(x_t, t_cur, condition=neg_condition, fwd_pred_type="x0")
            x0_pred = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
        else:
            x0_pred = x0_cond

        # ODE step: x0 -> eps -> forward_process to next timestep
        t_next = ode_t_list[step_idx + 1]
        if t_next > 0:
            eps = noise_scheduler.x0_to_eps(xt=x_t, x0=x0_pred, t=t_cur)
            x_t = noise_scheduler.forward_process(x0_pred, eps, t_next.unsqueeze(0))
        else:
            x_t = x0_pred

        trajectory.append(x_t.clone())

    # Stack: [1, num_steps+1, C, T, H, W]
    trajectory = torch.stack(trajectory, dim=1)

    # Resolve negative indices and subsample
    total_states = trajectory.shape[1]
    resolved_indices = [idx if idx >= 0 else total_states + idx for idx in subsample_indices]
    subsampled = trajectory[:, resolved_indices]

    return subsampled.squeeze(0)  # [num_subsample, C, T, H, W]


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

    # ── Log trajectory subsampling plan ──
    if global_rank == 0:
        ode_t_schedule = noise_scheduler.get_t_list(
            sample_steps=args.num_inference_steps, device="cpu"
        )
        subsample_idx = compute_subsample_indices(ode_t_schedule, args.t_list)
        logger.info(f"Target t_list: {args.t_list}")
        logger.info(f"Computed subsample indices: {subsample_idx}")
        total_states = len(ode_t_schedule)
        for i, idx in enumerate(subsample_idx):
            resolved = idx if idx >= 0 else total_states + idx
            actual_t = ode_t_schedule[resolved].item()
            target_t = args.t_list[i] if i < len(args.t_list) else 0.0
            logger.info(f"  Step {i}: target t={target_t:.4f} -> index {idx}, actual t={actual_t:.4f}")

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
        logger.info(f"Negative text embedding shape: {neg_text_embeds.shape}")

    # ── Gather sample directories ──
    with open(args.data_list_path) as f:
        all_dirs = [line.strip() for line in f if line.strip()]

    # Apply index range
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(all_dirs)
    all_dirs = all_dirs[start:end]

    # Apply max_samples
    if args.max_samples is not None:
        all_dirs = all_dirs[:args.max_samples]

    # Filter: check required files exist
    required_files = ["vae_latents_mask_all.pt", "audio_emb_omniavatar.pt", "text_emb.pt"]
    valid_dirs = []
    for d in all_dirs:
        missing = [fn for fn in required_files if not os.path.exists(os.path.join(d, fn))]
        if missing:
            if global_rank == 0:
                logger.warning(f"Skipping {d}: missing {missing}")
        else:
            valid_dirs.append(d)

    if global_rank == 0:
        logger.info(
            f"Processing {len(valid_dirs)} samples "
            f"(range [{start}:{end}], {len(all_dirs) - len(valid_dirs)} skipped)"
        )

    # Skip existing if requested
    if args.skip_existing:
        before = len(valid_dirs)
        valid_dirs = [
            d for d in valid_dirs
            if not os.path.exists(os.path.join(d, args.output_key))
        ]
        if global_rank == 0:
            logger.info(f"Skipped {before - len(valid_dirs)} samples with existing {args.output_key}")

    # ── Process samples ──
    # Distribute across ranks
    rank_dirs = valid_dirs[global_rank::world_size]

    if global_rank == 0:
        logger.info(f"Rank {global_rank}: processing {len(rank_dirs)} samples")

    total_time = 0.0
    success_count = 0
    fail_count = 0

    pbar = tqdm(rank_dirs, disable=global_rank != 0, desc="Generating ODE trajectories")
    for sample_dir in pbar:
        t_start = time.time()

        # Load sample
        sample = load_sample(
            sample_dir=sample_dir,
            mask=mask,
            neg_text_embeds=neg_text_embeds,
            use_ref_sequence=args.use_ref_sequence,
            num_video_frames=args.num_video_frames,
            device=device,
            dtype=dtype,
        )

        if sample is None:
            fail_count += 1
            continue

        input_latents = sample["input_latents"]   # [16, 21, H, W]
        condition = sample["condition"]
        neg_condition = sample["neg_condition"]

        latent_shape = tuple(input_latents.shape)  # (16, 21, H, W)

        try:
            # Extract ODE trajectory
            trajectory = extract_ode_trajectory(
                teacher=teacher,
                noise_scheduler=noise_scheduler,
                latent_shape=latent_shape,
                condition=condition,
                neg_condition=neg_condition,
                num_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                target_t_list=args.t_list,
                device=device,
                dtype=dtype,
            )
            # trajectory: [num_subsample, C, T, H, W] — includes clean as last element

            # Split: noisy states (exclude clean) = ode_path
            ode_path = trajectory[:-1]  # [4, 16, 21, H, W]

            # Save to sample directory
            save_path = os.path.join(sample_dir, args.output_key)
            torch.save(ode_path.to(torch.bfloat16).cpu(), save_path)

            t_elapsed = time.time() - t_start
            total_time += t_elapsed
            success_count += 1

            pbar.set_postfix({
                "done": success_count,
                "fail": fail_count,
                "shape": list(ode_path.shape),
                "time": f"{t_elapsed:.1f}s",
            })

        except Exception as e:
            logger.warning(f"Failed ODE extraction for {sample_dir}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            continue

        # Free memory
        del sample, trajectory, ode_path
        torch.cuda.empty_cache()

    # ── Summary ──
    if world_size > 1:
        dist.barrier()

    avg_time = total_time / max(success_count, 1)
    if global_rank == 0:
        logger.info(
            f"ODE trajectory generation complete. "
            f"Success: {success_count}, Failed: {fail_count}, "
            f"Avg time: {avg_time:.1f}s/sample"
        )
        if success_count > 0:
            # Verify a saved file
            test_dir = rank_dirs[0] if rank_dirs else valid_dirs[0]
            test_path = os.path.join(test_dir, args.output_key)
            if os.path.exists(test_path):
                saved = torch.load(test_path, map_location="cpu", weights_only=True)
                logger.info(
                    f"Verification: {test_path} -> shape={list(saved.shape)}, "
                    f"dtype={saved.dtype}, "
                    f"min={saved.float().min():.4f}, max={saved.float().max():.4f}"
                )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
