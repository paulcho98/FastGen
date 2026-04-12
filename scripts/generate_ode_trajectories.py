# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate ODE trajectory pairs from a bidirectional teacher model for knowledge distillation.

This script runs a multi-step ODE solve with classifier-free guidance (CFG) on a
pretrained teacher model, collects intermediate denoising states, and saves them
as WebDataset shards compatible with FastGen's PathLoaderConfig.

Supports two input modes:
  - "latent": Load pre-computed VAE latents (skips VAE encoding)
  - "video":  Load raw video files and VAE-encode on the fly

Usage (distributed, 8 GPUs):
    torchrun --nproc_per_node=8 scripts/generate_ode_trajectories.py \
        --config fastgen/configs/experiments/WanT2V/config_sf.py \
        --teacher_ckpt /path/to/teacher.pt \
        --input_data /path/to/latents_or_videos/ \
        --caption_file /path/to/captions.json \
        --output_dir /path/to/output_shards/ \
        --input_format latent \
        --num_inference_steps 50 \
        --guidance_scale 6.0

Output format (per sample in WebDataset shard):
    - latent.pth:  Clean target latent [C, T, H, W]
    - path.pth:    ODE trajectory [num_subsample_steps, C, T, H, W] (noise → clean)
    - txt_emb.pth: Pre-encoded text embedding tensor
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from typing import Any, List, Optional

import torch
import torch.distributed as dist
import webdataset as wds
from tqdm import tqdm

from fastgen.configs.config_utils import import_config_from_python_file
from fastgen.methods.model import FastGenModel
from fastgen.utils import instantiate
import fastgen.utils.logging_utils as logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ODE trajectory pairs for KD training")

    # Model & weights
    parser.add_argument("--config", type=str, required=True, help="FastGen config file with teacher network definition")
    parser.add_argument("--teacher_ckpt", type=str, required=True, help="Path to pretrained teacher checkpoint")

    # Input data
    parser.add_argument("--input_data", type=str, required=True, help="Directory of pre-encoded latents or raw videos")
    parser.add_argument(
        "--input_format",
        type=str,
        choices=["latent", "video"],
        default="latent",
        help="Input format: 'latent' for pre-computed VAE latents, 'video' for raw video files",
    )
    parser.add_argument(
        "--caption_file",
        type=str,
        required=True,
        help="Path to captions: JSON {filename: prompt} or .txt (one prompt per line)",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="",
        help="Negative prompt for classifier-free guidance (empty string for unconditional)",
    )

    # ODE parameters
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of ODE solver steps")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Classifier-free guidance scale")
    parser.add_argument(
        "--t_list",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Target noise levels for the trajectory snapshots, matching the KD training t_list "
            "(e.g., 0.999 0.937 0.833 0.624 0.0). The script auto-computes which ODE trajectory "
            "indices correspond to these noise levels. If not provided, reads from config.model.sample_t_cfg.t_list."
        ),
    )
    parser.add_argument(
        "--subsample_indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "ADVANCED: Manually specify indices to subsample from the ODE trajectory. "
            "Overrides --t_list. Only use if you know exactly which trajectory states to extract."
        ),
    )

    # Output
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for WebDataset shards")
    parser.add_argument("--samples_per_shard", type=int, default=100, help="Number of samples per WebDataset shard")

    # Precision
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])

    args = parser.parse_args()
    return args


def load_captions(caption_file: str) -> dict[str, str] | list[str]:
    """Load captions from JSON or text file."""
    if caption_file.endswith(".json"):
        with open(caption_file, "r") as f:
            return json.load(f)
    elif caption_file.endswith(".txt"):
        with open(caption_file, "r") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported caption file format: {caption_file}. Use .json or .txt")


def gather_input_files(input_dir: str, input_format: str) -> list[str]:
    """Gather input files from directory."""
    if input_format == "latent":
        extensions = (".pt", ".pth", ".npy")
    else:
        extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    files = []
    for f in sorted(os.listdir(input_dir)):
        if f.lower().endswith(extensions):
            files.append(os.path.join(input_dir, f))
    return files


def load_latent(path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Load a pre-computed VAE latent from a .pt/.pth file."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        # Handle CausVid-style {prompt: tensor} format
        key = next(iter(data))
        tensor = data[key]
    elif isinstance(data, torch.Tensor):
        tensor = data
    else:
        raise ValueError(f"Unexpected data type in {path}: {type(data)}")

    # Ensure shape is [C, T, H, W] (remove batch dim if present)
    if tensor.ndim == 5:
        tensor = tensor.squeeze(0)
    assert tensor.ndim == 4, f"Expected [C, T, H, W] but got shape {tensor.shape}"

    return tensor.to(device=device, dtype=dtype)


def load_video_as_latent(
    path: str, vae: torch.nn.Module, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Load a raw video and encode it with VAE."""
    try:
        import imageio.v3 as iio
    except ImportError:
        raise ImportError("imageio is required for video loading. Install with: pip install imageio[pyav]")

    frames = iio.imread(path, plugin="pyav")  # [T, H, W, C] uint8
    video = torch.tensor(frames, dtype=torch.float32, device=device)
    video = video.permute(3, 0, 1, 2).unsqueeze(0) / 255.0  # [1, C, T, H, W] in [0, 1]
    video = video * 2 - 1  # Normalize to [-1, 1]
    video = video.to(dtype=dtype)

    with torch.no_grad():
        latent = vae.encode(video).squeeze(0)  # [C, T, H, W]
    return latent


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes for WebDataset storage."""
    buf = io.BytesIO()
    torch.save(tensor.cpu(), buf)
    return buf.getvalue()


def compute_subsample_indices(
    ode_t_list: torch.Tensor,
    target_t_values: list[float],
) -> list[int]:
    """Compute which ODE trajectory indices best match the target noise levels.

    The ODE trajectory has (num_steps + 1) states at noise levels given by ode_t_list.
    This function finds the closest trajectory index for each target noise level.

    Args:
        ode_t_list: Tensor of shape [num_steps+1] with noise levels from the ODE schedule
        target_t_values: List of target noise levels to match (excluding terminal 0.0)

    Returns:
        List of trajectory indices, plus -1 for the clean state
    """
    # The last element of target_t_values is typically 0.0 (clean) — handle separately
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
    teacher: torch.nn.Module,
    noise_scheduler,
    latent_shape: tuple,
    condition: Any,
    neg_condition: Any,
    num_steps: int = 50,
    guidance_scale: float = 6.0,
    subsample_indices: list[int] | None = None,
    target_t_list: list[float] | None = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run teacher ODE solve and return subsampled trajectory.

    The trajectory states are selected to match the noise levels in target_t_list,
    ensuring consistency with FastGen's KD training (CausalKDModel).

    Args:
        teacher: Bidirectional teacher network
        noise_scheduler: Noise scheduler with get_t_list, x0_to_eps, forward_process, latents
        latent_shape: Shape of the latent tensor [C, T, H, W]
        condition: Text conditioning (tensor or dict)
        neg_condition: Negative conditioning for CFG
        num_steps: Number of ODE solver steps
        guidance_scale: CFG scale
        subsample_indices: If provided, directly use these trajectory indices (advanced)
        target_t_list: Target noise levels (e.g., [0.999, 0.937, 0.833, 0.624, 0.0]).
            Auto-computes subsample_indices to match. Ignored if subsample_indices is given.
        device: Target device
        dtype: Compute dtype

    Returns:
        Tensor of shape [num_subsample, C, T, H, W] containing subsampled trajectory.
        The states are ordered from most noisy to clean, matching the KD t_list order.
        NOTE: The clean state (t=0) is excluded — it should be stored separately as
        'latent.pth' to match PathLoaderConfig's {path, latent} format.
    """
    # Get evenly-spaced timestep schedule: [max_t, ..., 0]
    ode_t_list = noise_scheduler.get_t_list(sample_steps=num_steps, device=device)

    # Compute subsample indices from target_t_list if not provided directly
    if subsample_indices is None:
        if target_t_list is None:
            raise ValueError("Either --subsample_indices or --t_list must be provided")
        subsample_indices = compute_subsample_indices(ode_t_list, target_t_list)

    # Start from pure noise
    noise = torch.randn(1, *latent_shape, device=device, dtype=dtype)
    x_t = noise_scheduler.latents(noise=noise, t_init=ode_t_list[0])

    # Collect trajectory states (num_steps + 1 total: initial noise + after each step)
    trajectory = [x_t.clone()]

    for step_idx in range(len(ode_t_list) - 1):
        t_cur = ode_t_list[step_idx].unsqueeze(0)  # [1]

        # Teacher prediction with classifier-free guidance
        x0_cond = teacher(x_t, t_cur, condition=condition, fwd_pred_type="x0")

        if guidance_scale != 1.0:
            x0_uncond = teacher(x_t, t_cur, condition=neg_condition, fwd_pred_type="x0")
            x0_pred = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
        else:
            x0_pred = x0_cond

        # ODE step: x0 → eps → forward_process to next timestep
        t_next = ode_t_list[step_idx + 1]
        if t_next > 0:
            eps = noise_scheduler.x0_to_eps(xt=x_t, x0=x0_pred, t=t_cur)
            x_t = noise_scheduler.forward_process(x0_pred, eps, t_next.unsqueeze(0))
        else:
            x_t = x0_pred

        trajectory.append(x_t.clone())

    # Stack and subsample: [1, num_steps+1, C, T, H, W] → [1, num_subsample, C, T, H, W]
    trajectory = torch.stack(trajectory, dim=1)
    # Resolve negative indices
    total_states = trajectory.shape[1]
    resolved_indices = [idx if idx >= 0 else total_states + idx for idx in subsample_indices]
    subsampled = trajectory[:, resolved_indices]

    return subsampled.squeeze(0)  # [num_subsample, C, T, H, W]


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
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.precision]

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load config to get teacher network definition
    config = import_config_from_python_file(args.config)
    teacher_config = config.model.teacher if config.model.teacher is not None else config.model.net

    # Instantiate teacher
    if global_rank == 0:
        logger.info("Instantiating teacher model...")
    teacher = instantiate(teacher_config)
    FastGenModel._load_pretrained_model(teacher, args.teacher_ckpt, device="cpu")
    teacher = teacher.to(device=device, dtype=dtype)
    teacher.eval().requires_grad_(False)

    noise_scheduler = teacher.noise_scheduler

    # Resolve target t_list for trajectory subsampling
    if args.subsample_indices is not None:
        # User explicitly provided indices — use directly
        target_t_list = None
        subsample_indices = args.subsample_indices
        if global_rank == 0:
            logger.info(f"Using explicit subsample_indices: {subsample_indices}")
    else:
        # Compute indices from t_list
        if args.t_list is not None:
            target_t_list = args.t_list
        elif hasattr(config.model, "sample_t_cfg") and config.model.sample_t_cfg.t_list is not None:
            target_t_list = list(config.model.sample_t_cfg.t_list)
        else:
            raise ValueError(
                "No t_list found. Provide --t_list, --subsample_indices, or ensure "
                "config.model.sample_t_cfg.t_list is set in the config file."
            )
        subsample_indices = None

        # Log the computed mapping
        if global_rank == 0:
            ode_t_schedule = noise_scheduler.get_t_list(sample_steps=args.num_inference_steps, device="cpu")
            computed_indices = compute_subsample_indices(ode_t_schedule, target_t_list)
            logger.info(f"Target t_list: {target_t_list}")
            logger.info(f"Computed subsample indices: {computed_indices}")
            for i, idx in enumerate(computed_indices):
                resolved = idx if idx >= 0 else len(ode_t_schedule) + idx
                actual_t = ode_t_schedule[resolved].item()
                target_t = target_t_list[i] if i < len(target_t_list) else 0.0
                logger.info(f"  Step {i}: target t={target_t:.4f} → index {idx}, actual t={actual_t:.4f}")

    # VAE for video mode
    vae = None
    if args.input_format == "video":
        if not hasattr(teacher, "vae") or teacher.vae is None:
            raise RuntimeError("Teacher model does not have a VAE. Required for --input_format=video")
        vae = teacher.vae

    # Text encoder
    text_encoder = None
    if hasattr(teacher, "text_encoder") and teacher.text_encoder is not None:
        text_encoder = teacher.text_encoder

    # Load captions
    captions = load_captions(args.caption_file)

    # Encode negative prompt
    if text_encoder is not None:
        neg_condition = text_encoder.encode([args.neg_prompt])
    else:
        neg_condition = None

    # Gather input files
    input_files = gather_input_files(args.input_data, args.input_format)
    if global_rank == 0:
        logger.info(f"Found {len(input_files)} input files")

    # Match files to captions
    if isinstance(captions, dict):
        # JSON: {filename: prompt}
        file_caption_pairs = []
        for fpath in input_files:
            fname = os.path.basename(fpath)
            # Try with and without extension
            prompt = captions.get(fname) or captions.get(os.path.splitext(fname)[0])
            if prompt is not None:
                file_caption_pairs.append((fpath, prompt))
            else:
                if global_rank == 0:
                    logger.warning(f"No caption found for {fname}, skipping")
    elif isinstance(captions, list):
        # Text file: 1:1 correspondence with input files
        if len(captions) != len(input_files):
            raise ValueError(
                f"Number of captions ({len(captions)}) != number of input files ({len(input_files)}). "
                "Use JSON format for non-1:1 mappings."
            )
        file_caption_pairs = list(zip(input_files, captions))
    else:
        raise ValueError(f"Unexpected captions type: {type(captions)}")

    if global_rank == 0:
        logger.info(f"Processing {len(file_caption_pairs)} file-caption pairs")
        os.makedirs(args.output_dir, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    # Shard work across ranks
    shard_writer = None
    shard_idx = 0
    samples_in_shard = 0

    def get_shard_writer():
        nonlocal shard_idx
        shard_path = os.path.join(args.output_dir, f"shard-{global_rank:04d}-{shard_idx:06d}.tar")
        shard_idx += 1
        return wds.TarWriter(shard_path)

    shard_writer = get_shard_writer()

    num_items = len(file_caption_pairs)
    items_per_rank = int(math.ceil(num_items / world_size))

    for local_idx in tqdm(range(items_per_rank), disable=global_rank != 0, desc="Generating ODE trajectories"):
        global_idx = local_idx * world_size + global_rank
        if global_idx >= num_items:
            continue

        fpath, prompt = file_caption_pairs[global_idx]

        try:
            # Load input
            if args.input_format == "latent":
                latent = load_latent(fpath, device=device, dtype=dtype)
            else:
                latent = load_video_as_latent(fpath, vae=vae, device=device, dtype=dtype)

            # Encode text
            if text_encoder is not None:
                condition = text_encoder.encode([prompt])
            else:
                # Assume pre-encoded embeddings are available alongside latents
                emb_path = fpath.replace(".pt", "_emb.pt").replace(".pth", "_emb.pth")
                if os.path.exists(emb_path):
                    condition = torch.load(emb_path, map_location=device, weights_only=True)
                else:
                    raise RuntimeError(f"No text encoder and no pre-encoded embedding found at {emb_path}")

            # Extract ODE trajectory
            trajectory = extract_ode_trajectory(
                teacher=teacher,
                noise_scheduler=noise_scheduler,
                latent_shape=latent.shape,
                condition=condition,
                neg_condition=neg_condition,
                num_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                subsample_indices=subsample_indices,
                target_t_list=target_t_list,
                device=device,
                dtype=dtype,
            )
            # trajectory shape: [num_subsample, C, T, H, W]
            # States ordered from most noisy to clean

            clean_latent = trajectory[-1]  # Last state = clean
            # path excludes the clean state (stored separately as latent.pth)
            path = trajectory[:-1]  # [num_steps, C, T, H, W] (noise levels only)

            # Write to WebDataset shard
            # path.pth: noisy states [num_steps, C, T, H, W] (excludes clean)
            # latent.pth: clean target [C, T, H, W]
            sample = {
                "__key__": f"{global_idx:08d}",
                "path.pth": tensor_to_bytes(path.to(torch.float16)),
                "latent.pth": tensor_to_bytes(clean_latent.to(torch.float16)),
            }
            # Save text embedding
            if isinstance(condition, torch.Tensor):
                sample["txt_emb.pth"] = tensor_to_bytes(condition.cpu())
            elif isinstance(condition, dict):
                # For dict-type conditions (e.g., with text_embeds key), save the whole dict
                sample["txt_emb.pth"] = tensor_to_bytes(condition)

            shard_writer.write(sample)
            samples_in_shard += 1

            # Rotate shard if needed
            if samples_in_shard >= args.samples_per_shard:
                shard_writer.close()
                shard_writer = get_shard_writer()
                samples_in_shard = 0

        except Exception as e:
            logger.warning(f"Failed to process {fpath}: {e}")
            continue

    # Close final shard
    if shard_writer is not None:
        shard_writer.close()

    if world_size > 1:
        dist.barrier()

    if global_rank == 0:
        logger.success(f"ODE trajectory generation complete. Shards saved to {args.output_dir}")


if __name__ == "__main__":
    main()
