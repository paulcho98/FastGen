#!/usr/bin/env python
"""Post-hoc convert FSDP-DCP LoRA-trained saves to trainable-only `.pth`.

Background:
    The 14B DF LoRA training run started before the trainable-only save
    filter (FSDPCheckpointer.save) landed in the codebase. The running
    Python process kept the pre-filter checkpointer in memory, so each
    save dumped the full 14B model state (~56 GB per `.net_model/` dir)
    instead of the ~5 GB of trainable LoRA + audio + patch_embedding
    parameters that were actually evolving.

    This script reads each `<step>.net_model/` DCP directory, filters
    the saved state_dict to LoRA + audio + patch_embedding keys (and
    any non-parameter buffer entries we can identify), and writes the
    result as a small `<step>.net_model_lora_only.pth` flat torch file
    sitting next to the original directory.

What is and isn't deleted:
    - Originals are NEVER touched. The script only creates new files.
    - After validation, the user may manually `rm -rf <step>.net_model/`
      to recover ~50 GB per save. The .pth file is sufficient for
      inference reconstruction (the inference loader at
      `inference_causal_14b.py` already handles flat .pth state_dicts via
      its `else` branch — `torch.load(path, map_location="cpu", ...)`).
    - The script also prints a recommended cleanup command per save —
      it does NOT execute it.

Trainable key patterns (matches the 14B LoRA + selective-unfreeze regime):
    1. LoRA adapter weights: `lora_A` or `lora_B` substring
    2. Audio path: `audio_proj` substring
    3. Audio conditioning: `audio_cond_projs` substring
    4. Patch embedding: `patch_embedding` substring

These match the `LORA_TARGET_MODULES` set + the `unfreeze_modules`
list used by the 14B configs. Frozen base parameters (Wan 2.1 attention
projections, MLP, layernorm, etc.) are filtered out.

Buffers: any key NOT matching the four patterns above is dropped. If
this drops a registered buffer that the model expects (rare for the
WanModel class — RoPE freqs are stored as Python attributes, not
buffers), inference falls back to the construction-time value via
`strict=False` load, which is the correct behavior since buffers are
deterministic from network hyperparameters.

Usage:
    # Convert all saves in a checkpoint dir:
    python scripts/post_hoc_convert_lora_saves.py \\
        --ckpt_dir /home/work/.local/.../df_..._14b_lora_..._5000iter/checkpoints

    # Convert specific steps only:
    python scripts/post_hoc_convert_lora_saves.py \\
        --ckpt_dir <dir> --steps 500 1000 1500

    # Dry run (report what would be converted, do nothing):
    python scripts/post_hoc_convert_lora_saves.py \\
        --ckpt_dir <dir> --dry_run

CPU-only — no GPU contention with concurrent training.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load


# Trainable key markers — keep any key whose name contains one of these
# substrings. Matches LORA_TARGET_MODULES + the 14B unfreeze list.
TRAINABLE_MARKERS = (
    "lora_A",
    "lora_B",
    "audio_proj",
    "audio_cond_projs",
    "patch_embedding",
)


def is_trainable_key(key: str) -> bool:
    return any(marker in key for marker in TRAINABLE_MARKERS)


def convert_one(net_model_dir: Path, out_path: Path, dry_run: bool = False) -> dict:
    """Read DCP dir, filter to trainable, save flat .pth."""
    reader = FileSystemReader(str(net_model_dir))
    md = reader.read_metadata()

    full_keys = list(md.state_dict_metadata.keys())
    trainable_keys = [k for k in full_keys if is_trainable_key(k)]

    if not trainable_keys:
        return {"status": "no_trainable_keys", "n_full": len(full_keys), "n_trainable": 0}

    # Build empty state_dict for the keys we want — DCP fills these in-place.
    state_dict = {}
    for key in trainable_keys:
        meta = md.state_dict_metadata[key]
        if hasattr(meta, "size"):
            # Use empty (uninitialized) tensor — values get overwritten by dcp_load.
            state_dict[key] = torch.empty(meta.size)

    if dry_run:
        return {
            "status": "dry_run",
            "n_full": len(full_keys),
            "n_trainable": len(trainable_keys),
            "out_path": str(out_path),
        }

    # Load filtered keys from DCP storage.
    dcp_load(state_dict, storage_reader=reader, no_dist=True)

    # Save as flat .pth (smaller than DCP, faster to load, matches the
    # inference script's else-branch fallback).
    torch.save(state_dict, str(out_path))

    out_size_bytes = out_path.stat().st_size
    in_size_bytes = sum(
        f.stat().st_size for f in net_model_dir.iterdir() if f.is_file()
    )

    return {
        "status": "ok",
        "n_full": len(full_keys),
        "n_trainable": len(trainable_keys),
        "in_size_gb": in_size_bytes / (1024**3),
        "out_size_gb": out_size_bytes / (1024**3),
        "out_path": str(out_path),
    }


def discover_steps(ckpt_dir: Path) -> list:
    """Find all <step>.net_model/ siblings to <step>.pth files."""
    steps = []
    for pth in sorted(ckpt_dir.glob("*.pth")):
        stem = pth.stem  # e.g. "0000500"
        try:
            step = int(stem)
        except ValueError:
            continue
        net_model_dir = ckpt_dir / f"{stem}.net_model"
        if net_model_dir.is_dir():
            steps.append(step)
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, type=Path)
    parser.add_argument(
        "--steps", nargs="+", type=int, default=None,
        help="Specific iteration numbers to convert. Default: all.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print plan without converting.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-convert and overwrite existing .net_model_lora_only.pth files.",
    )
    args = parser.parse_args()

    if not args.ckpt_dir.is_dir():
        print(f"ERROR: --ckpt_dir does not exist: {args.ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    steps = args.steps if args.steps else discover_steps(args.ckpt_dir)
    if not steps:
        print(f"No <step>.net_model/ dirs found under {args.ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"=== post_hoc_convert_lora_saves ===")
    print(f"  ckpt_dir:       {args.ckpt_dir}")
    print(f"  steps:          {steps}")
    print(f"  dry_run:        {args.dry_run}")
    print(f"  overwrite:      {args.overwrite}")
    print(f"  trainable markers: {TRAINABLE_MARKERS}")
    print()

    total_savings_gb = 0.0
    converted = 0
    skipped = 0

    for step in steps:
        step_p = f"{step:07d}"
        net_model_dir = args.ckpt_dir / f"{step_p}.net_model"
        out_path = args.ckpt_dir / f"{step_p}.net_model_lora_only.pth"

        if not net_model_dir.is_dir():
            print(f"step {step:>7d}: SKIP — no .net_model/ dir at {net_model_dir}")
            skipped += 1
            continue

        if out_path.exists() and not args.overwrite:
            print(f"step {step:>7d}: SKIP — output already exists at {out_path}")
            skipped += 1
            continue

        try:
            result = convert_one(net_model_dir, out_path, dry_run=args.dry_run)
        except Exception as e:
            print(f"step {step:>7d}: FAIL — {type(e).__name__}: {e}")
            continue

        if result["status"] == "ok":
            saved_gb = result["in_size_gb"] - result["out_size_gb"]
            total_savings_gb += saved_gb
            converted += 1
            print(
                f"step {step:>7d}: OK   — "
                f"{result['n_trainable']:>4d}/{result['n_full']} keys, "
                f"{result['in_size_gb']:>5.1f} GB -> {result['out_size_gb']:>5.2f} GB "
                f"(save {saved_gb:>5.1f} GB)"
            )
            print(f"           -> {out_path}")
            print(
                f"           manual cleanup (after validation): "
                f"rm -rf {net_model_dir}"
            )
        elif result["status"] == "dry_run":
            print(
                f"step {step:>7d}: DRY  — would convert "
                f"{result['n_trainable']}/{result['n_full']} keys -> "
                f"{out_path}"
            )
        else:
            print(f"step {step:>7d}: SKIP — {result['status']}")
            skipped += 1

    print()
    print(f"=== Summary ===")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped}")
    if not args.dry_run and converted > 0:
        print(f"  Total savings (after manual cleanup): {total_savings_gb:.1f} GB")
        print(
            f"  NOTE: originals are NOT deleted. The script printed a per-save "
            f"`rm -rf` command — run those manually after sanity-checking the "
            f".pth files load correctly."
        )


if __name__ == "__main__":
    main()
