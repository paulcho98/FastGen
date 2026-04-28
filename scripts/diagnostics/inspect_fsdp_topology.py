"""Inspect a saved FSDP DCP checkpoint's wrap topology.

Reports, per parameter, whether it was saved as a sharded DTensor (one
.distcp shard per rank) or as a non-sharded full tensor.  Used to confirm
that bug 1's fix (per-submodule fully_shard wraps in
CausalOmniAvatarWan.fully_shard) actually changed the wrap topology for
non-block submodules.

Usage:
    python scripts/diagnostics/inspect_fsdp_topology.py \\
        <path-to-NNNNNNN.net_model-dir>

The path argument should point to one of the *.net_model directories that
the FSDPCheckpointer creates next to the <step>.pth metadata file.  E.g.,
for an SF run:

    /tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT_FSMATCHED_T769/.../checkpoints/0000500.net_model

Pre-fix SF run (block-only sharding) — non-block params stored as full
tensors:
    [REPLICATED] _core.patch_embedding.weight                 shape=...
    [REPLICATED] _core.audio_proj.proj.weight                 shape=...
    [REPLICATED] _core.audio_cond_projs.0.weight              shape=...
    [SHARDED]    _core.blocks.0.self_attn.q.weight (4 shards) shape=...

Post-fix run (per-submodule FSDP) — non-block params now sharded:
    [SHARDED] _core.patch_embedding.weight     (4 shards)     shape=...
    [SHARDED] _core.audio_proj.proj.weight     (4 shards)     shape=...
    [SHARDED] _core.audio_cond_projs.0.weight  (4 shards)     shape=...
    [SHARDED] _core.blocks.0.self_attn.q.weight (4 shards)    shape=...

The topology change is itself the proof of fix scope: in the pre-fix
case those tensors had no FSDP hook -> no reduce-scatter on backward ->
gradient drift across ranks every iter.  Post-fix, the SHARDED status
implies an FSDP hook is now in place for them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)


# Submodules whose params we expect to differ in topology between pre-fix
# (block-only sharding) and post-fix (per-submodule sharding) runs.  These
# are the ones that bug 1's drift affected.
NON_BLOCK_SUBMODULE_PREFIXES = [
    "_core.patch_embedding",
    "_core.text_embedding",
    "_core.time_embedding",
    "_core.time_projection",
    "_core.head",
    "_core.audio_proj",
    "_core.audio_cond_projs",
]


def classify_metadata(meta: Metadata) -> dict:
    """Walk the .metadata file and classify every entry.

    Returns:
      {
        "sharded":    [(fqn, num_shards, shape)],
        "replicated": [(fqn, shape)],
        "bytes":      [fqn],          # non-tensor entries (rare)
      }
    """
    out = {"sharded": [], "replicated": [], "bytes": []}

    for fqn, plan in meta.state_dict_metadata.items():
        if isinstance(plan, BytesStorageMetadata):
            out["bytes"].append(fqn)
            continue

        if not isinstance(plan, TensorStorageMetadata):
            continue

        chunks = plan.chunks
        size = tuple(plan.size)
        if len(chunks) > 1:
            out["sharded"].append((fqn, len(chunks), size))
        else:
            chunk_size = tuple(chunks[0].sizes)
            if chunk_size == size:
                out["replicated"].append((fqn, size))
            else:
                out["sharded"].append((fqn, 1, size))

    return out


def summarize(classified: dict, expected_world_size: int | None = None) -> None:
    n_sharded = len(classified["sharded"])
    n_replicated = len(classified["replicated"])
    n_bytes = len(classified["bytes"])

    print(f"Total entries: {n_sharded + n_replicated + n_bytes}")
    print(f"  SHARDED:    {n_sharded}")
    print(f"  REPLICATED: {n_replicated}")
    print(f"  BYTES:      {n_bytes}")
    print()

    rep_by_prefix = defaultdict(list)
    for fqn, shape in classified["replicated"]:
        for prefix in NON_BLOCK_SUBMODULE_PREFIXES:
            if fqn.startswith(prefix):
                rep_by_prefix[prefix].append((fqn, shape))
                break

    print("Non-block submodules in REPLICATED (would be drift-prone in pre-fix runs):")
    if not rep_by_prefix:
        print("  (none — all non-block submodules are sharded)")
    else:
        for prefix in NON_BLOCK_SUBMODULE_PREFIXES:
            entries = rep_by_prefix.get(prefix, [])
            if entries:
                print(f"  {prefix}:")
                for fqn, shape in sorted(entries)[:5]:
                    print(f"    {fqn}  shape={shape}")
                if len(entries) > 5:
                    print(f"    ... ({len(entries) - 5} more)")
    print()

    sharded_block = [s for s in classified["sharded"] if "_core.blocks" in s[0]]
    sharded_nonblock = [s for s in classified["sharded"] if "_core.blocks" not in s[0]]

    print("Block sharded entries (sanity — should always be sharded):")
    print(f"  count: {len(sharded_block)}")
    if sharded_block:
        fqn, n_shards, shape = sharded_block[0]
        print(f"  example: {fqn}  shards={n_shards}  shape={shape}")
    print()

    print("Non-block sharded entries (post-fix evidence — these were REPLICATED pre-fix):")
    print(f"  count: {len(sharded_nonblock)}")
    for fqn, n_shards, shape in sorted(sharded_nonblock)[:8]:
        print(f"  {fqn}  shards={n_shards}  shape={shape}")
    if len(sharded_nonblock) > 8:
        print(f"  ... ({len(sharded_nonblock) - 8} more)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ckpt_path",
        help="Path to a *.net_model directory (or any DCP-saved dir) containing .metadata",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Expected world size (used only for cross-check on shard count)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_dir():
        print(f"ERROR: not a directory: {ckpt_path}", file=sys.stderr)
        return 1

    metadata_file = ckpt_path / ".metadata"
    if not metadata_file.exists():
        print(f"ERROR: no .metadata file at {metadata_file}", file=sys.stderr)
        return 1

    # PyTorch DCP writes .metadata via the FileSystemWriter using either
    # pickle or its own protocol depending on version.  Try DCP's own
    # FileSystemReader first (most robust), fall back to pickle.
    try:
        from torch.distributed.checkpoint import FileSystemReader

        reader = FileSystemReader(str(ckpt_path))
        metadata = reader.read_metadata()
    except Exception:
        import pickle
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
    classified = classify_metadata(metadata)
    print(f"Inspecting: {ckpt_path}")
    print()
    summarize(classified, expected_world_size=args.world_size)

    return 0


if __name__ == "__main__":
    sys.exit(main())
