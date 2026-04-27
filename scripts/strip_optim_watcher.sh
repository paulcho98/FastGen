#!/bin/bash
# =============================================================================
# Optim-shard watcher: strip `*_optim/` from non-latest checkpoints
# =============================================================================
#
# Polls a FSDP-style checkpoints directory. Whenever it sees a step N that
# has a fully-completed save (=> `<step>.pth` metadata file exists), AND
# there is some strictly-greater step N' > N also fully-completed, the
# watcher removes:
#   <CKPT_DIR>/<step>.net_optim/         (the student optimizer shards)
#   <CKPT_DIR>/<step>.fake_score_optim/  (if present, e.g. SF runs)
#
# The latest step's `_optim/` directories are PRESERVED — so if training
# crashes you can still RESUME=True from that step. Once a newer save
# completes, the previously-latest step's optim is in turn stripped.
#
# Net effect: at any given time, exactly one step retains the full quartet
# (model + optim, both for `net` and `fake_score` if applicable); all
# earlier steps are pruned to model-only (inference + warm-restart only,
# no exact resume).
#
# Why ".pth"-existence as the completeness gate: in the FSDP checkpointer
# (fastgen/utils/checkpointer.py:299-330), the `.pth` is written LAST in
# the save sequence — after all `_model/` and `_optim/` distcp dirs are
# committed. So `<step>.pth` existing implies the save is consistent.
# A partial save (mid-OOM-crash, mid-write) will have `_model/` and maybe
# `_optim/` dirs but NO `.pth` — the watcher correctly ignores those.
#
# Usage:
#   bash scripts/strip_optim_watcher.sh <CKPT_DIR>
#   # or with a custom poll interval (seconds, default 60):
#   INTERVAL=30 bash scripts/strip_optim_watcher.sh <CKPT_DIR>
#
# Run in a separate tmux session / nohup background so it survives the
# training process. Stop with Ctrl-C or `kill <pid>`.
#
# Idempotent and safe to start/stop at any time during training.
# =============================================================================

set -uo pipefail

CKPT_DIR="${1:?usage: $0 <checkpoints_dir> (CKPT_DIR ending in /checkpoints)}"
INTERVAL="${INTERVAL:-60}"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: $CKPT_DIR does not exist or is not a directory" >&2
    exit 1
fi

echo "[strip_watcher] watching:    $CKPT_DIR"
echo "[strip_watcher] poll every:  ${INTERVAL}s"
echo "[strip_watcher] policy:      strip <step>.{net,fake_score}_optim once a strictly-greater <step>.pth exists"
echo "[strip_watcher] gate:        <step>.pth presence (FSDP checkpointer writes .pth last => save is complete)"
echo ""

while true; do
    # All complete saves (have a top-level .pth metadata file).
    # Format of step labels: zero-padded integers like 0000500.
    complete_steps=$(ls "$CKPT_DIR"/*.pth 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.pth$//' | sort -u)

    if [ -z "$complete_steps" ]; then
        # Nothing saved yet — wait for the first checkpoint to land.
        sleep "$INTERVAL"
        continue
    fi

    # Highest-numbered fully-saved step. We preserve THIS one's optim.
    latest=$(echo "$complete_steps" | tail -n1)

    # Walk every complete step != latest. Strip optim if not yet stripped.
    for step in $complete_steps; do
        [ "$step" = "$latest" ] && continue
        for kind in net_optim fake_score_optim; do
            target="$CKPT_DIR/${step}.${kind}"
            if [ -d "$target" ]; then
                # Use du for a quick size before deleting, for the log line.
                sz=$(du -sh "$target" 2>/dev/null | cut -f1)
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] [strip_watcher] rm -rf ${step}.${kind} (${sz}); latest=${latest}"
                rm -rf "$target"
            fi
        done
    done

    sleep "$INTERVAL"
done
