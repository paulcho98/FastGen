#!/usr/bin/env bash
# Re-DMD sync-C reward 4-GPU smoke test.
# Runs scripts/train_sf_sink1_window7_redmd.sh's launch pattern with the smoke config.
# Goal: verify the reward path works end-to-end for ≤10 iterations.

set -euo pipefail

cd "$(dirname "$0")/.."

# Env vars mirrored from train_sf_sink1_window7_redmd.sh
export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_SF_OUTPUT"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

# Override wandb to offline so we don't pollute the real project
export WANDB_MODE=offline

mkdir -p logs logs/redmd_smoke_debug

echo "=== Starting Re-DMD smoke test (10 iters, batch_size=1, 4 GPUs) ==="
echo "Log: logs/redmd_smoke_run.log"

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_smoke.py \
    - trainer.resume=False \
    log_config.group="omniavatar_sf_smoke" \
    log_config.name="sf_sink1_window7_redmd_syncc_beta0p25_smoke" \
    log_config.project="OmniAvatar-FastGen-Smoke" \
    log_config.wandb_entity="paulhcho" \
    2>&1 | tee logs/redmd_smoke_run.log

# Post-run verification
echo
echo "=== Post-run checks ==="

if grep -qE "Traceback|Error " logs/redmd_smoke_run.log; then
    echo "FAIL: Traceback or Error found in log. First 30 hits:"
    grep -nE "Traceback|Error " logs/redmd_smoke_run.log | head -30
    exit 1
fi

if ! grep -q "reward_sync_c_mean" logs/redmd_smoke_run.log; then
    echo "FAIL: reward_sync_c_mean never appeared — reward path didn't fire."
    echo "Last 40 lines of log:"
    tail -40 logs/redmd_smoke_run.log
    exit 1
fi

echo "OK: reward_sync_c_mean found in log."

debug_mp4s=$(ls logs/redmd_smoke_debug/*.mp4 2>/dev/null | wc -l)
if [ "$debug_mp4s" -lt 1 ]; then
    echo "WARN: no debug MP4 written — check save_reward_debug_video config."
else
    echo "OK: $debug_mp4s debug MP4 file(s) written."
fi

echo "SMOKE TEST PASSED"
