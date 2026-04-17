#!/usr/bin/env bash
# 4-GPU smoke for Re-DMD + TAEW decoder.
# Run: bash scripts/smoke_test_redmd_taew.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_SF_OUTPUT"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

mkdir -p logs logs/redmd_smoke_debug_taew

echo "=== Starting Re-DMD + TAEW smoke (10 iters, batch=1, 4 GPUs) ==="
echo "Log: logs/redmd_smoke_run_taew.log"

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_taew_smoke.py \
    - trainer.resume=False \
    log_config.group="omniavatar_sf_smoke" \
    log_config.name="sf_sink1_window7_redmd_syncc_beta0p25_joonson_parity_taew_smoke" \
    log_config.project="OmniAvatar-FastGen-Smoke" \
    log_config.wandb_entity="paulhcho" \
    2>&1 | tee logs/redmd_smoke_run_taew.log

echo
echo "=== Post-run checks ==="

if grep -qE "Traceback|Error " logs/redmd_smoke_run_taew.log; then
    echo "FAIL: Traceback or Error in log. First 30 hits:"
    grep -nE "Traceback|Error " logs/redmd_smoke_run_taew.log | head -30
    exit 1
fi

if ! grep -q "reward_sync_c_mean" logs/redmd_smoke_run_taew.log; then
    echo "FAIL: reward_sync_c_mean never appeared — reward path didn't fire."
    tail -40 logs/redmd_smoke_run_taew.log
    exit 1
fi

echo "OK: reward_sync_c_mean appeared"

debug_mp4s=$(ls logs/redmd_smoke_debug_taew/*.mp4 2>/dev/null | wc -l)
echo "Debug MP4s written: $debug_mp4s"

echo "SMOKE TEST PASSED"
