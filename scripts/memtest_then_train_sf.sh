#!/bin/bash
# =============================================================================
# Step 1: Run 14B×3 memory test (6 iterations, ~10 min)
# Step 2: Start full 1.3B Self-Forcing training (bs=8, 5000 iters)
# =============================================================================
# Usage: nohup bash scripts/memtest_then_train_sf.sh > /tmp/memtest_then_train.log 2>&1 &

set -euo pipefail

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export NEG_TEXT_EMB_PATH="/home/work/stableavatar_data/neg_text_emb.pt"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800  # 30 min timeout for wandb uploads during validation
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_SF_OUTPUT"  # Use overlay FS (960GB free) instead of /home/work (17GB free)
export SKIP_GT_VAL_UPLOAD=1  # Skip GT validation video upload at startup (causes NCCL timeout)
export SKIP_EARLY_SAMPLE_LOG=1  # Skip sample video generation at iter 1 (slow, blocks NCCL)

MEMTEST_LOG="/tmp/sf_14b_memtest_$(date +%Y%m%d_%H%M%S).log"
TRAIN_LOG="/tmp/sf_train_$(date +%Y%m%d_%H%M%S).log"

# =============================================================================
# Step 1: 14B×3 Memory Test
# =============================================================================
echo "============================================="
echo "  STEP 1: 14B×3 Memory Test"
echo "  Log: ${MEMTEST_LOG}"
echo "============================================="

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf_14b_memtest.py \
    2>&1 | tee "${MEMTEST_LOG}"

MEMTEST_EXIT=$?

echo ""
echo "============================================="
echo "  14B×3 Memory Test Results"
echo "============================================="

if [ $MEMTEST_EXIT -eq 0 ]; then
    echo "STATUS: SUCCESS"
    echo ""
    echo "=== Peak GPU Memory ==="
    grep "peak_gpu_mem" "${MEMTEST_LOG}" | tail -5
    echo ""
    echo "=== Per-Stage Memory ==="
    grep "\[MEM\]" "${MEMTEST_LOG}" | tail -20
    echo ""
    echo "=== Per-Layer Memory (14B teacher forward) ==="
    grep "\[MEM-fwd\].*5120" "${MEMTEST_LOG}" | tail -10
    echo ""
    echo "=== Training Losses ==="
    grep "iter.*loss" "${MEMTEST_LOG}"
else
    echo "STATUS: FAILED (exit code ${MEMTEST_EXIT})"
    echo ""
    echo "=== Errors ==="
    grep -E "Error|OOM|CUDA" "${MEMTEST_LOG}" | tail -10
    echo ""
    echo "WARNING: 14B×3 did not fit. Proceeding with 1.3B training anyway."
fi

echo ""
echo "Memory test results saved to: ${MEMTEST_LOG}"
echo ""

# =============================================================================
# Step 2: Full 1.3B Self-Forcing Training
# =============================================================================
echo "============================================="
echo "  STEP 2: Starting 1.3B Self-Forcing Training"
echo "  Log: ${TRAIN_LOG}"
echo "============================================="
echo ""

# Clean up any leftover state from memtest
rm -rf FASTGEN_OUTPUT/fastgen/omniavatar_sf_14b_memtest/ 2>/dev/null

NGPU=4
RUN_NAME="sf_4gpu_bs8_lr1e5_5000iter_shift5"

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=${NGPU} \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf.py \
    - trainer.resume=False \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    2>&1 | tee "${TRAIN_LOG}"

echo ""
echo "Training finished. Log: ${TRAIN_LOG}"
