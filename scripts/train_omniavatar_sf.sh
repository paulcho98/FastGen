#!/bin/bash
# =============================================================================
# OmniAvatar Self-Forcing (Stage 2) Training
# =============================================================================
#
# 14B teacher + 1.3B causal student + 1.3B fake_score
# Starting from DF shift=5 checkpoint at step 5000
#
# Effective batch size: 8 * 4 GPUs * 2 grad_accum = 64
# Peak memory: ~90 GB/GPU on H200
#
# Usage:
#   bash scripts/train_omniavatar_sf.sh
#   NGPU=2 bash scripts/train_omniavatar_sf.sh
# =============================================================================

set -euo pipefail

NGPU="${NGPU:-4}"
RESUME="${RESUME:-False}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export NEG_TEXT_EMB_PATH="/home/work/stableavatar_data/neg_text_emb.pt"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800  # 30 min timeout for wandb uploads during validation

RUN_NAME="sf_${NGPU}gpu_bs8_lr1e5_5000iter_shift5"

echo "============================================="
echo "  OmniAvatar Self-Forcing Training"
echo "============================================="
echo "  GPUs:            ${NGPU}"
echo "  Batch size:      8/GPU × ${NGPU} GPUs × 2 accum = $((8 * NGPU * 2))"
echo "  Learning rate:   1e-5 (student), 2e-6 (critic)"
echo "  Max iterations:  5000"
echo "  Run name:        ${RUN_NAME}"
echo "============================================="
echo ""

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=${NGPU} \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf.py \
    - trainer.resume=${RESUME} \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen"
