#!/bin/bash
# =============================================================================
# OmniAvatar Diffusion Forcing (Stage 1) — Full Training Run
# =============================================================================
#
# Trains the causal 1.3B student on real data with inhomogeneous block-wise
# timesteps. No ODE trajectories or teacher model needed.
#
# Starting checkpoint: 1.3B V2V phase2 (step-19500, 65ch, LoRA merged)
# Effective batch size: 16 * 4 GPUs = 64
# LR: 5e-5 (matching OmniAvatar native training)
# Loss logging: every step
# Video visualization: every 500 steps (10 samples, 25fps with audio)
# Checkpoints: every 500 steps
#
# Memory: ~48 GB peak per GPU (bs=16 on H200 143GB)
#
# Usage:
#   bash scripts/train_omniavatar_df.sh              # Default: 4 GPUs, 5000 iters
#   NGPU=2 bash scripts/train_omniavatar_df.sh       # 2 GPUs
#   MAX_ITER=10000 bash scripts/train_omniavatar_df.sh  # 10K iters
# =============================================================================

set -euo pipefail

NGPU="${NGPU:-4}"
MAX_ITER="${MAX_ITER:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-500}"
VIZ_EVERY="${VIZ_EVERY:-500}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EFFECTIVE_BS=$((BATCH_SIZE * NGPU))
RUN_NAME="df_${NGPU}gpu_bs${BATCH_SIZE}_lr1e5_${MAX_ITER}iter"

echo "============================================="
echo "  OmniAvatar Diffusion Forcing Training"
echo "============================================="
echo "  GPUs:            ${NGPU}"
echo "  Batch size:      ${BATCH_SIZE}/GPU × ${NGPU} = ${EFFECTIVE_BS}"
echo "  Learning rate:   1e-5"
echo "  Max iterations:  ${MAX_ITER}"
echo "  Checkpoint:      ${OMNIAVATAR_STUDENT_CKPT:-/home/work/output_omniavatar_v2v_1.3B_phase2/step-19500.pt}"
echo "  Save every:      ${SAVE_EVERY} steps"
echo "  Visualize every: ${VIZ_EVERY} steps"
echo "  Run name:        ${RUN_NAME}"
echo "============================================="
echo ""

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=${NGPU} \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_df.py \
    - dataloader_train.batch_size=${BATCH_SIZE} \
    trainer.ddp=True \
    trainer.max_iter=${MAX_ITER} \
    trainer.save_ckpt_iter=${SAVE_EVERY} \
    log_config.group="omniavatar_df" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen"
