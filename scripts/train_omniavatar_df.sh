#!/bin/bash
# OmniAvatar Diffusion Forcing (Stage 1) — Single GPU training
#
# Trains the causal 1.3B student on real data with inhomogeneous block-wise timesteps.
# No ODE trajectories or teacher model needed.
#
# Max batch_size per H200 (143GB): 36  (105 GB peak)
# Recommended: batch_size=32, grad_accum=2 → effective batch size 64
#
# Usage:
#   bash scripts/train_omniavatar_df.sh [GPU_ID]
#   bash scripts/train_omniavatar_df.sh 2

GPU_ID="${1:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_ITER="${MAX_ITER:-5000}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="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"

echo "=== OmniAvatar Diffusion Forcing Training ==="
echo "GPU: ${GPU_ID}"
echo "Batch size: ${BATCH_SIZE} x grad_accum ${GRAD_ACCUM} = effective $(( BATCH_SIZE * GRAD_ACCUM ))"
echo "Max iterations: ${MAX_ITER}"
echo ""

/home/work/.local/miniconda3/envs/hb_fastgen/bin/python train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_df.py \
    - dataloader_train.batch_size=${BATCH_SIZE} \
    trainer.grad_accum_rounds=${GRAD_ACCUM} \
    trainer.max_iter=${MAX_ITER} \
    trainer.logging_iter=10 \
    trainer.save_ckpt_iter=500 \
    log_config.group="omniavatar_df" \
    log_config.name="omniavatar_df_bs${BATCH_SIZE}x${GRAD_ACCUM}"
