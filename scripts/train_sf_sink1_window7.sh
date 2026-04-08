#!/bin/bash
# SF training with sliding window attention: sink=1, window=7, dynamic RoPE, 2-step.
# Uses the stochastic-attention DF checkpoint (step 10000) as initialization.
#
# Usage: nohup bash scripts/train_sf_sink1_window7.sh > /tmp/train_sf_sink1_window7.log 2>&1 &
set -euo pipefail

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_SF_OUTPUT"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

RUN_NAME="sf_sink1_window7_dynrope_2step"

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7.py \
    - trainer.resume=False \
    log_config.group="omniavatar_sf" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho"
