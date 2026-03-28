#!/bin/bash
# Quick E2E test: 3 train iters + validation with 2 samples on single GPU
# Tests: DF training -> FlexAttention compile -> val loop -> AR generation -> VAE decode -> wandb video logging
set -euo pipefail

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export OMNIAVATAR_VAL_LIST="/home/work/stableavatar_data/v2v_training_data/video_square_val2_test.txt"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== E2E Validation Test ==="
echo "  3 train iters (bs=1), validate at iter 3 with 2 samples"
echo "  Single GPU, wandb enabled"
echo ""

CUDA_VISIBLE_DEVICES=2 /home/work/.local/miniconda3/envs/hb_fastgen/bin/python \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_df.py \
    - dataloader_train.batch_size=1 \
    trainer.max_iter=4 \
    trainer.logging_iter=1 \
    trainer.save_ckpt_iter=9999 \
    trainer.validation_iter=3 \
    trainer.skip_initial_validation=True \
    trainer.callbacks.wandb.sample_logging_iter=3 \
    log_config.group="omniavatar_df_test" \
    log_config.name="val_e2e_test" \
    log_config.project="OmniAvatar-FastGen"
