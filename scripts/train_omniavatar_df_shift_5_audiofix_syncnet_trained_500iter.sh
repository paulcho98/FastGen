#!/bin/bash
# =============================================================================
# OmniAvatar DF (shift=5) — recovery re-run for step-500 checkpoint only
# =============================================================================
#
# The original 5000-iter syncnet-trained DF run
# (train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh) produced ckpts at
# iters 500, 1000, ..., 4500, 5000 with SAVE_EVERY=500. Only 0005000.pth
# survived the cleanup; the intermediates were deleted. We need the step-500
# checkpoint for downstream eval/ablation, so this wrapper re-runs DF with
# MAX_ITER=500 to regenerate just that one checkpoint.
#
# Why a separate wrapper:
# - Unique RUN_NAME (..._500iter) -> unique output directory under
#   FASTGEN_OUTPUT/.../df_audiofix_syncnet_trained_shift_5_4gpu_bs16_lr1e5_500iter/
#   so the original 0005000.pth is NOT overwritten.
# - Unique output dir -> fresh wandb_id.txt -> new wandb run (not appended to
#   the original 5000-iter run's history).
# - Reproducibility: this is the exact invocation used to recover the ckpt;
#   checking it in documents the intent.
#
# Walltime: ~500 iters * training speed (a few hours on 4x H200). Uses the
# same hyperparameters, student init, dataset, and seed as the parent script.
#
# Usage:
#   bash scripts/train_omniavatar_df_shift_5_audiofix_syncnet_trained_500iter.sh
#
# Resume (if crashed midway; reuses the wandb run via persisted wandb_id.txt):
#   RESUME=True bash scripts/train_omniavatar_df_shift_5_audiofix_syncnet_trained_500iter.sh
#
# GPU conflict caveat: if the active SF training is still consuming the 4
# H200s, this will OOM. Either wait for SF to complete, kill SF, or override
# NGPU / CUDA_VISIBLE_DEVICES for a smaller run on free GPUs.
# =============================================================================

set -euo pipefail

# Delegate to the parent script with MAX_ITER=500. All other env vars
# (NGPU, BATCH_SIZE, SAVE_EVERY, OMNIAVATAR_STUDENT_CKPT, RESUME) pass through
# from the calling shell; defaults (NGPU=4, BS=16, SAVE=500, syncnet-trained
# 1.3B adapter) match the parent.
MAX_ITER=500 exec "$(dirname "$(readlink -f "$0")")/train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh" "$@"
