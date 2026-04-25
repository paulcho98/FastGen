#!/bin/bash
# =============================================================================
# OmniAvatar DF (shift=5) — t769 schedule (2-step, matches SF t769 inference)
# =============================================================================
#
# Same student init (syncnet-trained mouthweight 1.3B step-1000) and same
# hyperparameters as train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh,
# but trains with config_df_shift_5_t769.py:
#
#   sample_t_cfg.t_list:       [0.999, 0.937, 0.833, 0.624, 0.0]
#                            -> [0.999, 0.769,        0.0]
#   student_sample_steps:                4 -> 2
#
# Effect: the DF student is only exposed to the two noise levels that the
# matching SF run will use at inference (input on step 1 = t=0.999,
# input on step 2 = t=0.769), instead of being spread across 4 timesteps
# (the existing DF-5000 ckpt was trained on 0.937, 0.833, 0.624, 0.0 which
# includes levels SF t769 will never face). Removes train/test mismatch
# between DF and SF schedules.
#
# Output dir:
#   FASTGEN_OUTPUT/.../df_audiofix_syncnet_trained_shift_5_t769_4gpu_bs16_lr1e5_5000iter/
# Fresh wandb run.
#
# Walltime estimate: ~32 h on 4x H200 (matches the original 5000-iter run;
# per-step cost is independent of how many timesteps are sampled — same
# forward/backward pass cost, just narrower distribution).
#
# Usage:
#   bash scripts/train_omniavatar_df_shift_5_t769_audiofix_syncnet_trained.sh
#
# Quick smoke (e.g. 100 iters, save every 50):
#   MAX_ITER=100 SAVE_EVERY=50 VIZ_EVERY=50 \
#     bash scripts/train_omniavatar_df_shift_5_t769_audiofix_syncnet_trained.sh
#
# Resume:
#   RESUME=True bash scripts/train_omniavatar_df_shift_5_t769_audiofix_syncnet_trained.sh
# =============================================================================

set -euo pipefail

# Use the t769-specialized config (overrides sample_t_cfg.t_list and
# student_sample_steps to the 2-step schedule).
export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_df_shift_5_t769.py"

# Distinct RUN_NAME (note the _t769_ infix in the name template) so the
# output dir does NOT collide with the existing DF runs.
NGPU="${NGPU:-4}"
MAX_ITER="${MAX_ITER:-5000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
export RUN_NAME="${RUN_NAME:-df_audiofix_syncnet_trained_shift_5_t769_${NGPU}gpu_bs${BATCH_SIZE}_lr1e5_${MAX_ITER}iter}"

# Delegate to the parent (passes NGPU/MAX_ITER/BATCH_SIZE/SAVE_EVERY/RESUME
# through the shell environment).
exec "$(dirname "$(readlink -f "$0")")/train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh" "$@"
