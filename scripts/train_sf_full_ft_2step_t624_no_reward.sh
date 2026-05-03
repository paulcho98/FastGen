#!/bin/bash
# =============================================================================
# Symmetric full-FT 1.3B SF — 2-step at t=0.624 (from 4-step DF), no reward
# =============================================================================
set -euo pipefail

export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_sf_full_ft_2step_t624_no_reward.py"
export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_syncnet_trained_shift_5_4gpu_bs16_lr1e5_5000iter/checkpoints/0005000.pth}"
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT_FULL_FT_2STEP_T624_NO_REWARD}"
export RUN_NAME="${RUN_NAME:-sf_full_ft_2step_t624_no_reward}"

exec "$(dirname "$(readlink -f "$0")")/train_sf_parent.sh" "$@"
