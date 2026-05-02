#!/bin/bash
# =============================================================================
# Symmetric full-FT 1.3B SF — 4-step schedule, no reward, windowed CFG
# =============================================================================
#
# Uses the 4-step DF checkpoint (not t769) with the matching 4-step
# timestep distribution.  No reward, windowed CFG on.
#
# Schedule: t_list=[0.999, 0.937, 0.833, 0.624, 0.0], 4 student steps
# DF init:  syncnet-trained 4-step DF 5000-iter ckpt
#
# Usage:
#   bash scripts/train_sf_full_ft_4step_no_reward.sh
#
# Resume:
#   RESUME=True bash scripts/train_sf_full_ft_4step_no_reward.sh
# =============================================================================
set -euo pipefail

export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_sf_full_ft_4step_no_reward.py"

# Override DF init to the 4-step DF checkpoint (NOT the t769 one)
export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_syncnet_trained_shift_5_4gpu_bs16_lr1e5_5000iter/checkpoints/0005000.pth}"

export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT_FULL_FT_4STEP_NO_REWARD}"
export RUN_NAME="${RUN_NAME:-sf_full_ft_4step_no_reward}"

exec "$(dirname "$(readlink -f "$0")")/train_sf_parent.sh" "$@"
