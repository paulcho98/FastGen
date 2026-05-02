#!/bin/bash
# =============================================================================
# Symmetric full-FT 1.3B SF — no reward, NO CFG
# =============================================================================
#
# Same as train_sf_full_ft_t769_no_reward.sh but with guidance_scale=None:
# the negative teacher forward pass is completely skipped, halving
# teacher compute per iter.
#
# Usage:
#   bash scripts/train_sf_full_ft_t769_no_reward_no_cfg.sh
#
# Resume:
#   RESUME=True bash scripts/train_sf_full_ft_t769_no_reward_no_cfg.sh
# =============================================================================
set -euo pipefail

export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_sf_full_ft_t769_no_reward_no_cfg.py"
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT_FULL_FT_T769_NO_REWARD_NO_CFG}"
export RUN_NAME="${RUN_NAME:-sf_full_ft_t769_no_reward_no_cfg}"

exec "$(dirname "$(readlink -f "$0")")/train_sf_parent.sh" "$@"
