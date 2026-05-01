#!/bin/bash
# =============================================================================
# Symmetric full-FT 1.3B SF — no reward, reversed timestep CFG
# =============================================================================
#
# Same as train_sf_full_ft_t769_no_reward.sh but with timestep_cfg.reverse=True:
# CFG is ON outside [0.556, 0.882] and OFF inside (opposite of normal).
#
# For the t769 2-step schedule this means:
#   Step 1 (t=0.999): CFG ON   (high noise — teacher needs guidance)
#   Step 2 (t=0.769): CFG OFF  (mid noise — teacher already confident)
#
# Usage:
#   bash scripts/train_sf_full_ft_t769_no_reward_reverse_cfg.sh
#
# Resume:
#   RESUME=True bash scripts/train_sf_full_ft_t769_no_reward_reverse_cfg.sh
# =============================================================================
set -euo pipefail

export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_sf_full_ft_t769_no_reward_reverse_cfg.py"
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT_FULL_FT_T769_NO_REWARD_REVERSE_CFG}"
export RUN_NAME="${RUN_NAME:-sf_full_ft_t769_no_reward_reverse_cfg}"

exec "$(dirname "$(readlink -f "$0")")/train_sf_parent.sh" "$@"
