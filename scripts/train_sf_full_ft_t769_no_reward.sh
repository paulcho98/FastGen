#!/bin/bash
# =============================================================================
# Symmetric full-FT 1.3B SF — t769, reward DISABLED (corrected noredmd ablation)
# =============================================================================
#
# Replaces the legacy ``train_sf_sink1_window7_audiofix_taew_syncnet_mouthweight_fsmatched_t769_fsdpfix_noredmd.sh``
# (now deprecated; was using EXTRA_OVERRIDES + asymmetric parent).
#
# Same training regime as ``train_sf_full_ft_t769.sh`` but with
# ``model.reward.enabled=False`` baked into the config — the ablation
# variant of the corrected redmd run.
#
# Default: 1000 iters (set via EXTRA_OVERRIDES below for the ablation).
#
# Usage:
#   bash scripts/train_sf_full_ft_t769_no_reward.sh
#
# Resume:
#   RESUME=True bash scripts/train_sf_full_ft_t769_no_reward.sh
# =============================================================================
set -euo pipefail

export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_sf_full_ft_t769_no_reward.py"
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT_FULL_FT_T769_NO_REWARD}"
export RUN_NAME="${RUN_NAME:-sf_full_ft_t769_no_reward}"

# Ablation runs are short (1000 iters) — bake into EXTRA_OVERRIDES.
# This is a non-regime parameter (just iter cap), so cmdline override is fine.
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-trainer.max_iter=1000}"

exec "$(dirname "$(readlink -f "$0")")/train_sf_parent.sh" "$@"
