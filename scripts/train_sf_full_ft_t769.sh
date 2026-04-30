#!/bin/bash
# =============================================================================
# Symmetric full-FT 1.3B SF — Re-DMD beta=2 + TAEW + t769 (corrected redmd)
# =============================================================================
#
# Replaces the legacy ``train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_t769_fsdpfix.sh``
# (now in scripts/deprecated_asymmetric/) with the asymmetry fix.
#
# Key differences vs legacy:
#   - fake_score full-FT (~1.3B trainable) instead of LoRA-only (~175M).
#     Fixes the 8× critic-capacity asymmetry.
#   - Both LRs at 2e-6 (legacy hardcoded fake_score at 3e-6 as a half-fix).
#   - Reward weighting is ENABLED (Re-DMD beta=2 + TAEW decoder, same as
#     the legacy redmd path).
#   - Config-driven: regime is set in config_sf_full_ft_t769.py with
#     coherence asserts; no EXTRA_OVERRIDES regime flags.
#
# Usage:
#   bash scripts/train_sf_full_ft_t769.sh
#
#   # Smoke test:
#   EXTRA_OVERRIDES="trainer.max_iter=50 trainer.save_ckpt_iter=50" \
#     bash scripts/train_sf_full_ft_t769.sh
#
# Resume:
#   RESUME=True bash scripts/train_sf_full_ft_t769.sh
# =============================================================================
set -euo pipefail

export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_sf_full_ft_t769.py"
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT_FULL_FT_T769}"
export RUN_NAME="${RUN_NAME:-sf_full_ft_t769}"

exec "$(dirname "$(readlink -f "$0")")/train_sf_parent.sh" "$@"
