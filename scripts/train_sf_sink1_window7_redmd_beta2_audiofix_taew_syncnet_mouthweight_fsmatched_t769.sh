#!/bin/bash
# =============================================================================
# SF Re-DMD β=2 TAEW audiofix — fsmatched + DF-5000 init + t769 schedule
# =============================================================================
#
# Same as fsmatched_lr3e6 (the active 1djswvuo run) with ONE change:
#
#   Schedule t_list[1]: 0.833 -> 0.769 (the second-step input is now at
#   inference step 30/50 instead of step 25/50 in denoising order).
#   Linear unshifted schedule t = (50-i)/50, then shift=5: t' = 5t/(1+4t).
#       step 25/50 -> t=0.5 -> t'=0.833 (fsmatched_lr3e6 baseline)
#       step 30/50 -> t=0.4 -> t'=0.769 (this variant)
#   Effect: first denoising step covers more of the schedule (0.999 -> 0.769
#   is a larger jump than 0.999 -> 0.833); second step refines a less-noisy
#   input. Concentrates denoising effort earlier, where the variance is
#   largest. 0.769 still falls inside timestep_cfg [t_lo=0.556, t_hi=0.882]
#   so timestep-conditional CFG still fires on this step.
#
# Critic LR stays at fsmatched_lr3e6 default (3e-6, 1.5x student LR) — same
# as the active run. The earlier sibling-wrapper that swept critic LR down
# to 4e-7 was retracted; we now keep critic LR fixed and isolate the t_list
# change as the only varied dimension.
#
# Everything else identical to the fsmatched parent:
# - Student DF init: DF-5000 mouthweight
# - Teacher: mouthweight 14B step-6000
# - Fake_score init: syncnet-trained 1.3B (fsmatched)
# - Reward: sync-C β=2 enabled, TAEW decoder
# - Attention: sink=1 / window=7
# - timestep-conditional CFG, dynamic RoPE
# - student_sample_steps=2 (only 2 actual denoising steps)
# - Critic LR: 3e-6 (parent's hardcoded override)
#
# Distinct output dir + wandb run name (_t769 suffix) so it does NOT
# collide with fsmatched_lr3e6.
#
# Usage:
#   bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_t769.sh
#
# Resume:
#   RESUME=True bash scripts/...fsmatched_t769.sh
# =============================================================================

set -euo pipefail

# Distinct output dir and wandb run name.
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT_FSMATCHED_T769}"
export RUN_NAME="${RUN_NAME:-sf_sink1_window7_redmd_audiofix_beta2_taew_syncnet_mouthweight_fsmatched_t769}"

# Single Hydra-style override appended to torchrun via the parent's
# EXTRA_OVERRIDES hook. Critic LR stays at the parent's 3e-6 default
# (the parent hardcodes model.fake_score_optimizer.lr=3e-6 on its own
# torchrun cmdline; we don't touch it).
# OmegaConf list syntax for t_list: square brackets, comma-separated, no spaces.
export EXTRA_OVERRIDES="model.sample_t_cfg.t_list=[0.999,0.769,0.0]"

# Delegate to the fsmatched parent (DF-5000 mouthweight init by default).
exec "$(dirname "$(readlink -f "$0")")/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh" "$@"
