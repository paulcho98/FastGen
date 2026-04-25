#!/bin/bash
# =============================================================================
# SF Re-DMD β=2 TAEW audiofix — fsmatched + DF-5000 init + t882 + critic 4e-7
# =============================================================================
#
# Same as fsmatched_lr3e6 (the active 1djswvuo run) with two changes:
#
# 1) Schedule t_list[1]: 0.833 -> 0.882 (corresponds to unshifted step 30/50
#    instead of step 25/50, after the shift=5 transform t' = 5t/(1+4t)).
#    Effect: the second student denoising step now starts deeper in the noise
#    schedule (more residual noise to remove), which biases the student
#    toward more aggressive single-step denoising on the final step.
#    The 0.882 is right at timestep_cfg.t_hi=0.882, so timestep-conditional
#    CFG still fires on this step (inclusive bound).
#
# 2) Critic (fake_score) LR: 3e-6 -> 4e-7 (7.5x drop). Slows critic
#    adaptation; tests whether the previous critic LR was over-fitting to
#    student noise. Note this is now BELOW the student LR (2e-6, default),
#    so the critic trains 5x slower than the student — opposite of every
#    prior fsmatched variant.
#
# Everything else identical to the fsmatched parent:
# - Student DF init: DF-5000 mouthweight
# - Teacher: mouthweight 14B step-6000
# - Fake_score init: syncnet-trained 1.3B (fsmatched)
# - Reward: sync-C β=2 enabled, TAEW decoder
# - Attention: sink=1 / window=7
# - timestep-conditional CFG, dynamic RoPE
# - student_sample_steps=2 (only 2 actual denoising steps)
#
# Distinct output dir + wandb run name (_t882_lrcrit4e7 suffix) so it
# does NOT collide with fsmatched_lr3e6.
#
# Usage:
#   bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_t882_lrcrit4e7.sh
#
# Resume:
#   RESUME=True bash scripts/...t882_lrcrit4e7.sh
# =============================================================================

set -euo pipefail

# Distinct output dir and wandb run name.
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT_FSMATCHED_T882_LRCRIT4E7}"
export RUN_NAME="${RUN_NAME:-sf_sink1_window7_redmd_audiofix_beta2_taew_syncnet_mouthweight_fsmatched_t882_lrcrit4e7}"

# Two Hydra-style overrides appended to torchrun via the parent's
# EXTRA_OVERRIDES hook. Order: parent has model.fake_score_optimizer.lr=3e-6
# baked in; our 4e-7 override comes AFTER it on the cmdline so it wins.
# OmegaConf list syntax for t_list: square brackets, comma-separated, no spaces.
export EXTRA_OVERRIDES="model.sample_t_cfg.t_list=[0.999,0.882,0.0] model.fake_score_optimizer.lr=4e-7"

# Delegate to the fsmatched parent (DF-5000 mouthweight init by default).
exec "$(dirname "$(readlink -f "$0")")/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh" "$@"
