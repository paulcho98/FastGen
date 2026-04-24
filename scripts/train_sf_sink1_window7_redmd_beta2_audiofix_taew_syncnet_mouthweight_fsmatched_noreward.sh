#!/bin/bash
# =============================================================================
# SF Re-DMD β=2 TAEW audiofix — fsmatched + lr3e6 — REWARD DISABLED variant
# =============================================================================
#
# Same as train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh
# (syncnet DF-5000 init + mouthweight 14B teacher + fsmatched fake score +
# critic lr 3e-6) with the sync-C reward turned off, so training reverts to
# vanilla OmniAvatar Self-Forcing (DMD2 VSD loss, no reward weighting, no
# VAE decode in the student step).
#
# How the disable works: the Re-DMD model class
# (fastgen/methods/omniavatar_self_forcing_re_dmd.py:67-72) checks
# config.reward.enabled at build time; when False it sets
# self.reward_scorer = None and _student_update_step falls back to base DMD2
# behavior. model_class._target_ stays OmniAvatarSelfForcingReDMD but
# functionally runs the baseline path.
#
# Why a separate wrapper:
# - Ablation baseline: isolate the effect of the sync-C reward on student
#   quality by matching everything else (init, teacher, fake_score, lr,
#   schedule, seed) against the fsmatched_lr3e6 run 1djswvuo.
# - Distinct FASTGEN_OUTPUT_ROOT + RUN_NAME (_noreward suffix) -> isolated
#   output dir + fresh wandb run (no collision with 1djswvuo).
# - DF init: uses the parent's default (step-5000 of the syncnet-trained DF
#   run), NOT the _df500 variant.
#
# Usage:
#   bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_noreward.sh
#
# Resume (same wandb run via persisted wandb_id.txt):
#   RESUME=True bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_noreward.sh
# =============================================================================

set -euo pipefail

# Distinct output dir and wandb run name.
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT_FSMATCHED_LR3E6_NOREWARD}"
export RUN_NAME="${RUN_NAME:-sf_sink1_window7_redmd_audiofix_beta2_taew_syncnet_mouthweight_fsmatched_lr3e6_noreward}"

# Disable the sync-C reward scorer + stop loading raw audio into the batch
# (waveform is only used by the reward path; skipping it is a small perf win).
export EXTRA_OVERRIDES="model.reward.enabled=False dataloader_train.load_raw_audio=False"

# Delegate to the fsmatched parent (defaults to DF step-5000 init).
exec "$(dirname "$(readlink -f "$0")")/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh" "$@"
