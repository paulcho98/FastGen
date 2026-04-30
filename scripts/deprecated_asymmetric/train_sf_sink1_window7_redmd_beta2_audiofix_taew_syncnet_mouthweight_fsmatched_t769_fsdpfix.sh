#!/bin/bash
# =============================================================================
# DEPRECATED — ASYMMETRIC TRAINABLE CAPACITY (do not launch new runs)
# =============================================================================
# Student trains as full-FT (~1421M); fake_score trains as LoRA-only (~175M).
# 8x critic-capacity asymmetry — see scripts/deprecated_asymmetric/README.md
# for full diagnosis.  Replacement scripts: train_sf_full_ft_t769.sh and
# train_sf_full_ft_t769_no_reward.sh in scripts/.  This file is kept only
# for reproducibility of past runs.
# =============================================================================
# =============================================================================
# SF fsmatched_t769 — REFERENCE: matched-baseline for FSDP non-block-grad-sync fix
# =============================================================================
#
# Identical configuration to fsmatched_t769 (sibling script), but launched
# AFTER the fix to CausalOmniAvatarWan.fully_shard that wraps non-block
# submodules (patch_embedding, text_embedding, time_embedding, time_projection,
# head, audio_proj, each audio_cond_projs[i]) with FSDP individually.  See
# fastgen/networks/OmniAvatar/network_causal.py:2148-2210.
#
# Hypothesis being tested: that the prior block-only-sharding behavior
# (gradients on the audio path NOT reduce-scattered across ranks; rank-0's
# weights drift independently of ranks 1-3 and end up being what we save)
# was a non-trivial contributor to the student-vs-teacher sync-C gap we
# have observed across all SF runs to date.
#
# Comparison target (for the eval CSV): the fsmatched_t769 run currently
# in flight (active wandb run).  Same DF init, same teacher, same fake_score,
# same reward, same schedule, same critic LR — only the fully_shard override
# differs.  Any sustained Sync-C delta between the two runs at matched
# checkpoint steps is attributable to the fix.
#
# This script is for REFERENCE — keep it ready to launch when you are
# willing to commit GPUs to the comparison.  The 14B DF run takes priority
# right now.
#
# Distinct output dir + RUN_NAME so eval CSV rows don't collide with the
# pre-fix baseline.
#
# Usage:
#   bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_t769_fsdpfix.sh
#
# Resume:
#   RESUME=True bash scripts/...fsmatched_t769_fsdpfix.sh
# =============================================================================

set -euo pipefail

# DF init: the SAME schedule-matched DF-5000 ckpt the pre-fix fsmatched_t769
# baseline used.  We do NOT want to confound the comparison by switching DF
# inits.  (When the new 14B DF — which itself benefits from the same fix —
# completes, a separate comparison would test that new DF init.)
export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_syncnet_trained_shift_5_t769_4gpu_bs16_lr1e5_5000iter/checkpoints/0005000.pth}"

# Distinct output dir and RUN_NAME (`_fsdpfix` suffix).
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT_FSMATCHED_T769_FSDPFIX}"
export RUN_NAME="${RUN_NAME:-sf_sink1_window7_redmd_audiofix_beta2_taew_syncnet_mouthweight_fsmatched_t769_fsdpfix}"

# Same EXTRA_OVERRIDES as fsmatched_t769 — only the t_list, no other
# changes.  The fully_shard fix is a code change in network_causal.py, not
# a config flag, so it applies automatically as long as this branch HEAD
# includes the fix commit.
export EXTRA_OVERRIDES="model.sample_t_cfg.t_list=[0.999,0.769,0.0]"

# Delegate to the fsmatched parent.
exec "$(dirname "$(readlink -f "$0")")/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh" "$@"
