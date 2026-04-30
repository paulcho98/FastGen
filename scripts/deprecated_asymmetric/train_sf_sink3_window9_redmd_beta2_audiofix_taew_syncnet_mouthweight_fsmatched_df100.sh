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
# SF Re-DMD β=2 TAEW audiofix — fsmatched + lr3e6 — DF-100 init + sink3/window9
# =============================================================================
#
# Two changes vs the baseline fsmatched_lr3e6 parent:
#
# 1) Student DF init: step-100 of the syncnet-trained DF 100-iter run
#    (from train_omniavatar_df_shift_5_audiofix_syncnet_trained_100iter.sh)
#    instead of the DF step-5000 final ckpt.
#
# 2) Attention sink/window: sink_size 1 -> 3, local_attn_size 7 -> 9. The
#    rolling window stays 6 frames; only the sink grows. Rationale: more
#    permanent-visible anchor frames may help long-sequence coherence and
#    identity stability under sync-C reward pressure.
#
# The DF student saw (sink=3, window=9) attention as one of the 5
# stochastic configs during training (config_df_shift_5.py:62), so the
# weights are already exposed to this attention pattern — no dedicated
# specialization required.
#
# For inference on checkpoints from this run, remember to also switch
# --sink_size and --local_attn_size to 3 and 9 respectively (the existing
# infer_*.sh scripts hardcode sink=1/window=7).
#
# Everything else is identical to the fsmatched parent:
# - teacher: mouthweight 14B step-6000
# - fake_score init: syncnet-trained 1.3B adapter (fsmatched)
# - critic lr: 3e-6
# - sync-C reward: enabled (β=2, SyncNet-v2, TAEW decoder)
# - timestep-conditional CFG, dynamic RoPE
#
# Usage:
#   bash scripts/train_sf_sink3_window9_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_df100.sh
#
# Resume:
#   RESUME=True bash scripts/train_sf_sink3_window9_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_df100.sh
# =============================================================================

set -euo pipefail

# DF init: step-100 of the (stochastic-attention) 100-iter DF run.
# The DF training samples uniformly from 5 attention configs at each step,
# one of which is exactly (sink=3, window=9). So the DF student already
# saw this mode during training (1/5 of the time) — no dedicated DF
# specialization needed for this attention config.
export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_syncnet_trained_shift_5_4gpu_bs16_lr1e5_100iter/checkpoints/0000100.pth}"

# Distinct output dir and wandb run name (_df100_sink3_window9 suffix).
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT_FSMATCHED_LR3E6_DF100_SINK3_WINDOW9}"
export RUN_NAME="${RUN_NAME:-sf_sink3_window9_redmd_audiofix_beta2_taew_syncnet_mouthweight_fsmatched_lr3e6_df100}"

# Attention sink 1 -> 3, local_attn_size 7 -> 9 (rolling window stays 6).
# These override the sink1_window7_tscfg base chain at train.py argv time.
export EXTRA_OVERRIDES="model.net.sink_size=3 model.net.local_attn_size=9"

# Delegate to the fsmatched parent (reward stays enabled, critic lr 3e-6).
exec "$(dirname "$(readlink -f "$0")")/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh" "$@"
