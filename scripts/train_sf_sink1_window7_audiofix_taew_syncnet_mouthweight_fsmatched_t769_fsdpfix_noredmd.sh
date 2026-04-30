#!/bin/bash
# =============================================================================
# SF fsmatched_t769_fsdpfix — REDMD-OFF ablation
# =============================================================================
#
# Sibling of train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_t769_fsdpfix.sh
# with sync-C reward weighting DISABLED.  Everything else is identical:
#
#   * Same DF init: syncnet-trained 5000-iter DF ckpt
#   * Same teacher: mouthweight 14B step-6000
#   * Same fake_score init: syncnet-trained 1.3B mouthweight adapter
#   * Same critic LR: 3e-6 (vs student 2e-6)
#   * Same windowed CFG (sliding window: sink_size=1, local_attn_size=7,
#     use_dynamic_rope=True)
#   * Same 2-step distillation: t_list=[0.999, 0.769, 0.0],
#     student_sample_steps=2, timestep_cfg.enabled=True
#   * Same FSDP per-submodule wrap fix (network_causal.py:2396-2407)
#   * Same effective batch: BS=8 * GA=2 * 4 GPUs = 64
#   * Same TAEW reward decoder *path* (config.model.reward.decoder_kind="taew")
#     — but reward.enabled=False so the decoder is constructed but never
#     actually called.  We keep the config field set so this script differs
#     from the redmd-on baseline by ONLY the enabled flag, making the
#     ablation crisp.
#
# What's turned OFF:
#   * config.model.reward.enabled = False
#       => self.reward_scorer = None (omniavatar_self_forcing_re_dmd.py:68)
#       => reward_active = False at every step
#       => loss path falls through to plain DMD2:
#          weighted_loss = vsd_loss + 0.003 * gan_loss_gen   (no exp(beta*r))
#       => per-sample VSD reduction collapses to mean (no per-sample weights)
#   * config.model.reward_beta = 0
#       => belt-and-suspenders: even if reward.enabled flipped True somehow,
#          exp(0 * r) = 1 produces unit weights (no effect).  Keeps the
#          ablation robust to env/CLI override surprises.
#
# What stays ON (note for ablation analysis):
#   * Model class is still OmniAvatarSelfForcingReDMD (wraps the dmd2 method
#     class).  With reward off the override is a no-op except in init paths.
#   * TAEW decoder is loaded but unused (no decode happens with reward off).
#     Fine — the load is one-time cost; saves you a config fork.
#
# Distinct RUN_NAME and FASTGEN_OUTPUT_ROOT so wandb / disk don't collide
# with the redmd-on baseline.
#
# Usage:
#   bash scripts/train_sf_sink1_window7_audiofix_taew_syncnet_mouthweight_fsmatched_t769_fsdpfix_noredmd.sh
#
#   # Smoke (50 iters, validates init + first save):
#   MAX_ITER=50 SAVE_EVERY=50 \
#     bash scripts/...fsmatched_t769_fsdpfix_noredmd.sh
#
# Resume:
#   RESUME=True bash scripts/...fsmatched_t769_fsdpfix_noredmd.sh
# =============================================================================

set -euo pipefail

# Same DF init as the redmd-on baseline — preserves apples-to-apples.
export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_syncnet_trained_shift_5_t769_4gpu_bs16_lr1e5_5000iter/checkpoints/0005000.pth}"

# Distinct output dir + RUN_NAME (`_noredmd` suffix).
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/tmp/FASTGEN_SF_OUTPUT_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT_FSMATCHED_T769_FSDPFIX_NOREDMD}"
export RUN_NAME="${RUN_NAME:-sf_sink1_window7_audiofix_taew_syncnet_mouthweight_fsmatched_t769_fsdpfix_noredmd}"

# EXTRA_OVERRIDES: same t_list as fsdpfix (windowed t769 schedule), plus the
# two reward-disable flags.  We keep model.sample_t_cfg.t_list explicit even
# though the parent inherits it from config_sf_sink1_window7_tscfg.py — the
# fsdpfix sibling sets it explicitly via override and we do the same for
# parity.
export EXTRA_OVERRIDES="model.sample_t_cfg.t_list=[0.999,0.769,0.0] model.reward.enabled=False model.reward_beta=0 trainer.max_iter=1000"

# Delegate to the fsmatched parent.
exec "$(dirname "$(readlink -f "$0")")/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh" "$@"
