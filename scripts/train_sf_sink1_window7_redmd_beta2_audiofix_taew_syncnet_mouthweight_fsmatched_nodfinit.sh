#!/bin/bash
# =============================================================================
# SF Re-DMD β=2 TAEW audiofix — fsmatched + lr3e6 — NO DF INIT variant
# =============================================================================
#
# Same as train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh
# (sync-C reward + mouthweight 14B teacher + fsmatched fake score + critic
# lr 3e-6) but WITHOUT the DF-trained checkpoint overwrite on the student.
#
# Student init chain when this wrapper is active:
#   (1) Wan 1.3B base weights     (from config.model.net.base_model_paths)
#   (2) Syncnet-trained V2V adapter (from OMNIAVATAR_STUDENT_CKPT
#                                    = ...maskall_refseq_mouth_weight_2gpu/step-1000.pt)
#   (3) <SKIPPED> — no DF ckpt overwrite
#
# So the student starts at the same place the DF run itself started —
# i.e. "student init before DF training". This isolates whether DF
# pre-training is actually helpful vs going straight into SF + sync-C reward
# from the raw V2V adapter.
#
# How the skip works: setting OMNIAVATAR_DF_CKPT="" (explicit empty) makes
# os.getenv("OMNIAVATAR_DF_CKPT", default) return "" (not the default) in
# config_sf_sink1_window7_tscfg.py. That sets
# trainer.checkpointer.pretrained_ckpt_path="". trainer.py:82 then skips
# load_pretrained_ckpt because the truthy check is False on "".
#
# Everything else is identical to the parent:
# - teacher: mouthweight 14B step-6000
# - fake_score init: syncnet-trained 1.3B adapter (fsmatched)
# - critic lr: 3e-6
# - sync-C reward: ENABLED (β=2, SyncNet-v2 ckpt, TAEW decoder)
# - timestep-conditional CFG, sliding-window attention, etc.
#
# Usage:
#   bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_nodfinit.sh
#
# Resume (same wandb run via persisted wandb_id.txt):
#   RESUME=True bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_nodfinit.sh
# =============================================================================

set -euo pipefail

# Skip the DF-ckpt overwrite: empty env -> empty pretrained_ckpt_path ->
# trainer.py skips load_pretrained_ckpt.
export OMNIAVATAR_DF_CKPT=""

# Distinct output dir and wandb run name.
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT_FSMATCHED_LR3E6_NODFINIT}"
export RUN_NAME="${RUN_NAME:-sf_sink1_window7_redmd_audiofix_beta2_taew_syncnet_mouthweight_fsmatched_lr3e6_nodfinit}"

# Delegate to the fsmatched parent (reward stays ENABLED by default).
exec "$(dirname "$(readlink -f "$0")")/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh" "$@"
