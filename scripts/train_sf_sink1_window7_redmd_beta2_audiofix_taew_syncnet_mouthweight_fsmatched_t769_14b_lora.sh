#!/bin/bash
# =============================================================================
# SF Re-DMD beta=2 + TAEW — 14B causal student + 14B fake_score, LoRA + unfreeze, t769
# =============================================================================
#
# Near-identical configuration to fsmatched_t769_fsdpfix.sh except for:
#   - Student + fake_score: 1.3B -> 14B with LoRA + selective unfreeze
#     (via CONFIG_PATH=config_sf_14b_lora_t769.py)
#   - Effective batch: 64 -> 16 (BS=1 GA=4 instead of BS=8 GA=2)
#   - FSDP enabled (necessary for 14B; same EXTRA_OVERRIDES pattern as
#     train_omniavatar_df_shift_5_14b_lora_t769.sh)
#   - Mouthweight 14B step-6000 init for both student and fake_score's V2V
#     LoRA values (the SF base config's STUDENT_CKPT defaults to 1.3B, so
#     we explicitly point OMNIAVATAR_STUDENT_CKPT_14B at the 14B mouthweight
#     ckpt — read by config_sf_14b_lora_t769.py)
#
# DF init: the 14B DF LoRA t769 run's final checkpoint (5000 iters), which
# itself uses the same regime (LoRA + selective unfreeze + t769 schedule).
# Default path baked in below assumes the run completes successfully; can
# be overridden via OMNIAVATAR_DF_CKPT.
#
# Memory budget: with three 14B networks FSDP-sharded (student, fake_score,
# teacher) plus LoRA optim (~1.2 GB per network) plus 2-step student rollout
# activations + teacher forward + fake_score forward (×2), BS=1 is the safe
# starting point.  Smoke first to verify before committing to a long run.
#
# Distinct RUN_NAME and FASTGEN_OUTPUT_ROOT (`_14b_lora` suffix).
#
# Usage:
#   bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_t769_14b_lora.sh
#
#   # Smoke test (50 iters, validates wrap+forward+backward+save format):
#   MAX_ITER=50 SAVE_EVERY=50 \
#     bash scripts/...fsmatched_t769_14b_lora.sh
#
# Resume:
#   RESUME=True bash scripts/...fsmatched_t769_14b_lora.sh
# =============================================================================

set -euo pipefail

# Use the 14B-LoRA-SF specialized config.
export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_sf_14b_lora_t769.py"

# DF init: trained 14B DF LoRA t769 ckpt.  Default assumes the running
# 14B DF LoRA t769 run (started 2026-04-28 19:22 KST) reaches iter 5000.
# Override via OMNIAVATAR_DF_CKPT to point at an earlier save (e.g. 4500)
# if you want to launch SF before DF reaches 5000.
# Note: the route uses ${VAR-default} (no colon) so an explicit-empty
# OMNIAVATAR_DF_CKPT="" is preserved (means "skip DF init"), only an
# unset env uses the default.
export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_syncnet_trained_shift_5_14b_lora_t769_4gpu_bs4_grad1_eff16_lr1e5_5000iter/checkpoints/0005000.pth}"

# Mouthweight 14B step-6000 — used by config_sf_14b_lora_t769.py to
# provide initial LoRA + audio + patch values for both student and
# fake_score (before the DF ckpt overwrites the student via the
# checkpointer).  Same ckpt as the SF teacher.
export OMNIAVATAR_STUDENT_CKPT_14B="${OMNIAVATAR_STUDENT_CKPT_14B:-/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/step-6000.pt}"

# Distinct output dir + RUN_NAME (`_14b_lora` infix).
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT_FSMATCHED_T769_14B_LORA}"
export RUN_NAME="${RUN_NAME:-sf_sink1_window7_redmd_audiofix_beta2_taew_syncnet_mouthweight_fsmatched_t769_14b_lora}"

# EXTRA_OVERRIDES: same DDP -> FSDP flip as the 14B DF wrapper, plus the
# t_list override (matches fsmatched_t769_fsdpfix.sh).  The config already
# sets fsdp=True and grad_accum_rounds=4, but we mirror the override style
# from the DF wrapper for parallelism / belt-and-suspenders against any
# parent cmdline that re-sets these.
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-trainer.ddp=False trainer.fsdp=True trainer.grad_accum_rounds=4 model.sample_t_cfg.t_list=[0.999,0.769,0.0]}"

echo "============================================="
echo "  SF 14B LoRA + unfreeze + t769 launch settings"
echo "============================================="
echo "  Config:          ${CONFIG_PATH}"
echo "  DF init ckpt:    ${OMNIAVATAR_DF_CKPT}"
echo "  Student/FS V2V:  ${OMNIAVATAR_STUDENT_CKPT_14B}"
echo "  Run name:        ${RUN_NAME}"
echo "  Output root:     ${FASTGEN_OUTPUT_ROOT}"
echo "  EXTRA_OVERRIDES: ${EXTRA_OVERRIDES}"
echo "  Effective batch: 16  (BS=1 x 4 GPUs x GA=4)"
echo "  Schedule:        t_list=[0.999, 0.769, 0.0], student_sample_steps=2"
echo "============================================="

# Delegate to the fsmatched parent (which will see CONFIG_PATH override).
exec "$(dirname "$(readlink -f "$0")")/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh" "$@"
