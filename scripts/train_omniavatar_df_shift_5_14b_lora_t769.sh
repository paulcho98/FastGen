#!/bin/bash
# =============================================================================
# OmniAvatar DF (shift=5) — 14B causal student, LoRA + unfreeze + t769 schedule
# =============================================================================
#
# Same training stack as train_omniavatar_df_shift_5_14b_lora.sh (LoRA on
# transformer blocks + selective full fine-tune on audio path + patch
# embedding, mouthweight 14B step-6000 init, FSDP, bf16/fp32 mixed
# precision) but with the t769 2-step schedule:
#
#   sample_t_cfg.t_list:   [0.999, 0.937, 0.833, 0.624, 0.0]
#                       -> [0.999, 0.769,        0.0]
#   student_sample_steps:                4 -> 2
#
# Effect: trains the student only at noise levels SF t769 inference uses,
# combined with the LoRA + selective-unfreeze regime that targets the
# audio path with full capacity while constraining the bulk of the
# network to a low-rank update.
#
# Why this combination: the t769 1.3B SF runs (fsmatched_t769) showed
# the largest sync gap to the teacher specifically at the t=0.769 noise
# level, where the student's audio adaptation matters most.  This config
# targets that exact training distribution with a regime that prioritizes
# audio-path adaptation.
#
# Disk: optim state is tiny (<1 GB per save) since most params are
# frozen.  No need for the strip-watcher.  Save is dominated by the full
# 14B model shards (~54 GB per save, vs ~161 GB for full-FT).
#
# Walltime: probably faster per iter than full-FT due to smaller optim
# step + smaller all-gather buffers.  Measure on the smoke run.
#
# Usage:
#   bash scripts/train_omniavatar_df_shift_5_14b_lora_t769.sh
#
#   # Smoke test (50 iters, validates wrap+forward+backward+save format):
#   MAX_ITER=50 SAVE_EVERY=50 \
#     bash scripts/train_omniavatar_df_shift_5_14b_lora_t769.sh
# =============================================================================

set -euo pipefail

# Use the LoRA + t769 specialized config.
export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_df_shift_5_14b_lora_t769.py"

# Per-GPU batch + grad accum: effective batch 16 like the full-FT 14B
# wrapper, but a different per-GPU/accum split (matches the LoRA
# wrapper's defaults).  Full-FT uses BATCH_SIZE=1 GRAD_ACCUM=4 because
# Adam state on 14B fp32 dominates per-GPU memory; LoRA's optim state is
# tiny so we have headroom for BATCH_SIZE=2 GRAD_ACCUM=2.  After the
# first launch confirms memory headroom, BATCH_SIZE=4 GRAD_ACCUM=1 may
# be feasible.
export BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

export SAVE_EVERY="${SAVE_EVERY:-500}"
export VIZ_EVERY="${VIZ_EVERY:-${SAVE_EVERY}}"

NGPU="${NGPU:-4}"
MAX_ITER="${MAX_ITER:-3000}"
EFFECTIVE_BATCH=$((BATCH_SIZE * NGPU * GRAD_ACCUM))
export RUN_NAME="${RUN_NAME:-df_audiofix_syncnet_trained_shift_5_14b_lora_t769_${NGPU}gpu_bs${BATCH_SIZE}_grad${GRAD_ACCUM}_eff${EFFECTIVE_BATCH}_lr1e5_${MAX_ITER}iter}"

# DDP -> FSDP and grad_accum override (config sets grad_accum_rounds=4
# inherited from the parent; we override on cmdline to match GRAD_ACCUM
# env).  Identical pattern to train_omniavatar_df_shift_5_14b_lora.sh.
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-trainer.ddp=False trainer.fsdp=True trainer.grad_accum_rounds=${GRAD_ACCUM}}"

echo "============================================="
echo "  14B DF FSDP launch settings — LoRA + unfreeze + t769"
echo "============================================="
echo "  Per-GPU batch:    ${BATCH_SIZE}"
echo "  GPUs:             ${NGPU}"
echo "  Grad accum:       ${GRAD_ACCUM}"
echo "  Effective batch:  ${EFFECTIVE_BATCH}  (= ${BATCH_SIZE} x ${NGPU} x ${GRAD_ACCUM})"
echo "  Max iter:         ${MAX_ITER}"
echo "  Save every:       ${SAVE_EVERY}"
echo "  Run name:         ${RUN_NAME}"
echo "  Config:           ${CONFIG_PATH}"
echo "  Schedule:         t_list=[0.999, 0.769, 0.0], student_sample_steps=2"
echo "  EXTRA_OVERRIDES:  ${EXTRA_OVERRIDES}"
echo "============================================="

# Delegate to the existing parent (passes NGPU/MAX_ITER/SAVE_EVERY/RESUME
# through env).
exec "$(dirname "$(readlink -f "$0")")/train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh" "$@"
