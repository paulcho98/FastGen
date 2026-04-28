#!/bin/bash
# =============================================================================
# OmniAvatar DF (shift=5) — 14B causal student, LoRA blocks + selective unfreeze
# =============================================================================
#
# Same training stack as train_omniavatar_df_shift_5_14b_audiofix_syncnet_trained.sh
# but uses config_df_shift_5_14b_lora.py instead of config_df_shift_5_14b.py.
# That config sets merge_lora=False and unfreeze_modules=["_core.audio_proj",
# "_core.audio_cond_projs", "_core.patch_embedding"], so the run trains
# LoRA A/B on the transformer blocks plus full fine-tunes of the audio path
# and patch embedding.
#
# Why this regime: see docs/lora_selective_unfreeze.md.  Brief: the
# lip-sync gap we observe in SF runs appears to depend strongly on
# audio-path adaptation; a hybrid LoRA-on-blocks + full-FT-on-audio-path
# tests whether targeting the bottleneck with full capacity (while
# constraining the bulk of the network to a low-rank update) closes
# the gap.
#
# Disk: optim state is tiny (<1 GB per save) since most params are
# frozen.  No need for the strip-watcher.
#
# Walltime: probably faster per iter than full-FT due to smaller optim
# step + smaller all-gather buffers.  Measure on the smoke run.
#
# Usage:
#   bash scripts/train_omniavatar_df_shift_5_14b_lora.sh
#
#   # Smoke test (50 iters, validates wrap+forward+backward+save format):
#   MAX_ITER=50 SAVE_EVERY=50 \
#     bash scripts/train_omniavatar_df_shift_5_14b_lora.sh
# =============================================================================

set -euo pipefail

# Use the LoRA-specialized config.
export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_df_shift_5_14b_lora.py"

# Per-GPU batch + grad accum: same as the full-FT 14B wrapper by default.
# The user can override either via env.  With smaller optim state, we
# may have headroom for BATCH_SIZE=4 or higher in a follow-up run, but
# keep it conservative for the first launch.
export BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

export SAVE_EVERY="${SAVE_EVERY:-500}"
export VIZ_EVERY="${VIZ_EVERY:-${SAVE_EVERY}}"

NGPU="${NGPU:-4}"
MAX_ITER="${MAX_ITER:-3000}"
EFFECTIVE_BATCH=$((BATCH_SIZE * NGPU * GRAD_ACCUM))
export RUN_NAME="${RUN_NAME:-df_audiofix_syncnet_trained_shift_5_14b_lora_${NGPU}gpu_bs${BATCH_SIZE}_grad${GRAD_ACCUM}_eff${EFFECTIVE_BATCH}_lr1e5_${MAX_ITER}iter}"

# DDP -> FSDP and grad_accum override (config sets grad_accum_rounds=4
# inherited from the parent; we override on cmdline to match GRAD_ACCUM
# env).  Identical pattern to train_omniavatar_df_shift_5_14b_audiofix_syncnet_trained.sh.
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-trainer.ddp=False trainer.fsdp=True trainer.grad_accum_rounds=${GRAD_ACCUM}}"

echo "============================================="
echo "  14B DF FSDP launch settings — LoRA + unfreeze"
echo "============================================="
echo "  Per-GPU batch:    ${BATCH_SIZE}"
echo "  GPUs:             ${NGPU}"
echo "  Grad accum:       ${GRAD_ACCUM}"
echo "  Effective batch:  ${EFFECTIVE_BATCH}  (= ${BATCH_SIZE} x ${NGPU} x ${GRAD_ACCUM})"
echo "  Max iter:         ${MAX_ITER}"
echo "  Save every:       ${SAVE_EVERY}"
echo "  Run name:         ${RUN_NAME}"
echo "  Config:           ${CONFIG_PATH}"
echo "  EXTRA_OVERRIDES:  ${EXTRA_OVERRIDES}"
echo "============================================="

# Delegate to the existing parent (passes NGPU/MAX_ITER/SAVE_EVERY/RESUME
# through env).
exec "$(dirname "$(readlink -f "$0")")/train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh" "$@"
