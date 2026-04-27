#!/bin/bash
# =============================================================================
# OmniAvatar DF (shift=5) — 14B causal student, FSDP, mouthweight init
# =============================================================================
#
# Mirror of the existing 1.3B DF wrapper but trains the 14B causal student
# variant, using:
# - config_df_shift_5_14b.py     (model_size=14B, FSDP, bf16/fp32 mixed prec,
#                                  Wan 2.1 T2V 14B base + mouthweight 14B
#                                  step-6000 V2V adapter as init)
# - per-GPU BATCH_SIZE=1 (vs 16 for 1.3B)
# - grad_accum_rounds=4 in the config -> effective batch = 1*4*4 = 16
#   (override grad_accum_rounds via EXTRA_OVERRIDES if you want 8 or 32)
# - MAX_ITER=3000 default (vs 5000 for 1.3B): disk math at SAVE_EVERY=500
#   only fits 6 saves with on-save optim strip (5 model-only + 1 full ~=
#   448 GB, marginal in ~409 GB available). At 5000 iters it doesn't fit
#   even with strip; revisit once external storage is set up.
# - SAVE_EVERY=500 default. 6 saves at 500-iter cadence; pair this script
#   with strip_optim_watcher.sh in a separate shell/tmux pane to keep disk
#   usage bounded during the run (always exactly ONE step retains optim).
# - EXTRA_OVERRIDES sets trainer.ddp=False trainer.fsdp=True so the parent's
#   hardcoded `trainer.ddp=True` line is overridden by the later cmdline arg.
#
# Walltime: ~4-6 days for 3000 iters on 4x H200 (FSDP all-gather adds
# meaningful overhead vs. the 1.3B DDP run's ~32 h). Smoke first with e.g.
# MAX_ITER=200 SAVE_EVERY=100 to verify it trains stably before committing.
#
# Usage:
#   nohup bash scripts/train_omniavatar_df_shift_5_14b_audiofix_syncnet_trained.sh \
#     > /tmp/train_df_14b_5000iter.log 2>&1 &
#
# Smoke-test (200 iters, ~6-8 hours):
#   MAX_ITER=200 SAVE_EVERY=100 \
#     bash scripts/train_omniavatar_df_shift_5_14b_audiofix_syncnet_trained.sh
#
# Resume:
#   RESUME=True bash scripts/train_omniavatar_df_shift_5_14b_audiofix_syncnet_trained.sh
#
# Override effective batch via env vars (cleaner than hand-crafting
# EXTRA_OVERRIDES — the wrapper builds the override string from
# BATCH_SIZE and GRAD_ACCUM):
#
#   # Default: per-GPU 1, grad_accum 4 -> effective 16, ~70-90 GB peak/GPU
#   bash scripts/train_omniavatar_df_shift_5_14b_audiofix_syncnet_trained.sh
#
#   # Faster (10-20% wall-clock) but tighter memory (~85-120 GB peak/GPU,
#   # OOM risk): per-GPU 2, grad_accum 2 -> effective 16
#   BATCH_SIZE=2 GRAD_ACCUM=2 bash scripts/...14b_audiofix_syncnet_trained.sh
#
#   # Effective batch 8 (drop if first run OOMs even at default): per-GPU
#   # 1, grad_accum 2
#   GRAD_ACCUM=2 bash scripts/...14b_audiofix_syncnet_trained.sh
#
# Recommendation: start at the default (1 x 4) for the first FSDP run on
# the causal class at 14B scale.  After iter 50, peak memory is realized;
# read it from `nvidia-smi`, and if there's >40 GB headroom on every rank,
# a follow-up run at BATCH_SIZE=2 GRAD_ACCUM=2 is safe and ~10-20% faster.
# =============================================================================

set -euo pipefail

# Use the 14B-specialized config.
export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_df_shift_5_14b.py"

# Per-GPU batch 1 (vs 16 for 1.3B). With NGPU=4 and GRAD_ACCUM=4,
# effective batch = 1*4*4 = 16.  Override via BATCH_SIZE env.
export BATCH_SIZE="${BATCH_SIZE:-1}"

# Gradient accumulation rounds.  Default 4 (matches config_df_shift_5_14b.py
# default).  Override here so the wrapper can also reach down into both
# trainer.grad_accum_rounds and model.grad_accum_rounds via cmdline (the
# config sets both; we mirror that on the override path).
GRAD_ACCUM="${GRAD_ACCUM:-4}"

# Save cadence: every 500 iters. Pairs with strip_optim_watcher.sh — that
# helper strips _optim/ shards from non-latest saves once a strictly-greater
# save lands, keeping disk usage bounded throughout the run. Without the
# watcher, 6 raw saves at 500 cadence over MAX_ITER=3000 would be ~1 TB
# (won't fit in NFS); WITH the watcher, peak usage stays ~280-450 GB.
export SAVE_EVERY="${SAVE_EVERY:-500}"

# Mirror SAVE_EVERY for visualization / validation.
export VIZ_EVERY="${VIZ_EVERY:-${SAVE_EVERY}}"

# Distinct RUN_NAME so the output dir does NOT collide with the 1.3B runs.
# Includes BATCH_SIZE and GRAD_ACCUM in the name so different effective-batch
# variants land in distinct dirs.  MAX_ITER=3000 default for 14B (vs 5000
# for 1.3B) — see header for disk math; bump back to 5000 only after
# enabling external storage or implementing bf16-state save format.
NGPU="${NGPU:-4}"
MAX_ITER="${MAX_ITER:-3000}"
EFFECTIVE_BATCH=$((BATCH_SIZE * NGPU * GRAD_ACCUM))
export RUN_NAME="${RUN_NAME:-df_audiofix_syncnet_trained_shift_5_14b_${NGPU}gpu_bs${BATCH_SIZE}_grad${GRAD_ACCUM}_eff${EFFECTIVE_BATCH}_lr1e5_${MAX_ITER}iter}"

# Build EXTRA_OVERRIDES from the env-toggled GRAD_ACCUM.  Appended last on
# the parent's torchrun cmdline so it wins on conflict.  Includes:
#   - DDP -> FSDP flip (parent hardcodes trainer.ddp=True earlier)
#   - grad_accum on both `trainer` and `model` (the config sets both;
#     we override both on the cmdline)
# If you set EXTRA_OVERRIDES in env, your value wins entirely (this default
# is skipped).
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-trainer.ddp=False trainer.fsdp=True trainer.grad_accum_rounds=${GRAD_ACCUM} model.grad_accum_rounds=${GRAD_ACCUM}}"

echo "============================================="
echo "  14B DF FSDP launch settings"
echo "============================================="
echo "  Per-GPU batch:    ${BATCH_SIZE}"
echo "  GPUs:             ${NGPU}"
echo "  Grad accum:       ${GRAD_ACCUM}"
echo "  Effective batch:  ${EFFECTIVE_BATCH}  (= ${BATCH_SIZE} x ${NGPU} x ${GRAD_ACCUM})"
echo "  Max iter:         ${MAX_ITER}"
echo "  Save every:       ${SAVE_EVERY}"
echo "  Run name:         ${RUN_NAME}"
echo "  EXTRA_OVERRIDES:  ${EXTRA_OVERRIDES}"
echo "============================================="

# Delegate to the parent (passes NGPU/MAX_ITER/SAVE_EVERY/RESUME through env).
exec "$(dirname "$(readlink -f "$0")")/train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh" "$@"
