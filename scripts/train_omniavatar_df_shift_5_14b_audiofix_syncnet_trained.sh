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
# - SAVE_EVERY=2500 default — only 2 saves over a 5000-iter run, since each
#   FSDP DF save is ~85 GB and /home/work has 279 GB free. Bump if you've
#   freed disk; drop to 1000 only if you've worked out a strip-on-save.
# - EXTRA_OVERRIDES sets trainer.ddp=False trainer.fsdp=True so the parent's
#   hardcoded `trainer.ddp=True` line is overridden by the later cmdline arg.
#
# Walltime: ~6-10 days for 5000 iters on 4x H200 (FSDP all-gather adds
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
# Override effective batch (if you need to drop to 8 due to OOM, or push
# to 32 if H200s have headroom):
#   EXTRA_OVERRIDES="trainer.ddp=False trainer.fsdp=True trainer.grad_accum_rounds=2 model.grad_accum_rounds=2" \
#     bash scripts/...14b_audiofix_syncnet_trained.sh
# =============================================================================

set -euo pipefail

# Use the 14B-specialized config.
export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_df_shift_5_14b.py"

# Per-GPU batch 1 (vs 16 for 1.3B). With NGPU=4 and grad_accum_rounds=4 in
# config, effective batch = 1*4*4 = 16.
export BATCH_SIZE="${BATCH_SIZE:-1}"

# Save cadence: every 2500 iters (so 2 saves over a 5000-iter run, fits
# /home/work's 279 GB free with margin). Override if you've freed disk
# or want to checkpoint more aggressively.
export SAVE_EVERY="${SAVE_EVERY:-2500}"

# Mirror SAVE_EVERY for visualization / validation.
export VIZ_EVERY="${VIZ_EVERY:-${SAVE_EVERY}}"

# Distinct RUN_NAME so the output dir does NOT collide with the 1.3B runs.
NGPU="${NGPU:-4}"
MAX_ITER="${MAX_ITER:-5000}"
export RUN_NAME="${RUN_NAME:-df_audiofix_syncnet_trained_shift_5_14b_${NGPU}gpu_bs${BATCH_SIZE}_grad4_lr1e5_${MAX_ITER}iter}"

# Flip DDP -> FSDP. Order matters: parent's torchrun cmdline has
# `trainer.ddp=True` baked in; this EXTRA_OVERRIDES is appended AFTER it,
# so the later trainer.ddp=False wins.
# (trainer.fsdp=True is also set by config_df_shift_5_14b.py, but include
# it here too as belt-and-suspenders against any cmdline override that
# might re-set it.)
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-trainer.ddp=False trainer.fsdp=True}"

# Delegate to the parent (passes NGPU/MAX_ITER/SAVE_EVERY/RESUME through env).
exec "$(dirname "$(readlink -f "$0")")/train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh" "$@"
