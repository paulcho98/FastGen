#!/bin/bash
# =============================================================================
# OmniAvatar DF (shift=5) — short recovery run for early-stage checkpoints
# =============================================================================
#
# Sibling of train_omniavatar_df_shift_5_audiofix_syncnet_trained_500iter.sh.
# Produces TWO checkpoints — step 50 and step 100 — by setting MAX_ITER=100
# and SAVE_EVERY=50. Useful for SF ablations where the student is initialized
# from a *very* lightly DF-trained adapter (between "no DF" and "DF=500").
#
# Output dir (auto-derived from RUN_NAME via the parent's templating):
#   FASTGEN_OUTPUT/.../df_audiofix_syncnet_trained_shift_5_4gpu_bs16_lr1e5_100iter/
# Fresh wandb run (new wandb_id.txt) — no collision with the 500iter or
# 5000iter runs.
#
# Walltime: ~100 iters * 20 s/iter on 4x H200 + 2 save spikes ~ 40-45 min.
#
# Usage:
#   bash scripts/train_omniavatar_df_shift_5_audiofix_syncnet_trained_100iter.sh
#
# Resume after a crash:
#   RESUME=True bash scripts/train_omniavatar_df_shift_5_audiofix_syncnet_trained_100iter.sh
#
# GPU conflict caveat: same as the 500iter wrapper — needs 4 H200s, currently
# consumed by the active SF training. Wait, kill SF, or override NGPU.
# =============================================================================

set -euo pipefail

# Delegate to the parent script with MAX_ITER=100 and SAVE_EVERY=50 so we
# get checkpoints at step 50 and step 100. VIZ_EVERY mirrors SAVE_EVERY so
# wandb visualizations land at each save. All other env vars (NGPU,
# BATCH_SIZE, OMNIAVATAR_STUDENT_CKPT, RESUME) pass through.
MAX_ITER=100 SAVE_EVERY=50 VIZ_EVERY=50 \
  exec "$(dirname "$(readlink -f "$0")")/train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh" "$@"
