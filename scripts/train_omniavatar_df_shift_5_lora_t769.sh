#!/bin/bash
# =============================================================================
# OmniAvatar DF (shift=5) — 1.3B causal student, LoRA + unfreeze + t769
# =============================================================================
#
# 1.3B counterpart of train_omniavatar_df_shift_5_14b_lora_t769.sh.  Trains
# the 1.3B causal student as LoRA on transformer blocks + full-FT on
# audio_proj + audio_cond_projs + patch_embedding, on the t769 2-step
# schedule.
#
# Pairs with: scripts/train_sf_lora_t769.sh — produces a 1.3B DF init
# in the SAME LoRA + selective-unfreeze regime that the SF LoRA ablation
# expects.  Avoids the train/test regime mismatch you'd get from
# initializing the SF LoRA run with a full-FT DF ckpt.
#
# Effective batch: same as the existing 1.3B DF runs (BATCH_SIZE=16 *
# NGPU=4 = 64).  Optim state is tiny (~150M trainable params * 8 bytes
# Adam state = ~1.2 GB total) so per-GPU memory has plenty of headroom
# vs full-FT.
#
# Why the matching DF init matters:
#     The SF student loads the DF ckpt's state.  If DF was trained
#     full-FT (all 1.3B params evolving) and SF is then constrained to
#     LoRA-only (only ~150M evolving), the SF run starts from a state
#     where the audio path / blocks have all drifted together — the LoRA
#     freeze then locks blocks at the DF endpoint while leaving the
#     audio path free, which is fine but not regime-pure.  A LoRA DF
#     init keeps the regime consistent across stages.
#
# Output dir:
#   /home/work/.local/.../FASTGEN_OUTPUT/.../df_..._lora_t769_*/
#
# Usage:
#   bash scripts/train_omniavatar_df_shift_5_lora_t769.sh
#
# Smoke:
#   MAX_ITER=100 SAVE_EVERY=50 \
#     bash scripts/train_omniavatar_df_shift_5_lora_t769.sh
#
# Resume:
#   RESUME=True bash scripts/train_omniavatar_df_shift_5_lora_t769.sh
# =============================================================================

set -euo pipefail

# Use the LoRA + t769 specialized config.
export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_df_shift_5_lora_t769.py"

# Distinct RUN_NAME with _lora_ infix so output dir does NOT collide
# with the full-FT t769 DF run.
NGPU="${NGPU:-4}"
MAX_ITER="${MAX_ITER:-5000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
export RUN_NAME="${RUN_NAME:-df_audiofix_syncnet_trained_shift_5_lora_t769_${NGPU}gpu_bs${BATCH_SIZE}_lr1e5_${MAX_ITER}iter}"

# Delegate to the existing 1.3B DF parent (handles env vars, torchrun
# launch, output dir).  Same parent the t769 full-FT DF wrapper uses;
# the only difference is CONFIG_PATH (and resulting wandb name).
exec "$(dirname "$(readlink -f "$0")")/train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh" "$@"
