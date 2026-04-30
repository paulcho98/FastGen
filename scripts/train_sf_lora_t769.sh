#!/bin/bash
# =============================================================================
# Symmetric LoRA + selective-unfreeze 1.3B SF — Re-DMD beta=2 + TAEW + t769
# =============================================================================
#
# 1.3B counterpart of train_sf_..._t769_14b_lora.sh.  LoRA ablation arm
# of the t769 SF experiment — pairs with train_sf_full_ft_t769.sh
# (full-FT both nets) for direct apples-to-apples comparison of
# regime effects on Sync-C convergence.
#
# Both student and fake_score:
#   - merge_lora=False (PEFT layers stay)
#   - LoRA on q/k/v/o/ffn linears (rank=128, alpha=64)
#   - Full FT on audio_proj + audio_cond_projs + patch_embedding
#
# Trainable count: ~150M per network instead of ~1421M (~10x smaller).
# Optim state shrinks proportionally; saves are tiny per the trainable-
# only filter in checkpointer.save.
#
# Same as full_ft_t769:
#   - DF init from syncnet-trained 1.3B 5000-iter t769 ckpt
#   - Mouthweight 14B teacher
#   - syncnet-trained 1.3B mouthweight as fake_score V2V
#   - Matched 2e-6 LRs
#   - Effective batch 64
#   - max_iter 5000, save_every 100
#
# Usage:
#   bash scripts/train_sf_lora_t769.sh
#
#   # Smoke test:
#   EXTRA_OVERRIDES="trainer.max_iter=50 trainer.save_ckpt_iter=50" \
#     bash scripts/train_sf_lora_t769.sh
#
# Resume:
#   RESUME=True bash scripts/train_sf_lora_t769.sh
# =============================================================================
set -euo pipefail

export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_sf_lora_t769.py"
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT_LORA_T769}"
export RUN_NAME="${RUN_NAME:-sf_lora_t769}"

exec "$(dirname "$(readlink -f "$0")")/train_sf_parent.sh" "$@"
