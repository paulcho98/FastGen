#!/bin/bash
# =============================================================================
# Re-DMD beta=2 + TAEW decoder + GAN (adversarial loss) — Audio-Fix Re-Run
# =============================================================================
# Mirror of train_sf_sink1_window7_redmd_beta2_audiofix_taew.sh with GAN
# enabled in the critic step. Inherits the canonical Wan 14B SF GAN recipe
# (gan_loss_weight_gen=0.003, Discriminator_Wan_14B, feature_indices=[21,30,39]).
#
# This is only safe to run on top of the Z(c) self-normalization landed at
# 7f96e6f. Under the prior `mean(w*L)` coupling, at beta=2 with typical
# sync_c ~3.9, weighted VSD had magnitude ~O(1000), which would have
# clobbered the 0.003-weighted GAN term by five orders of magnitude.
# Self-normalization bounds weighted VSD in [min(L), max(L)], restoring
# the GAN term's intended relative strength.
#
# Memory budget: batch 8 -> 4, grad_accum 2 -> 4 (effective batch stays 64).
# Expected VRAM: ~119 GB/H200. If OOM, drop batch further (2 + accum 8).
#
# Prereqs:
#   - /home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth
#   - DF shift=5 audiofix checkpoint (same default as non-GAN variant).
#
# Usage (inside tmux):
#   bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan.sh \
#     2>&1 | tee /tmp/train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan.log
#
# Resume after a crash:
#   RESUME=True bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan.sh \
#     2>&1 | tee -a /tmp/train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan.log
# =============================================================================
set -euo pipefail

RESUME="${RESUME:-False}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Separate output root so this run doesn't clobber the no-GAN TAEW beta=2 run.
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_GAN"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

# Same DF init as the no-GAN TAEW variant.
export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_shift_5_4gpu_bs16_lr1e5_10000iter/checkpoints/0010000.pth}"

if [[ ! -f "${OMNIAVATAR_DF_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_DF_CKPT does not exist: ${OMNIAVATAR_DF_CKPT}" >&2
    exit 1
fi

RUN_NAME="sf_sink1_window7_redmd_audiofix_beta2_taew_gan"

echo "============================================="
echo "  Re-DMD beta=2 + TAEW + GAN"
echo "============================================="
echo "  DF init ckpt:       ${OMNIAVATAR_DF_CKPT}"
echo "  TAEW ckpt:          /home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth"
echo "  gan_loss_weight:    0.003"
echo "  Discriminator:      Wan_14B (40 blocks, feature_indices=[21,30,39])"
echo "  Disc optimizer LR:  5e-6"
echo "  Dataloader batch:   4 (halved from 8 for GAN VRAM budget)"
echo "  Grad accum rounds:  4 (effective batch 4 * 4 GPUs * 4 accum = 64)"
echo "  Run name:           ${RUN_NAME}"
echo "  Output root:        ${FASTGEN_OUTPUT_ROOT}"
echo "  Resume:             ${RESUME}"
echo "============================================="
echo ""

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_beta2_taew_gan.py \
    - trainer.resume=${RESUME} \
    log_config.group="omniavatar_sf_audiofix" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho"
