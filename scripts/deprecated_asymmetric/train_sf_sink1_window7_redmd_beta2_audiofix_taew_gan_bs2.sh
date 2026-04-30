#!/bin/bash
# =============================================================================
# DEPRECATED — ASYMMETRIC TRAINABLE CAPACITY (do not launch new runs)
# =============================================================================
# Student trains as full-FT (~1421M); fake_score trains as LoRA-only (~175M).
# 8x critic-capacity asymmetry — see scripts/deprecated_asymmetric/README.md
# for full diagnosis.  Replacement scripts: train_sf_full_ft_t769.sh and
# train_sf_full_ft_t769_no_reward.sh in scripts/.  This file is kept only
# for reproducibility of past runs.
# =============================================================================
# =============================================================================
# Re-DMD beta=2 + TAEW + GAN @ batch=2 (post-bs=4-OOM on student step)
# =============================================================================
# bs=8 OOMed on critic step's disc forward (1.64 GiB short).
# bs=4 passed critic steps but OOMed on student step's teacher-with-grad
#   forward (3.28 GiB allocation; 1 GiB free).
# bs=2 halves again — predicted comfortable fit with ~17 GiB headroom.
#
# Effective batch preserved: 2 * 4 GPUs * 8 accum = 64 (same as no-GAN
# baseline and the bs=8/bs=4 variants).
#
# Usage:
#   bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan_bs2.sh \
#     2>&1 | tee /tmp/train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan_bs2.log
#
# Resume:
#   RESUME=True bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan_bs2.sh \
#     2>&1 | tee -a /tmp/train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan_bs2.log
# =============================================================================
set -euo pipefail

RESUME="${RESUME:-False}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_GAN_BS2"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_shift_5_4gpu_bs16_lr1e5_10000iter/checkpoints/0010000.pth}"

if [[ ! -f "${OMNIAVATAR_DF_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_DF_CKPT does not exist: ${OMNIAVATAR_DF_CKPT}" >&2
    exit 1
fi

RUN_NAME="sf_sink1_window7_redmd_audiofix_beta2_taew_gan_bs2"

echo "============================================="
echo "  Re-DMD beta=2 + TAEW + GAN @ batch=2"
echo "============================================="
echo "  DF init ckpt:       ${OMNIAVATAR_DF_CKPT}"
echo "  gan_loss_weight:    0.003"
echo "  Discriminator:      Wan_14B"
echo "  Dataloader batch:   2 (halved from bs=4 after student-step OOM)"
echo "  Grad accum rounds:  8 (effective batch 2 * 4 GPUs * 8 accum = 64)"
echo "  Run name:           ${RUN_NAME}"
echo "  Output root:        ${FASTGEN_OUTPUT_ROOT}"
echo "  Resume:             ${RESUME}"
echo "============================================="
echo ""

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_beta2_taew_gan_bs2.py \
    - trainer.resume=${RESUME} \
    log_config.group="omniavatar_sf_audiofix" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho"
