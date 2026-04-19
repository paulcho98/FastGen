#!/bin/bash
# =============================================================================
# Re-DMD beta=2 + TAEW + GAN @ batch=8 — Memory Stress Probe
# =============================================================================
# Same GAN recipe as the batch=4 variant, but keeps dataloader_batch=8 and
# grad_accum_rounds=2 (the no-GAN TAEW baseline's sizing). Purpose: check
# empirically whether GAN+batch=8 OOMs on H200 before committing to the
# reduced-batch variant.
#
# If this OOMs: use train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan.sh
# If this fits: it's the preferred variant (matches no-GAN baseline's
#               effective-batch and iteration rate).
#
# Recommend running with a low max_iter / quick smoke to probe memory
# before committing to a full run:
#   bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan_bs8.sh \
#     2>&1 | tee /tmp/train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan_bs8.log
#
# Resume:
#   RESUME=True bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan_bs8.sh \
#     2>&1 | tee -a /tmp/train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan_bs8.log
# =============================================================================
set -euo pipefail

RESUME="${RESUME:-False}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_GAN_BS8"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_shift_5_4gpu_bs16_lr1e5_10000iter/checkpoints/0010000.pth}"

if [[ ! -f "${OMNIAVATAR_DF_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_DF_CKPT does not exist: ${OMNIAVATAR_DF_CKPT}" >&2
    exit 1
fi

RUN_NAME="sf_sink1_window7_redmd_audiofix_beta2_taew_gan_bs8"

echo "============================================="
echo "  Re-DMD beta=2 + TAEW + GAN @ batch=8 (memory probe)"
echo "============================================="
echo "  DF init ckpt:       ${OMNIAVATAR_DF_CKPT}"
echo "  TAEW ckpt:          /home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth"
echo "  gan_loss_weight:    0.003"
echo "  Discriminator:      Wan_14B (40 blocks, feature_indices=[21,30,39])"
echo "  Dataloader batch:   8 (matches no-GAN baseline)"
echo "  Grad accum rounds:  2 (effective batch 8 * 4 GPUs * 2 accum = 64)"
echo "  Run name:           ${RUN_NAME}"
echo "  Output root:        ${FASTGEN_OUTPUT_ROOT}"
echo "  Resume:             ${RESUME}"
echo "============================================="
echo ""

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_beta2_taew_gan_bs8.py \
    - trainer.resume=${RESUME} \
    log_config.group="omniavatar_sf_audiofix" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho"
