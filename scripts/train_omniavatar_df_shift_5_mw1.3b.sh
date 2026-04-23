#!/bin/bash
# =============================================================================
# OmniAvatar DF (shift=5) — MW1.3B checkpoint (aux losses, better sync)
# =============================================================================
#
# Same as train_omniavatar_df_shift_5_audiofix.sh, but starting from the
# MW1.3B s1000 checkpoint (trained with SyncNet+LPIPS+TREPA aux losses,
# mouth_weight=2.0, in_dim=65, ref_sequence=True).
#
# This checkpoint has Sync-C=8.04 vs 6.57 for Phase2 s19500 (MSE only).
#
# Usage:
#   bash scripts/train_omniavatar_df_shift_5_mw1.3b.sh
#   NGPU=2 bash scripts/train_omniavatar_df_shift_5_mw1.3b.sh
#   MAX_ITER=10000 bash scripts/train_omniavatar_df_shift_5_mw1.3b.sh
# =============================================================================

set -euo pipefail

NGPU="${NGPU:-4}"
MAX_ITER="${MAX_ITER:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-500}"
VIZ_EVERY="${VIZ_EVERY:-500}"
RESUME="${RESUME:-False}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export OMNIAVATAR_STUDENT_CKPT="/home/work/output_omniavatar_v2v_1.3B_maskall_refseq_mouth_weight_2gpu/step-1000.pt"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EFFECTIVE_BS=$((BATCH_SIZE * NGPU))
RUN_NAME="df_mw1.3b_shift_5_${NGPU}gpu_bs${BATCH_SIZE}_lr1e5_${MAX_ITER}iter"

echo "============================================="
echo "  OmniAvatar DF shift=5 (MW1.3B checkpoint)"
echo "============================================="
echo "  GPUs:            ${NGPU}"
echo "  Batch size:      ${BATCH_SIZE}/GPU × ${NGPU} = ${EFFECTIVE_BS}"
echo "  Learning rate:   1e-5"
echo "  Max iterations:  ${MAX_ITER}"
echo "  Checkpoint:      ${OMNIAVATAR_STUDENT_CKPT}"
echo "  Save every:      ${SAVE_EVERY} steps"
echo "  Visualize every: ${VIZ_EVERY} steps"
echo "  Validate every:  ${SAVE_EVERY} steps (10 fixed samples)"
echo "  Run name:        ${RUN_NAME}"
echo "============================================="
echo ""

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=${NGPU} \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_df_shift_5.py \
    - dataloader_train.batch_size=${BATCH_SIZE} \
    trainer.ddp=True \
    trainer.max_iter=${MAX_ITER} \
    trainer.save_ckpt_iter=${SAVE_EVERY} \
    trainer.resume=${RESUME} \
    log_config.group="omniavatar_df_audiofix" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho"
