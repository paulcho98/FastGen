#!/bin/bash
# =============================================================================
# OmniAvatar Diffusion Forcing (Stage 1) — Audio-Fix Re-Run
# =============================================================================
#
# Same as train_omniavatar_df.sh, but run AFTER the concat(dim=0)->stack(dim=1)
# fix in fastgen/networks/OmniAvatar/{wan_model.py, network_causal.py}. At
# per-rank batch >= 2 the old code scrambled (batch, projection) pairs for
# audio conditioning; this run retrains on correctly-routed audio.
#
# Starting checkpoint: same 1.3B V2V phase2 (step-19500) — that one was
# pretrained against the canonical OmniAvatar reference (stack(dim=1)) and is
# itself clean.
#
# Usage:
#   bash scripts/train_omniavatar_df_audiofix.sh
#   NGPU=2 bash scripts/train_omniavatar_df_audiofix.sh
#   MAX_ITER=10000 bash scripts/train_omniavatar_df_audiofix.sh
# =============================================================================

set -euo pipefail

NGPU="${NGPU:-4}"
MAX_ITER="${MAX_ITER:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-500}"
VIZ_EVERY="${VIZ_EVERY:-500}"
RESUME="${RESUME:-False}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EFFECTIVE_BS=$((BATCH_SIZE * NGPU))
RUN_NAME="df_audiofix_${NGPU}gpu_bs${BATCH_SIZE}_lr1e5_${MAX_ITER}iter"

echo "============================================="
echo "  OmniAvatar DF Training (audio-fix re-run)"
echo "============================================="
echo "  GPUs:            ${NGPU}"
echo "  Batch size:      ${BATCH_SIZE}/GPU × ${NGPU} = ${EFFECTIVE_BS}"
echo "  Learning rate:   1e-5"
echo "  Max iterations:  ${MAX_ITER}"
echo "  Checkpoint:      ${OMNIAVATAR_STUDENT_CKPT:-/home/work/output_omniavatar_v2v_1.3B_phase2/step-19500.pt}"
echo "  Save every:      ${SAVE_EVERY} steps"
echo "  Visualize every: ${VIZ_EVERY} steps"
echo "  Validate every:  ${SAVE_EVERY} steps (10 fixed samples)"
echo "  Run name:        ${RUN_NAME}"
echo "============================================="
echo ""

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=${NGPU} \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_df.py \
    - dataloader_train.batch_size=${BATCH_SIZE} \
    trainer.ddp=True \
    trainer.max_iter=${MAX_ITER} \
    trainer.save_ckpt_iter=${SAVE_EVERY} \
    trainer.resume=${RESUME} \
    log_config.group="omniavatar_df_audiofix" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho"
