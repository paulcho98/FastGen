#!/bin/bash
# =============================================================================
# OmniAvatar DF (shift=5) — Audio-Fix, init from syncnet-trained V2V adapter
# =============================================================================
#
# Same training setup as train_omniavatar_df_shift_5_audiofix.sh (audio-fix
# re-run: concat(dim=0) -> stack(dim=1) in wan_model.py + network_causal.py)
# but initializes the student from a V2V adapter that was separately fine-
# tuned with syncnet / mask-all / ref-seq / mouth-weight objectives, rather
# than the plain phase2 step-19500 V2V adapter.
#
# Starting checkpoint (baked in as default):
#   /home/work/output_omniavatar_v2v_1.3B_maskall_refseq_mouth_weight_2gpu/step-1000.pt
#
# This is a clean V2V LoRA state dict (verified — compatible with
# torch.load(weights_only=True) via config_df_shift_5.py's STUDENT_CKPT
# route at line 24-26, which passes it into CausalOmniAvatarWan's
# omniavatar_ckpt_path kwarg). Override via OMNIAVATAR_STUDENT_CKPT env if
# you want a different init.
#
# Usage:
#   bash scripts/train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh
#   NGPU=2 bash scripts/train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh
#   MAX_ITER=10000 bash scripts/train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh
#   RESUME=True bash scripts/train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh
# =============================================================================

set -euo pipefail

NGPU="${NGPU:-4}"
MAX_ITER="${MAX_ITER:-5000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-500}"
VIZ_EVERY="${VIZ_EVERY:-500}"
RESUME="${RESUME:-False}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Bake in the syncnet-trained V2V adapter as the default starting point.
# This is what distinguishes this run from train_omniavatar_df_shift_5_audiofix.sh
# (which defaults to the plain phase2 step-19500 adapter).
export OMNIAVATAR_STUDENT_CKPT="${OMNIAVATAR_STUDENT_CKPT:-/home/work/output_omniavatar_v2v_1.3B_maskall_refseq_mouth_weight_2gpu/step-1000.pt}"

if [[ ! -f "${OMNIAVATAR_STUDENT_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_STUDENT_CKPT does not exist: ${OMNIAVATAR_STUDENT_CKPT}" >&2
    exit 1
fi

EFFECTIVE_BS=$((BATCH_SIZE * NGPU))
# RUN_NAME (optional override): wrappers can set a custom name to avoid
# colliding with default-named runs (e.g. _t769 schedule variant).
RUN_NAME="${RUN_NAME:-df_audiofix_syncnet_trained_shift_5_${NGPU}gpu_bs${BATCH_SIZE}_lr1e5_${MAX_ITER}iter}"

echo "============================================="
echo "  OmniAvatar DF shift=5 (audio-fix, syncnet-trained init)"
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
echo "  Resume:          ${RESUME}"
echo "============================================="
echo ""

# CONFIG_PATH (optional): override the train-config Python file. Useful for
# variants with different sample_t_cfg / student_sample_steps (e.g.
# config_df_shift_5_t769.py for the 2-step SF schedule).
CONFIG_PATH="${CONFIG_PATH:-fastgen/configs/experiments/OmniAvatar/config_df_shift_5.py}"

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=${NGPU} \
    train.py \
    --config=${CONFIG_PATH} \
    - dataloader_train.batch_size=${BATCH_SIZE} \
    trainer.ddp=True \
    trainer.max_iter=${MAX_ITER} \
    trainer.save_ckpt_iter=${SAVE_EVERY} \
    trainer.resume=${RESUME} \
    log_config.group="omniavatar_df_audiofix" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho"
