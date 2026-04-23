#!/bin/bash
# =============================================================================
# Re-DMD Stage 2 (beta=2) with TAEW decoder — syncnet-trained DF init +
#                                               mouthweight 14B teacher
# =============================================================================
# Same training setup as train_sf_sink1_window7_redmd_beta2_audiofix_taew.sh
# (shift=5 audio-fix, TAEW decoder for the reward path, 2-step student,
# beta=2 Re-DMD) but swaps two checkpoints:
#
#   1. Student DF init: the final ckpt from the syncnet-trained DF run
#      (train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh, 5000 iters)
#      instead of the plain V2V-phase2 DF ckpt.
#
#   2. Teacher:         the mouthweight 14B step-6000 checkpoint
#      (/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/
#      step-6000.pt) instead of the default phase2 step-10500 teacher.
#      Both are 14B (~1.24 GB). The "_4gpu" naming convention distinguishes
#      14B from "_2gpu" which is 1.3B.
#
# Override either via env vars:
#   OMNIAVATAR_DF_CKPT=/path/to/df.pth   -> trainer.checkpointer.pretrained_ckpt_path
#   OMNIAVATAR_TEACHER_CKPT=/path/to/teacher.pt -> config_sf.py TEACHER_CKPT
#
# IMPORTANT: student routes via OMNIAVATAR_DF_CKPT (checkpointer unwraps
# FastGen training metadata, handles numpy scalars). Do NOT use
# OMNIAVATAR_STUDENT_CKPT for the DF init — that route expects a clean
# V2V adapter state dict and fails on weights_only=True.
#
# Prereqs:
#   - /home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth
#   - Student DF ckpt (baked default path below — verify it exists)
#   - Teacher mouthweight 14B ckpt (baked default path below)
#
# Usage (inside tmux):
#   bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight.sh \
#     2>&1 | tee /tmp/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight.log
#
# Resume after a crash (loads latest ckpt + continues the same wandb run via
# persisted wandb_id.txt):
#   RESUME=True bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight.sh \
#     2>&1 | tee -a /tmp/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight.log
# =============================================================================
set -euo pipefail

# Self-locate: the script uses relative paths (train.py, fastgen/configs/...).
# Make CWD the repo root regardless of where we were invoked from.
cd "$(dirname "$(readlink -f "$0")")/.."

RESUME="${RESUME:-False}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

# Student DF init: final ckpt of the syncnet-trained DF run (5000 iters).
export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_syncnet_trained_shift_5_4gpu_bs16_lr1e5_5000iter/checkpoints/0005000.pth}"

if [[ ! -f "${OMNIAVATAR_DF_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_DF_CKPT does not exist: ${OMNIAVATAR_DF_CKPT}" >&2
    echo "       (the syncnet-trained DF run may still be in progress)" >&2
    exit 1
fi

# Teacher: mouthweight 14B step-6000 (vs default phase2 step-10500).
export OMNIAVATAR_TEACHER_CKPT="${OMNIAVATAR_TEACHER_CKPT:-/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/step-6000.pt}"

if [[ ! -f "${OMNIAVATAR_TEACHER_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_TEACHER_CKPT does not exist: ${OMNIAVATAR_TEACHER_CKPT}" >&2
    exit 1
fi

RUN_NAME="sf_sink1_window7_redmd_audiofix_beta2_taew_syncnet_mouthweight"

echo "============================================="
echo "  Re-DMD beta=2 Training (audio-fix, TAEW, syncnet DF + mouthweight 14B teacher)"
echo "============================================="
echo "  DF init ckpt:    ${OMNIAVATAR_DF_CKPT}"
echo "  Teacher ckpt:    ${OMNIAVATAR_TEACHER_CKPT}"
echo "  TAEW ckpt:       /home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth"
echo "  Run name:        ${RUN_NAME}"
echo "  Output root:     ${FASTGEN_OUTPUT_ROOT}"
echo "  Resume:          ${RESUME}"
echo "============================================="
echo ""

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_beta2_taew.py \
    - trainer.resume=${RESUME} \
    log_config.group="omniavatar_sf_audiofix" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho"
