#!/bin/bash
# =============================================================================
# Re-DMD Stage 2 (beta=2) with TAEW decoder — Audio-Fix Re-Run
# =============================================================================
# Mirror of train_sf_sink1_window7_redmd_beta2_audiofix.sh but the reward
# path uses TAEHVDecoderWrapper (11.3M-param tiny VAE) instead of the full
# Wan 2.1 VAE (127M) for pixel decode. Value-wise MAE vs Wan is ~0.011 on
# [-1, 1], so the reward signal should be effectively identical while the
# per-student-step decode is much cheaper.
#
# Prereqs:
#   - /home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth
#   - DF shift=5 audiofix checkpoint (default baked in below; override via env
#     OMNIAVATAR_DF_CKPT=/new/path if you want a different init).
#
# Usage (inside tmux):
#   bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew.sh \
#     2>&1 | tee /tmp/train_sf_sink1_window7_redmd_beta2_audiofix_taew.log
#
# Resume after a crash (loads latest ckpt + continues the same wandb run via
# persisted wandb_id.txt):
#   RESUME=True bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew.sh \
#     2>&1 | tee -a /tmp/train_sf_sink1_window7_redmd_beta2_audiofix_taew.log
# =============================================================================
set -euo pipefail

RESUME="${RESUME:-False}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

# Default DF init for the student: final ckpt of the DF shift=5 audiofix run.
# IMPORTANT: this must be OMNIAVATAR_DF_CKPT (routes to
# trainer.checkpointer.pretrained_ckpt_path — the checkpointer knows how to
# unwrap FastGen training metadata), NOT OMNIAVATAR_STUDENT_CKPT (that routes
# to omniavatar_ckpt_path which expects a clean V2V-adapter state dict and
# fails on numpy scalars via torch.load(weights_only=True)).
# Override by exporting OMNIAVATAR_DF_CKPT before launch.
export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_shift_5_4gpu_bs16_lr1e5_10000iter/checkpoints/0010000.pth}"

if [[ ! -f "${OMNIAVATAR_DF_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_DF_CKPT does not exist: ${OMNIAVATAR_DF_CKPT}" >&2
    exit 1
fi

RUN_NAME="sf_sink1_window7_redmd_audiofix_beta2_taew"

echo "============================================="
echo "  Re-DMD beta=2 Training (audio-fix, TAEW decoder)"
echo "============================================="
echo "  DF init ckpt:    ${OMNIAVATAR_DF_CKPT}"
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
