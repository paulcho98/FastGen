#!/bin/bash
# =============================================================================
# Re-DMD Stage 2 (β=2 config) — Audio-Fix Re-Run
# =============================================================================
# Same as train_sf_sink1_window7_redmd_beta2.sh, but run AFTER the
# concat(dim=0)->stack(dim=1) fix in
# fastgen/networks/OmniAvatar/{wan_model.py, network_causal.py}.
#
# Default DF init: final ckpt from the DF shift=5 audio-fix re-run. Override by
# exporting OMNIAVATAR_DF_CKPT=/new/path before launch.
#
# IMPORTANT: this must be OMNIAVATAR_DF_CKPT (routes to
# trainer.checkpointer.pretrained_ckpt_path — the checkpointer knows how to
# unwrap FastGen training metadata), NOT OMNIAVATAR_STUDENT_CKPT (that routes
# to omniavatar_ckpt_path which expects a clean V2V-adapter state dict and
# fails on numpy scalars via torch.load(weights_only=True) in PyTorch 2.6+).
#
# Usage (inside tmux):
#   bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix.sh \
#     2>&1 | tee /tmp/train_sf_sink1_window7_redmd_beta2_audiofix.log
#
# Resume after a crash (loads latest ckpt + continues the same wandb run via
# persisted wandb_id.txt):
#   RESUME=True bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix.sh \
#     2>&1 | tee -a /tmp/train_sf_sink1_window7_redmd_beta2_audiofix.log
# =============================================================================
set -euo pipefail

RESUME="${RESUME:-False}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

# Default DF init: final ckpt of the DF shift=5 audiofix run.
# (train_omniavatar_df_shift_5_audiofix.sh writes here via relative FASTGEN_OUTPUT).
export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_shift_5_4gpu_bs16_lr1e5_10000iter/checkpoints/0010000.pth}"

if [[ ! -f "${OMNIAVATAR_DF_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_DF_CKPT does not exist: ${OMNIAVATAR_DF_CKPT}" >&2
    exit 1
fi

RUN_NAME="sf_sink1_window7_redmd_audiofix_beta2"

echo "============================================="
echo "  Re-DMD β=2 Training (audio-fix re-run)"
echo "============================================="
echo "  DF init ckpt:    ${OMNIAVATAR_DF_CKPT}"
echo "  Run name:        ${RUN_NAME}"
echo "  Output root:     ${FASTGEN_OUTPUT_ROOT}"
echo "  Resume:          ${RESUME}"
echo "============================================="
echo ""

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_beta2.py \
    - trainer.resume=${RESUME} \
    log_config.group="omniavatar_sf_audiofix" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho"
