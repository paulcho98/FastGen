#!/bin/bash
# =============================================================================
# Re-DMD Stage 2 (β=2 config) — Audio-Fix Re-Run
# =============================================================================
# Same as train_sf_sink1_window7_redmd_beta2.sh, but run AFTER the
# concat(dim=0)->stack(dim=1) fix in
# fastgen/networks/OmniAvatar/{wan_model.py, network_causal.py}.
#
# IMPORTANT: point OMNIAVATAR_STUDENT_CKPT at the NEW DF (shift=5) checkpoint
# from the audio-fix DF re-run (train_omniavatar_df_shift_5_audiofix.sh). The
# prior DF checkpoint was trained against scrambled audio (B=16 + concat bug)
# and should NOT be used as init — it has under-learned audio projections.
#
# Usage:
#   OMNIAVATAR_STUDENT_CKPT=/path/to/df_shift5_audiofix/step-XXXXX.pt \
#     nohup bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix.sh \
#     > /tmp/train_sf_sink1_window7_redmd_beta2_audiofix.log 2>&1 &
# =============================================================================
set -euo pipefail

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

# Fail fast if the user forgot to point at the audio-fix DF checkpoint.
if [[ -z "${OMNIAVATAR_STUDENT_CKPT:-}" ]]; then
    echo "ERROR: OMNIAVATAR_STUDENT_CKPT is unset." >&2
    echo "  Set it to the new DF (shift=5) checkpoint produced by" >&2
    echo "  train_omniavatar_df_shift_5_audiofix.sh before launching." >&2
    exit 1
fi

RUN_NAME="sf_sink1_window7_redmd_audiofix_beta2"

echo "============================================="
echo "  Re-DMD β=2 Training (audio-fix re-run)"
echo "============================================="
echo "  Student init:    ${OMNIAVATAR_STUDENT_CKPT}"
echo "  Run name:        ${RUN_NAME}"
echo "  Output root:     ${FASTGEN_OUTPUT_ROOT}"
echo "============================================="
echo ""

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_beta2.py \
    - trainer.resume=False \
    log_config.group="omniavatar_sf_audiofix" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho"
