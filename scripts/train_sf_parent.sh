#!/bin/bash
# =============================================================================
# Clean parent for symmetric-full-FT 1.3B SF runs.
# =============================================================================
#
# Replaces the legacy ``train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh``
# parent with these correctness changes:
#
#   1. NO hardcoded ``model.fake_score_optimizer.lr=3e-6`` — the 1.5×
#      student LR was a half-fix for the LoRA-only fake_score capacity
#      asymmetry, and is incorrect once the asymmetry is removed via
#      symmetric full-FT.  The new configs (config_sf_full_ft_t769*)
#      explicitly set both LRs to 2e-6.
#
#   2. NO EXTRA_OVERRIDES default — child wrappers set CONFIG_PATH and
#      that's it.  Anything regime-affecting should live in the config
#      file (where it's type-checked + asserted) rather than in a
#      cmdline override string.
#
#   3. CONFIG_PATH is REQUIRED (no default) — children must opt into a
#      specific recipe via config rather than inheriting an opinionated
#      base.  Prevents accidentally launching with a stale parent default.
#
# All env-var ckpt routes (DF, teacher, student-V2V, etc.) are preserved
# from the legacy parent — same paths, same semantics, just without the
# asymmetric hardcodes layered on top.
#
# Usage (via a child wrapper that sets CONFIG_PATH):
#   bash scripts/train_sf_full_ft_t769.sh
#   bash scripts/train_sf_full_ft_t769_no_reward.sh
#
# Resume:
#   RESUME=True bash scripts/train_sf_full_ft_t769.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

if [[ -z "${CONFIG_PATH:-}" ]]; then
    echo "ERROR: CONFIG_PATH must be set by a child wrapper before exec'ing this script." >&2
    echo "       This parent does NOT carry a default config — child must opt in explicitly." >&2
    exit 1
fi

RESUME="${RESUME:-False}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT_FULL_FT}"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1
export NEG_TEXT_EMB_PATH="${NEG_TEXT_EMB_PATH:-/home/work/stableavatar_data/neg_text_emb.pt}"

# Student DF init: same syncnet-trained 5000-iter DF ckpt used by the
# legacy fsmatched runs.  ${VAR-default} (no colon) preserves explicit empty.
export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_syncnet_trained_shift_5_t769_4gpu_bs16_lr1e5_5000iter/checkpoints/0005000.pth}"

if [[ -n "${OMNIAVATAR_DF_CKPT}" && ! -f "${OMNIAVATAR_DF_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_DF_CKPT does not exist: ${OMNIAVATAR_DF_CKPT}" >&2
    exit 1
fi

# Teacher: mouthweight 14B step-6000.
export OMNIAVATAR_TEACHER_CKPT="${OMNIAVATAR_TEACHER_CKPT:-/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/step-6000.pt}"
if [[ ! -f "${OMNIAVATAR_TEACHER_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_TEACHER_CKPT does not exist: ${OMNIAVATAR_TEACHER_CKPT}" >&2
    exit 1
fi

# Fake-score V2V adapter: syncnet-trained 1.3B mouthweight (fsmatched).
export OMNIAVATAR_STUDENT_CKPT="${OMNIAVATAR_STUDENT_CKPT:-/home/work/output_omniavatar_v2v_1.3B_maskall_refseq_mouth_weight_2gpu/step-1000.pt}"
if [[ ! -f "${OMNIAVATAR_STUDENT_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_STUDENT_CKPT does not exist: ${OMNIAVATAR_STUDENT_CKPT}" >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-$(basename ${CONFIG_PATH} .py)}"

echo "============================================="
echo "  SF full-FT 1.3B Training (clean parent)"
echo "============================================="
echo "  CONFIG_PATH:      ${CONFIG_PATH}"
echo "  DF init ckpt:     ${OMNIAVATAR_DF_CKPT}"
echo "  Teacher ckpt:     ${OMNIAVATAR_TEACHER_CKPT}"
echo "  Fake-score V2V:   ${OMNIAVATAR_STUDENT_CKPT}"
echo "  LRs:              both 2e-6 (set in config; symmetric)"
echo "  Run name:         ${RUN_NAME}"
echo "  Output root:      ${FASTGEN_OUTPUT_ROOT}"
echo "  Resume:           ${RESUME}"
echo "  EXTRA_OVERRIDES:  ${EXTRA_OVERRIDES:-(none)}"
echo "============================================="
echo ""

# EXTRA_OVERRIDES kept available for ad-hoc debugging (e.g.
# `EXTRA_OVERRIDES="trainer.max_iter=100"` for a quick smoke), but the
# child wrappers should NOT bake regime-affecting flags into it.
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=${CONFIG_PATH} \
    - trainer.resume=${RESUME} \
    log_config.group="omniavatar_sf_full_ft" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho" \
    ${EXTRA_OVERRIDES}
