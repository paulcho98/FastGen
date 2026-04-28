#!/bin/bash
# =============================================================================
# Re-DMD Stage 2 (beta=2) with TAEW decoder — syncnet-trained DF init +
#                                               mouthweight 14B teacher +
#                                               syncnet-matched fake score init
# =============================================================================
# Fixes a bug in train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight.sh:
# that script only overrode the DF ckpt (which loads into the student via the
# checkpointer) and the teacher ckpt, but did NOT override OMNIAVATAR_STUDENT_CKPT.
# `config_sf.py:60` routes the *fake score* network's V2V adapter load from
# STUDENT_CKPT, so without the override the fake score was initialized from
# /home/work/output_omniavatar_v2v_1.3B_phase2/step-19500.pt (plain phase2)
# instead of the syncnet-trained adapter that the DF run started from.
# Misaligned fake-score init means slower convergence while the fake score
# travels from the plain-phase2 basin toward the student's DF-trained basin.
#
# Three aligned checkpoints in this run:
#   1. Student DF init (via checkpointer): the syncnet-trained DF final ckpt
#      (train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh, 5000 iters)
#   2. Teacher:  mouthweight 14B step-6000
#      (/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/
#      step-6000.pt)
#   3. Fake score V2V adapter (NEW): the same syncnet-trained 1.3B adapter
#      that the DF run itself was initialized from
#      (/home/work/output_omniavatar_v2v_1.3B_maskall_refseq_mouth_weight_2gpu/
#      step-1000.pt)
#
# Additional tuning: critic (fake_score) learning rate raised from 2e-6 to
# 3e-6 (1.5x student LR) so the critic tracks the moving student distribution
# slightly faster. Injected via torchrun override below; see Hydra override
# syntax `model.fake_score_optimizer.lr=3e-6`.
#
# RUN_NAME and FASTGEN_OUTPUT_ROOT carry a `_lr3e6` suffix so wandb shows
# this as a fresh run (new run ID) distinct from the prior OOM-crashed run
# `63ehs3bc` (which trained up to iter 138 at the default 2e-6 critic LR
# before being evicted by a concurrent job on GPU 0).
#
# Side effect of setting OMNIAVATAR_STUDENT_CKPT here: the student's V2V
# adapter is *initially* loaded from the syncnet-trained ckpt too. But the
# checkpointer immediately overrides config.model.net with the DF ckpt
# (0005000.pth), so the student's final state is unchanged by this script
# vs the non-fsmatched version — only the fake score differs.
#
# Override any ckpt via env:
#   OMNIAVATAR_DF_CKPT=/path       -> trainer.checkpointer.pretrained_ckpt_path (student)
#   OMNIAVATAR_TEACHER_CKPT=/path  -> config_sf.py TEACHER_CKPT
#   OMNIAVATAR_STUDENT_CKPT=/path  -> config_sf.py STUDENT_CKPT (fake score adapter; also
#                                     student base pre-checkpointer)
#
# IMPORTANT: DF ckpt routes via OMNIAVATAR_DF_CKPT (checkpointer unwraps
# FastGen training metadata, handles numpy scalars). Do NOT put the DF
# training ckpt at OMNIAVATAR_STUDENT_CKPT — that route expects a clean
# V2V adapter state dict and fails on weights_only=True.
#
# Prereqs:
#   - /home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth
#   - Student DF ckpt (baked default path below)
#   - Teacher mouthweight 14B ckpt (baked default path below)
#   - Syncnet-trained 1.3B V2V adapter (baked default path below)
#
# Usage (inside tmux):
#   bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh \
#     2>&1 | tee /tmp/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.log
#
# Resume after a crash (loads latest ckpt + continues the same wandb run via
# persisted wandb_id.txt):
#   RESUME=True bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh \
#     2>&1 | tee -a /tmp/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.log
# =============================================================================
set -euo pipefail

# Self-locate: the script uses relative paths (train.py, fastgen/configs/...).
# Make CWD the repo root regardless of where we were invoked from.
cd "$(dirname "$(readlink -f "$0")")/.."

RESUME="${RESUME:-False}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT_FSMATCHED_LR3E6}"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

# Student DF init: final ckpt of the syncnet-trained DF run (5000 iters).
# Loaded via the FastGen checkpointer -> config.model.net.
# Note: uses ${VAR-default} (no colon) so an explicit-empty
# OMNIAVATAR_DF_CKPT="" from a caller is PRESERVED (means "skip DF init");
# only an unset env uses the default. ${VAR:-default} would clobber "" with
# the default, silently re-enabling DF init.
export OMNIAVATAR_DF_CKPT="${OMNIAVATAR_DF_CKPT-/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_syncnet_trained_shift_5_4gpu_bs16_lr1e5_5000iter/checkpoints/0005000.pth}"

if [[ -n "${OMNIAVATAR_DF_CKPT}" && ! -f "${OMNIAVATAR_DF_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_DF_CKPT does not exist: ${OMNIAVATAR_DF_CKPT}" >&2
    echo "       (the syncnet-trained DF run may still be in progress)" >&2
    exit 1
fi
# Empty OMNIAVATAR_DF_CKPT is explicitly allowed: it propagates to an empty
# pretrained_ckpt_path, so trainer.py:82's truthy check skips load_pretrained_ckpt
# and the student stays at the Wan-base + V2V-adapter init (pre-DF).

# Teacher: mouthweight 14B step-6000 (vs default phase2 step-10500).
export OMNIAVATAR_TEACHER_CKPT="${OMNIAVATAR_TEACHER_CKPT:-/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/step-6000.pt}"

if [[ ! -f "${OMNIAVATAR_TEACHER_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_TEACHER_CKPT does not exist: ${OMNIAVATAR_TEACHER_CKPT}" >&2
    exit 1
fi

# Fake-score (and student-base-pre-checkpointer) V2V adapter: syncnet-trained
# 1.3B mouthweight adapter — the same one the DF run itself was initialized from.
# This is the central fix of the _fsmatched variant.
export OMNIAVATAR_STUDENT_CKPT="${OMNIAVATAR_STUDENT_CKPT:-/home/work/output_omniavatar_v2v_1.3B_maskall_refseq_mouth_weight_2gpu/step-1000.pt}"

if [[ ! -f "${OMNIAVATAR_STUDENT_CKPT}" ]]; then
    echo "ERROR: OMNIAVATAR_STUDENT_CKPT does not exist: ${OMNIAVATAR_STUDENT_CKPT}" >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-sf_sink1_window7_redmd_audiofix_beta2_taew_syncnet_mouthweight_fsmatched_lr3e6}"

echo "============================================="
echo "  Re-DMD beta=2 Training (audio-fix, TAEW)"
echo "  syncnet DF init + mouthweight 14B teacher + fsmatched fake score"
echo "============================================="
echo "  DF init ckpt:     ${OMNIAVATAR_DF_CKPT}"
echo "  Teacher ckpt:     ${OMNIAVATAR_TEACHER_CKPT}"
echo "  Fake-score V2V:   ${OMNIAVATAR_STUDENT_CKPT}"
echo "  Fake-score LR:    3e-6 (1.5x student LR of 2e-6)"
echo "  TAEW ckpt:        /home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth"
echo "  Run name:         ${RUN_NAME}"
echo "  Output root:      ${FASTGEN_OUTPUT_ROOT}"
echo "  Resume:           ${RESUME}"
echo "============================================="
echo ""

# EXTRA_OVERRIDES (optional, space-separated key=val pairs): appended to the
# torchrun command so sibling wrappers can toggle features without copying
# the whole launcher. Example: EXTRA_OVERRIDES="model.reward.enabled=False"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# CONFIG_PATH (optional): override the train-config Python file. Default is
# the 1.3B beta=2 + TAEW config; sibling wrappers (e.g. the 14B LoRA SF
# variant) override this to point at a different config without forking
# the whole launcher.
CONFIG_PATH="${CONFIG_PATH:-fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_beta2_taew.py}"

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=${CONFIG_PATH} \
    - trainer.resume=${RESUME} \
    model.fake_score_optimizer.lr=3e-6 \
    log_config.group="omniavatar_sf_audiofix" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho" \
    ${EXTRA_OVERRIDES}
