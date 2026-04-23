#!/bin/bash
# =============================================================================
# Inference for the Re-DMD beta=2 + TAEW audio-fix + syncnet-DF-init +
# mouthweight-14B-teacher SF run (the "first" variant, with MIS-INITIALIZED
# fake score — phase2/step-19500 instead of syncnet-trained 1.3B).
#
# Use this BEFORE killing that run and launching the _fsmatched variant, to
# establish a baseline for comparison. Runs inference on multiple checkpoint
# steps in parallel across GPUs.
#
# Usage:
#   nohup bash scripts/infer_redmd_taew_audiofix_syncnet_mouthweight.sh \
#     > /tmp/infer_redmd_taew_audiofix_syncnet_mouthweight.log 2>&1 &
#
# Override defaults:
#   CKPT_STEPS="100 200 300 400" GPUS="0 1 2 3" bash scripts/...
#
# Default: steps 300 400 500 on GPUs 0 1 2. Waits for 500.pth to exist before
# launching (fails fast if any requested step is missing on disk).
# =============================================================================
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

CKPT_DIR="/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT/OmniAvatar-FastGen/omniavatar_sf_audiofix/sf_sink1_window7_redmd_audiofix_beta2_taew_syncnet_mouthweight/checkpoints"
OUT_ROOT="/home/work/output_hdtf_sf_redmd_beta2_taew_syncnet_mouthweight"
HDTF="/home/work/.local/HDTF/HDTF_original_testset_81frames"
TEXT_EMB="/home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt"

# Default: 3 ckpts on 3 GPUs. GPU 3 idle so training can continue (if
# training is still running when this launches — otherwise use 0 1 2 3 and
# override CKPT_STEPS to 4 values).
CKPT_STEPS=(${CKPT_STEPS:-300 400 500})
GPUS=(${GPUS:-0 1 2})
NUM=${#CKPT_STEPS[@]}

if [ "$NUM" -ne "${#GPUS[@]}" ]; then
    echo "ERROR: CKPT_STEPS (${NUM}) and GPUS (${#GPUS[@]}) must have equal length." >&2
    exit 1
fi

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"

if [ ! -d "$HDTF/videos_cfr" ]; then
    echo "ERROR: HDTF dir missing: $HDTF/videos_cfr" >&2
    exit 1
fi
TOTAL_VIDEOS=$(ls "$HDTF"/videos_cfr/*.mp4 2>/dev/null | wc -l)

echo "============================================="
echo "  Re-DMD beta=2 TAEW (syncnet + mouthweight) — Inference"
echo "============================================="
echo "  CKPT_DIR:   $CKPT_DIR"
echo "  OUT_ROOT:   $OUT_ROOT"
echo "  HDTF:       $HDTF ($TOTAL_VIDEOS videos)"
echo "  Assignment:"
for i in "${!CKPT_STEPS[@]}"; do
    step="${CKPT_STEPS[$i]}"
    gpu="${GPUS[$i]}"
    step_p=$(printf "%07d" "$step")
    ckpt="$CKPT_DIR/${step_p}.pth"
    echo "    GPU ${gpu}: step ${step} -> ${ckpt}"
    if [ ! -f "$ckpt" ]; then
        echo "    ERROR: ckpt missing: ${ckpt}" >&2
        echo "           (if step 500 is pending, wait for it to be saved first.)" >&2
        exit 1
    fi
    if [ ! -d "$CKPT_DIR/${step_p}.net_model" ]; then
        echo "    ERROR: distcp dir missing: ${CKPT_DIR}/${step_p}.net_model" >&2
        exit 1
    fi
done
echo "============================================="
echo ""

run_checkpoint() {
    local GPU_ID=$1
    local STEP=$2
    local STEP_P=$(printf "%07d" "$STEP")
    local CKPT="$CKPT_DIR/${STEP_P}.pth"
    local OUT_DIR="$OUT_ROOT/step_${STEP_P}"

    local n=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | grep -v aligned | wc -l)
    if [ "$n" -ge "$TOTAL_VIDEOS" ]; then
        echo "[GPU $GPU_ID | step $STEP] Already complete ($n/$TOTAL_VIDEOS) — skipping"
        return
    fi

    mkdir -p "$OUT_DIR"
    echo "[GPU $GPU_ID | step $STEP] Starting inference ($n/$TOTAL_VIDEOS done)"

    local i=0
    for video in "$HDTF"/videos_cfr/*.mp4; do
        local name=$(basename "$video" _cfr25.mp4)
        i=$((i + 1))
        [ -f "$OUT_DIR/${name}.mp4" ] && continue

        echo "[GPU $GPU_ID | step $STEP] [$i/$TOTAL_VIDEOS] $name"
        CUDA_VISIBLE_DEVICES=$GPU_ID /home/work/.local/miniconda3/envs/hb_fastgen/bin/python \
            scripts/inference/inference_causal.py \
            --ckpt_path "$CKPT" \
            --vae_path "$OMNIAVATAR_ROOT/pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
            --wav2vec_path "$OMNIAVATAR_ROOT/pretrained_models/wav2vec2-base-960h" \
            --mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
            --base_model_paths "$OMNIAVATAR_ROOT/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
            --omniavatar_ckpt_path /home/work/output_omniavatar_v2v_1.3B_phase2/step-19500.pt \
            --text_embeds_path "$TEXT_EMB" \
            --video_path "$video" \
            --output_path "$OUT_DIR/${name}.mp4" \
            --t_list 0.999 0.833 0.0 \
            --local_attn_size 7 \
            --sink_size 1 \
            --use_dynamic_rope \
            --latentsync \
            --face_cache_dir /home/work/.local/HDTF/face_cache \
            --skip_existing \
        || echo "  FAILED: step $STEP | $name (continuing)"
    done
    echo "[GPU $GPU_ID | step $STEP] Inference done"
}

for i in "${!CKPT_STEPS[@]}"; do
    run_checkpoint "${GPUS[$i]}" "${CKPT_STEPS[$i]}" &
done
wait

echo ""
echo "============================================="
echo "  All inference complete."
echo "  Outputs under: $OUT_ROOT/step_{0000300,0000400,0000500}/"
echo "  Next: existing eval scripts (e.g. scripts/eval_redmd_taew_audiofix.sh)"
echo "        against OUT_ROOT to compute sync-C / FID / etc."
echo "============================================="
