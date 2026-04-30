#!/bin/bash
# =============================================================================
# Inference for the 14B-LoRA SF Re-DMD beta=2 + TAEW audiofix + syncnet +
# mouthweight + fsmatched + t769 run.
#
# Sibling of infer_redmd_taew_audiofix_syncnet_mouthweight.sh (which targets
# the 1.3B run).  Differences:
#   - CKPT_DIR points at the 14B SF run output dir
#   - Uses inference_causal_14b.py (PEFT-aware loader, --model_size=14B,
#     --merge_lora_post_load=True by default)
#   - Base model paths point at the 6 14B safetensor shards
#   - omniavatar_ckpt_path points at the 14B mouthweight ckpt (matches the
#     SF run's STUDENT_CKPT_14B default — same ckpt the SF run used as
#     student/fake_score base init)
#
# Usage:
#   nohup bash scripts/infer_redmd_taew_audiofix_syncnet_mouthweight_14b.sh \
#     > /tmp/infer_14b_lora.log 2>&1 &
#
# Override defaults:
#   CKPT_STEPS="200 300 400" GPUS="0 1 2" bash scripts/...
# =============================================================================
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

CKPT_DIR="/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT_FSMATCHED_T769_14B_LORA/OmniAvatar-FastGen/omniavatar_sf_audiofix/sf_sink1_window7_redmd_audiofix_beta2_taew_syncnet_mouthweight_fsmatched_t769_14b_lora/checkpoints"
OUT_ROOT="/home/work/output_hdtf_sf_redmd_beta2_taew_syncnet_mouthweight_fsmatched_t769_14b_lora"
HDTF="/home/work/.local/HDTF/HDTF_original_testset_81frames"
TEXT_EMB="/home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt"

CKPT_STEPS=(${CKPT_STEPS:-200 300 400})
GPUS=(${GPUS:-0 1 2})
NUM=${#CKPT_STEPS[@]}

if [ "$NUM" -ne "${#GPUS[@]}" ]; then
    echo "ERROR: CKPT_STEPS (${NUM}) and GPUS (${#GPUS[@]}) must have equal length." >&2
    exit 1
fi

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"

# 6-shard 14B base
WAN_14B_BASE=$(IFS=,; echo "${OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,${OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,${OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,${OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,${OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,${OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors")

# 14B V2V mouthweight ckpt — matches the SF run's STUDENT_CKPT_14B default.
OMNIAVATAR_CKPT_14B="${OMNIAVATAR_CKPT_14B:-/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/step-6000.pt}"

if [ ! -d "$HDTF/videos_cfr" ]; then
    echo "ERROR: HDTF dir missing: $HDTF/videos_cfr" >&2
    exit 1
fi
TOTAL_VIDEOS=$(ls "$HDTF"/videos_cfr/*.mp4 2>/dev/null | wc -l)

echo "============================================="
echo "  14B SF Re-DMD beta=2 TAEW (mouthweight + fsmatched + t769) — Inference"
echo "============================================="
echo "  CKPT_DIR:   $CKPT_DIR"
echo "  OUT_ROOT:   $OUT_ROOT"
echo "  HDTF:       $HDTF ($TOTAL_VIDEOS videos)"
echo "  Base 14B:   ${WAN_14B_BASE%%,*}, ... (6 shards)"
echo "  V2V mouth:  $OMNIAVATAR_CKPT_14B"
echo "  Assignment:"
for i in "${!CKPT_STEPS[@]}"; do
    step="${CKPT_STEPS[$i]}"
    gpu="${GPUS[$i]}"
    step_p=$(printf "%07d" "$step")
    ckpt="$CKPT_DIR/${step_p}.pth"
    echo "    GPU ${gpu}: step ${step} -> ${ckpt}"
    if [ ! -f "$ckpt" ]; then
        echo "    ERROR: ckpt missing: ${ckpt}" >&2
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
            scripts/inference/inference_causal_14b.py \
            --model_size 14B \
            --ckpt_path "$CKPT" \
            --vae_path "$OMNIAVATAR_ROOT/pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
            --wav2vec_path "$OMNIAVATAR_ROOT/pretrained_models/wav2vec2-base-960h" \
            --mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
            --base_model_paths "$WAN_14B_BASE" \
            --omniavatar_ckpt_path "$OMNIAVATAR_CKPT_14B" \
            --text_embeds_path "$TEXT_EMB" \
            --video_path "$video" \
            --output_path "$OUT_DIR/${name}.mp4" \
            --t_list 0.999 0.769 0.0 \
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
echo "  All 14B inference complete."
echo "  Outputs under: $OUT_ROOT/step_{0000200,0000300,0000400}/"
echo "============================================="
