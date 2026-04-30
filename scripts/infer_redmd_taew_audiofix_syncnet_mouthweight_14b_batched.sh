#!/bin/bash
# =============================================================================
# Batched inference for the 14B-LoRA SF run.
# =============================================================================
#
# 14B counterpart of infer_redmd_taew_audiofix_syncnet_mouthweight_batched.sh.
#
# Key differences:
#   - Uses inference_causal_14b.py (--model_size=14B, PEFT-aware loader,
#     --merge_lora_post_load for fused inference)
#   - 6-shard 14B base model paths
#   - 14B mouthweight V2V ckpt
#   - CKPT_DIR points at the 14B SF run output
#   - t_list=[0.999, 0.769, 0.0] (t769 schedule, matches 14B training)
#
# ONE python process per (ckpt, GPU) — model + VAE + wav2vec loaded ONCE,
# then loops over all HDTF samples in /tmp/hdtf_staging via --input_dir.
# --skip_existing makes this safe to re-run after a partial run.
#
# NOTE: the 14B model needs ~28 GB VRAM for inference (bf16).  Each GPU
# runs one ckpt at a time; do not assign multiple ckpts to the same GPU.
#
# Usage:
#   nohup bash scripts/infer_redmd_taew_audiofix_syncnet_mouthweight_14b_batched.sh \
#     > /tmp/infer_14b_batched.log 2>&1 &
#
# Override defaults:
#   CKPT_STEPS="200 300 400 500" GPUS="0 1 2 3" bash scripts/...
# =============================================================================
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

CKPT_DIR="/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT_FSMATCHED_T769_14B_LORA/OmniAvatar-FastGen/omniavatar_sf_audiofix/sf_sink1_window7_redmd_audiofix_beta2_taew_syncnet_mouthweight_fsmatched_t769_14b_lora/checkpoints"
OUT_ROOT="/home/work/output_hdtf_sf_redmd_beta2_taew_syncnet_mouthweight_fsmatched_t769_14b_lora"
STAGE_DIR="/tmp/hdtf_staging"

CKPT_STEPS=(${CKPT_STEPS:-200 300 400})
GPUS=(${GPUS:-0 1 2})

if [ ! -d "$STAGE_DIR" ] || [ "$(ls -d "$STAGE_DIR"/*/ 2>/dev/null | wc -l)" -lt 1 ]; then
    echo "ERROR: $STAGE_DIR not populated. Run the staging step first." >&2
    exit 1
fi
if [ "${#CKPT_STEPS[@]}" -ne "${#GPUS[@]}" ]; then
    echo "ERROR: CKPT_STEPS (${#CKPT_STEPS[@]}) and GPUS (${#GPUS[@]}) must match." >&2
    exit 1
fi

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"

# 6-shard 14B base
WAN_14B_BASE="${OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,${OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,${OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,${OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,${OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,${OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors"

# 14B V2V mouthweight ckpt
OMNIAVATAR_CKPT_14B="${OMNIAVATAR_CKPT_14B:-/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/step-6000.pt}"

N_SAMPLES=$(ls -d "$STAGE_DIR"/*/ | wc -l)

echo "============================================="
echo "  14B SF LoRA — Batched Inference (HDTF staging)"
echo "============================================="
echo "  CKPT_DIR:   $CKPT_DIR"
echo "  STAGE_DIR:  $STAGE_DIR ($N_SAMPLES samples)"
echo "  OUT_ROOT:   $OUT_ROOT"
echo "  Base 14B:   ${WAN_14B_BASE%%,*}, ... (6 shards)"
echo "  V2V mouth:  $OMNIAVATAR_CKPT_14B"
echo "  Assignment:"
for i in "${!CKPT_STEPS[@]}"; do
    step="${CKPT_STEPS[$i]}"
    gpu="${GPUS[$i]}"
    step_p=$(printf "%07d" "$step")
    ckpt="$CKPT_DIR/${step_p}.pth"
    echo "    GPU ${gpu}: step ${step}  ->  ${ckpt}"
    [ -f "$ckpt" ] || { echo "    ERROR: ckpt missing: $ckpt" >&2; exit 1; }
    [ -d "$CKPT_DIR/${step_p}.net_model" ] || { echo "    ERROR: distcp dir missing" >&2; exit 1; }
done
echo "============================================="
echo ""

run_ckpt() {
    local GPU_ID=$1
    local STEP=$2
    local STEP_P=$(printf "%07d" "$STEP")
    local CKPT="$CKPT_DIR/${STEP_P}.pth"
    local OUT_DIR="$OUT_ROOT/step_${STEP_P}"
    mkdir -p "$OUT_DIR"

    local have=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | grep -v aligned | wc -l)
    echo "[GPU $GPU_ID | step $STEP] Starting (${have}/${N_SAMPLES} already done; --skip_existing will reuse)"

    CUDA_VISIBLE_DEVICES=$GPU_ID /home/work/.local/miniconda3/envs/hb_fastgen/bin/python \
        scripts/inference/inference_causal_14b.py \
        --model_size 14B \
        --ckpt_path "$CKPT" \
        --vae_path "$OMNIAVATAR_ROOT/pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
        --wav2vec_path "$OMNIAVATAR_ROOT/pretrained_models/wav2vec2-base-960h" \
        --mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
        --base_model_paths "$WAN_14B_BASE" \
        --omniavatar_ckpt_path "$OMNIAVATAR_CKPT_14B" \
        --text_embeds_path /home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt \
        --input_dir "$STAGE_DIR" \
        --output_dir "$OUT_DIR" \
        --skip_existing \
        --t_list 0.999 0.769 0.0 \
        --local_attn_size 7 \
        --sink_size 1 \
        --use_dynamic_rope \
        --latentsync \
        --face_cache_dir /home/work/.local/HDTF/face_cache \
    && echo "[GPU $GPU_ID | step $STEP] DONE" \
    || echo "[GPU $GPU_ID | step $STEP] FAILED (see log for stack trace)"
}

for i in "${!CKPT_STEPS[@]}"; do
    run_ckpt "${GPUS[$i]}" "${CKPT_STEPS[$i]}" &
done
wait

echo ""
echo "============================================="
echo "  14B batched inference complete."
echo "  Outputs: $OUT_ROOT/"
echo "============================================="
