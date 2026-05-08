#!/bin/bash
# Inference + correct evaluation for the Re-DMD + SyncNet reward run.
# (sf_sink1_window7_redmd_syncc_beta0p25_joonson_parity)
#
# Usage: nohup bash scripts/infer_eval_redmd.sh > /tmp/infer_eval_redmd.log 2>&1 &
set -uo pipefail

CKPT_DIR="/tmp/FASTGEN_SF_OUTPUT/OmniAvatar-FastGen/omniavatar_sf/sf_sink1_window7_redmd_syncc_beta0p25_joonson_parity/checkpoints"
HDTF=/home/work/.local/HDTF/HDTF_original_testset_81frames
TEXT_EMB=/home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt
OUT_ROOT="/home/work/output_hdtf_sf_redmd"
GPUS=(${INFER_GPUS:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
TOTAL_VIDEOS=$(ls "$HDTF"/videos_cfr/*.mp4 | wc -l)

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"

mapfile -t ALL_CKPTS < <(ls "$CKPT_DIR"/*.pth 2>/dev/null | sort)

echo "============================================="
echo "  Re-DMD Sweep: ${#ALL_CKPTS[@]} ckpts × ${NUM_GPUS} GPUs"
echo "============================================="

# =============================================
#  Phase 1: Inference
# =============================================
run_inference() {
    local GPU_ID=$1
    local CKPT=$2
    local STEP=$(basename "$CKPT" .pth)
    local OUT_DIR="$OUT_ROOT/step_${STEP}"

    # Skip if complete
    local n=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | grep -v aligned | wc -l)
    if [ "$n" -ge "$TOTAL_VIDEOS" ]; then
        echo "[GPU $GPU_ID | step $STEP] Inference complete ($n/$TOTAL_VIDEOS) — skipping"
        return
    fi

    mkdir -p "$OUT_DIR"
    echo "[GPU $GPU_ID | step $STEP] Starting inference..."

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
        || echo "  FAILED: $name (continuing)"
    done
    echo "[GPU $GPU_ID | step $STEP] Inference done"
}

run_gpu_infer() {
    local GPU_ID=$1
    shift
    for CKPT in "$@"; do
        run_inference "$GPU_ID" "$CKPT"
    done
}

# Distribute inference round-robin
declare -A GPU_INFER
for gpu in "${GPUS[@]}"; do GPU_INFER[$gpu]=""; done
for i in "${!ALL_CKPTS[@]}"; do
    gpu=${GPUS[$((i % NUM_GPUS))]}
    GPU_INFER[$gpu]+="${ALL_CKPTS[$i]} "
done

echo ""
echo "--- Inference distribution ---"
for gpu in "${GPUS[@]}"; do
    ckpts=(${GPU_INFER[$gpu]})
    names=""; for c in "${ckpts[@]}"; do names+="$(basename $c .pth) "; done
    echo "GPU $gpu (${#ckpts[@]}): $names"
done
echo ""

for gpu in "${GPUS[@]}"; do
    ckpts=(${GPU_INFER[$gpu]})
    [ ${#ckpts[@]} -gt 0 ] && run_gpu_infer "$gpu" "${ckpts[@]}" &
done
wait

echo ""
echo "============================================="
echo "  Phase 1 complete. Starting evaluation."
echo "============================================="

# =============================================
#  Phase 2: Correct evaluation
# =============================================
declare -a EVAL_LABELS=()
declare -a EVAL_DIRS=()
for CKPT in "${ALL_CKPTS[@]}"; do
    STEP=$(basename "$CKPT" .pth)
    EVAL_LABELS+=("ReDMD_s${STEP}")
    EVAL_DIRS+=("$OUT_ROOT/step_${STEP}")
done

run_gpu_eval() {
    local gpu=$1
    shift
    while [ $# -ge 2 ]; do
        local label="$1"
        local dir="$2"
        shift 2
        bash scripts/eval_correct.sh "$dir" "$label" "$gpu"
    done
}

declare -A GPU_EVAL
for gpu in "${GPUS[@]}"; do GPU_EVAL[$gpu]=""; done
for i in "${!EVAL_LABELS[@]}"; do
    gpu=${GPUS[$((i % NUM_GPUS))]}
    GPU_EVAL[$gpu]+="${EVAL_LABELS[$i]} ${EVAL_DIRS[$i]} "
done

for gpu in "${GPUS[@]}"; do
    args=(${GPU_EVAL[$gpu]})
    [ ${#args[@]} -gt 0 ] && run_gpu_eval "$gpu" "${args[@]}" &
done
wait

# =============================================
#  Aggregate
# =============================================
echo ""
echo "============================================="
echo "  Results"
echo "============================================="

OUT_EVAL="/home/work/.local/hyunbin/FastGen/eval_results"

echo ""
echo "=== Standard Metrics ==="
echo "method,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD"
for CKPT in "${ALL_CKPTS[@]}"; do
    STEP=$(basename "$CKPT" .pth)
    LABEL="ReDMD_s${STEP}"
    LOG="$OUT_EVAL/metrics_standard/${LABEL}/metrics.log"
    LMD_LOG="$OUT_EVAL/metrics_standard/${LABEL}/ssim_lmd_per_video.log"

    FID=$(grep -oP 'FID:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    SSIM_VAL=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$LMD_LOG" 2>/dev/null || echo "N/A")
    FVD=$(grep -oP 'FVD:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    CSIM=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    SYNC_C=$(grep -oP 'Mean SyncNet Confidence.*?:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    SYNC_D=$(grep -oP 'Mean SyncNet Min Distance.*?:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    LMD=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$LMD_LOG" 2>/dev/null || echo "N/A")

    echo "${LABEL},${FID},${SSIM_VAL},${FVD},${CSIM},${SYNC_C},${SYNC_D},${LMD}"
done | column -t -s','

echo ""
echo "=== GT-Aligned Metrics ==="
echo "method,FID,SSIM,FVD"
for CKPT in "${ALL_CKPTS[@]}"; do
    STEP=$(basename "$CKPT" .pth)
    LABEL="ReDMD_s${STEP}"
    LOG="$OUT_EVAL/metrics_gt_aligned/${LABEL}/metrics_aligned.log"

    FID=$(grep -oP 'FID:\s*\K[\d.]+' "$LOG" 2>/dev/null || echo "N/A")
    SSIM_VAL=$(grep -oP 'SSIM:\s*\K[\d.]+' "$LOG" 2>/dev/null || echo "N/A")
    FVD=$(grep -oP 'FVD:\s*\K[\d.]+' "$LOG" 2>/dev/null || echo "N/A")

    echo "${LABEL},${FID},${SSIM_VAL},${FVD}"
done | column -t -s','

echo ""
echo "All done."
