#!/bin/bash
# Parallel inference + evaluation for sliding-window SF checkpoints.
# Distributes checkpoints across GPUs (round-robin). Skips completed work.
#
# Usage: nohup bash scripts/infer_eval_sf_sw_parallel.sh > /tmp/infer_eval_sf_sw_parallel.log 2>&1 &
set -uo pipefail

# --- Configuration ---
CKPT_DIR="/tmp/FASTGEN_SF_OUTPUT/OmniAvatar-FastGen/omniavatar_sf/sf_sink1_window7_redmd_syncc_beta0p25/checkpoints"
HDTF=/home/work/.local/HDTF/HDTF_original_testset_81frames
TEXT_EMB=/home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt
OUT_ROOT="/home/work/output_hdtf_sf_sw"
GPUS=(${INFER_GPUS:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"

# Eval setup
EVAL_METRICS_DIR="/home/work/.local/eval_metrics"
SHAPE_PREDICTOR="${EVAL_METRICS_DIR}/shape_predictor_68_face_landmarks.dat"
GT_DIR="$HDTF/videos_cfr"

mapfile -t ALL_CKPTS < <(ls "$CKPT_DIR"/*.pth 2>/dev/null | sort)
TOTAL_VIDEOS=$(ls "$HDTF"/videos_cfr/*.mp4 | wc -l)

echo "============================================="
echo "  SF SW Parallel Sweep: ${#ALL_CKPTS[@]} ckpts × ${NUM_GPUS} GPUs"
echo "============================================="

run_checkpoint() {
    local GPU_ID=$1
    local CKPT=$2
    local STEP=$(basename "$CKPT" .pth)
    local OUT_DIR="$OUT_ROOT/step_${STEP}"
    local METRICS_OUT="$OUT_ROOT/metrics/step_${STEP}"

    # Skip if eval already complete
    if [ -f "$METRICS_OUT/gt_aligned/metrics.log" ] && grep -q "FVD:" "$METRICS_OUT/gt_aligned/metrics.log" 2>/dev/null; then
        echo "[GPU $GPU_ID | step $STEP] Already complete — skipping"
        return
    fi

    # --- Inference ---
    mkdir -p "$OUT_DIR"
    echo "[GPU $GPU_ID | step $STEP] Starting inference..."

    local i=0
    for video in "$HDTF"/videos_cfr/*.mp4; do
        local name=$(basename "$video" _cfr25.mp4)
        i=$((i + 1))

        if [ -f "$OUT_DIR/${name}.mp4" ]; then
            continue
        fi

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

    local num_generated=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | wc -l)
    echo "[GPU $GPU_ID | step $STEP] Inference done: ${num_generated} videos"

    # --- Evaluation ---
    mkdir -p "$METRICS_OUT"
    pushd "$EVAL_METRICS_DIR" > /dev/null

    echo "[GPU $GPU_ID | step $STEP] Eval Pass 1: SyncNet, LMD, CSIM"
    export PATH="/home/work/.local/miniconda3/envs/latentsync-metrics/bin:$PATH"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    bash eval/run_metrics.sh \
        --real_videos_dir "$GT_DIR" \
        --fake_videos_dir "$OUT_DIR" \
        --shape_predictor_path "$SHAPE_PREDICTOR" \
        --output_dir "$METRICS_OUT/composited" \
        --log_path "$METRICS_OUT/composited/metrics.log" \
        --fallback_detection_confidence 0.2 \
        --fake_videos_top_level \
        --syncnet \
        --ssim-lmd \
        --csim \
    || echo "[GPU $GPU_ID | step $STEP] Pass 1 had failures (continuing)"

    echo "[GPU $GPU_ID | step $STEP] Eval Pass 2: GT-aligned SSIM, FID, FVD"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    bash eval/run_metrics.sh \
        --real_videos_dir "$GT_DIR" \
        --fake_videos_dir "$OUT_DIR" \
        --output_dir "$METRICS_OUT/gt_aligned" \
        --log_path "$METRICS_OUT/gt_aligned/metrics.log" \
        --fallback_detection_confidence 0.2 \
        --fake_videos_top_level \
        --gt-aligned \
    || echo "[GPU $GPU_ID | step $STEP] Pass 2 had failures (continuing)"

    popd > /dev/null
    echo "[GPU $GPU_ID | step $STEP] Complete."
}

run_gpu() {
    local GPU_ID=$1
    shift
    local CKPTS=("$@")
    for CKPT in "${CKPTS[@]}"; do
        run_checkpoint "$GPU_ID" "$CKPT"
    done
}

# Distribute checkpoints across GPUs, prioritizing step 600 on GPU 0
# Reorder: put 600 first, then remaining in round-robin
PRIORITIZED=()
REMAINING=()
for ckpt in "${ALL_CKPTS[@]}"; do
    if [[ "$(basename "$ckpt" .pth)" == "0000600" ]]; then
        PRIORITIZED+=("$ckpt")
    else
        REMAINING+=("$ckpt")
    fi
done
ORDERED=("${PRIORITIZED[@]}" "${REMAINING[@]}")

declare -A GPU_WORK
for gpu in "${GPUS[@]}"; do
    GPU_WORK[$gpu]=""
done
for i in "${!ORDERED[@]}"; do
    gpu_idx=$((i % NUM_GPUS))
    gpu=${GPUS[$gpu_idx]}
    GPU_WORK[$gpu]+="${ORDERED[$i]} "
done

echo ""
for gpu in "${GPUS[@]}"; do
    ckpts=(${GPU_WORK[$gpu]})
    names=""
    for c in "${ckpts[@]}"; do names+="$(basename $c .pth) "; done
    echo "GPU $gpu (${#ckpts[@]} ckpts): $names"
done
echo ""

# Launch all GPUs in parallel
for gpu in "${GPUS[@]}"; do
    ckpts=(${GPU_WORK[$gpu]})
    if [ ${#ckpts[@]} -gt 0 ]; then
        run_gpu "$gpu" "${ckpts[@]}" &
    fi
done

wait

# --- Aggregate ---
echo ""
echo "============================================="
echo "  Summary"
echo "============================================="

SUMMARY="$OUT_ROOT/metrics/summary.csv"
echo "step,Sync-C,Sync-D,LMD,CSIM,SSIM_gt,FID_gt,FVD_gt" > "$SUMMARY"

for d in "$OUT_ROOT"/metrics/step_*; do
    STEP=$(basename "$d")
    COMP_LOG="$d/composited/metrics.log"
    GT_LOG="$d/gt_aligned/metrics.log"

    SYNC_C=$(grep -oP 'Mean SyncNet Confidence.*?:\s*\K[\d.]+' "$COMP_LOG" 2>/dev/null | head -1 || echo "N/A")
    SYNC_D=$(grep -oP 'Mean SyncNet Min Distance.*?:\s*\K[\d.]+' "$COMP_LOG" 2>/dev/null | head -1 || echo "N/A")
    LMD=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$d/composited/ssim_lmd_per_video.log" 2>/dev/null || echo "N/A")
    CSIM=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$COMP_LOG" 2>/dev/null | head -1 || echo "N/A")
    GT_SSIM=$(grep -oP '^\s*SSIM:\s*\K[\d.]+' "$GT_LOG" 2>/dev/null | tail -1 || echo "N/A")
    GT_FID=$(grep -oP '^\s*FID:\s*\K[\d.]+' "$GT_LOG" 2>/dev/null | tail -1 || echo "N/A")
    GT_FVD=$(grep -oP '^\s*FVD:\s*\K[\d.]+' "$GT_LOG" 2>/dev/null | tail -1 || echo "N/A")

    echo "${STEP},${SYNC_C},${SYNC_D},${LMD},${CSIM},${GT_SSIM},${GT_FID},${GT_FVD}" >> "$SUMMARY"
done

echo ""
column -t -s',' "$SUMMARY"
echo ""
echo "Saved to: $SUMMARY"
echo "All done."
