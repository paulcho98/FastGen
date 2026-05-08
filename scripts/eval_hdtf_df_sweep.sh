#!/bin/bash
# Evaluate ALL DF attention config sweep outputs on HDTF.
# 25 configs (5 attn configs × 5 checkpoints), distributed across GPUs.
#
# Usage: nohup bash scripts/eval_hdtf_df_sweep.sh > /tmp/eval_hdtf_df_sweep.log 2>&1 &
set -uo pipefail

export PATH="/home/work/.local/miniconda3/envs/latentsync-metrics/bin:$PATH"

SWEEP_DIR="/home/work/output_hdtf_df_attn_sweep"
GT_DIR="/home/work/.local/HDTF/HDTF_original_testset_81frames/videos_cfr"
EVAL_METRICS_DIR="/home/work/.local/eval_metrics"
SHAPE_PREDICTOR="${EVAL_METRICS_DIR}/shape_predictor_68_face_landmarks.dat"
METRICS_OUT="${SWEEP_DIR}/metrics"
GPUS=(${EVAL_GPUS:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}

mapfile -t CONFIG_DIRS < <(ls -d "$SWEEP_DIR"/df_* 2>/dev/null | sort)
NUM_CONFIGS=${#CONFIG_DIRS[@]}

if [ "$NUM_CONFIGS" -eq 0 ]; then
    echo "ERROR: No df_* directories found in $SWEEP_DIR"
    exit 1
fi

echo "============================================="
echo "  DF Sweep Evaluation: ${NUM_CONFIGS} configs × ${NUM_GPUS} GPUs"
echo "============================================="

mkdir -p "$METRICS_OUT"

eval_config() {
    local GPU_ID=$1
    local CONFIG_DIR=$2
    local CONFIG_NAME=$(basename "$CONFIG_DIR")
    local OUT="${METRICS_OUT}/${CONFIG_NAME}"
    mkdir -p "$OUT"

    # Skip if eval already complete
    if [ -f "$OUT/gt_aligned/metrics.log" ] && grep -q "FVD:" "$OUT/gt_aligned/metrics.log" 2>/dev/null; then
        echo "[GPU $GPU_ID] ${CONFIG_NAME} already complete — skipping"
        return
    fi

    echo "[GPU $GPU_ID] Evaluating ${CONFIG_NAME} ..."

    pushd "$EVAL_METRICS_DIR" > /dev/null

    # Pass 1: SyncNet + LMD + CSIM
    echo "[GPU $GPU_ID | ${CONFIG_NAME}] Pass 1: SyncNet, LMD, CSIM"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    bash eval/run_metrics.sh \
        --real_videos_dir "$GT_DIR" \
        --fake_videos_dir "$CONFIG_DIR" \
        --shape_predictor_path "$SHAPE_PREDICTOR" \
        --output_dir "$OUT/composited" \
        --log_path "$OUT/composited/metrics.log" \
        --fallback_detection_confidence 0.2 \
        --fake_videos_top_level \
        --syncnet \
        --ssim-lmd \
        --csim \
    || echo "[GPU $GPU_ID | ${CONFIG_NAME}] Pass 1 had failures (continuing)"

    # Pass 2: GT-aligned SSIM, FID, FVD
    echo "[GPU $GPU_ID | ${CONFIG_NAME}] Pass 2: GT-aligned SSIM, FID, FVD"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    bash eval/run_metrics.sh \
        --real_videos_dir "$GT_DIR" \
        --fake_videos_dir "$CONFIG_DIR" \
        --output_dir "$OUT/gt_aligned" \
        --log_path "$OUT/gt_aligned/metrics.log" \
        --fallback_detection_confidence 0.2 \
        --fake_videos_top_level \
        --gt-aligned \
    || echo "[GPU $GPU_ID | ${CONFIG_NAME}] Pass 2 had failures (continuing)"

    popd > /dev/null
    echo "[GPU $GPU_ID] ${CONFIG_NAME} done."
}

run_gpu() {
    local GPU_ID=$1
    shift
    local DIRS=("$@")
    for DIR in "${DIRS[@]}"; do
        eval_config "$GPU_ID" "$DIR"
    done
}

# Distribute round-robin
declare -A GPU_WORK
for gpu in "${GPUS[@]}"; do
    GPU_WORK[$gpu]=""
done
for i in "${!CONFIG_DIRS[@]}"; do
    gpu_idx=$((i % NUM_GPUS))
    gpu=${GPUS[$gpu_idx]}
    GPU_WORK[$gpu]+="${CONFIG_DIRS[$i]} "
done

echo ""
for gpu in "${GPUS[@]}"; do
    ckpts=(${GPU_WORK[$gpu]})
    echo "GPU $gpu (${#ckpts[@]} configs): $(printf '%s ' "${ckpts[@]##*/}")"
done
echo ""

# Launch
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
echo "  Aggregating results"
echo "============================================="

SUMMARY="${METRICS_OUT}/summary.csv"
echo "config,Sync-C,Sync-D,LMD,CSIM,SSIM_gt,FID_gt,FVD_gt" > "$SUMMARY"

for d in "$METRICS_OUT"/df_*; do
    [ -d "$d" ] || continue
    CONFIG=$(basename "$d")
    COMP_LOG="$d/composited/metrics.log"
    GT_LOG="$d/gt_aligned/metrics.log"

    SYNC_C=$(grep -oP 'Mean SyncNet Confidence.*?:\s*\K[\d.]+' "$COMP_LOG" 2>/dev/null | head -1 || echo "N/A")
    SYNC_D=$(grep -oP 'Mean SyncNet Min Distance.*?:\s*\K[\d.]+' "$COMP_LOG" 2>/dev/null | head -1 || echo "N/A")
    LMD=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$d/composited/ssim_lmd_per_video.log" 2>/dev/null || echo "N/A")
    CSIM=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$COMP_LOG" 2>/dev/null | head -1 || echo "N/A")
    GT_SSIM=$(grep -oP '^\s*SSIM:\s*\K[\d.]+' "$GT_LOG" 2>/dev/null | tail -1 || echo "N/A")
    GT_FID=$(grep -oP '^\s*FID:\s*\K[\d.]+' "$GT_LOG" 2>/dev/null | tail -1 || echo "N/A")
    GT_FVD=$(grep -oP '^\s*FVD:\s*\K[\d.]+' "$GT_LOG" 2>/dev/null | tail -1 || echo "N/A")

    echo "${CONFIG},${SYNC_C},${SYNC_D},${LMD},${CSIM},${GT_SSIM},${GT_FID},${GT_FVD}" >> "$SUMMARY"
done

echo ""
column -t -s',' "$SUMMARY"
echo ""
echo "Saved to: $SUMMARY"
echo "All done."
