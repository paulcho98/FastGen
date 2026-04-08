#!/bin/bash
# Evaluate all SF checkpoint inference outputs on HDTF.
#
# Two evaluation passes per checkpoint:
#   1. Composited vs full-res GT → SyncNet (Sync-C/D), LMD, CSIM
#   2. Composited vs full-res GT → GT-aligned SSIM, FID, FVD
#
# Distributes checkpoints across 4 GPUs (round-robin).
#
# Usage: bash scripts/eval_hdtf_sf_sweep.sh
#   or:  nohup bash scripts/eval_hdtf_sf_sweep.sh > /tmp/eval_sf_sweep.log 2>&1 &
set -uo pipefail

# Eval scripts need latentsync-metrics env (has cv2, dlib, mediapipe, pytorch-fid, etc.)
export PATH="/home/work/.local/miniconda3/envs/latentsync-metrics/bin:$PATH"

SWEEP_DIR="${1:-/home/work/output_hdtf_sf_sweep_v2}"
GT_DIR="/home/work/.local/HDTF/HDTF_original_testset_81frames/videos_cfr"
EVAL_METRICS_DIR="/home/work/.local/eval_metrics"
SHAPE_PREDICTOR="${EVAL_METRICS_DIR}/shape_predictor_68_face_landmarks.dat"
METRICS_OUT="${SWEEP_DIR}/metrics"
# GPU list — skip any GPUs that are full (e.g., GPU 0 occupied by other processes)
GPUS=(${EVAL_GPUS:-1 2 3})
NUM_GPUS=${#GPUS[@]}

# Collect checkpoint dirs
mapfile -t CKPT_DIRS < <(ls -d "$SWEEP_DIR"/step_* 2>/dev/null | sort)
NUM_CKPTS=${#CKPT_DIRS[@]}

if [ "$NUM_CKPTS" -eq 0 ]; then
    echo "ERROR: No step_* directories found in $SWEEP_DIR"
    exit 1
fi

echo "============================================="
echo "  SF Sweep Evaluation: ${NUM_CKPTS} checkpoints"
echo "  GT: ${GT_DIR}"
echo "  Output: ${METRICS_OUT}"
echo "============================================="

mkdir -p "$METRICS_OUT"

eval_checkpoint() {
    local GPU_ID=$1
    local CKPT_DIR=$2
    local STEP=$(basename "$CKPT_DIR")
    local OUT="${METRICS_OUT}/${STEP}"
    mkdir -p "$OUT"

    echo "[GPU $GPU_ID] Evaluating ${STEP} ..."

    # run_metrics.sh uses relative paths for model checkpoints (syncnet, arcface, etc.)
    # so we must cd into the eval_metrics directory
    pushd "$EVAL_METRICS_DIR" > /dev/null

    # --- Pass 1: SyncNet + LMD + CSIM on composited outputs ---
    echo "[GPU $GPU_ID | ${STEP}] Pass 1: SyncNet, LMD, CSIM"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    bash eval/run_metrics.sh \
        --real_videos_dir "$GT_DIR" \
        --fake_videos_dir "$CKPT_DIR" \
        --shape_predictor_path "$SHAPE_PREDICTOR" \
        --output_dir "$OUT/composited" \
        --log_path "$OUT/composited/metrics.log" \
        --fallback_detection_confidence 0.2 \
        --fake_videos_top_level \
        --syncnet \
        --ssim-lmd \
        --csim \
    || echo "[GPU $GPU_ID | ${STEP}] Pass 1 had failures (continuing)"

    # --- Pass 2: GT-aligned SSIM, FID, FVD ---
    echo "[GPU $GPU_ID | ${STEP}] Pass 2: GT-aligned SSIM, FID, FVD"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    bash eval/run_metrics.sh \
        --real_videos_dir "$GT_DIR" \
        --fake_videos_dir "$CKPT_DIR" \
        --output_dir "$OUT/gt_aligned" \
        --log_path "$OUT/gt_aligned/metrics.log" \
        --fallback_detection_confidence 0.2 \
        --fake_videos_top_level \
        --gt-aligned \
    || echo "[GPU $GPU_ID | ${STEP}] Pass 2 had failures (continuing)"

    popd > /dev/null

    echo "[GPU $GPU_ID] ${STEP} done."
}

run_gpu() {
    local GPU_ID=$1
    shift
    local DIRS=("$@")

    for CKPT_DIR in "${DIRS[@]}"; do
        eval_checkpoint "$GPU_ID" "$CKPT_DIR"
    done
}

# Distribute checkpoints across available GPUs (round-robin)
declare -A GPU_CKPTS
for gpu in "${GPUS[@]}"; do
    GPU_CKPTS[$gpu]=""
done
for i in "${!CKPT_DIRS[@]}"; do
    gpu_idx=$((i % NUM_GPUS))
    gpu=${GPUS[$gpu_idx]}
    GPU_CKPTS[$gpu]+="${CKPT_DIRS[$i]} "
done

echo ""
for gpu in "${GPUS[@]}"; do
    ckpts=(${GPU_CKPTS[$gpu]})
    echo "GPU $gpu (${#ckpts[@]} ckpts): $(printf '%s ' "${ckpts[@]##*/}")"
done
echo ""

# Launch all GPUs in parallel
for gpu in "${GPUS[@]}"; do
    ckpts=(${GPU_CKPTS[$gpu]})
    run_gpu "$gpu" "${ckpts[@]}" &
done

wait

# --- Aggregate results ---
echo ""
echo "============================================="
echo "  Aggregating results"
echo "============================================="

SUMMARY="${METRICS_OUT}/summary.csv"
echo "step,Sync-C,Sync-D,LMD,CSIM,SSIM_gt,FID_gt,FVD_gt" > "$SUMMARY"

for d in "$METRICS_OUT"/step_*; do
    STEP=$(basename "$d")
    COMP_LOG="$d/composited/metrics.log"
    GT_LOG="$d/gt_aligned/metrics.log"

    # Composited metrics
    SYNC_C=$(grep -oP 'Mean SyncNet Confidence.*?:\s*\K[\d.]+' "$COMP_LOG" 2>/dev/null | head -1 || echo "N/A")
    SYNC_D=$(grep -oP 'Mean SyncNet Min Distance.*?:\s*\K[\d.]+' "$COMP_LOG" 2>/dev/null | head -1 || echo "N/A")
    LMD=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$d/composited/ssim_lmd_per_video.log" 2>/dev/null || echo "N/A")
    CSIM=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$COMP_LOG" 2>/dev/null | head -1 || echo "N/A")

    # GT-aligned metrics (format: "  SSIM: 0.6654", "FID: 62.0444", "FVD: 403.4963")
    GT_SSIM=$(grep -oP '^\s*SSIM:\s*\K[\d.]+' "$GT_LOG" 2>/dev/null | tail -1 || echo "N/A")
    GT_FID=$(grep -oP '^\s*FID:\s*\K[\d.]+' "$GT_LOG" 2>/dev/null | tail -1 || echo "N/A")
    GT_FVD=$(grep -oP '^\s*FVD:\s*\K[\d.]+' "$GT_LOG" 2>/dev/null | tail -1 || echo "N/A")

    echo "${STEP},${SYNC_C},${SYNC_D},${LMD},${CSIM},${GT_SSIM},${GT_FID},${GT_FVD}" >> "$SUMMARY"
done

echo ""
echo "Results saved to: $SUMMARY"
column -t -s',' "$SUMMARY"
echo ""
echo "All done."
