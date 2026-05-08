#!/bin/bash
# Parallel evaluation sweep using the CORRECT pipeline (matching OmniAvatar baseline).
# Distributes across GPUs round-robin.
#
# Usage: nohup bash scripts/eval_correct_sweep.sh > /tmp/eval_correct_sweep.log 2>&1 &
set -uo pipefail

GPUS=(${EVAL_GPUS:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}

# --- Build job list ---
declare -a LABELS=()
declare -a DIRS=()

# SW SF checkpoints
SW_ROOT="/home/work/output_hdtf_sf_sw"
for d in "$SW_ROOT"/step_*; do
    [ -d "$d" ] || continue
    step=$(basename "$d" | sed 's/step_//')
    LABELS+=("SF_SW_s${step##+(0)}")
    DIRS+=("$d")
done

TOTAL=${#LABELS[@]}
echo "============================================="
echo "  Correct Eval Sweep: $TOTAL runs × ${NUM_GPUS} GPUs"
echo "============================================="

for i in "${!LABELS[@]}"; do
    echo "  ${LABELS[$i]} -> ${DIRS[$i]}"
done
echo ""

# --- Run ---
run_gpu() {
    local gpu=$1
    shift
    while [ $# -ge 2 ]; do
        local label="$1"
        local dir="$2"
        shift 2
        bash scripts/eval_correct.sh "$dir" "$label" "$gpu"
    done
}

# Build per-GPU work
declare -A GPU_ARGS
for gpu in "${GPUS[@]}"; do
    GPU_ARGS[$gpu]=""
done
for i in "${!LABELS[@]}"; do
    gpu_idx=$((i % NUM_GPUS))
    gpu=${GPUS[$gpu_idx]}
    GPU_ARGS[$gpu]+="${LABELS[$i]} ${DIRS[$i]} "
done

for gpu in "${GPUS[@]}"; do
    args=(${GPU_ARGS[$gpu]})
    if [ ${#args[@]} -gt 0 ]; then
        run_gpu "$gpu" "${args[@]}" &
    fi
done

wait

# --- Aggregate ---
echo ""
echo "============================================="
echo "  Aggregating results"
echo "============================================="

OUT_ROOT="/home/work/.local/hyunbin/FastGen/eval_results"

# Standard summary
STD_SUMMARY="${OUT_ROOT}/summary_standard.csv"
echo "method,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD" > "$STD_SUMMARY"

for d in "$OUT_ROOT"/metrics_standard/*; do
    [ -d "$d" ] || continue
    LABEL=$(basename "$d")
    LOG="$d/metrics.log"
    LMD_LOG="$d/ssim_lmd_per_video.log"

    FID=$(grep -oP 'FID:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    SSIM_VAL=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$LMD_LOG" 2>/dev/null || echo "N/A")
    FVD=$(grep -oP 'FVD:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    CSIM=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    SYNC_C=$(grep -oP 'Mean SyncNet Confidence.*?:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    SYNC_D=$(grep -oP 'Mean SyncNet Min Distance.*?:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    LMD=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$LMD_LOG" 2>/dev/null || echo "N/A")

    echo "${LABEL},${FID},${SSIM_VAL},${FVD},${CSIM},${SYNC_C},${SYNC_D},${LMD}" >> "$STD_SUMMARY"
done

# GT-aligned summary
AC_SUMMARY="${OUT_ROOT}/summary_gt_aligned.csv"
echo "method,FID,SSIM,FVD" > "$AC_SUMMARY"

for d in "$OUT_ROOT"/metrics_gt_aligned/*; do
    [ -d "$d" ] || continue
    LABEL=$(basename "$d")
    LOG="$d/metrics_aligned.log"

    FID=$(grep -oP 'FID:\s*\K[\d.]+' "$LOG" 2>/dev/null || echo "N/A")
    SSIM_VAL=$(grep -oP 'SSIM:\s*\K[\d.]+' "$LOG" 2>/dev/null || echo "N/A")
    FVD=$(grep -oP 'FVD:\s*\K[\d.]+' "$LOG" 2>/dev/null || echo "N/A")

    echo "${LABEL},${FID},${SSIM_VAL},${FVD}" >> "$AC_SUMMARY"
done

echo ""
echo "=== Standard Metrics ==="
column -t -s',' "$STD_SUMMARY"
echo ""
echo "=== GT-Aligned Metrics ==="
column -t -s',' "$AC_SUMMARY"
echo ""
echo "Saved to: $STD_SUMMARY, $AC_SUMMARY"
echo "All done."
