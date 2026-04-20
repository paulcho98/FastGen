#!/bin/bash
# =============================================================================
# Evaluation sweep for Re-DMD β=2 + TAEW audio-fix run.
# Runs the standard FastGen eval pipeline (scripts/eval_correct.sh) across
# all /home/work/output_hdtf_sf_redmd_beta2_taew/step_* dirs, one per GPU.
#
# Results go to /home/work/.local/hyunbin/FastGen/eval_results/ (same tree as
# the SF_SW baseline runs, for direct comparison).
#
# Labels: SF_RM_TAEW_s{step}  (e.g. SF_RM_TAEW_s100)
#
# Usage: nohup bash scripts/eval_redmd_taew_audiofix.sh > /tmp/eval_redmd_taew_audiofix.log 2>&1 &
# =============================================================================
set -uo pipefail

OUT_ROOT="/home/work/output_hdtf_sf_redmd_beta2_taew"
EVAL_SH="/home/work/.local/hyunbin/FastGen/scripts/eval_correct.sh"
EVAL_RESULTS="/home/work/.local/hyunbin/FastGen/eval_results"
GPUS=(${EVAL_GPUS:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}

if [ ! -x "$EVAL_SH" ]; then
    echo "ERROR: eval_correct.sh not found or not executable: $EVAL_SH" >&2
    exit 1
fi

# --- Collect job list ---
declare -a LABELS=()
declare -a DIRS=()
for d in "$OUT_ROOT"/step_*; do
    [ -d "$d" ] || continue
    step=$(basename "$d" | sed 's/step_0*//')
    [ -z "$step" ] && step=0
    LABELS+=("SF_RM_TAEW_s${step}")
    DIRS+=("$d")
done

TOTAL=${#LABELS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: no step_* dirs found under $OUT_ROOT. Run inference first." >&2
    exit 1
fi

echo "============================================="
echo "  Re-DMD β=2 TAEW (audiofix) — Eval Sweep"
echo "============================================="
echo "  $TOTAL runs across ${NUM_GPUS} GPUs"
for i in "${!LABELS[@]}"; do
    echo "    ${LABELS[$i]} -> ${DIRS[$i]}"
done
echo "============================================="
echo ""

# --- Round-robin dispatch across GPUs ---
run_gpu() {
    local gpu=$1
    shift
    while [ $# -ge 2 ]; do
        local label="$1"
        local dir="$2"
        shift 2
        echo "[GPU $gpu] eval $label"
        bash "$EVAL_SH" "$dir" "$label" "$gpu"
    done
}

declare -A GPU_ARGS
for gpu in "${GPUS[@]}"; do GPU_ARGS[$gpu]=""; done
for i in "${!LABELS[@]}"; do
    gpu=${GPUS[$((i % NUM_GPUS))]}
    GPU_ARGS[$gpu]+="${LABELS[$i]} ${DIRS[$i]} "
done

for gpu in "${GPUS[@]}"; do
    args=(${GPU_ARGS[$gpu]})
    [ ${#args[@]} -gt 0 ] && run_gpu "$gpu" "${args[@]}" &
done
wait

# --- Aggregate this run's metrics into a CSV ---
echo ""
echo "============================================="
echo "  Results"
echo "============================================="

SUMMARY="$EVAL_RESULTS/summary_redmd_taew_audiofix.csv"
echo "method,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD,FID_aligned,SSIM_aligned,FVD_aligned" > "$SUMMARY"

for i in "${!LABELS[@]}"; do
    LABEL="${LABELS[$i]}"
    STD="$EVAL_RESULTS/metrics_standard/$LABEL"
    AC="$EVAL_RESULTS/metrics_gt_aligned/$LABEL"
    LOG="$STD/metrics.log"
    LMD_LOG="$STD/ssim_lmd_per_video.log"
    AC_LOG="$AC/metrics_aligned.log"

    FID=$(grep -oP 'FID:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    SSIM_VAL=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$LMD_LOG" 2>/dev/null || echo "N/A")
    FVD=$(grep -oP 'FVD:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    CSIM=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    SYNC_C=$(grep -oP 'Mean SyncNet Confidence.*?:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    SYNC_D=$(grep -oP 'Mean SyncNet Min Distance.*?:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    LMD=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$LMD_LOG" 2>/dev/null || echo "N/A")
    FID_A=$(grep -oP 'FID:\s*\K[\d.]+' "$AC_LOG" 2>/dev/null || echo "N/A")
    SSIM_A=$(grep -oP 'SSIM:\s*\K[\d.]+' "$AC_LOG" 2>/dev/null || echo "N/A")
    FVD_A=$(grep -oP 'FVD:\s*\K[\d.]+' "$AC_LOG" 2>/dev/null || echo "N/A")

    echo "${LABEL},${FID},${SSIM_VAL},${FVD},${CSIM},${SYNC_C},${SYNC_D},${LMD},${FID_A},${SSIM_A},${FVD_A}" >> "$SUMMARY"
done

echo ""
echo "=== Metrics (standard + GT-aligned) ==="
column -t -s',' "$SUMMARY"
echo ""
echo "Saved: $SUMMARY"
echo "All done."
