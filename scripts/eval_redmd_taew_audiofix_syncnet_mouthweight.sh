#!/bin/bash
# =============================================================================
# Evaluation sweep for Re-DMD beta=2 + TAEW + syncnet-DF + mouthweight-teacher
# SF run. Inline two-pass eval (no detour through eval_correct.sh — that script
# hardcodes --all which includes redundant FVD/FID/SSIM that we skip here):
#
#   Pass 1 (standard, face-agnostic): CSIM + SyncNet + LMD-only
#     - Skips FVD, FID (redundant with GT-aligned pass 2)
#     - Skips standard SSIM (expensive mediapipe-crop + skimage SSIM per
#       frame — use SSIM_aligned from pass 2 instead)
#     - LMD is computed via a dedicated scripts/eval/eval_lmd_only.py helper
#       that reuses compute_lmd() from eval_metrics but does NOT compute SSIM
#   Pass 2 (GT-aligned): FID_aligned, SSIM_aligned, FVD_aligned
#     - Same as eval_correct.sh pass 2, unchanged.
#
# Sequential on GPU 3 by default. Override via EVAL_GPUS="0 1 2 3" for parallel.
#
# Labels: SF_RM_TAEW_SNM_s{step}. The _fsmatched follow-up run should use a
# distinct prefix (e.g. SF_RM_TAEW_SNM_FSM_s{step}) so results don't collide.
#
# Results:
#   /home/work/.local/hyunbin/FastGen/eval_results/metrics_standard/SF_RM_TAEW_SNM_sNNN/
#   /home/work/.local/hyunbin/FastGen/eval_results/metrics_gt_aligned/SF_RM_TAEW_SNM_sNNN/
#   /home/work/.local/hyunbin/FastGen/eval_results/summary_redmd_taew_audiofix_syncnet_mouthweight.csv
#
# Usage:
#   nohup bash scripts/eval_redmd_taew_audiofix_syncnet_mouthweight.sh \
#     > /tmp/eval_redmd_taew_audiofix_syncnet_mouthweight.log 2>&1 &
# =============================================================================
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

OUT_ROOT="/home/work/output_hdtf_sf_redmd_beta2_taew_syncnet_mouthweight"
EVAL_RESULTS="/home/work/.local/hyunbin/FastGen/eval_results"
# Default: sequential on GPU 3. Override via e.g. EVAL_GPUS="0 1 2" for parallel.
GPUS=(${EVAL_GPUS:-3})
NUM_GPUS=${#GPUS[@]}

# --- Shared paths (copied from /home/work/.local/hyunbin/FastGen/scripts/eval_correct.sh) ---
CONDA_METRICS="/home/work/.local/miniconda3/envs/latentsync-metrics/bin"
METRICS_REPO="/home/work/.local/eval_metrics"
METRICS_PYTHON="${CONDA_METRICS}/python"
SHAPE_PRED="${METRICS_REPO}/shape_predictor_68_face_landmarks.dat"
REAL="/home/work/.local/OmniAvatar/demo_out/comprehensive_eval/originals/hdtf"
ARCFACE_WEIGHT="${METRICS_REPO}/checkpoints/auxiliary/models/arcface/ms1mv3_arcface_r100_fp16.pth"
ARCFACE_DIR="${METRICS_REPO}/arcface_torch"
I3D_PATH="${METRICS_REPO}/checkpoints/auxiliary/i3d_torchscript.pt"
EVAL_ALIGNED="/home/work/.local/OmniAvatar/scripts/eval_aligned_crops.py"

# --- Collect runs ---
declare -a LABELS=()
declare -a DIRS=()
for d in "$OUT_ROOT"/step_*; do
    [ -d "$d" ] || continue
    step=$(basename "$d" | sed 's/step_0*//')
    [ -z "$step" ] && step=0
    LABELS+=("SF_RM_TAEW_SNM_s${step}")
    DIRS+=("$d")
done

TOTAL=${#LABELS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: no step_* dirs found under $OUT_ROOT. Run inference first." >&2
    exit 1
fi

echo "============================================="
echo "  Re-DMD beta=2 TAEW (syncnet + mouthweight) — Eval Sweep (light)"
echo "  Pass 1: CSIM + LMD + SyncNet (FVD/FID skipped)"
echo "  Pass 2: GT-aligned FID/SSIM/FVD"
echo "============================================="
echo "  $TOTAL runs across ${NUM_GPUS} GPU(s): ${GPUS[*]}"
for i in "${!LABELS[@]}"; do
    echo "    ${LABELS[$i]} -> ${DIRS[$i]}"
done
echo "============================================="
echo ""

# --- Per-label two-pass eval ---
eval_one() {
    local FAKE_DIR="$1"
    local LABEL="$2"
    local GPU_ID="$3"

    local STD_DIR="${EVAL_RESULTS}/metrics_standard/${LABEL}"
    local AC_DIR="${EVAL_RESULTS}/metrics_gt_aligned/${LABEL}"

    # --- Prep temp dir with only composited .mp4 (exclude _aligned) ---
    local TMP_DIR="/tmp/eval_snm_${LABEL}"
    rm -rf "$TMP_DIR"
    mkdir -p "$TMP_DIR"
    for f in "$FAKE_DIR"/*.mp4; do
        [ -f "$f" ] || continue
        base=$(basename "$f")
        echo "$base" | grep -q "_aligned" && continue
        ln -sf "$f" "$TMP_DIR/$base"
    done

    local VID_COUNT=$(ls "$TMP_DIR"/*.mp4 2>/dev/null | wc -l)
    if [ "$VID_COUNT" -lt 10 ]; then
        echo "[GPU $GPU_ID | $LABEL] SKIP (only $VID_COUNT videos)"
        return 0
    fi
    echo "[GPU $GPU_ID | $LABEL] Starting evaluation ($VID_COUNT videos)"

    # --- Pass 1: CSIM + SyncNet + LMD-only (SSIM skipped) ---
    local STD_DONE=false
    grep -q "completed\|Completed" "$STD_DIR/metrics.log" 2>/dev/null && STD_DONE=true
    if ! $STD_DONE; then
        rm -rf "$STD_DIR"
        mkdir -p "$STD_DIR"
        echo "[GPU $GPU_ID | $LABEL] Pass 1a: CSIM + SyncNet (no FVD/FID/SSIM)..."
        pushd "$METRICS_REPO" > /dev/null
        CUDA_VISIBLE_DEVICES=$GPU_ID PATH="${CONDA_METRICS}:$PATH" \
            bash eval/run_metrics.sh \
            --real_videos_dir "$REAL" \
            --fake_videos_dir "$TMP_DIR" \
            --output_dir "$STD_DIR" --log_path "$STD_DIR/metrics.log" \
            --fallback_detection_confidence 0.2 \
            --csim --syncnet \
            > "$STD_DIR/eval.log" 2>&1
        popd > /dev/null

        # CSIM fallback (sometimes the combined flag misses it)
        if ! grep -q "cosine similarity" "$STD_DIR/metrics.log" 2>/dev/null; then
            CUDA_VISIBLE_DEVICES=$GPU_ID ARCFACE_TORCH_DIR="$ARCFACE_DIR" \
                ${METRICS_PYTHON} "${METRICS_REPO}/eval/eval_csim.py" \
                --real_videos_dir "$REAL" \
                --fake_videos_dir "$TMP_DIR" \
                --weight "$ARCFACE_WEIGHT" \
                --arcface_dir "$ARCFACE_DIR" \
                --model_name r100 --batch_size 512 \
                2>&1 | tee "$STD_DIR/csim_rerun.log"
            csim=$(grep "cosine similarity:" "$STD_DIR/csim_rerun.log" 2>/dev/null | awk '{print $3}')
            [ -n "$csim" ] && echo "cosine similarity: $csim" >> "$STD_DIR/metrics.log"
        fi

        # Pass 1b: LMD-only (dlib landmarks, no SSIM compute)
        echo "[GPU $GPU_ID | $LABEL] Pass 1b: LMD-only (dlib)..."
        CUDA_VISIBLE_DEVICES=$GPU_ID ${METRICS_PYTHON} \
            /home/work/.local/hyunbin/FastGen-redmd/scripts/eval/eval_lmd_only.py \
            --real_videos_dir "$REAL" \
            --fake_videos_dir "$TMP_DIR" \
            --shape_predictor_path "$SHAPE_PRED" \
            --log_path "$STD_DIR/ssim_lmd_per_video.log" \
            > "$STD_DIR/lmd_only.log" 2>&1

        echo "[GPU $GPU_ID | $LABEL] Pass 1 done"
    else
        echo "[GPU $GPU_ID | $LABEL] Pass 1 already complete — skipping"
    fi

    # --- Pass 2: GT-aligned metrics ---
    local AC_DONE=false
    [ -f "$AC_DIR/metrics_aligned.log" ] && AC_DONE=true
    if ! $AC_DONE; then
        rm -rf "$AC_DIR"
        mkdir -p "$AC_DIR"
        echo "[GPU $GPU_ID | $LABEL] Pass 2: GT-aligned (SSIM, FID, FVD)..."
        CUDA_VISIBLE_DEVICES=$GPU_ID $METRICS_PYTHON "$EVAL_ALIGNED" \
            --real_videos_dir "$REAL" \
            --fake_videos_dir "$TMP_DIR" \
            --output_dir "$AC_DIR" \
            --device cuda:0 --metrics ssim fid fvd \
            --i3d_path "$I3D_PATH" \
            > "$AC_DIR/eval.log" 2>&1
        echo "[GPU $GPU_ID | $LABEL] Pass 2 done"
    else
        echo "[GPU $GPU_ID | $LABEL] Pass 2 already complete — skipping"
    fi

    # --- Cleanup ---
    rm -rf "$TMP_DIR"
    echo "[GPU $GPU_ID | $LABEL] ALL DONE"
}

# --- Dispatch: sequential per GPU (round-robin if multiple GPUs given) ---
run_gpu() {
    local gpu=$1
    shift
    while [ $# -ge 2 ]; do
        local label="$1"
        local dir="$2"
        shift 2
        echo "[GPU $gpu] eval $label"
        eval_one "$dir" "$label" "$gpu"
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

# --- Aggregate ---
echo ""
echo "============================================="
echo "  Results"
echo "============================================="

SUMMARY="$EVAL_RESULTS/summary_redmd_taew_audiofix_syncnet_mouthweight.csv"
echo "method,CSIM,Sync-C,Sync-D,LMD,FID_aligned,SSIM_aligned,FVD_aligned" > "$SUMMARY"

for i in "${!LABELS[@]}"; do
    LABEL="${LABELS[$i]}"
    STD="$EVAL_RESULTS/metrics_standard/$LABEL"
    AC="$EVAL_RESULTS/metrics_gt_aligned/$LABEL"
    LOG="$STD/metrics.log"
    LMD_LOG="$STD/ssim_lmd_per_video.log"
    AC_LOG="$AC/metrics_aligned.log"

    CSIM=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    SYNC_C=$(grep -oP 'Mean SyncNet Confidence.*?:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    SYNC_D=$(grep -oP 'Mean SyncNet Min Distance.*?:\s*\K[\d.]+' "$LOG" 2>/dev/null | head -1 || echo "N/A")
    LMD=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$LMD_LOG" 2>/dev/null || echo "N/A")
    FID_A=$(grep -oP 'FID:\s*\K[\d.]+' "$AC_LOG" 2>/dev/null || echo "N/A")
    SSIM_A=$(grep -oP 'SSIM:\s*\K[\d.]+' "$AC_LOG" 2>/dev/null || echo "N/A")
    FVD_A=$(grep -oP 'FVD:\s*\K[\d.]+' "$AC_LOG" 2>/dev/null || echo "N/A")

    echo "${LABEL},${CSIM},${SYNC_C},${SYNC_D},${LMD},${FID_A},${SSIM_A},${FVD_A}" >> "$SUMMARY"
done

echo ""
echo "=== Metrics ==="
column -t -s',' "$SUMMARY"
echo ""
echo "Saved: $SUMMARY"
echo "All done."
