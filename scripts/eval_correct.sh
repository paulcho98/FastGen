#!/bin/bash
# Correct evaluation pipeline matching the OmniAvatar baseline.
# Two passes:
#   1. Standard: run_metrics.sh --all (FVD, FID, CSIM, SSIM-LMD, SyncNet)
#   2. GT-aligned: eval_aligned_crops.py (SSIM, FID, FVD on GT-bbox-aligned crops)
#
# Uses the same GT, eval code, and parameters as the comprehensive_eval baseline.
#
# Usage:
#   bash scripts/eval_correct.sh <fake_videos_dir> <label> [gpu_id]
#   bash scripts/eval_correct.sh /home/work/output_hdtf_sf_sw/step_0000600 SF_SW_s600 0
#
# Parallel sweep:
#   nohup bash scripts/eval_correct_sweep.sh > /tmp/eval_correct_sweep.log 2>&1 &
set -uo pipefail

FAKE_DIR="$1"
LABEL="$2"
GPU_ID="${3:-0}"

CONDA_METRICS="/home/work/.local/miniconda3/envs/latentsync-metrics/bin"
METRICS_REPO="/home/work/.local/eval_metrics"
METRICS_PYTHON="${CONDA_METRICS}/python"
SHAPE_PRED="${METRICS_REPO}/shape_predictor_68_face_landmarks.dat"
REAL="/home/work/.local/OmniAvatar/demo_out/comprehensive_eval/originals/hdtf"
ARCFACE_WEIGHT="${METRICS_REPO}/checkpoints/auxiliary/models/arcface/ms1mv3_arcface_r100_fp16.pth"
ARCFACE_DIR="${METRICS_REPO}/arcface_torch"
I3D_PATH="${METRICS_REPO}/checkpoints/auxiliary/i3d_torchscript.pt"
EVAL_ALIGNED="/home/work/.local/OmniAvatar/scripts/eval_aligned_crops.py"

OUT_ROOT="/home/work/.local/hyunbin/FastGen/eval_results"
STD_DIR="${OUT_ROOT}/metrics_standard/${LABEL}"
AC_DIR="${OUT_ROOT}/metrics_gt_aligned/${LABEL}"

# --- Prepare temp dir with only composited .mp4 (exclude _aligned) ---
TMP_DIR="/tmp/eval_correct_${LABEL}"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
for f in "$FAKE_DIR"/*.mp4; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    echo "$base" | grep -q "_aligned" && continue
    ln -sf "$f" "$TMP_DIR/$base"
done

VID_COUNT=$(ls "$TMP_DIR"/*.mp4 2>/dev/null | wc -l)
if [ "$VID_COUNT" -lt 10 ]; then
    echo "[GPU $GPU_ID] SKIP $LABEL (only $VID_COUNT videos)"
    exit 0
fi

echo "[GPU $GPU_ID | $LABEL] Starting evaluation ($VID_COUNT videos)"

# --- Pass 1: Standard metrics ---
STD_DONE=false
grep -q "completed\|Completed" "$STD_DIR/metrics.log" 2>/dev/null && STD_DONE=true

if ! $STD_DONE; then
    rm -rf "$STD_DIR"
    mkdir -p "$STD_DIR"
    echo "[GPU $GPU_ID | $LABEL] Pass 1: Standard metrics (FVD, FID, CSIM, SSIM-LMD, SyncNet)..."
    pushd "$METRICS_REPO" > /dev/null
    CUDA_VISIBLE_DEVICES=$GPU_ID PATH="${CONDA_METRICS}:$PATH" \
        bash eval/run_metrics.sh \
        --real_videos_dir "$REAL" \
        --fake_videos_dir "$TMP_DIR" \
        --shape_predictor_path "$SHAPE_PRED" \
        --output_dir "$STD_DIR" --log_path "$STD_DIR/metrics.log" \
        --fallback_detection_confidence 0.2 --all \
        > "$STD_DIR/eval.log" 2>&1
    popd > /dev/null

    # CSIM fallback (sometimes --all misses it)
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
    echo "[GPU $GPU_ID | $LABEL] Pass 1 done"
else
    echo "[GPU $GPU_ID | $LABEL] Pass 1 already complete — skipping"
fi

# --- Pass 2: GT-aligned metrics ---
AC_DONE=false
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
