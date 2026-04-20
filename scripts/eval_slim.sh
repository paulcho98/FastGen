#!/bin/bash
# =============================================================================
# Slim evaluation pipeline: GT-aligned FID/SSIM/FVD + non-aligned CSIM/Sync/LMD.
# Skips non-aligned FID/FVD/SSIM because their GT-aligned counterparts (face
# crops only) are the cleaner signal for lip-sync quality.
#
# Two passes:
#   Pass A: run_metrics.sh --csim --syncnet --ssim-lmd  (CSIM, Sync-C, Sync-D, LMD)
#     (SSIM from ssim-lmd is non-aligned and ignored when reading results.)
#   Pass B: eval_aligned_crops.py  (GT-aligned SSIM, FID, FVD)
#
# Usage:
#   bash scripts/eval_slim.sh <fake_videos_dir> <label> [gpu_id]
# =============================================================================
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

TMP_DIR="/tmp/eval_slim_${LABEL}"
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

echo "[GPU $GPU_ID | $LABEL] Starting SLIM evaluation ($VID_COUNT videos)"

# --- Pass A: CSIM + SyncNet + SSIM-LMD ---
rm -rf "$STD_DIR"
mkdir -p "$STD_DIR"
echo "[GPU $GPU_ID | $LABEL] Pass A: CSIM + SyncNet + SSIM-LMD (skipping FVD, FID) ..."
pushd "$METRICS_REPO" > /dev/null
CUDA_VISIBLE_DEVICES=$GPU_ID PATH="${CONDA_METRICS}:$PATH" \
    bash eval/run_metrics.sh \
    --real_videos_dir "$REAL" \
    --fake_videos_dir "$TMP_DIR" \
    --shape_predictor_path "$SHAPE_PRED" \
    --output_dir "$STD_DIR" --log_path "$STD_DIR/metrics.log" \
    --fallback_detection_confidence 0.2 \
    --csim --syncnet --ssim-lmd \
    > "$STD_DIR/eval.log" 2>&1

# CSIM fallback (sometimes it misses; mirror logic from eval_correct.sh)
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
popd > /dev/null

# SyncNet fallback: run_metrics.sh --syncnet silently fails in some envs.
# Run eval_sync.py directly — it just needs --videos_dir.
if ! grep -q "Mean SyncNet Confidence" "$STD_DIR/metrics.log" 2>/dev/null; then
    echo "[GPU $GPU_ID | $LABEL] SyncNet (direct fallback) ..."
    pushd "$METRICS_REPO" > /dev/null
    CUDA_VISIBLE_DEVICES=$GPU_ID ${METRICS_PYTHON} eval/eval_sync.py \
        --videos_dir "$TMP_DIR" \
        2>&1 | tee "$STD_DIR/syncnet_direct.log"
    # Append summary lines to metrics.log
    sync_c=$(grep "Mean SyncNet Confidence" "$STD_DIR/syncnet_direct.log" 2>/dev/null)
    sync_d=$(grep "Mean SyncNet Min Distance" "$STD_DIR/syncnet_direct.log" 2>/dev/null)
    [ -n "$sync_c" ] && echo "$sync_c" >> "$STD_DIR/metrics.log"
    [ -n "$sync_d" ] && echo "$sync_d" >> "$STD_DIR/metrics.log"
    popd > /dev/null
fi
echo "[GPU $GPU_ID | $LABEL] Pass A done"

# --- Pass B: GT-aligned (SSIM, FID, FVD on aligned crops) ---
rm -rf "$AC_DIR"
mkdir -p "$AC_DIR"
echo "[GPU $GPU_ID | $LABEL] Pass B: GT-aligned (SSIM, FID, FVD) ..."
CUDA_VISIBLE_DEVICES=$GPU_ID $METRICS_PYTHON "$EVAL_ALIGNED" \
    --real_videos_dir "$REAL" \
    --fake_videos_dir "$TMP_DIR" \
    --output_dir "$AC_DIR" \
    --device cuda:0 --metrics ssim fid fvd \
    --i3d_path "$I3D_PATH" \
    > "$AC_DIR/eval.log" 2>&1
echo "[GPU $GPU_ID | $LABEL] Pass B done"

rm -rf "$TMP_DIR"
echo "[GPU $GPU_ID | $LABEL] ALL DONE"
