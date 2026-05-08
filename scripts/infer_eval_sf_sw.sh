#!/bin/bash
# Inference + evaluation for the sliding-window SF run (sf_sink1_window7_redmd_syncc_beta0p25).
# Uses the latest checkpoint, 2-step t_list, sliding window (sink=1, window=7), dynamic RoPE.
# Runs on a single GPU alongside training.
#
# Usage: nohup bash scripts/infer_eval_sf_sw.sh > /tmp/infer_eval_sf_sw.log 2>&1 &
set -uo pipefail

# --- Configuration ---
CKPT_DIR="/tmp/FASTGEN_SF_OUTPUT/OmniAvatar-FastGen/omniavatar_sf/sf_sink1_window7_redmd_syncc_beta0p25/checkpoints"
HDTF=/home/work/.local/HDTF/HDTF_original_testset_81frames
TEXT_EMB=/home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt
GPU_ID="${INFER_GPU:-3}"

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Find latest checkpoint
LATEST=$(ls "$CKPT_DIR"/*.pth 2>/dev/null | sort | tail -1)
if [ -z "$LATEST" ]; then
    echo "ERROR: No .pth checkpoint found in $CKPT_DIR"
    exit 1
fi
STEP=$(basename "$LATEST" .pth)
echo "Using checkpoint: $LATEST (step $STEP)"

OUT_DIR="/home/work/output_hdtf_sf_sw/step_${STEP}"
mkdir -p "$OUT_DIR"

# =============================================
#  Phase 1: Inference
# =============================================
echo ""
echo "============================================="
echo "  Phase 1: Inference (step ${STEP}, GPU ${GPU_ID})"
echo "  t_list: 0.999 0.833 0.0 (2-step)"
echo "  sliding window: sink=1, local_attn_size=7"
echo "============================================="

total=$(ls "$HDTF"/videos_cfr/*.mp4 | wc -l)
i=0
for video in "$HDTF"/videos_cfr/*.mp4; do
    name=$(basename "$video" _cfr25.mp4)
    i=$((i + 1))

    if [ -f "$OUT_DIR/${name}.mp4" ]; then
        echo "[$i/$total] $name — skipping (exists)"
        continue
    fi

    echo "[$i/$total] $name"
    /home/work/.local/miniconda3/envs/hb_fastgen/bin/python \
        scripts/inference/inference_causal.py \
        --ckpt_path "$LATEST" \
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

num_generated=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | wc -l)
echo ""
echo "Inference complete: ${num_generated}/${total} videos in $OUT_DIR"

# =============================================
#  Phase 2: Evaluation
# =============================================
echo ""
echo "============================================="
echo "  Phase 2: Evaluation (step ${STEP})"
echo "============================================="

export PATH="/home/work/.local/miniconda3/envs/latentsync-metrics/bin:$PATH"

GT_DIR="$HDTF/videos_cfr"
EVAL_METRICS_DIR="/home/work/.local/eval_metrics"
SHAPE_PREDICTOR="${EVAL_METRICS_DIR}/shape_predictor_68_face_landmarks.dat"
METRICS_OUT="/home/work/output_hdtf_sf_sw/metrics/step_${STEP}"
mkdir -p "$METRICS_OUT"

pushd "$EVAL_METRICS_DIR" > /dev/null

# Pass 1: SyncNet + LMD + CSIM
echo "[step ${STEP}] Pass 1: SyncNet, LMD, CSIM"
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
|| echo "Pass 1 had failures (continuing)"

# Pass 2: GT-aligned SSIM, FID, FVD
echo "[step ${STEP}] Pass 2: GT-aligned SSIM, FID, FVD"
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
|| echo "Pass 2 had failures (continuing)"

popd > /dev/null

# --- Print results ---
echo ""
echo "============================================="
echo "  Results: step ${STEP}"
echo "============================================="

COMP_LOG="$METRICS_OUT/composited/metrics.log"
GT_LOG="$METRICS_OUT/gt_aligned/metrics.log"

SYNC_C=$(grep -oP 'Mean SyncNet Confidence.*?:\s*\K[\d.]+' "$COMP_LOG" 2>/dev/null | head -1 || echo "N/A")
SYNC_D=$(grep -oP 'Mean SyncNet Min Distance.*?:\s*\K[\d.]+' "$COMP_LOG" 2>/dev/null | head -1 || echo "N/A")
LMD=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$METRICS_OUT/composited/ssim_lmd_per_video.log" 2>/dev/null || echo "N/A")
CSIM=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$COMP_LOG" 2>/dev/null | head -1 || echo "N/A")
GT_SSIM=$(grep -oP '^\s*SSIM:\s*\K[\d.]+' "$GT_LOG" 2>/dev/null | tail -1 || echo "N/A")
GT_FID=$(grep -oP '^\s*FID:\s*\K[\d.]+' "$GT_LOG" 2>/dev/null | tail -1 || echo "N/A")
GT_FVD=$(grep -oP '^\s*FVD:\s*\K[\d.]+' "$GT_LOG" 2>/dev/null | tail -1 || echo "N/A")

printf "  Sync-C: %s\n  Sync-D: %s\n  LMD: %s\n  CSIM: %s\n  SSIM_gt: %s\n  FID_gt: %s\n  FVD_gt: %s\n" \
    "$SYNC_C" "$SYNC_D" "$LMD" "$CSIM" "$GT_SSIM" "$GT_FID" "$GT_FVD"

echo ""
echo "All done."
