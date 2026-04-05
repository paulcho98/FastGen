#!/bin/bash
# Run HDTF inference for the last 16 SF checkpoints, distributed across 4 GPUs.
# Each GPU processes 4 checkpoints sequentially.
# Reuses face detection caches from --face_cache_dir.
#
# Usage: bash scripts/infer_hdtf_sf_batch16.sh
set -euo pipefail

CKPT_DIR="/tmp/FASTGEN_SF_OUTPUT/OmniAvatar-FastGen/omniavatar_sf/sf_4gpu_bs8_lr2e6_5000iter_shift5_combined_v3/checkpoints"
HDTF=/home/work/.local/HDTF/HDTF_original_testset_81frames
TEXT_EMB=/home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt
OUT_ROOT=/home/work/output_hdtf_sf_sweep

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"

# Get last 16 checkpoints
mapfile -t ALL_CKPTS < <(ls "$CKPT_DIR"/*.pth 2>/dev/null | sort | tail -16)

if [ ${#ALL_CKPTS[@]} -lt 16 ]; then
    echo "WARNING: Only ${#ALL_CKPTS[@]} checkpoints found (expected 16)"
fi

echo "============================================="
echo "  SF Checkpoint Sweep: ${#ALL_CKPTS[@]} checkpoints × 4 GPUs"
echo "============================================="

# Distribute checkpoints across 4 GPUs (round-robin)
run_gpu() {
    local GPU_ID=$1
    shift
    local CKPTS=("$@")

    for CKPT in "${CKPTS[@]}"; do
        STEP=$(basename "$CKPT" .pth)
        OUT_DIR="$OUT_ROOT/step_${STEP}"
        mkdir -p "$OUT_DIR"

        echo "[GPU $GPU_ID] Starting step $STEP → $OUT_DIR"

        for video in "$HDTF"/videos_cfr/*.mp4; do
            name=$(basename "$video" _cfr25.mp4)
            out_file="$OUT_DIR/${name}.mp4"

            if [ -f "$out_file" ]; then
                continue  # skip existing
            fi

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
                --output_path "$out_file" \
                --t_list 0.999 0.937 0.833 0.624 0.0 \
                --use_dynamic_rope \
                --latentsync \
                --face_cache_dir /home/work/.local/HDTF/face_cache \
                --skip_existing
        done 2>&1 | tee "$OUT_DIR/inference.log"

        echo "[GPU $GPU_ID] Finished step $STEP"
    done
}

# Split checkpoints: GPU 0 gets indices 0,4,8,12; GPU 1 gets 1,5,9,13; etc.
GPU0_CKPTS=()
GPU1_CKPTS=()
GPU2_CKPTS=()
GPU3_CKPTS=()

for i in "${!ALL_CKPTS[@]}"; do
    case $((i % 4)) in
        0) GPU0_CKPTS+=("${ALL_CKPTS[$i]}") ;;
        1) GPU1_CKPTS+=("${ALL_CKPTS[$i]}") ;;
        2) GPU2_CKPTS+=("${ALL_CKPTS[$i]}") ;;
        3) GPU3_CKPTS+=("${ALL_CKPTS[$i]}") ;;
    esac
done

echo ""
echo "GPU 0: ${GPU0_CKPTS[*]##*/}"
echo "GPU 1: ${GPU1_CKPTS[*]##*/}"
echo "GPU 2: ${GPU2_CKPTS[*]##*/}"
echo "GPU 3: ${GPU3_CKPTS[*]##*/}"
echo ""

# Launch all 4 GPUs in parallel
run_gpu 0 "${GPU0_CKPTS[@]}" &
run_gpu 1 "${GPU1_CKPTS[@]}" &
run_gpu 2 "${GPU2_CKPTS[@]}" &
run_gpu 3 "${GPU3_CKPTS[@]}" &

wait
echo ""
echo "All done. Output: $OUT_ROOT"
