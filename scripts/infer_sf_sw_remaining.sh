#!/bin/bash
# Inference ONLY for remaining SW SF checkpoints (skips completed).
# No eval — run eval_correct_sweep.sh after this completes.
#
# Usage: nohup bash scripts/infer_sf_sw_remaining.sh > /tmp/infer_sf_sw_remaining.log 2>&1 &
set -uo pipefail

CKPT_DIR="/tmp/FASTGEN_SF_OUTPUT/OmniAvatar-FastGen/omniavatar_sf/sf_sink1_window7_redmd_syncc_beta0p25/checkpoints"
HDTF=/home/work/.local/HDTF/HDTF_original_testset_81frames
TEXT_EMB=/home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt
OUT_ROOT="/home/work/output_hdtf_sf_sw"
GPUS=(${INFER_GPUS:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
TOTAL_VIDEOS=$(ls "$HDTF"/videos_cfr/*.mp4 | wc -l)

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"

mapfile -t ALL_CKPTS < <(ls "$CKPT_DIR"/*.pth 2>/dev/null | sort)

# Filter to only checkpoints that need work
NEED_WORK=()
for CKPT in "${ALL_CKPTS[@]}"; do
    STEP=$(basename "$CKPT" .pth)
    OUT_DIR="$OUT_ROOT/step_${STEP}"
    n=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | grep -v aligned | wc -l)
    if [ "$n" -lt "$TOTAL_VIDEOS" ]; then
        NEED_WORK+=("$CKPT")
        echo "Need work: step $STEP ($n/$TOTAL_VIDEOS)"
    else
        echo "Complete: step $STEP ($n/$TOTAL_VIDEOS)"
    fi
done

echo ""
echo "============================================="
echo "  Inference: ${#NEED_WORK[@]} checkpoints need work"
echo "============================================="

run_checkpoint() {
    local GPU_ID=$1
    local CKPT=$2
    local STEP=$(basename "$CKPT" .pth)
    local OUT_DIR="$OUT_ROOT/step_${STEP}"
    mkdir -p "$OUT_DIR"

    echo "[GPU $GPU_ID | step $STEP] Starting inference..."
    local i=0
    for video in "$HDTF"/videos_cfr/*.mp4; do
        local name=$(basename "$video" _cfr25.mp4)
        i=$((i + 1))
        [ -f "$OUT_DIR/${name}.mp4" ] && continue

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
    echo "[GPU $GPU_ID | step $STEP] Inference done"
}

run_gpu() {
    local GPU_ID=$1
    shift
    for CKPT in "$@"; do
        run_checkpoint "$GPU_ID" "$CKPT"
    done
}

# Distribute round-robin
declare -A GPU_WORK
for gpu in "${GPUS[@]}"; do GPU_WORK[$gpu]=""; done
for i in "${!NEED_WORK[@]}"; do
    gpu=${GPUS[$((i % NUM_GPUS))]}
    GPU_WORK[$gpu]+="${NEED_WORK[$i]} "
done

for gpu in "${GPUS[@]}"; do
    ckpts=(${GPU_WORK[$gpu]})
    [ ${#ckpts[@]} -gt 0 ] && run_gpu "$gpu" "${ckpts[@]}" &
done

wait
echo ""
echo "All inference done. Now run: bash scripts/eval_correct_sweep.sh"
