#!/bin/bash
# Inference on HDTF test set using the latest SF checkpoint
set -euo pipefail

CKPT_DIR="/tmp/FASTGEN_SF_OUTPUT/OmniAvatar-FastGen/omniavatar_sf/sf_4gpu_bs8_lr2e6_5000iter_shift5_combined_v3/checkpoints"
LATEST=$(ls "$CKPT_DIR"/*.pth 2>/dev/null | sort | tail -1)

if [ -z "$LATEST" ]; then
    echo "ERROR: No .pth checkpoint found in $CKPT_DIR"
    exit 1
fi

STEP=$(basename "$LATEST" .pth)
echo "Using latest checkpoint: $LATEST (step $STEP)"

HDTF=/home/work/.local/HDTF/HDTF_original_testset_81frames
OUT=/home/work/output_hdtf_sf_${STEP}
TEXT_EMB=/home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUT"

total=$(ls "$HDTF"/videos_cfr/*.mp4 | wc -l)
i=0

for video in "$HDTF"/videos_cfr/*.mp4; do
    name=$(basename "$video" _cfr25.mp4)
    i=$((i + 1))
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
        --output_path "$OUT/${name}.mp4" \
        --t_list 0.999 0.937 0.833 0.624 0.0 \
        --use_dynamic_rope \
        --latentsync \
        --face_cache_dir /home/work/.local/HDTF/face_cache \
        --skip_existing
done

echo "Done. Output: $OUT"
