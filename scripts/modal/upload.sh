#!/bin/bash
# One-shot upload of heavyweight assets to Modal Volume `fastgen-assets`.
# Run once (or whenever you swap the SF checkpoint / test clip).
# Total upload: ~10 GB. First run will take a while.
#
# Usage: bash scripts/modal/upload.sh

set -euo pipefail

VOL=fastgen-assets

OMNIAVATAR_ROOT=${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}
SF_CKPT_DIR=/tmp/FASTGEN_SF_OUTPUT/OmniAvatar-FastGen/omniavatar_sf/sf_sink1_window7_redmd_syncc_beta0p25_joonson_parity/checkpoints
CKPT_STEP=${CKPT_STEP:-0000600}
VIDEO=${VIDEO:-RD_Radio18_000_cfr25}

# Create the volume (idempotent).
modal volume create "$VOL" 2>/dev/null || true

echo ">> Wan 2.1 VAE (484 MB)"
modal volume put "$VOL" \
    "$OMNIAVATAR_ROOT/pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
    /wan_vae/Wan2.1_VAE.pth

echo ">> Wan base diffusion (~5.4 GB)"
modal volume put "$VOL" \
    "$OMNIAVATAR_ROOT/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    /wan_base/diffusion_pytorch_model.safetensors

echo ">> wav2vec2-base-960h (1.1 GB)"
modal volume put "$VOL" \
    "$OMNIAVATAR_ROOT/pretrained_models/wav2vec2-base-960h" \
    /wav2vec2-base-960h

echo ">> OmniAvatar LoRA (339 MB)"
modal volume put "$VOL" \
    /home/work/output_omniavatar_v2v_1.3B_phase2/step-19500.pt \
    /omniavatar/step-19500.pt

echo ">> SF checkpoint $CKPT_STEP (5.3 GB, sharded DCP)"
modal volume put "$VOL" "$SF_CKPT_DIR/${CKPT_STEP}.pth" "/sf_ckpts/${CKPT_STEP}.pth"
modal volume put "$VOL" "$SF_CKPT_DIR/${CKPT_STEP}.net_model" "/sf_ckpts/${CKPT_STEP}.net_model"

echo ">> TAEHV (22 MB)"
modal volume put "$VOL" \
    /home/work/.local/hyunbin/FastGen-redmd/checkpoints/taehv/taew2_1.pth \
    /taehv/taew2_1.pth

echo ">> insightface buffalo_l (327 MB)"
modal volume put "$VOL" /home/work/.insightface/models/buffalo_l /insightface/buffalo_l

echo ">> mask.png"
modal volume put "$VOL" \
    /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
    /mask/mask.png

echo ">> text embedding"
modal volume put "$VOL" \
    /home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt \
    /text_emb/text_emb.pt

echo ">> HDTF video + face cache ($VIDEO)"
modal volume put "$VOL" \
    "/home/work/.local/HDTF/HDTF_original_testset_81frames/videos_cfr/${VIDEO}.mp4" \
    "/hdtf/videos/${VIDEO}.mp4"
modal volume put "$VOL" \
    "/home/work/.local/HDTF/face_cache/${VIDEO}_face_cache.pt" \
    "/hdtf/face_cache/${VIDEO}_face_cache.pt"

echo
echo "Upload complete. Now run:"
echo "  modal run scripts/modal/app.py --ckpt-name $CKPT_STEP --video-name ${VIDEO%_cfr25}"
echo "  modal run scripts/modal/app.py --ckpt-name $CKPT_STEP --video-name ${VIDEO%_cfr25} --use-taehv"
