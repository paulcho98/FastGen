#!/bin/bash
# =============================================================================
# Batched inference for the Re-DMD beta=2 + TAEW + syncnet-DF + mouthweight-
# teacher SF run. Mirrors scripts/infer_redmd_taew_audiofix_batched.sh but
# points at this run's CKPT_DIR / OUT_ROOT, and defaults to 3 ckpts on 3 GPUs
# (vs 4 in the original).
#
# Key win over the per-video variant: ONE python process per (ckpt, GPU),
# model + VAE + wav2vec load ONCE, then loops over all HDTF samples in
# /tmp/hdtf_staging via --input_dir. Expected wall time ~15-30 min vs
# ~90-120 min for the per-video script.
#
# --skip_existing makes this safe to run after a partial per-video run:
# videos already on disk in OUT_ROOT/step_NNNNNNN/ are reused, only missing
# ones are generated.
#
# Usage:
#   nohup bash scripts/infer_redmd_taew_audiofix_syncnet_mouthweight_batched.sh \
#     > /tmp/infer_redmd_taew_audiofix_syncnet_mouthweight_batched.log 2>&1 &
# =============================================================================
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

CKPT_DIR="/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT/OmniAvatar-FastGen/omniavatar_sf_audiofix/sf_sink1_window7_redmd_audiofix_beta2_taew_syncnet_mouthweight/checkpoints"
OUT_ROOT="/home/work/output_hdtf_sf_redmd_beta2_taew_syncnet_mouthweight"
STAGE_DIR="/tmp/hdtf_staging"

CKPT_STEPS=(${CKPT_STEPS:-300 400 500})
GPUS=(${GPUS:-0 1 2})

if [ ! -d "$STAGE_DIR" ] || [ "$(ls -d "$STAGE_DIR"/*/ 2>/dev/null | wc -l)" -lt 1 ]; then
    echo "ERROR: $STAGE_DIR not populated. Run the staging step first." >&2
    exit 1
fi
if [ "${#CKPT_STEPS[@]}" -ne "${#GPUS[@]}" ]; then
    echo "ERROR: CKPT_STEPS (${#CKPT_STEPS[@]}) and GPUS (${#GPUS[@]}) must match." >&2
    exit 1
fi

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
N_SAMPLES=$(ls -d "$STAGE_DIR"/*/ | wc -l)

echo "============================================="
echo "  Re-DMD beta=2 TAEW (syncnet + mouthweight) — Batched Inference"
echo "============================================="
echo "  CKPT_DIR:   $CKPT_DIR"
echo "  STAGE_DIR:  $STAGE_DIR ($N_SAMPLES samples)"
echo "  OUT_ROOT:   $OUT_ROOT"
echo "  Assignment:"
for i in "${!CKPT_STEPS[@]}"; do
    step="${CKPT_STEPS[$i]}"
    gpu="${GPUS[$i]}"
    step_p=$(printf "%07d" "$step")
    ckpt="$CKPT_DIR/${step_p}.pth"
    echo "    GPU ${gpu}: step ${step}  ->  ${ckpt}"
    [ -f "$ckpt" ] || { echo "    ERROR: ckpt missing: $ckpt" >&2; exit 1; }
    [ -d "$CKPT_DIR/${step_p}.net_model" ] || { echo "    ERROR: distcp dir missing" >&2; exit 1; }
done
echo "============================================="
echo ""

run_ckpt() {
    local GPU_ID=$1
    local STEP=$2
    local STEP_P=$(printf "%07d" "$STEP")
    local CKPT="$CKPT_DIR/${STEP_P}.pth"
    local OUT_DIR="$OUT_ROOT/step_${STEP_P}"
    mkdir -p "$OUT_DIR"

    local have=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | grep -v aligned | wc -l)
    echo "[GPU $GPU_ID | step $STEP] Starting (${have}/${N_SAMPLES} already done; --skip_existing will reuse)"

    CUDA_VISIBLE_DEVICES=$GPU_ID /home/work/.local/miniconda3/envs/hb_fastgen/bin/python \
        scripts/inference/inference_causal.py \
        --ckpt_path "$CKPT" \
        --vae_path "$OMNIAVATAR_ROOT/pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
        --wav2vec_path "$OMNIAVATAR_ROOT/pretrained_models/wav2vec2-base-960h" \
        --mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
        --base_model_paths "$OMNIAVATAR_ROOT/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
        --omniavatar_ckpt_path /home/work/output_omniavatar_v2v_1.3B_phase2/step-19500.pt \
        --text_embeds_path /home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt \
        --input_dir "$STAGE_DIR" \
        --output_dir "$OUT_DIR" \
        --skip_existing \
        --t_list 0.999 0.833 0.0 \
        --local_attn_size 7 \
        --sink_size 1 \
        --use_dynamic_rope \
        --latentsync \
        --face_cache_dir /home/work/.local/HDTF/face_cache \
    && echo "[GPU $GPU_ID | step $STEP] DONE" \
    || echo "[GPU $GPU_ID | step $STEP] FAILED (see log for stack trace)"
}

for i in "${!CKPT_STEPS[@]}"; do
    run_ckpt "${GPUS[$i]}" "${CKPT_STEPS[$i]}" &
done
wait

echo ""
echo "============================================="
echo "  Batched inference complete."
echo "  Outputs: $OUT_ROOT/step_{0000300,0000400,0000500}/"
echo "============================================="
