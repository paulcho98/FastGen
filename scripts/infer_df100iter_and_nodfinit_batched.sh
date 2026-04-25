#!/bin/bash
# =============================================================================
# Batched inference: DF-100iter (steps 50, 100) + nodfinit SF (steps 200, 300)
# =============================================================================
#
# 4 jobs, one per GPU. Each job loads model+VAE+wav2vec ONCE, then loops over
# the 33 staged HDTF samples in /tmp/hdtf_staging via --input_dir mode.
#
# GPU 0: SF nodfinit step 200
# GPU 1: SF nodfinit step 300
# GPU 2: DF 100iter step 50
# GPU 3: DF 100iter step 100
#
# Coordination: jobs poll for their ckpts to land. Since the DF 100iter run
# is currently consuming all 4 GPUs, no inference can start until DF training
# finishes. We gate the launch on the *final* DF ckpt (0000100.pth) being
# saved — at which point training is wrapping up — plus a 30 s buffer so the
# training process fully exits and frees GPU memory before we start.
#
# --skip_existing makes this safe to re-run (already-generated mp4s are
# reused, only missing ones regenerate).
#
# Usage:
#   nohup bash scripts/infer_df100iter_and_nodfinit_batched.sh \
#     > /tmp/infer_df100iter_nodfinit_batched.log 2>&1 &
# =============================================================================
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"

STAGE_DIR="/tmp/hdtf_staging"

NODFINIT_CKPT_DIR="/tmp/FASTGEN_SF_OUTPUT_BETA2_AUDIOFIX_TAEW_SYNCNET_MOUTHWEIGHT_FSMATCHED_LR3E6_NODFINIT/OmniAvatar-FastGen/omniavatar_sf_audiofix/sf_sink1_window7_redmd_audiofix_beta2_taew_syncnet_mouthweight_fsmatched_lr3e6_nodfinit/checkpoints"
DF100_CKPT_DIR="/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_syncnet_trained_shift_5_4gpu_bs16_lr1e5_100iter/checkpoints"

NODFINIT_OUT_ROOT="/home/work/output_hdtf_sf_redmd_beta2_taew_fsmatched_lr3e6_nodfinit"
DF100_OUT_ROOT="/home/work/output_hdtf_df_audiofix_syncnet_trained_100iter"

# Static inference inputs (match existing batched scripts).
VAE_PATH="$OMNIAVATAR_ROOT/pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
WAV2VEC_PATH="$OMNIAVATAR_ROOT/pretrained_models/wav2vec2-base-960h"
MASK_PATH="/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png"
BASE_MODEL_PATHS="$OMNIAVATAR_ROOT/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
# Use the syncnet-trained adapter (matches the training-time student/fake_score
# init for these runs). The SF/DF ckpt overwrites the student weights anyway,
# but matching at construction time keeps things consistent.
OMNIAVATAR_CKPT="/home/work/output_omniavatar_v2v_1.3B_maskall_refseq_mouth_weight_2gpu/step-1000.pt"
TEXT_EMB_PATH="/home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt"
FACE_CACHE="/home/work/.local/HDTF/face_cache"

PYTHON_BIN="/home/work/.local/miniconda3/envs/hb_fastgen/bin/python"

if [ ! -d "$STAGE_DIR" ] || [ "$(ls -d "$STAGE_DIR"/*/ 2>/dev/null | wc -l)" -lt 1 ]; then
    echo "ERROR: $STAGE_DIR not populated. Run the staging step first." >&2
    exit 1
fi
N_SAMPLES=$(ls -d "$STAGE_DIR"/*/ | wc -l)

# (gpu, label, ckpt_path, out_dir) tuples.
JOBS=(
    "0|nodfinit_s200|${NODFINIT_CKPT_DIR}/0000200.pth|${NODFINIT_OUT_ROOT}/step_0000200"
    "1|nodfinit_s300|${NODFINIT_CKPT_DIR}/0000300.pth|${NODFINIT_OUT_ROOT}/step_0000300"
    "2|df100_s050   |${DF100_CKPT_DIR}/0000050.pth|${DF100_OUT_ROOT}/step_0000050"
    "3|df100_s100   |${DF100_CKPT_DIR}/0000100.pth|${DF100_OUT_ROOT}/step_0000100"
)

echo "============================================="
echo "  4-GPU batched inference"
echo "============================================="
echo "  STAGE_DIR:  $STAGE_DIR ($N_SAMPLES samples)"
echo "  Jobs:"
for j in "${JOBS[@]}"; do
    IFS='|' read -r gpu label ckpt out <<< "$j"
    echo "    GPU $gpu : $label : $ckpt"
    echo "                          -> $out"
done
echo "============================================="
echo ""

# Block until the FINAL DF 100iter ckpt exists (signals DF training has hit
# its save-and-exit phase), then a brief buffer so training fully releases GPU.
FINAL_DF_CKPT="${DF100_CKPT_DIR}/0000100.pth"
if [ ! -f "$FINAL_DF_CKPT" ]; then
    echo "[$(date)] Waiting for final DF ckpt to land at $FINAL_DF_CKPT ..."
    while [ ! -f "$FINAL_DF_CKPT" ]; do sleep 30; done
    echo "[$(date)] Final DF ckpt detected. Sleeping 30s for training to exit cleanly..."
    sleep 30
else
    echo "[$(date)] Final DF ckpt already present, proceeding."
fi

run_one() {
    local job=$1
    IFS='|' read -r gpu label ckpt out <<< "$job"
    label="$(echo "$label" | xargs)"   # trim whitespace from padded labels
    mkdir -p "$out"

    if [ ! -f "$ckpt" ]; then
        echo "[$(date)] [$label | GPU $gpu] ERROR: ckpt missing: $ckpt" >&2
        return 1
    fi

    local have=$(ls "$out"/*.mp4 2>/dev/null | grep -v aligned | wc -l)
    echo "[$(date)] [$label | GPU $gpu] start ($have/$N_SAMPLES already done; --skip_existing reuses them)"

    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" \
        scripts/inference/inference_causal.py \
        --ckpt_path "$ckpt" \
        --vae_path "$VAE_PATH" \
        --wav2vec_path "$WAV2VEC_PATH" \
        --mask_path "$MASK_PATH" \
        --base_model_paths "$BASE_MODEL_PATHS" \
        --omniavatar_ckpt_path "$OMNIAVATAR_CKPT" \
        --text_embeds_path "$TEXT_EMB_PATH" \
        --input_dir "$STAGE_DIR" \
        --output_dir "$out" \
        --skip_existing \
        --t_list 0.999 0.833 0.0 \
        --local_attn_size 7 \
        --sink_size 1 \
        --use_dynamic_rope \
        --latentsync \
        --face_cache_dir "$FACE_CACHE" \
    && echo "[$(date)] [$label | GPU $gpu] DONE" \
    || echo "[$(date)] [$label | GPU $gpu] FAILED (see trace above)"
}

for j in "${JOBS[@]}"; do
    run_one "$j" &
done
wait

echo ""
echo "============================================="
echo "  All 4 inference jobs complete."
echo "  Outputs:"
echo "    $NODFINIT_OUT_ROOT/step_{0000200,0000300}/"
echo "    $DF100_OUT_ROOT/step_{0000050,0000100}/"
echo "============================================="
