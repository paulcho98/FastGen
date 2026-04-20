#!/bin/bash
# Same as infer_hdtf_sink1_window7_redmd.sh but uses TAEHV tiny decoder
# (inference_causal_taehv.py + --taehv_ckpt).
#
# Usage: bash scripts/infer_hdtf_sink1_window7_redmd_taehv.sh
set -euo pipefail

RUN_NAME="sf_sink1_window7_redmd_syncc_beta0p25_joonson_parity"
CKPT_DIR="/tmp/FASTGEN_SF_OUTPUT/OmniAvatar-FastGen/omniavatar_sf/${RUN_NAME}/checkpoints"
HDTF=/home/work/.local/HDTF/HDTF_original_testset_81frames
TEXT_EMB=/home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt
OUT_ROOT=/home/work/output_hdtf_sink1_window7_redmd_taehv
TAEHV_CKPT=/home/work/.local/hyunbin/FastGen-redmd/checkpoints/taehv/taew2_1.pth

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"

mapfile -t ALL_CKPTS < <(ls "$CKPT_DIR"/*.pth 2>/dev/null | sort | tail -12)

if [ ${#ALL_CKPTS[@]} -lt 1 ]; then
    echo "ERROR: No checkpoints found in $CKPT_DIR"
    exit 1
fi
if [ ${#ALL_CKPTS[@]} -lt 12 ]; then
    echo "WARNING: Only ${#ALL_CKPTS[@]} checkpoints found (expected 12)"
fi
if [ ! -f "$TAEHV_CKPT" ]; then
    echo "ERROR: TAEHV checkpoint not found: $TAEHV_CKPT"
    exit 1
fi

echo "============================================="
echo "  ${RUN_NAME} (TAEHV): ${#ALL_CKPTS[@]} checkpoints × 4 GPUs"
echo "============================================="

run_gpu() {
    local GPU_ID=$1
    shift
    local CKPTS=("$@")

    for CKPT in "${CKPTS[@]}"; do
        STEP=$(basename "$CKPT" .pth)
        OUT_DIR="$OUT_ROOT/step_${STEP}"
        mkdir -p "$OUT_DIR"

        echo "[GPU $GPU_ID] Starting step $STEP → $OUT_DIR"

        total=$(ls "$HDTF"/videos_cfr/*.mp4 | wc -l)
        i=0

        for video in "$HDTF"/videos_cfr/*.mp4; do
            name=$(basename "$video" _cfr25.mp4)
            i=$((i + 1))

            if [ -f "$OUT_DIR/${name}.mp4" ]; then
                continue
            fi

            echo "[GPU $GPU_ID | step $STEP] [$i/$total] $name"
            CUDA_VISIBLE_DEVICES=$GPU_ID /home/work/.local/miniconda3/envs/hb_fastgen/bin/python \
                scripts/inference/inference_causal_taehv.py \
                --ckpt_path "$CKPT" \
                --vae_path "$OMNIAVATAR_ROOT/pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
                --taehv_ckpt "$TAEHV_CKPT" \
                --wav2vec_path "$OMNIAVATAR_ROOT/pretrained_models/wav2vec2-base-960h" \
                --mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
                --base_model_paths "$OMNIAVATAR_ROOT/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
                --omniavatar_ckpt_path /home/work/output_omniavatar_v2v_1.3B_phase2/step-19500.pt \
                --text_embeds_path "$TEXT_EMB" \
                --video_path "$video" \
                --output_path "$OUT_DIR/${name}.mp4" \
                --t_list 0.999 0.833 0.0 \
                --chunk_size 3 \
                --local_attn_size 7 \
                --sink_size 1 \
                --use_dynamic_rope \
                --latentsync \
                --face_cache_dir /home/work/.local/HDTF/face_cache \
                --skip_existing
        done

        echo "[GPU $GPU_ID] Finished step $STEP"
    done
}

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
echo "GPU 0 (${#GPU0_CKPTS[@]} ckpts): $(printf '%s ' "${GPU0_CKPTS[@]##*/}")"
echo "GPU 1 (${#GPU1_CKPTS[@]} ckpts): $(printf '%s ' "${GPU1_CKPTS[@]##*/}")"
echo "GPU 2 (${#GPU2_CKPTS[@]} ckpts): $(printf '%s ' "${GPU2_CKPTS[@]##*/}")"
echo "GPU 3 (${#GPU3_CKPTS[@]} ckpts): $(printf '%s ' "${GPU3_CKPTS[@]##*/}")"
echo ""

run_gpu 0 "${GPU0_CKPTS[@]}" &
run_gpu 1 "${GPU1_CKPTS[@]}" &
run_gpu 2 "${GPU2_CKPTS[@]}" &
run_gpu 3 "${GPU3_CKPTS[@]}" &

wait
echo ""
echo "All done. Output: $OUT_ROOT"
