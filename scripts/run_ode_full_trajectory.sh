#!/bin/bash
# Generate full ODE trajectories (all 50 timesteps + denoised outputs) for 10 recon samples.
#
# Step 1: Precompute vae_latents_mask_all.pt (frame 0 masked) if missing
# Step 2: Run full ODE trajectory extraction
#
# Usage:
#   bash scripts/run_ode_full_trajectory.sh              # 14B teacher (default)
#   bash scripts/run_ode_full_trajectory.sh 1.3B          # 1.3B model
#   bash scripts/run_ode_full_trajectory.sh 14B 2         # 14B, 2 GPUs
#
# Output: $OUTPUT_ROOT/{14B,1.3B}/<sample_name>/step_{000..049}_{xt,x0}.pt
# Total:  10 samples × 50 steps × 2 = 1000 files (~2.6 GB)

set -euo pipefail

MODEL_SIZE="${1:-14B}"
NUM_GPUS="${2:-1}"

# ── Paths ──
PRETRAINED="/home/work/.local/OmniAvatar/pretrained_models"
OMNIAVATAR_ROOT="/home/work/.local/OmniAvatar"
DATA_DIR="/home/work/stableavatar_data/v2v_validation_data/recon"
MASK_PATH="/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png"
NEG_TEXT_EMB="/home/work/stableavatar_data/neg_text_emb.pt"
OUTPUT_ROOT="/home/work/ode_full_trajectories"
FASTGEN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Step 1: Precompute vae_latents_mask_all.pt if missing ──
# Check if any sample is missing the mask_all version
NEED_PRECOMPUTE=false
for d in "$DATA_DIR"/*/; do
    if [ -f "$d/vae_latents.pt" ] && [ ! -f "$d/vae_latents_mask_all.pt" ]; then
        NEED_PRECOMPUTE=true
        break
    fi
done

if [ "$NEED_PRECOMPUTE" = true ]; then
    echo "=== Step 1: Precomputing vae_latents_mask_all.pt ==="

    # Create temp data list for precomputation
    DATA_LIST=$(mktemp /tmp/ode_precompute_XXXXXX.txt)
    cat "$DATA_DIR/video_square_path.txt" > "$DATA_LIST"

    VAE_PATH="${PRETRAINED}/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"

    pushd "$OMNIAVATAR_ROOT" > /dev/null
    CUDA_VISIBLE_DEVICES=0 python scripts/precompute_vae_latents_masked.py \
        --vae_path "$VAE_PATH" \
        --latentsync_mask_path "$MASK_PATH" \
        --data_list_path "$DATA_LIST" \
        --batch_size 10
    popd > /dev/null

    rm -f "$DATA_LIST"
    echo ""
else
    echo "=== Step 1: vae_latents_mask_all.pt already exists for all samples, skipping ==="
    echo ""
fi

# ── Step 2: Run ODE trajectory extraction ──
echo "=== Step 2: Generating full ODE trajectories ==="

# Model-specific config
if [ "$MODEL_SIZE" = "14B" ]; then
    BASE_PATHS="${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors"
    CKPT="/home/work/output_omniavatar_v2v_phase2/step-10500.pt"
    OUTPUT_DIR="${OUTPUT_ROOT}/14B"
elif [ "$MODEL_SIZE" = "1.3B" ]; then
    BASE_PATHS="${PRETRAINED}/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
    CKPT="/home/work/output_omniavatar_v2v_1.3B_phase2/step-19500.pt"
    OUTPUT_DIR="${OUTPUT_ROOT}/1.3B"
else
    echo "ERROR: model_size must be 14B or 1.3B, got: $MODEL_SIZE"
    exit 1
fi

echo "Model:  ${MODEL_SIZE}"
echo "GPUs:   ${NUM_GPUS}"
echo "Data:   ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

cd "$FASTGEN_ROOT"

COMMON_ARGS=(
    scripts/generate_omniavatar_ode_pairs_full.py
    --model_size "$MODEL_SIZE"
    --in_dim 65
    --base_model_paths "$BASE_PATHS"
    --omniavatar_ckpt_path "$CKPT"
    --data_dir "$DATA_DIR"
    --latentsync_mask_path "$MASK_PATH"
    --neg_text_emb_path "$NEG_TEXT_EMB"
    --output_dir "$OUTPUT_DIR"
    --num_inference_steps 50
    --guidance_scale 4.5
    --shift 5.0
    --max_samples 10
    --skip_existing
)

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$NUM_GPUS" "${COMMON_ARGS[@]}"
else
    CUDA_VISIBLE_DEVICES=0 python "${COMMON_ARGS[@]}"
fi
