#!/bin/bash
# Monitor SF training until step 3000, then:
# 1. Kill SF training
# 2. Run SF checkpoint inference sweep (12 checkpoints × 4 GPUs)
# 3. Start DF training with stochastic sliding window attention
#
# Usage: nohup bash scripts/monitor_sf_then_infer_then_df.sh > /tmp/monitor_sf_df.log 2>&1 &
set -euo pipefail

CKPT_DIR="/tmp/FASTGEN_SF_OUTPUT/OmniAvatar-FastGen/omniavatar_sf/sf_4gpu_bs8_lr2e6_5000iter_shift5_combined_v3/checkpoints"
WAIT_FOR_STEP=3100  # Next checkpoint after 3000 — guarantees 3000 is fully saved

echo "============================================="
echo "  Monitoring SF training for step ${WAIT_FOR_STEP}"
echo "  (kill after step 3000 checkpoint is confirmed saved)"
echo "============================================="

# Step 1: Wait for step 3100 checkpoint to appear
while true; do
    if [ -f "${CKPT_DIR}/$(printf '%07d' ${WAIT_FOR_STEP}).pth" ]; then
        echo "$(date): Step ${WAIT_FOR_STEP} checkpoint found — step 3000 is safe."
        break
    fi
    LATEST=$(ls "${CKPT_DIR}"/*.pth 2>/dev/null | sort | tail -1)
    if [ -n "$LATEST" ]; then
        STEP=$(basename "$LATEST" .pth)
        echo "$(date): Latest checkpoint: step ${STEP}, waiting for ${WAIT_FOR_STEP}..."
    fi
    sleep 60
done

# Step 2: Kill SF training
echo ""
echo "============================================="
echo "  Killing SF training"
echo "============================================="
ps aux | grep "train.py" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
sleep 10
echo "SF training killed."

# Step 3: Run SF checkpoint inference sweep
echo ""
echo "============================================="
echo "  Running SF inference sweep (12 checkpoints)"
echo "============================================="
cd /home/work/.local/hyunbin/FastGen
bash scripts/infer_hdtf_sf_batch12.sh
echo "SF inference sweep complete."

# Step 4: Start DF training with stochastic sliding window
echo ""
echo "============================================="
echo "  Starting DF training with stochastic attention"
echo "============================================="

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_DF_OUTPUT"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

RUN_NAME="df_4gpu_bs16_stochastic_attn_shift5"

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_df_shift_5.py \
    - dataloader_train.batch_size=16 \
    trainer.ddp=True \
    trainer.max_iter=10000 \
    trainer.save_ckpt_iter=500 \
    trainer.resume=False \
    log_config.group="omniavatar_df" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho"

echo ""
echo "DF training complete."
