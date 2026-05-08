#!/bin/bash
# Train OmniAvatar Self-Forcing with SyncNet Reward Forcing.
#
# Same pipeline as scripts/train_omniavatar_sf.sh but weights VSD loss
# by exp(beta * Sync-C) to reward better lip-sync quality.
#
# Usage:
#   bash syncreward/train_sync_reward_sf.sh
#
# Optional environment variables:
#   SYNC_BETA            - Reward temperature (default: 0.5)
#   SYNCNET_CKPT_PATH    - Path to syncnet_v2.model
#   NUM_GPUS             - Number of GPUs (default: 4)
#   MASTER_PORT          - Distributed master port (default: 29501)
#   DATA_LIST_PATH       - Override training data list
#   LATENTSYNC_MASK_PATH - Override mask path

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

# Optional CLI overrides for data paths
EXTRA_ARGS=""
if [ -n "${DATA_LIST_PATH:-}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} dataloader_train.data_list_path='${DATA_LIST_PATH}'"
fi
if [ -n "${LATENTSYNC_MASK_PATH:-}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} dataloader_train.latentsync_mask_path='${LATENTSYNC_MASK_PATH}'"
fi

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" train.py \
    --config=syncreward/config_experiment.py \
    - ${EXTRA_ARGS} \
      log_config.name=omniavatar_sync_reward_sf
