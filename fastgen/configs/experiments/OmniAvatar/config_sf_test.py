# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test config: Self-Forcing quick validation test.

Tests both training iterations AND validation with video generation + wandb logging.
Uses bs=1 for speed, fires validation at iter 5 (first student update).

Usage:
    torchrun --nproc_per_node=4 train.py \
        --config fastgen/configs/experiments/OmniAvatar/config_sf_test.py
"""

import os
from omegaconf import DictConfig
from fastgen.configs.experiments.OmniAvatar.config_sf import create_config as create_base_config
from fastgen.utils import LazyCall as L
from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataLoader, create_omniavatar_dataloader
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
    GPUStats_CALLBACK,
    ParamCount_CALLBACK,
    WANDB_CALLBACK,
)
from fastgen.callbacks.stdout_logger import StdoutLoggerCallback
from fastgen.callbacks.wandb import WandbCallback

STDOUT_LOG_CALLBACK = {"stdout_logger": L(StdoutLoggerCallback)()}

OMNI_ROOT = os.getenv(
    "OMNIAVATAR_ROOT",
    "/home/work/.local/OmniAvatar",
)
DATA_ROOT = os.getenv(
    "OMNIAVATAR_DATA_ROOT",
    "/home/work/stableavatar_data/v2v_training_data",
)
TEST_DATA = os.getenv(
    "OMNIAVATAR_TEST_DATA",
    "/home/work/stableavatar_data/v2v_validation_data/recon",
)
MASK_PATH = "/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png"
NEG_TEXT_EMB = "/home/work/stableavatar_data/neg_text_emb.pt"
VAE_PATH = os.path.join(OMNI_ROOT, "pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")


def create_config():
    config = create_base_config()

    # Training data (full set, needed for bs>1 with DistributedSampler)
    config.dataloader_train = L(OmniAvatarDataLoader)(
        data_list_path=os.path.join(DATA_ROOT, "video_square_path.txt"),
        latentsync_mask_path=MASK_PATH,
        batch_size=1,
        num_workers=0,
        use_ref_sequence=True,
        load_ode_path=False,
        neg_text_emb_path=NEG_TEXT_EMB,
    )

    # Validation data (10 fixed samples)
    config.dataloader_val = L(create_omniavatar_dataloader)(
        data_list_path=os.path.join(TEST_DATA, "video_square_path.txt"),
        latentsync_mask_path=MASK_PATH,
        batch_size=1,
        num_workers=0,
        use_ref_sequence=True,
        load_ode_path=False,
        neg_text_emb_path=NEG_TEXT_EMB,
    )
    config.model.vae_path = VAE_PATH

    # Callbacks: stdout + wandb + gpu stats
    config.trainer.callbacks = DictConfig({
        **GradClip_CALLBACK,
        **GPUStats_CALLBACK,
        **ParamCount_CALLBACK,
        **STDOUT_LOG_CALLBACK,
        "wandb": L(WandbCallback)(sample_logging_iter=5),
    })

    # FSDP
    config.trainer.fsdp = True
    config.model.precision_fsdp = "bfloat16"

    # Quick test: 6 iterations, validate at iter 5 (first student update)
    config.trainer.max_iter = 6
    config.trainer.grad_accum_rounds = 1
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 999999
    config.trainer.validation_iter = 5
    config.trainer.skip_initial_validation = True

    # Wandb — same entity/project as OmniAvatar training
    config.log_config.project = "OmniAvatar-FastGen"
    config.log_config.group = "omniavatar_sf_test"
    config.log_config.name = "sf_val_test"
    config.log_config.wandb_mode = "online"

    return config
