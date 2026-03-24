# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test config: Diffusion Forcing on 3 Hallo3 samples, 20 iterations.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config fastgen/configs/experiments/OmniAvatar/config_df_test.py
"""

import os
from omegaconf import DictConfig
from fastgen.configs.experiments.OmniAvatar.config_df import create_config as create_base_config
from fastgen.utils import LazyCall as L
from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataLoader
from fastgen.configs.callbacks import GradClip_CALLBACK, GPUStats_CALLBACK, ParamCount_CALLBACK
from fastgen.callbacks.stdout_logger import StdoutLoggerCallback

STDOUT_LOG_CALLBACK = {"stdout_logger": L(StdoutLoggerCallback)()}

OMNI_ROOT = os.getenv(
    "OMNIAVATAR_ROOT",
    "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_OmniAvatar/OmniAvatar-Train",
)
TEST_DATA = os.getenv(
    "OMNIAVATAR_TEST_DATA",
    "/data/karlo-research_715/workspace/kinemaar/datasets/sample_hallo3_latentsync",
)


def create_config():
    config = create_base_config()

    # Point to test data
    config.dataloader_train = L(OmniAvatarDataLoader)(
        data_list_path=os.path.join(TEST_DATA, "video_list.txt"),
        latentsync_mask_path=os.path.join(OMNI_ROOT, "OmniAvatar/utils/latentsync/mask.png"),
        batch_size=1,
        num_workers=0,
        use_ref_sequence=True,
        load_ode_path=False,
    )

    # Disable wandb for local testing
    config.trainer.callbacks = DictConfig({
        **GradClip_CALLBACK,
        **GPUStats_CALLBACK,
        **ParamCount_CALLBACK,
        **STDOUT_LOG_CALLBACK,
    })

    # Quick test: 20 iterations, log every step, save at end
    config.trainer.max_iter = 20
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 20
    config.trainer.validation_iter = 999999  # skip validation

    config.log_config.group = "omniavatar_df_test"
    config.log_config.name = "df_test_run"
    config.log_config.wandb_mode = "disabled"

    return config
