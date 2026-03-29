# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test config: Self-Forcing on 3 Hallo3 samples, 3 iterations (no grad accum).

Usage (non-FSDP):
    torchrun --nproc_per_node=2 train.py \
        --config fastgen/configs/experiments/OmniAvatar/config_sf_test.py

Usage (FSDP):
    torchrun --nproc_per_node=2 train.py \
        --config fastgen/configs/experiments/OmniAvatar/config_sf_test.py \
        trainer.fsdp=True
"""

import os
from omegaconf import DictConfig
from fastgen.configs.experiments.OmniAvatar.config_sf import create_config as create_base_config
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

    # Multi-GPU: FSDP required (DDP OOMs on student update at ~79GB/GPU)
    config.trainer.fsdp = True
    # Use bf16 for FSDP reduction to save memory (fp32 default doubles param storage)
    config.model.precision_fsdp = "bfloat16"

    # Quick test: 11 iterations (covers 2 student updates at freq=5), no grad accum
    config.trainer.max_iter = 11
    config.trainer.grad_accum_rounds = 1
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 999999
    config.trainer.validation_iter = 999999  # skip validation
    config.trainer.skip_initial_validation = True

    config.log_config.group = "omniavatar_sf_test"
    config.log_config.name = "sf_test_run"
    config.log_config.wandb_mode = "disabled"

    return config
