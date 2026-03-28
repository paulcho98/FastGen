# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Method config for OmniAvatar Diffusion Forcing (Stage 1 alternative to ODE KD)."""

import attrs
from omegaconf import DictConfig

from fastgen.utils import LazyCall as L
from fastgen.configs.config import BaseConfig, BaseModelConfig
from fastgen.methods.omniavatar_diffusion_forcing import OmniAvatarDiffusionForcingModel
from fastgen.callbacks.wandb import WandbCallback
from fastgen.configs.callbacks import (
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    ParamCount_CALLBACK,
    TrainProfiler_CALLBACK,
    GPUStats_CALLBACK,
)


@attrs.define(slots=False)
class ModelConfig(BaseModelConfig):
    context_noise: float = 0.0
    vae_path: str = ""  # Path to WanVAE for visual logging (optional)


@attrs.define(slots=False)
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = L(OmniAvatarDiffusionForcingModel)(
        config=None,
    )


def create_config():
    config = Config()
    # OmniAvatar uses 25fps video
    OMNIAVATAR_WANDB = dict(wandb=L(WandbCallback)(sample_logging_iter=None, fps=25))
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            **OMNIAVATAR_WANDB,
        }
    )

    config.dataloader_train.batch_size = 1
    config.model.student_sample_steps = 4
    config.model.net_scheduler.warm_up_steps = [0]

    return config
