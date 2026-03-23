# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Method config for OmniAvatar Self-Forcing distillation."""

import attrs
from omegaconf import DictConfig

from fastgen.utils import LazyCall as L
from typing import Optional

from fastgen.configs.methods.config_self_forcing import (
    Config as SFConfig,
    ModelConfig as SFModelConfig,
)
from fastgen.methods.omniavatar_self_forcing import OmniAvatarSelfForcingModel
from fastgen.configs.callbacks import (
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    ParamCount_CALLBACK,
    TrainProfiler_CALLBACK,
    GPUStats_CALLBACK,
    EMA_CALLBACK,
)


@attrs.define(slots=False)
class OmniAvatarModelConfig(SFModelConfig):
    # Separate fake_score config (allows 1.3B fake_score with 14B teacher)
    fake_score: Optional[DictConfig] = None


@attrs.define(slots=False)
class Config(SFConfig):
    model: OmniAvatarModelConfig = attrs.field(factory=OmniAvatarModelConfig)
    model_class: DictConfig = L(OmniAvatarSelfForcingModel)(
        config=None,
    )


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            **EMA_CALLBACK,
            **WANDB_CALLBACK,
        }
    )

    config.dataloader_train.batch_size = 1
    config.model.student_sample_steps = 4
    config.model.discriminator_scheduler.warm_up_steps = [0]
    config.model.fake_score_scheduler.warm_up_steps = [0]
    config.model.net_scheduler.warm_up_steps = [0]

    return config
