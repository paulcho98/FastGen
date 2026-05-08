"""Method config for OmniAvatar Self-Forcing with SyncNet Reward Forcing."""

import attrs
from omegaconf import DictConfig
from typing import Optional

from fastgen.utils import LazyCall as L
from fastgen.configs.methods.config_omniavatar_sf import (
    Config as OmniAvatarSFConfig,
    OmniAvatarModelConfig,
    create_config as create_omniavatar_sf_config,
)
from syncreward.sync_reward_model import OmniAvatarSyncRewardSFModel


@attrs.define(slots=False)
class SyncRewardModelConfig(OmniAvatarModelConfig):
    sync_beta: float = 0.5
    syncnet_ckpt_path: str = "/home/work/.local/eval_metrics/checkpoints/auxiliary/syncnet_v2.model"


@attrs.define(slots=False)
class Config(OmniAvatarSFConfig):
    model: SyncRewardModelConfig = attrs.field(factory=SyncRewardModelConfig)
    model_class: DictConfig = L(OmniAvatarSyncRewardSFModel)(
        config=None,
    )


def create_config():
    config = Config()

    # Inherit all defaults from the OmniAvatar SF base config
    base = create_omniavatar_sf_config()
    config.trainer = base.trainer
    config.log_config = base.log_config

    config.model.sync_beta = 0.5
    config.model.student_sample_steps = 4
    config.model.discriminator_scheduler.warm_up_steps = [0]
    config.model.fake_score_scheduler.warm_up_steps = [0]
    config.model.net_scheduler.warm_up_steps = [0]

    return config
