# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-sized Re-DMD + TAEW variant.

Inherits config_sf_sink1_window7_redmd_taew.py, narrows to 10 iters /
batch_size=1 / debug MP4 dump, and renames the run.
"""

from fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd_taew import (
    create_config as _taew_create_config,
)


def create_config():
    config = _taew_create_config()

    config.trainer.max_iter = 11
    config.trainer.grad_accum_rounds = 1
    config.dataloader_train.batch_size = 1

    config.model.save_reward_debug_video = True
    config.model.reward_debug_dir = "logs/redmd_smoke_debug_taew"

    config.log_config.name = "sf_sink1_window7_redmd_syncc_beta0p25_joonson_parity_taew_smoke"

    if hasattr(config.trainer, "eval_period"):
        config.trainer.eval_period = 999999
    return config


config = create_config()
