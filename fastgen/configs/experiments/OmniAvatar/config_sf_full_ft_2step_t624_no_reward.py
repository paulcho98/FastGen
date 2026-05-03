# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symmetric full-FT 1.3B SF — 2-step at t=0.624 (from 4-step DF), no reward, windowed CFG.

Uses the 4-step DF checkpoint as init but narrows to a 2-step schedule
where the intermediate timestep is the 4th boundary of the 4-step
schedule (t=0.624):

    4-step: [0.999, 0.937, 0.833, 0.624, 0.0]   (DF training distribution)
    2-step: [0.999,                0.624, 0.0]    (this config)

Compared to the t833 sibling [0.999, 0.833, 0.0], this puts the
intermediate further down the noise schedule — the student's first
step covers a wider noise range (0.999→0.624) and the second step
is shorter (0.624→0.0).

DF init: 4-step syncnet-trained DF 5000-iter ckpt
CFG: windowed [0.556, 0.882], guidance_scale=4.5
Reward: OFF

Pairs with: scripts/train_sf_full_ft_2step_t624_no_reward.sh.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_full_ft_t769_no_reward as _base


def create_config():
    config = _base.create_config()

    config.model.sample_t_cfg.t_list = [0.999, 0.624, 0.0]
    config.model.student_sample_steps = 2

    config.model.timestep_cfg.enabled = True
    config.model.timestep_cfg.reverse = False

    config.log_config.name = "sf_full_ft_2step_t624_no_reward"
    return config


config = create_config()
