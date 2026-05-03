# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symmetric full-FT 1.3B SF — 2-step at t=0.833 (from 4-step DF), no reward, windowed CFG.

Uses the 4-step DF checkpoint as init but narrows to a 2-step schedule
where the intermediate timestep is the 3rd boundary of the 4-step
schedule (t=0.833):

    4-step: [0.999, 0.937, 0.833, 0.624, 0.0]   (DF training distribution)
    2-step: [0.999,        0.833,        0.0]     (this config)

Compared to the t769 schedule [0.999, 0.769, 0.0], this uses a native
boundary point from the DF's training distribution rather than an
interpolated one.  Tests whether matching a DF boundary reduces the
train/test mismatch between DF and SF stages.

DF init: 4-step syncnet-trained DF 5000-iter ckpt (same as config_sf_full_ft_4step_no_reward.py)
CFG: windowed [0.556, 0.882], guidance_scale=4.5
Reward: OFF

Everything else identical to the t769 no-reward configs: symmetric
full-FT both nets, matched 2e-6 LRs, 600-iter cap.

Pairs with: scripts/train_sf_full_ft_2step_t833_no_reward.sh.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_full_ft_t769_no_reward as _base


def create_config():
    config = _base.create_config()

    # ---- 2-STEP SCHEDULE at t=0.833 ----
    config.model.sample_t_cfg.t_list = [0.999, 0.833, 0.0]
    config.model.student_sample_steps = 2

    # ---- WINDOWED CFG ON ----
    config.model.timestep_cfg.enabled = True
    config.model.timestep_cfg.reverse = False

    config.log_config.name = "sf_full_ft_2step_t833_no_reward"
    return config


config = create_config()
