# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symmetric full-FT 1.3B SF — 4-step schedule, no reward, windowed CFG.

Uses the original 4-step DF checkpoint and its matching timestep
distribution, rather than the t769 2-step schedule.  All other settings
match the full_ft_t769_no_reward config (symmetric full-FT, matched LRs,
reward disabled).

Schedule:
    t_list = [0.999, 0.937, 0.833, 0.624, 0.0]
    student_sample_steps = 4
    (derived from shift=5.0: new_t = 5*t / (1 + 4*t) applied to
    linspace(1, 0, 5))

DF init:
    The 4-step syncnet-trained DF 5000-iter ckpt (NOT the t769 variant):
    df_audiofix_syncnet_trained_shift_5_4gpu_bs16_lr1e5_5000iter/0005000.pth

Windowed CFG:
    timestep_cfg.enabled = True
    t_lo = 0.556, t_hi = 0.882  (inherited from config_sf.py)
    CFG ON inside [0.556, 0.882], OFF outside

Reward: OFF (model.reward.enabled=False, reward_beta=0)

Everything else identical to config_sf_full_ft_t769_no_reward.py:
symmetric full-FT both nets, matched 2e-6 LRs, 600-iter cap.

Pairs with: scripts/train_sf_full_ft_4step_no_reward.sh.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_full_ft_t769_no_reward as _base


def create_config():
    config = _base.create_config()

    # ---- 4-STEP SCHEDULE (matches the 4-step DF checkpoint) ----
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]
    config.model.student_sample_steps = 4

    # ---- WINDOWED CFG ON ----
    # Inherited from the chain (config_sf.py sets t_lo/t_hi; the t769
    # no_reward parent enables timestep_cfg).  Just confirm it's on.
    config.model.timestep_cfg.enabled = True
    config.model.timestep_cfg.reverse = False

    # ---- REWARD OFF (inherited, but explicit for clarity) ----
    assert config.model.reward.enabled is False
    assert config.model.reward_beta == 0

    config.log_config.name = "sf_full_ft_4step_no_reward"
    return config


config = create_config()
