# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symmetric full-FT 1.3B SF — no reward, REVERSED timestep CFG.

Sibling of ``config_sf_full_ft_t769_no_reward.py``.  Only difference:

    config.model.timestep_cfg.reverse = True

Effect: CFG is applied OUTSIDE the [t_lo, t_hi] = [0.556, 0.882]
range and turned OFF inside it.  For the t769 2-step schedule:

    Step 1 input (t=0.999): CFG ON  (was OFF in normal mode)
    Step 2 input (t=0.769): CFG OFF (was ON in normal mode)

This tests the hypothesis that CFG helps more at extreme noise levels
(high t where the denoiser has less signal) than in the mid-range
where the teacher prediction is already confident.

Everything else is identical to config_sf_full_ft_t769_no_reward.py:
symmetric full-FT both nets (1421M each), matched 2e-6 LRs, reward
disabled, 600-iter cap, same DF init / teacher / fake_score.

Pairs with: scripts/train_sf_full_ft_t769_no_reward_reverse_cfg.sh.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_full_ft_t769_no_reward as _base


def create_config():
    config = _base.create_config()

    config.model.timestep_cfg.reverse = True

    # Sanity: confirm the base settings we're reversing
    assert config.model.timestep_cfg.enabled is True, (
        "timestep_cfg must be enabled for reverse to have any effect"
    )
    assert config.model.timestep_cfg.t_lo == 0.556, (
        f"Expected t_lo=0.556, got {config.model.timestep_cfg.t_lo}"
    )
    assert config.model.timestep_cfg.t_hi == 0.882, (
        f"Expected t_hi=0.882, got {config.model.timestep_cfg.t_hi}"
    )

    config.log_config.name = "sf_full_ft_t769_no_reward_reverse_cfg"
    return config


config = create_config()
