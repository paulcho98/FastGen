# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symmetric full-FT 1.3B SF with sync-C reward DISABLED — ablation.

Sibling of ``config_sf_full_ft_t769.py``.  Differences:

    1. ``config.model.reward.enabled = False`` — drops self.reward_scorer to
       None at construction time, makes ``reward_active = False`` every
       step, and falls back to plain DMD2 loss (no exp(beta * r) weighting,
       no per-sample reduction).
    2. ``config.model.reward_beta = 0`` — belt-and-suspenders: even if the
       enabled flag flips True, exp(0*r) = 1 produces unit weights.

Everything else is identical to ``config_sf_full_ft_t769.py``: symmetric
full-FT (student + fake_score both ~1.3B trainable), matched 2e-6 LR,
t769 schedule, mouthweight 14B teacher, 5000-iter syncnet-trained DF
init, FSDP per-submodule wrap fix.

Pairs with: scripts/train_sf_full_ft_t769_no_reward.sh.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_full_ft_t769 as _base


def create_config():
    config = _base.create_config()

    config.model.reward.enabled = False
    config.model.reward_beta = 0

    config.log_config.name = "sf_full_ft_t769_no_reward"
    return config


config = create_config()
