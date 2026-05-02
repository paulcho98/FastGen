# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symmetric full-FT 1.3B SF — no reward, NO CFG at all.

Sibling of ``config_sf_full_ft_t769_no_reward.py``.  Differences:

    config.model.guidance_scale = None
    config.model.timestep_cfg.enabled = False

Effect: the negative teacher forward pass is completely skipped
(the guard at dmd2._student_update_step checks
``guidance_scale is not None`` before calling
``_apply_classifier_free_guidance``).  This halves the teacher
compute per iteration — only the positive-conditioned forward runs.

For the t769 2-step schedule:
    Step 1 (t=0.999): no CFG (teacher runs once, positive only)
    Step 2 (t=0.769): no CFG (same)

Tests the hypothesis that CFG is not needed for distillation quality
— the teacher's unconditional prediction gap may already be captured
by the VSD loss without explicit guidance.

Everything else identical to config_sf_full_ft_t769_no_reward.py:
symmetric full-FT both nets (1421M each), matched 2e-6 LRs, reward
disabled, 600-iter cap, same DF init / teacher / fake_score.

Pairs with: scripts/train_sf_full_ft_t769_no_reward_no_cfg.sh.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_full_ft_t769_no_reward as _base


def create_config():
    config = _base.create_config()

    config.model.guidance_scale = None
    config.model.timestep_cfg.enabled = False

    config.log_config.name = "sf_full_ft_t769_no_reward_no_cfg"
    return config


config = create_config()
