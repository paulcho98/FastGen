# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Short diagnostic Re-DMD run for smoke testing.

Inherits the full Re-DMD config and narrows training: short max_iter,
grad_accum=1, batch_size=1, and enables the rank-0 VAE-decoded debug MP4
dump so we can visually inspect what the SyncCScorer sees.

Launch via scripts/smoke_test_redmd.sh.
"""

from fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd import (
    create_config as _redmd_create_config,
)


def create_config():
    config = _redmd_create_config()

    # Narrow to a smoke-sized run.
    # Trainer loop is range(1, max_iter), so max_iter=11 gives training
    # iterations 1..10, including student steps at iter 5 and 10 (the two
    # reward calls we want to observe).
    config.trainer.max_iter = 11
    config.trainer.grad_accum_rounds = 1
    config.dataloader_train.batch_size = 1

    # Turn on the diagnostic dump
    config.model.save_reward_debug_video = True
    config.model.reward_debug_dir = "logs/redmd_smoke_debug"

    # Distinct run name
    config.log_config.name = "sf_sink1_window7_redmd_syncc_beta0p25_smoke"

    # Reduce eval noise — no validation during smoke if the base config had it
    if hasattr(config.trainer, "eval_period"):
        config.trainer.eval_period = 999999

    return config


config = create_config()
