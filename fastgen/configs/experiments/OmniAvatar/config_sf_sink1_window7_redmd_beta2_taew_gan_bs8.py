# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-DMD beta=2 + TAEW + GAN at dataloader_batch=8 (memory stress probe).

Inherits from config_sf_sink1_window7_redmd_beta2_taew_gan.py and reverts
the memory-budget overrides (batch 4 -> 8, grad_accum 4 -> 2), so this
matches the no-GAN TAEW baseline's batch sizing.

Purpose: empirically verify whether GAN+batch=8 OOMs on our H200 setup
rather than relying on the ~35 GB comment in config_sf.py:120. If this
fits, use it; if it OOMs, fall back to the batch=4 variant.

Everything else — GAN weights, discriminator config, TAEW decoder, reward
beta=2 — is identical to the batch=4 variant.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd_beta2_taew_gan as _gan_base


def create_config():
    config = _gan_base.create_config()

    # Revert memory-budget overrides back to the no-GAN baseline's sizing.
    config.dataloader_train.batch_size = 8
    config.trainer.grad_accum_rounds = 2
    config.model.grad_accum_rounds = 2  # mirror for fake_score loss scaling

    config.log_config.name = "sf_sink1_window7_redmd_audiofix_beta2_taew_gan_bs8"
    return config


config = create_config()
