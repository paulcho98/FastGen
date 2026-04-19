# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-DMD beta=2 + TAEW decoder + GAN (adversarial) loss.

Inherits from config_sf_sink1_window7_redmd_beta2_taew.py and adds the GAN
settings from WanT2V/config_sf_14b.py (the canonical Wan 14B Self-Forcing
recipe with GAN on). OmniAvatar's base `config_sf.py:122` explicitly disables
GAN to save ~35 GB VRAM; we re-enable it here with a reduced dataloader batch
and bumped grad_accum_rounds to stay within the H200 141 GB budget.

Why this is safe on top of reward-weighted VSD:
  - `fc56e4a` fixed per-sample reward-loss coupling at batch>1.
  - `7f96e6f` added Z(c) self-normalization to `_apply_reward_weighting`.
    Weighted VSD magnitude is now bounded in [min(L), max(L)], so the
    `gan_loss_weight_gen=0.003` term retains its tuned relative strength
    (wasn't the case under the prior unnormalized `mean(w*L)` form, which
    at beta=2 was O(1000x) the GAN term).

Memory plan:
  - Dataloader batch 8 -> 4 (halves student+critic activations).
  - grad_accum_rounds 2 -> 4 (effective batch stays at 64 = 4*4*4).
  - Discriminator (Wan 14B config: 40 blocks, inner_dim=1280): ~250M params,
    ~3-5 GB in bf16 + grads + Adam moments.
  - Additional teacher-feature activations during GAN feature extraction:
    ~15-25 GB depending on where the bottleneck lands.
  - Expected VRAM: ~109 GB (baseline) - ~20 GB (batch halved) + ~30 GB (GAN)
    = ~119 GB. Leaves ~22 GB H200 headroom — smoke test to verify.

Launch via scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_gan.sh.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd_beta2_taew as _taew_base
from fastgen.configs.discriminator import Discriminator_Wan_14B_Config


def create_config():
    config = _taew_base.create_config()

    # ---- GAN recipe (matches WanT2V/config_sf_14b.py defaults) ----
    config.model.gan_loss_weight_gen = 0.003
    config.model.discriminator = Discriminator_Wan_14B_Config
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [21, 30, 39]  # 14B teacher: 40 blocks
    config.model.discriminator_optimizer.lr = 5e-6
    config.model.gan_use_same_t_noise = True
    # R1 regularization on the discriminator left off for the smoke; enable
    # later if discriminator collapses:
    # config.model.gan_r1_reg_weight = 0.1

    # ---- Memory budget: shrink dataloader batch, preserve effective batch ----
    config.dataloader_train.batch_size = 4
    config.trainer.grad_accum_rounds = 4
    config.model.grad_accum_rounds = 4   # mirror for fake_score loss scaling

    # ---- Run metadata ----
    config.log_config.name = "sf_sink1_window7_redmd_audiofix_beta2_taew_gan"
    return config


config = create_config()
