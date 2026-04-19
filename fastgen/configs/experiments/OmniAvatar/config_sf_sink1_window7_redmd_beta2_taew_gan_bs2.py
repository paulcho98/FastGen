# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-DMD beta=2 + TAEW + GAN at batch=2/accum=8 (post-bs=4-OOM).

bs=4 (config_sf_sink1_window7_redmd_beta2_taew_gan.py) survived 4 critic
steps but OOMed on the first student step (iter 5): the student step's
teacher forward is NOT in torch.no_grad() (gradient flows back through
fake_feat -> teacher layers -> perturbed_data -> student), so it retains
full activations through layer 39. Critic-step teacher forwards ARE in
no_grad, so bs=4 was fine there. Peak footprint is the student step.

bs=2 halves activations again: the 3.28 GiB student-step allocation that
failed at bs=4 becomes ~1.64 GiB, which should fit (we had ~1 GiB free at
bs=4, gain ~16 GiB per batch-halving).

Effective batch preserved: 2 * 4 GPUs * 8 accum = 64.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd_beta2_taew_gan as _gan_base


def create_config():
    config = _gan_base.create_config()

    config.dataloader_train.batch_size = 2
    config.trainer.grad_accum_rounds = 8
    config.model.grad_accum_rounds = 8   # mirror for fake_score loss scaling

    config.log_config.name = "sf_sink1_window7_redmd_audiofix_beta2_taew_gan_bs2"
    return config


config = create_config()
