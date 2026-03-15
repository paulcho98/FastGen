# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment config for OmniAvatar Causal KD (Stage 1 of Self-Forcing pipeline).

Pre-trains the causal 1.3B student on ODE trajectories from the 14B teacher.
"""

import os
import fastgen.configs.methods.config_omniavatar_kd as config_kd_default

from fastgen.utils import LazyCall as L
from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

# ---- Paths ----
OMNIAVATAR_ROOT = os.getenv("OMNIAVATAR_ROOT", "/home/work/.local/OmniAvatar")
STUDENT_CKPT = os.getenv("OMNIAVATAR_STUDENT_CKPT", None)

# ---- Student network config ----
CausalOmniAvatar_V2V_1_3B_Config: dict = L(CausalOmniAvatarWan)(
    model_size="1.3B",
    in_dim=65,
    mode="v2v",
    use_audio=True,
    audio_hidden_size=32,
    chunk_size=3,
    total_num_frames=21,
    base_model_paths=f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    omniavatar_ckpt_path=STUDENT_CKPT,
    net_pred_type="flow",
    schedule_type="rf",
)


def create_config():
    config = config_kd_default.create_config()

    # Precision
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"

    # Input shape: 512x512 @ 81 frames → latent [16, 21, 64, 64]
    config.model.input_shape = [16, 21, 64, 64]

    # Student network
    config.model.net = CausalOmniAvatar_V2V_1_3B_Config
    config.model.net.total_num_frames = config.model.input_shape[1]

    # Timestep schedule (must match ODE trajectory generation)
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # KD settings
    config.model.student_sample_steps = 4

    # Training
    config.dataloader_train.batch_size = 1
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 10
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "omniavatar_kd"
    return config
