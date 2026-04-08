# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SF config with sliding window attention: 1 sink frame + 7 frame window + dynamic RoPE.

local_attn_size=7 means each chunk sees at most 7 frames total (including the 1 sink).
Effective rolling window = 7 - 1 = 6 frames.

Inherits everything from config_sf.py, only overrides:
  - local_attn_size: -1 → 7
  - sink_size: 0 → 1
  - use_dynamic_rope: already True (kept)
  - pretrained_ckpt_path: new stochastic-attention DF checkpoint
"""

import os
import fastgen.configs.experiments.OmniAvatar.config_sf as config_sf_base


def create_config():
    config = config_sf_base.create_config()

    # Sliding window: 1 sink + 6 rolling = 7 total visible frames
    config.model.net.local_attn_size = 7
    config.model.net.sink_size = 1
    config.model.net.use_dynamic_rope = True

    # DF checkpoint trained with stochastic attention (includes sink1+window7 config)
    config.trainer.checkpointer.pretrained_ckpt_path = os.getenv(
        "OMNIAVATAR_DF_CKPT",
        "/home/work/.local/hyunbin/checkpoints/df_4gpu_bs16_stochastic_attn_shift5/0010000.pth",
    )

    # 2-step distillation: t_list[0] and t_list[2] from the 4-step schedule
    config.model.sample_t_cfg.t_list = [0.999, 0.833, 0.0]
    config.model.student_sample_steps = 2

    config.log_config.name = "sf_sink1_window7_dynrope_2step"

    return config
