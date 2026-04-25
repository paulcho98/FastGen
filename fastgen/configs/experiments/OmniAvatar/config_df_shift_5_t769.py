# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DF (shift=5) specialized for the SF t769 (2-step, intermediate at step 30) schedule.

Inherits everything from config_df_shift_5.py but collapses the noise-level
training distribution from the default 4 student denoising steps
(t_list=[0.999, 0.937, 0.833, 0.624, 0.0], student_sample_steps=4) down to
the 2-step schedule that the matching SF run uses
(t_list=[0.999, 0.769, 0.0], student_sample_steps=2).

Effect: the DF student is only ever trained at the noise levels SF
inference will hit (t=0.999 input on step 1, t=0.769 input on step 2),
instead of being spread across 4 different training timesteps that include
levels (0.937, 0.624) the SF run will never see.

Pairs with:
- scripts/train_omniavatar_df_shift_5_t769_audiofix_syncnet_trained.sh
  (the DF wrapper that trains with this config)
- scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_t769.sh
  (the SF wrapper that consumes the resulting DF ckpt as student init)
"""

import fastgen.configs.experiments.OmniAvatar.config_df_shift_5 as _df_base


def create_config():
    config = _df_base.create_config()

    # 2-step schedule: t_list values are boundary timesteps; with
    # student_sample_steps=2 the student is trained on the two intervals
    # (0.999 -> 0.769) and (0.769 -> 0.0).
    config.model.sample_t_cfg.t_list = [0.999, 0.769, 0.0]
    config.model.student_sample_steps = 2

    return config


config = create_config()
