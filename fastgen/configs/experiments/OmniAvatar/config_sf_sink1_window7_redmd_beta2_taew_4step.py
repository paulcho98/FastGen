# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SF + Re-DMD beta=2 with TAEW decoder — 4-step student distillation.

Differs from config_sf_sink1_window7_redmd_beta2_taew.py only in:
  - config.model.student_sample_steps: 2 -> 4
  - config.model.sample_t_cfg.t_list: [0.999, 0.833, 0.0]
                                     -> [0.999, 0.937, 0.833, 0.624, 0.0]
  - config.log_config.name suffixed with "_4step"

Everything else (sink_size=1, local_attn_size=7, use_dynamic_rope=True,
timestep_cfg.enabled=True, shift=5.0, beta=2 Re-DMD reward, TAEW decoder,
DF checkpoint init) is inherited unchanged.

The 2-step base (config_sf_sink1_window7_tscfg.py:35-37) hard-codes the
"distill 4 -> 2 steps" simplification with the comment "2-step
distillation: t_list[0] and t_list[2] from the 4-step schedule". We undo
that simplification here so the student trains with the original 4-step
schedule. The assertion at
`fastgen/methods/distribution_matching/self_forcing.py:139`
(`len(t_list) - 1 == student_sample_steps`) still holds: 5 - 1 == 4.

Trade-offs vs the 2-step variant:
  - Each student rollout runs 4 denoising passes instead of 2, so the
    student forward portion of each iteration is ~2x slower.
  - Peak VRAM during rollout grows with the number of cached intermediate
    latents; expect ~1.5-2x memory of the 2-step variant at the same batch
    size. If the 2-step run was near the memory ceiling, consider halving
    train.batch_size_per_gpu or switching to grad_accum_rounds=4.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd_beta2_taew as _taew_base


def create_config():
    config = _taew_base.create_config()

    # Undo the 2-step override from config_sf_sink1_window7_tscfg.py:35-37.
    # These values match the base config_sf.py 4-step schedule (shift=5.0).
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    config.log_config.name = "sf_sink1_window7_redmd_audiofix_beta2_taew_4step"
    return config


config = create_config()
