# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SF + Re-DMD (sync-C reward) with TAEW decoder for the reward path.

Differs from config_sf_sink1_window7_redmd.py only in:
  - config.model.reward.decoder_kind = "taew"
  - config.model.reward.taew_checkpoint_path = <taew2_1.pth>
  - config.log_config.name suffixed with "_taew"

Everything else (beta=0.25, 2-step distillation, timestep-conditional CFG,
sliding-window attention, joonson-parity SyncCScorer preprocessing) is
inherited unchanged from the base Re-DMD config.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd as _redmd_base


TAEW_CKPT = "/home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth"


def create_config():
    config = _redmd_base.create_config()

    config.model.reward.decoder_kind = "taew"
    config.model.reward.taew_checkpoint_path = TAEW_CKPT

    config.log_config.name = "sf_sink1_window7_redmd_syncc_beta0p25_joonson_parity_taew"
    return config


config = create_config()
