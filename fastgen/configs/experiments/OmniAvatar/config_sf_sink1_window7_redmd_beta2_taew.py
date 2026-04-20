# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SF + Re-DMD beta=2 with TAEW decoder for the reward path.

Differs from config_sf_sink1_window7_redmd_beta2.py only in:
  - config.model.reward.decoder_kind = "taew"
  - config.model.reward.taew_checkpoint_path = <taew2_1.pth>
  - config.log_config.name suffixed with "_taew"

Reward-path pixel decode goes through TAEHVDecoderWrapper (11.3M-param tiny
VAE) instead of the full Wan 2.1 VAE (127M params). Value-wise MAE vs Wan
VAE is ~0.011 on [-1, 1] scale — indistinguishable for SyncNet-v2's 224x224
face-crop-level reward computation.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd_beta2 as _beta2_base


TAEW_CKPT = "/home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth"


def create_config():
    config = _beta2_base.create_config()

    config.model.reward.decoder_kind = "taew"
    config.model.reward.taew_checkpoint_path = TAEW_CKPT

    # The beta2 base config leaves log_config.name as the beta0p25 string
    # (a copy-paste holdover). Set the correct name here so the TAEW variant
    # is unambiguous.
    config.log_config.name = "sf_sink1_window7_redmd_audiofix_beta2_taew"
    return config


config = create_config()
