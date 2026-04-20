# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SF + Re-DMD (sync-C reward) on top of config_sf_sink1_window7_tscfg.

Changes vs the baseline SF config (config_sf_sink1_window7_tscfg.py):
  - model_class._target_ switches to OmniAvatarSelfForcingReDMD
  - model.reward sub-config added (SyncNet-v2 ckpt, input_fps=25, audio_sr=16k, vshift=15)
  - model.reward_beta=2  (Baseline beta=2)
  - model.center_reward=False
  - model.clamp_reward=None
  - dataloader_train gets load_raw_audio=True + matching audio kwargs
  - vae_path inherited from base chain (config_sf.py sets it; assert verifies it is truthy)

Design reference:
  /home/work/.local/hyunbin/Reward-Forcing/docs/sync_c_scorer_design.md
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_tscfg as _tscfg_base
from fastgen.configs.methods.config_omniavatar_sf import RewardConfig
from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD


def create_config():
    config = _tscfg_base.create_config()

    # ------------------------------------------------------------------ #
    # Switch model class to the Re-DMD variant                            #
    # ------------------------------------------------------------------ #
    # config.model_class is an OmegaConf DictConfig created by LazyCall.
    # Its _target_ key is what fastgen.utils.instantiate dispatches on.
    config.model_class._target_ = OmniAvatarSelfForcingReDMD

    # ------------------------------------------------------------------ #
    # Reward sub-config                                                   #
    # ------------------------------------------------------------------ #
    # Use RewardConfig (attrs class) so the value survives OmegaConf
    # serialization when train.py does config.model_class.config = config.model.
    # SimpleNamespace is NOT OmegaConf-compatible and gets stripped.
    config.model.reward = RewardConfig(
        enabled=True,
        checkpoint_path="/home/work/.local/eval_metrics/checkpoints/auxiliary/syncnet_v2.model",
        input_fps=25.0,
        audio_sample_rate=16000,
        vshift=15,
    )

    # Top-level reward knobs (read by _apply_reward_weighting)
    config.model.reward_beta = 2
    config.model.center_reward = False   # set True to subtract EMA(sync_c) for mean weight ≈ 1
    config.model.clamp_reward = None     # e.g. (0.0, 15.0) to bound exp(beta * r)

    # ------------------------------------------------------------------ #
    # VAE path (required for reward decode in _decode_gen_to_pixels)      #
    # ------------------------------------------------------------------ #
    # The base chain (config_sf.py) already sets config.model.vae_path via
    # OMNIAVATAR_VAE_PATH env or the default OmniAvatar install path.
    # Assert here so a misconfigured environment fails fast at import time
    # rather than deep inside the training loop.
    assert getattr(config.model, "vae_path", "") != "", (
        "config.model.vae_path must be set for Re-DMD — the reward path VAE-decodes "
        "the generator output to pixels. Either set OMNIAVATAR_VAE_PATH env or "
        "ensure the default OmniAvatar install path is present."
    )

    # ------------------------------------------------------------------ #
    # Data: raw waveform loading (required for audio_waveform in batch)   #
    # ------------------------------------------------------------------ #
    # config.dataloader_train is an OmegaConf DictConfig (from LazyCall);
    # these are passed as **kwargs through OmniAvatarDataLoader to OmniAvatarDataset.
    config.dataloader_train.load_raw_audio = True
    config.dataloader_train.raw_audio_sample_rate = 16000
    config.dataloader_train.raw_audio_num_frames = 81
    config.dataloader_train.raw_audio_fps = 25.0

    # ------------------------------------------------------------------ #
    # W&B run name                                                        #
    # ------------------------------------------------------------------ #
    config.log_config.name = "sf_sink1_window7_redmd_syncc_beta0p25"

    return config


# Module-level config object for smoke-test imports and interactive use.
# The training launcher (train.py) calls create_config() via config_utils; this
# attribute is not used by the launcher but allows:
#   from fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd import config
config = create_config()
