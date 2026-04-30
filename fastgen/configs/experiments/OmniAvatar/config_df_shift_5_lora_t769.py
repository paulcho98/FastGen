# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DF (shift=5) for the 1.3B causal student — LoRA + selective unfreeze + t769.

The 1.3B counterpart of ``config_df_shift_5_14b_lora_t769.py``.  Trains
the 1.3B causal student as LoRA on transformer blocks + full-FT on the
audio path + patch_embedding, on the t769 2-step schedule.

Why useful:
    - The DF init the 1.3B SF runs use was always full-FT (saved in the
      legacy ``df_audiofix_syncnet_trained_shift_5_t769_4gpu_bs16_lr1e5_5000iter``
      run).  This config adds a LoRA-only DF training option so the SF
      LoRA ablation (config_sf_lora_t769.py) can be initialized from a
      DF run that's already in the LoRA regime, instead of full-FT.
    - Lets us measure whether the SF LoRA regime benefits from a LoRA
      DF init vs a full-FT DF init.

Differences vs config_df_shift_5_t769.py:
    1. ``merge_lora=False``: keep V2V LoRA as PEFT layers instead of
       fusing into base.
    2. ``unfreeze_modules=[_core.audio_proj, _core.audio_cond_projs,
       _core.patch_embedding]``: full-FT the audio path.
    3. ``lora_rank=128``, ``lora_alpha=64`` (match V2V mouthweight).

The apply_lora_freeze hook in OmniAvatarDiffusionForcingModel.build_model
sees unfreeze_modules non-empty -> fires -> base frozen, LoRA + audio +
patch trainable (per f049693's gating).  Optim state shrinks to
~150M trainable params instead of ~1421M.

Pairs with: scripts/train_omniavatar_df_shift_5_lora_t769.sh.
"""

import fastgen.configs.experiments.OmniAvatar.config_df_shift_5_t769 as _t769_base


# Same paths as the 14B variant (relative to CausalOmniAvatarWan instance).
DEFAULT_UNFREEZE_MODULES = [
    "_core.audio_proj",
    "_core.audio_cond_projs",
    "_core.patch_embedding",
]


def create_config():
    config = _t769_base.create_config()

    # ---- Switch from full-FT to LoRA + selective unfreeze ----
    config.model.net.merge_lora = False
    config.model.net.unfreeze_modules = DEFAULT_UNFREEZE_MODULES
    config.model.net.lora_rank = 128
    config.model.net.lora_alpha = 64

    # Coherence asserts.
    assert config.model.net.merge_lora is False, (
        "config_df_shift_5_lora_t769 requires net.merge_lora=False"
    )
    assert list(config.model.net.unfreeze_modules) == DEFAULT_UNFREEZE_MODULES, (
        "config_df_shift_5_lora_t769 requires net.unfreeze_modules set so "
        "apply_lora_freeze fires (per gating in f049693)"
    )

    # ---- ITER CAP ----
    config.trainer.max_iter = 600

    return config


config = create_config()
