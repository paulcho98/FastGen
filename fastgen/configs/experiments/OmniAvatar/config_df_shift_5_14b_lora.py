# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DF (shift=5) for the 14B causal student — LoRA blocks + selective unfreeze.

Inherits from config_df_shift_5_14b.py.  Substantive differences:

1) ``merge_lora=False``: instead of fusing the V2V adapter into the base
   weights and full-fine-tuning all 14B params, keep the LoRA adapters
   separate (PEFT-injected) and train only the LoRA A/B matrices on the
   transformer blocks.  The base 14B weights stay frozen.

2) ``unfreeze_modules`` selectively re-enables ``requires_grad`` on
   submodules that DO need to fully adapt to the causal-student dynamics:
   - ``_core.audio_proj``: AudioPack input projection (audio -> hidden)
   - ``_core.audio_cond_projs``: per-block audio cross-attn projections
   - ``_core.patch_embedding``: input Conv3d for the V2V channels

   The rationale: the lip-sync gap we observe across SF runs appears to
   be most strongly tied to the audio path's ability to adapt under
   distillation, and constraining the bulk of the network to a low-rank
   update while keeping the audio path full-rank is a hypothesis-aligned
   experiment.  See ``docs/lora_selective_unfreeze.md`` for context.

3) Optimizer state shrinks dramatically: with ``merge_lora=True`` Adam
   m+v on 14B fp32 is ~107 GB per save.  With this config the trainable
   params are LoRA(rank=128) on the q/k/v/o/ffn linears plus the audio
   path (~100M) + patch_embedding (~3M) = roughly 50-150M trainable
   params.  Optim state per save drops to <1 GB.  This sidesteps the
   disk-pressure issue around save 6 in the full-FT run entirely.

4) Effective batch and FSDP knobs are unchanged from the parent.
"""

import fastgen.configs.experiments.OmniAvatar.config_df_shift_5_14b as _full_ft_base


# Submodules to keep fully trainable alongside LoRA on the transformer blocks.
# Paths are dotted, relative to the CausalOmniAvatarWan instance (so they
# include the "_core." prefix where the actual modules live).
DEFAULT_UNFREEZE_MODULES = [
    "_core.audio_proj",
    "_core.audio_cond_projs",
    "_core.patch_embedding",
]


def create_config():
    config = _full_ft_base.create_config()

    # ---- Switch from full FT to LoRA + selective unfreeze ----
    config.model.net.merge_lora = False
    config.model.net.unfreeze_modules = DEFAULT_UNFREEZE_MODULES

    # LoRA hyperparameters.  Match the V2V adapter we're loading from
    # (rank=128, alpha=64) — values come from the OmniAvatar V2V training
    # recipe and are what the saved adapter weights were trained at.
    # Changing these would require re-initializing the LoRA matrices
    # from scratch.
    config.model.net.lora_rank = 128
    config.model.net.lora_alpha = 64

    return config


config = create_config()
