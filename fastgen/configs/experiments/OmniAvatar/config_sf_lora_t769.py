# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoRA + selective-unfreeze 1.3B SF — student + fake_score symmetric LoRA.

The 1.3B counterpart of ``config_sf_14b_lora_t769.py``.  Both networks
train as LoRA on transformer blocks + full-FT on the audio path +
patch_embedding (the same recipe used at 14B).  This is the LoRA
ablation arm of the t769 SF experiment — pairs with
``config_sf_full_ft_t769.py`` for the full-FT ablation, with everything
else (DF init, teacher, schedule, batch, FSDP fix) held identical.

Why useful:
    - Tests whether constraining the bulk of the 1.3B student to a
      low-rank update (with full-FT only on the audio-path submodules)
      reduces the student-vs-teacher Sync-C gap relative to full-FT
      training.  Lip-sync quality has historically been most sensitive
      to the audio path; a LoRA regime focused there may converge faster
      or to a better minimum.
    - The 14B SF run uses this exact regime — symmetric 1.3B LoRA gives
      a same-regime, smaller-model comparison point.

Differences vs config_sf_full_ft_t769.py:
    1. Student (config.model.net): merge_lora=True -> merge_lora=False
       with PEFT injection (LoRA on q/k/v/o/ffn linears, rank=128,
       alpha=64).
    2. Student gets unfreeze_modules covering the audio path + patch.
    3. Fake_score: merge_lora=False (already; legacy default), but
       unfreeze_modules added so apply_lora_freeze fires symmetrically
       (matches student regime; keeps audio path fully trainable).
    4. Both LRs at 2e-6 (matched, same as full_ft).

The apply_lora_freeze gating fix (commit f049693) ensures the freeze
hook only fires when unfreeze_modules is non-empty, so this config
correctly enters the LoRA + unfreeze regime while
config_sf_full_ft_t769.py (which does NOT set unfreeze_modules on
either net) stays in full-FT.

Pairs with: scripts/train_sf_lora_t769.sh.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd_beta2_taew as _base


# Submodule paths to keep fully trainable alongside LoRA.  Path-prefix
# differs between the causal and bidirectional classes:
#   - CausalOmniAvatarWan (student): WanModel lives at self._core
#   - OmniAvatarWan (fake_score):    WanModel lives at self.model
STUDENT_UNFREEZE = [
    "_core.audio_proj",
    "_core.audio_cond_projs",
    "_core.patch_embedding",
]
FAKE_SCORE_UNFREEZE = [
    "model.audio_proj",
    "model.audio_cond_projs",
    "model.patch_embedding",
]


def create_config():
    config = _base.create_config()

    # ---- Student: full-FT -> LoRA + selective unfreeze ----
    config.model.net.merge_lora = False
    config.model.net.unfreeze_modules = STUDENT_UNFREEZE
    config.model.net.lora_rank = 128
    config.model.net.lora_alpha = 64

    # ---- Fake_score: already merge_lora=False; add unfreeze_modules ----
    # Without unfreeze_modules, apply_lora_freeze gates out (per
    # f049693) and fake_score falls through to the legacy 1.3B
    # asymmetric LoRA-only regime.  Setting unfreeze_modules engages
    # the gate -> apply_lora_freeze runs -> base frozen + LoRA + audio +
    # patch trainable, symmetric to student.
    config.model.fake_score_net.merge_lora = False
    config.model.fake_score_net.unfreeze_modules = FAKE_SCORE_UNFREEZE
    config.model.fake_score_net.lora_rank = 128
    config.model.fake_score_net.lora_alpha = 64

    # ---- Matched LRs (both 2e-6) ----
    config.model.fake_score_optimizer.lr = 2e-6

    # ---- t769 schedule ----
    config.model.sample_t_cfg.t_list = [0.999, 0.769, 0.0]

    # ---- Coherence asserts ----
    assert config.model.fake_score_net.merge_lora is False, (
        "config_sf_lora_t769 requires fake_score_net.merge_lora=False (LoRA regime)"
    )
    assert list(config.model.fake_score_net.unfreeze_modules) == FAKE_SCORE_UNFREEZE, (
        "config_sf_lora_t769 requires fake_score_net.unfreeze_modules to be set "
        "for the apply_lora_freeze gate to engage"
    )
    assert config.model.net_optimizer.lr == config.model.fake_score_optimizer.lr, (
        f"config_sf_lora_t769 requires matched LRs; got "
        f"net={config.model.net_optimizer.lr} != "
        f"fake_score={config.model.fake_score_optimizer.lr}"
    )

    # ---- ITER CAP ----
    config.trainer.max_iter = 600

    config.log_config.name = "sf_lora_t769"
    return config


config = create_config()
