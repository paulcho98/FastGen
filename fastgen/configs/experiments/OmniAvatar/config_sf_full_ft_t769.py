# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symmetric full-FT 1.3B SF — student + fake_score both full-FT.

Background — the asymmetry bug this config fixes:
    The base SF config (config_sf.py) sets the fake_score with
    ``merge_lora=False`` while the student inherits ``merge_lora=True``
    (CausalOmniAvatarWan default).  At init_optimizers time, this causes:

        - Student: V2V LoRA fused into base at construction → no PEFT
          layers → all 1.3B params trainable → optim has full Adam state
          (~1.3B params).
        - Fake_score: V2V LoRA stays as PEFT layers → PEFT defaults set
          base ``requires_grad=False``, ``lora_A/B requires_grad=True`` →
          init_optimizers builds Adam state for ONLY the LoRA + audio +
          patch params (~175M).

    The per-iter ``requires_grad_(True)`` wipes in
    ``dmd2._setup_grad_requirements`` flip everything to True, but the
    optimizer is already built — those wipes only affect the save filter,
    not what actually trains.

    Net effect: the critic has 8× LESS trainable capacity than the student.
    The fake_score cannot keep up with the student's distribution shift,
    producing biased VSD gradients and contributing to the persistent
    student-vs-teacher Sync-C gap observed across all legacy 1.3B SF runs.

What this config changes vs config_sf_sink1_window7_redmd_beta2_taew.py:
    1. ``config.model.fake_score_net.merge_lora = True`` — V2V LoRA fuses
       into fake_score base at construction → no PEFT → full-FT all 1.3B
       params (matches student's full-FT regime).
    2. ``config.model.fake_score_optimizer.lr = 2e-6`` — match student's
       LR.  The legacy 1.3B parent wrapper hardcoded the critic LR at 3e-6
       (1.5× student) as a half-fix for the asymmetry.  With symmetric
       capacity, the asymmetric LR is no longer needed (and could in fact
       harm convergence by making the critic outpace the student).
    3. ``t_list = [0.999, 0.769, 0.0]`` — t769 schedule (matches the
       recent fsmatched_t769 runs).
    4. Sanity asserts on merge_lora coherence at config-load time.

What stays the same:
    - DF init via OMNIAVATAR_DF_CKPT (parent script env)
    - Teacher: 14B mouthweight step-6000 (frozen, merge_lora=True)
    - Reward: enabled (Re-DMD beta=2 + TAEW decoder)
    - Effective batch: BS=8 * GA=2 * 4 GPUs = 64
    - max_iter, save_ckpt_iter inherited from base (5000, 100)
    - FSDP per-submodule wrap fix (network_causal.py:2386-2407) — code-level

Pairs with: scripts/train_sf_full_ft_t769.sh (the wrapper script).
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd_beta2_taew as _base


def create_config():
    config = _base.create_config()

    # ---- SYMMETRIC FULL-FT ----
    # Force fake_score to full-FT by fusing the V2V LoRA at construction.
    # Without this override, fake_score is asymmetric LoRA-only (~175M).
    config.model.fake_score_net.merge_lora = True

    # ---- MATCHED LR ----
    # Both networks at 2e-6 (legacy parent hardcoded 3e-6 for fake_score
    # to compensate for the capacity asymmetry; with symmetric capacity
    # the LR asymmetry is no longer warranted).
    config.model.fake_score_optimizer.lr = 2e-6

    # ---- t769 SCHEDULE ----
    config.model.sample_t_cfg.t_list = [0.999, 0.769, 0.0]

    # ---- COHERENCE ASSERTS ----
    # Catch misconfigurations at config-load time.  fake_score_net.merge_lora
    # was just set above, so direct attribute access is safe.  Student
    # merge_lora is not asserted because CausalOmniAvatar_V2V_1_3B_Student
    # in config_sf.py doesn't set it explicitly — it relies on the
    # CausalOmniAvatarWan constructor default of True (full-FT regime).
    assert config.model.fake_score_net.merge_lora is True, (
        f"config_sf_full_ft_t769 requires fake_score_net.merge_lora=True for "
        f"symmetric full-FT (otherwise fake_score is LoRA-only ~175M, "
        f"vs student full-FT ~1.3B — asymmetric).  Got: "
        f"{config.model.fake_score_net.merge_lora}"
    )

    assert config.model.net_optimizer.lr == config.model.fake_score_optimizer.lr, (
        f"config_sf_full_ft_t769 requires matched LRs; got "
        f"net={config.model.net_optimizer.lr} != "
        f"fake_score={config.model.fake_score_optimizer.lr}"
    )

    config.log_config.name = "sf_full_ft_t769"
    return config


config = create_config()
