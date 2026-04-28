# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SF (Re-DMD beta=2 + TAEW) for the 14B causal student + 14B fake_score, LoRA + selective unfreeze, t769 schedule.

Inherits from config_sf_sink1_window7_redmd_beta2_taew.py and switches:

1) Student (config.model.net): 1.3B causal -> 14B causal, with
   ``merge_lora=False`` + ``unfreeze_modules`` selectively re-enabling
   the audio path + patch_embedding.  Init from the mouthweight 14B
   step-6000 LoRA adapter (same as the SF teacher).  The actual student
   state at training time will be overwritten by ``OMNIAVATAR_DF_CKPT``
   (the trained 14B DF LoRA checkpoint, fed via the wrapper into the
   trainer's checkpointer).

2) Fake_score (config.model.fake_score_net): 1.3B bidirectional -> 14B
   bidirectional, also LoRA + selective unfreeze.  Init from the same
   mouthweight 14B step-6000 LoRA adapter.

3) Teacher (config.model.teacher): unchanged — 14B bidirectional, frozen,
   merge_lora=True (the V2V LoRA fused into base at construction).

4) Schedule (matches t769): ``sample_t_cfg.t_list = [0.999, 0.769, 0.0]``,
   ``student_sample_steps = 2``.

5) Effective batch: BS=1/GPU * NGPU * grad_accum=4 = 16 (vs 1.3B SF's
   effective 64).  At 14B, three networks all FSDP-sharded plus
   activations from a 2-step student rollout + teacher forward + 2x
   fake_score forward push memory well above the 1.3B regime.  BS=1 is
   the safe starting point; BS=2 may be feasible after smoke confirms
   memory headroom.

6) FSDP knobs: bf16 forward, fp32 master/optim, fsdp_meta_init=False
   (RoPE Python-attr issue — same as 14B DF).

Why LoRA is mandatory at 14B SF: full-FT both student + fake_score
would need ~28 GB sharded Adam state per network = 56 GB just for
optim, plus 3 * 14 GB params = 42 GB, plus all-gather buffers + 2-step
student activations.  Doesn't fit on H200 (143 GB).  LoRA collapses
optim to ~1.2 GB per network, opening up enough memory budget to fit.

Pairs with:
- scripts/train_sf_..._14b_lora_t769.sh (the SF wrapper)
- The trained 14B DF LoRA t769 ckpt as DF init (running 2026-04-28,
  ETA 2026-04-30 ~08:00 KST)
"""

import os

import fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd_beta2_taew as _sf_taew_base


# Mouthweight 14B step-6000: same checkpoint as the SF teacher, used here
# to provide initial LoRA values + audio/patch weights for both the
# student and the fake_score before training.
STUDENT_CKPT_14B = os.getenv(
    "OMNIAVATAR_STUDENT_CKPT_14B",
    "/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/step-6000.pt",
)

OMNIAVATAR_ROOT = os.getenv("OMNIAVATAR_ROOT", "/home/work/.local/OmniAvatar")

WAN_14B_BASE = ",".join(
    f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-{i:05d}-of-00006.safetensors"
    for i in range(1, 7)
)


# Submodule paths (relative to each network) to keep fully trainable
# alongside the LoRA A/B matrices on the transformer blocks.  Different
# prefix between the causal and bidirectional classes:
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
    config = _sf_taew_base.create_config()

    # ---- Student: 1.3B causal -> 14B causal LoRA ----
    config.model.net.model_size = "14B"
    config.model.net.base_model_paths = WAN_14B_BASE
    config.model.net.omniavatar_ckpt_path = STUDENT_CKPT_14B
    config.model.net.merge_lora = False
    config.model.net.unfreeze_modules = STUDENT_UNFREEZE
    config.model.net.lora_rank = 128
    config.model.net.lora_alpha = 64

    # ---- Fake_score: 1.3B bidirectional -> 14B bidirectional LoRA ----
    config.model.fake_score_net.model_size = "14B"
    config.model.fake_score_net.base_model_paths = WAN_14B_BASE
    config.model.fake_score_net.omniavatar_ckpt_path = STUDENT_CKPT_14B
    # fake_score in the SF base config ALREADY uses merge_lora=False
    # (intentional — it's the trainable critic), but we set it explicitly
    # for clarity and add the unfreeze list (which the base config doesn't set).
    config.model.fake_score_net.merge_lora = False
    config.model.fake_score_net.unfreeze_modules = FAKE_SCORE_UNFREEZE
    config.model.fake_score_net.lora_rank = 128
    config.model.fake_score_net.lora_alpha = 64

    # ---- Teacher: stays 14B + merge_lora=True (frozen, full state) ----
    # No changes — inherited from the SF base config.

    # ---- Schedule: t769 (matches the 14B DF LoRA t769 run's training distribution) ----
    config.model.sample_t_cfg.t_list = [0.999, 0.769, 0.0]
    config.model.student_sample_steps = 2

    # ---- Effective batch 16 = 1/GPU * 4 GPUs * grad_accum=4 ----
    config.dataloader_train.batch_size = 1
    config.trainer.grad_accum_rounds = 4
    config.model.grad_accum_rounds = 4  # mirrors trainer.grad_accum_rounds for fake_score loss scaling

    # ---- FSDP knobs (mirror 14B DF LoRA setup) ----
    config.trainer.ddp = False
    config.trainer.fsdp = True
    config.trainer.fsdp_min_num_params = int(1e8)
    config.trainer.fsdp_cpu_offload = False
    config.trainer.fsdp_sharding_group_size = None
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"
    config.model.fsdp_meta_init = False

    return config


config = create_config()
