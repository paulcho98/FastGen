# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DF (shift=5) for the 14B causal student.

Inherits from config_df_shift_5.py. Three substantive changes:

1) Student network: 1.3B -> 14B (model_size, base_model_paths,
   omniavatar_ckpt_path). Uses the mouthweight 14B step-6000 V2V adapter
   (the same teacher we use for SF distillation) as construction-time
   init. With merge_lora=True the adapter is fused into the base 14B
   weights for downstream training.

2) Distributed strategy: DDP -> FSDP. 14B in fp32 is ~56 GB params +
   ~112 GB Adam state = ~170 GB per replica, which doesn't fit per-GPU
   under DDP even on H200 (143 GB). FSDP shards model+optim across the
   4 GPUs. Precision: bfloat16 forward, float32 master/optim — same
   recipe the SF stage uses for its 14B teacher.

3) Effective batch: 16 (default) via per-GPU batch_size=1 +
   grad_accum_rounds=4 across 4 GPUs. The 1.3B run used effective 64
   (16*4*1); 14B can't hold that even sharded, so we drop 4x. User can
   bump to grad_accum_rounds=2 for effective 8 if memory still tight,
   or to 8 for effective 32 if there's headroom.

Walltime estimate (4x H200): roughly 100-200 s/iter at effective batch
16 with FSDP all-gather overhead, => ~5000 iters in 6-10 days. Plan
accordingly.

Disk: each FSDP DF save is ~85 GB (model+optim, sharded). With 279 GB
free on /home/work, pair with SAVE_EVERY=2500 in the wrapper to cap at
2-3 saves. Strip optim immediately after each save if you want to
keep more steps.
"""

import os

import fastgen.configs.experiments.OmniAvatar.config_df_shift_5 as _df_base


OMNIAVATAR_ROOT = os.getenv("OMNIAVATAR_ROOT", "/home/work/.local/OmniAvatar")

# Student adapter for 14B: the mouthweight 14B step-6000 ckpt — same as our
# SF teacher. Override via OMNIAVATAR_STUDENT_CKPT_14B if you want to swap
# (e.g., to plain phase2 14B step-10500 for an ablation).
STUDENT_CKPT_14B = os.getenv(
    "OMNIAVATAR_STUDENT_CKPT_14B",
    "/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/step-6000.pt",
)

WAN_14B_BASE = ",".join(
    f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-{i:05d}-of-00006.safetensors"
    for i in range(1, 7)
)


def create_config():
    config = _df_base.create_config()

    # ---- Switch student to 14B ----
    config.model.net.model_size = "14B"
    config.model.net.base_model_paths = WAN_14B_BASE
    config.model.net.omniavatar_ckpt_path = STUDENT_CKPT_14B
    # The 14B teacher uses merge_lora=True to fuse the adapter into the base
    # before training. Mirror it for the 14B student so DF starts from the
    # fused state, not a base+LoRA stacked state.
    config.model.net.merge_lora = True

    # ---- DDP -> FSDP ----
    config.trainer.ddp = False
    config.trainer.fsdp = True
    # FSDP knobs (mirror SF stage's working settings).
    config.trainer.fsdp_min_num_params = int(1e8)
    config.trainer.fsdp_cpu_offload = False
    config.trainer.fsdp_sharding_group_size = None  # default = world_size
    # Mixed-precision FSDP: bf16 fwd/bwd, fp32 master+optim (so Adam's m/v
    # don't lose precision). Non-sharded modules get cast to bf16 too.
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"
    # Meta-init disabled: OmniAvatar's wan_model stores RoPE as a Python attr
    # (`self._core.freqs = precompute_freqs_cis_3d(...)` in wan_model.py:305),
    # not a registered buffer, and the (Causal)OmniAvatarWan classes don't
    # implement `reset_parameters()`. With fsdp_meta_init=True, ranks 1+
    # build freqs as meta complex tensors at construction; the FSDP wrap
    # path's `to_empty` and state_dict broadcast both skip Python attrs, so
    # the first forward's `freqs.to(cuda)` would allocate uninitialized GPU
    # memory on non-rank-0 ranks => garbage RoPE => training broken from
    # iter 1.  All SF configs leave this False (default) for the same
    # reason; we follow suit here. Cost: 4x host RAM at construction (~120
    # GB across 4 ranks for fp32 14B), well within the host's capacity.
    config.model.fsdp_meta_init = False

    # ---- Effective batch 16 = 1/GPU * 4 GPUs * grad_accum 4 ----
    # Wrapper overrides BATCH_SIZE on the cmdline; we set grad_accum here.
    config.trainer.grad_accum_rounds = 4
    config.model.grad_accum_rounds = 4

    return config


config = create_config()
