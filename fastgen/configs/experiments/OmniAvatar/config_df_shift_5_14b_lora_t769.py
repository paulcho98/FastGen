# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DF (shift=5) for the 14B causal student — LoRA + selective unfreeze + t769 schedule.

Combines two existing variants:

1) config_df_shift_5_14b_lora.py provides:
   - model_size="14B" student
   - mouthweight 14B step-6000 init (via STUDENT_CKPT_14B env var
     defaulting to /home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/step-6000.pt)
   - merge_lora=False (PEFT injects LoRA on transformer blocks)
   - unfreeze_modules=["_core.audio_proj", "_core.audio_cond_projs",
     "_core.patch_embedding"] (full FT on the audio path + patch embedding)
   - lora_rank=128, lora_alpha=64
   - FSDP + bf16 fwd / fp32 master+optim

2) config_df_shift_5_t769.py's schedule narrowing:
   - sample_t_cfg.t_list = [0.999, 0.769, 0.0]
   - student_sample_steps = 2

Effect: the DF student is only ever trained at the noise levels the SF
t769 run will hit at inference (input on step 1 = t=0.999, input on
step 2 = t=0.769), removing train/test mismatch between DF and SF
schedules — same rationale as the 1.3B t769 run.

Pairs with:
- scripts/train_omniavatar_df_shift_5_14b_lora_t769.sh (this DF wrapper)
- scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_t769.sh
  (the SF wrapper that would consume the resulting DF ckpt as student init)
"""

import fastgen.configs.experiments.OmniAvatar.config_df_shift_5_14b_lora as _lora_base


def create_config():
    config = _lora_base.create_config()

    # 2-step schedule overrides.  With student_sample_steps=2, the student
    # is trained on the two intervals (0.999 -> 0.769) and (0.769 -> 0.0)
    # — the exact intervals SF t769 inference uses.  All other settings
    # (LoRA, unfreeze_modules, FSDP, mixed precision, mouthweight init,
    # grad_accum, FSDP knobs) come unchanged from _lora_base.
    config.model.sample_t_cfg.t_list = [0.999, 0.769, 0.0]
    config.model.student_sample_steps = 2

    return config


config = create_config()
