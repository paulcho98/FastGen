# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-DMD beta=2 + TAEW + GAN at batch=1/accum=16 (matches reference recipe).

The reference FastGen WanT2V/config_sf_14b.py uses batch_size=1 for 14B
teacher + GAN. Our OmniAvatar config_sf.py uses batch_size=8 (tuned for
no-GAN Re-DMD), which is why bs=8/4/2 probes all OOMed: the reference
recipe already anticipated that a 14B teacher with grad-retaining GAN
feature extraction is a 1-sample-per-GPU workload.

Effective batch preserved: 1 * 4 GPUs * 16 accum = 64.

Note: our V2V setup adds audio cross-attention + masked-video + ref-seq
conditioning on top of the plain Wan T2V forward — the teacher has
higher per-sample activation cost than the reference. If bs=1 is still
tight, the next step is a code-level fix (split `_compute_teacher_
prediction_gan_loss` into a no_grad teacher_x0 forward + a grad-retaining
return_features_early=True fake_feat forward, so the student step keeps
activations for only layers 0..39 instead of the full 40+output).
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd_beta2_taew_gan as _gan_base


def create_config():
    config = _gan_base.create_config()

    config.dataloader_train.batch_size = 1
    config.trainer.grad_accum_rounds = 16
    config.model.grad_accum_rounds = 16   # mirror for fake_score loss scaling

    # Memory optimization: move audio-cond add inside checkpoint boundary so
    # its pre-add x is recomputed rather than retained. Mathematically
    # equivalent to default path (guarded by
    # tests/test_wan_audio_checkpoint_scope.py). Only fires when the teacher
    # forward is grad-required, which is the student-step+GAN path — exactly
    # where bs=4 OOMed and bs=2/1 were tight.
    config.model.teacher.expand_audio_checkpoint_scope = True
    config.model.fake_score_net.expand_audio_checkpoint_scope = True

    config.log_config.name = "sf_sink1_window7_redmd_audiofix_beta2_taew_gan_bs1"
    return config


config = create_config()
