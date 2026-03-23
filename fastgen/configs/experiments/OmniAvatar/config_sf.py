# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment config for OmniAvatar Self-Forcing distillation.

14B teacher (bidirectional) → 1.3B student (causal) using Self-Forcing Stage 2.
"""

import os
from fastgen.configs.discriminator import Discriminator_Wan_14B_Config
import fastgen.configs.methods.config_omniavatar_sf as config_sf_default

from fastgen.configs.net import CKPT_ROOT_DIR
from fastgen.utils import LazyCall as L

from fastgen.networks.OmniAvatar.network import OmniAvatarWan
from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
from fastgen.datasets.omniavatar_dataloader import create_omniavatar_dataloader

# ---- Paths (override via CLI or env) ----
OMNIAVATAR_ROOT = os.getenv("OMNIAVATAR_ROOT", "/home/work/.local/OmniAvatar")
DATA_ROOT = os.getenv("OMNIAVATAR_DATA_ROOT", "/home/work/stableavatar_data/v2v_training_data")
TEACHER_CKPT = os.getenv(
    "OMNIAVATAR_TEACHER_CKPT",
    "/home/work/output_omniavatar_v2v_maskall_refseq_new_data_loss_weights_mouth_weights/step-1500.pt",
)
STUDENT_CKPT = os.getenv("OMNIAVATAR_STUDENT_CKPT", None)  # Set when 1.3B refseq is trained

# ---- Network configs ----
OmniAvatar_V2V_14B_Teacher: dict = L(OmniAvatarWan)(
    model_size="14B",
    in_dim=65,
    mode="v2v",
    use_audio=True,
    audio_hidden_size=32,
    base_model_paths=f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,"
                     f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,"
                     f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,"
                     f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,"
                     f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,"
                     f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
    omniavatar_ckpt_path=TEACHER_CKPT,
    merge_lora=True,
    net_pred_type="flow",
    schedule_type="rf",
)

OmniAvatar_V2V_1_3B_FakeScore: dict = L(OmniAvatarWan)(
    model_size="1.3B",
    in_dim=65,
    mode="v2v",
    use_audio=True,
    audio_hidden_size=32,
    base_model_paths=f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    omniavatar_ckpt_path=STUDENT_CKPT,
    merge_lora=False,  # Fake score is trainable, keep LoRA separate
    net_pred_type="flow",
    schedule_type="rf",
)

CausalOmniAvatar_V2V_1_3B_Student: dict = L(CausalOmniAvatarWan)(
    model_size="1.3B",
    in_dim=65,
    mode="v2v",
    use_audio=True,
    audio_hidden_size=32,
    chunk_size=3,
    total_num_frames=21,
    base_model_paths=f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    omniavatar_ckpt_path=STUDENT_CKPT,
    net_pred_type="flow",
    schedule_type="rf",
)


def create_config():
    config = config_sf_default.create_config()

    # Learning rates
    config.model.net_optimizer.lr = 5e-6
    config.model.discriminator_optimizer.lr = 5e-6
    config.model.fake_score_optimizer.lr = 5e-6

    # Precision
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"

    # Input shape: 512x512 @ 81 frames → latent [16, 21, 64, 64]
    config.model.input_shape = [16, 21, 64, 64]
    config.model.fake_score_pred_type = "x0"
    config.model.guidance_scale = 4.5

    # Networks
    config.model.net = CausalOmniAvatar_V2V_1_3B_Student
    config.model.net.total_num_frames = config.model.input_shape[1]
    config.model.teacher = OmniAvatar_V2V_14B_Teacher
    config.model.fake_score = OmniAvatar_V2V_1_3B_FakeScore

    # GAN settings — discriminator inner_dim must match teacher (14B, dim=5120)
    config.model.gan_loss_weight_gen = 0.003
    config.model.discriminator = Discriminator_Wan_14B_Config
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [15, 22, 29]
    config.model.gan_use_same_t_noise = True

    # Pretrained student from KD Stage 1
    # config.model.pretrained_student_net_path = f"{CKPT_ROOT_DIR}/OmniAvatar/checkpoints/ode_init.pt"

    # Timestep schedule — shift=3.0 matches the OmniAvatar teacher's training distribution
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 3.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Self-Forcing specific
    config.model.enable_gradient_in_rollout = True
    config.model.start_gradient_frame = 0
    config.model.same_step_across_blocks = True
    config.model.context_noise = 0.0

    # Dataloader
    config.dataloader_train = L(create_omniavatar_dataloader)(
        data_list_path=f"{DATA_ROOT}/video_square_path.txt",
        latentsync_mask_path=os.getenv(
            "LATENTSYNC_MASK_PATH",
            "/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png",
        ),
        batch_size=1,
        num_workers=4,
        neg_text_emb_path=os.getenv("NEG_TEXT_EMB_PATH", None),
        use_ref_sequence=True,
    )

    # Training
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 10
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "omniavatar_sf"
    return config
