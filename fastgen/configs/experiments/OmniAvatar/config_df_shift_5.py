# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment config for OmniAvatar Diffusion Forcing (Stage 1 alternative to ODE KD).

Trains the causal 1.3B student on real data with inhomogeneous block-wise timesteps.
No pre-computed ODE trajectories from the teacher are needed.

Default: 4 GPU DDP, bs=16/GPU (effective 64), lr=5e-5, 5000 iters.
Max per-GPU batch: 36 on H200 (143GB). Adjust via CLI if needed.
"""

import os
import fastgen.configs.methods.config_omniavatar_df as config_df_default

from fastgen.utils import LazyCall as L
from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataLoader, create_omniavatar_dataloader

# ---- Paths (override via env vars) ----
OMNIAVATAR_ROOT = os.getenv("OMNIAVATAR_ROOT", "/home/work/.local/OmniAvatar")
DATA_ROOT = os.getenv("OMNIAVATAR_DATA_ROOT", "/home/work/stableavatar_data/v2v_training_data")
STUDENT_CKPT = os.getenv(
    "OMNIAVATAR_STUDENT_CKPT",
    "/home/work/output_omniavatar_v2v_1.3B_phase2/step-19500.pt",
)
DATA_LIST = os.getenv("OMNIAVATAR_DATA_LIST", f"{DATA_ROOT}/video_square_path.txt")
VAL_LIST = os.getenv("OMNIAVATAR_VAL_LIST", f"{DATA_ROOT}/video_square_val10.txt")
MASK_PATH = os.getenv(
    "OMNIAVATAR_MASK_PATH",
    "/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png",
)
VAE_PATH = os.getenv(
    "OMNIAVATAR_VAE_PATH",
    os.path.join(OMNIAVATAR_ROOT, "pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
)

# ---- Student network config ----
CausalOmniAvatar_V2V_1_3B_Config: dict = L(CausalOmniAvatarWan)(
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
    # Sliding window attention (default: full causal, no window)
    local_attn_size=-1,
    sink_size=0,
    use_dynamic_rope=False,
    # Stochastic attention configs (uncomment to enable):
    # stochastic_attn_configs=[
    #     {"local_attn_size": -1, "sink_size": 0, "weight": 0.25},   # full causal
    #     {"local_attn_size": 6,  "sink_size": 0, "weight": 0.25},   # tight window
    #     {"local_attn_size": 9,  "sink_size": 0, "weight": 0.25},   # medium window
    #     {"local_attn_size": 7,  "sink_size": 1, "weight": 0.25},   # window + sink
    # ],
)


def create_config():
    config = config_df_default.create_config()

    # Learning rate
    config.model.net_optimizer.lr = 1e-5

    # Precision
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"

    # Input shape: 512x512 @ 81 frames -> latent [16, 21, 64, 64]
    config.model.input_shape = [16, 21, 64, 64]

    # Student network
    config.model.net = CausalOmniAvatar_V2V_1_3B_Config
    config.model.net.total_num_frames = config.model.input_shape[1]

    # Timestep schedule — shift=5.0 matches OmniAvatar's default scheduler
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 5.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    # t_list derived from shift=5.0: new_t = 3*t / (1 + 2*t) applied to linspace(1,0,5)
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Diffusion forcing settings
    config.model.student_sample_steps = 4

    # Dataloader
    config.dataloader_train = L(OmniAvatarDataLoader)(
        data_list_path=DATA_LIST,
        latentsync_mask_path=MASK_PATH,
        batch_size=16,
        num_workers=4,
        neg_text_emb_path=os.getenv("NEG_TEXT_EMB_PATH", None),
        use_ref_sequence=True,
        load_ode_path=False,
    )

    # Validation dataloader — 10 fixed samples, finite iterator, batch_size=1
    config.dataloader_val = L(create_omniavatar_dataloader)(
        data_list_path=VAL_LIST,
        latentsync_mask_path=MASK_PATH,
        batch_size=1,
        num_workers=2,
        neg_text_emb_path=os.getenv("NEG_TEXT_EMB_PATH", None),
        use_ref_sequence=True,
        load_ode_path=False,
    )

    # VAE for visual logging (decodes latents to video for wandb)
    config.model.vae_path = VAE_PATH

    # Training
    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 500
    config.trainer.validation_iter = 500
    config.trainer.skip_initial_validation = True
    config.trainer.callbacks.wandb.sample_logging_iter = 500

    config.log_config.group = "omniavatar_df"
    return config
