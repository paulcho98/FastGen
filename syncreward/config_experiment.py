"""
Experiment config for OmniAvatar Self-Forcing with SyncNet Reward Forcing.

Mirrors fastgen/configs/experiments/OmniAvatar/config_sf.py exactly,
but uses OmniAvatarSyncRewardSFModel and adds sync_beta parameter.
"""

import os

from fastgen.utils import LazyCall as L
from fastgen.configs.experiments.OmniAvatar.config_sf import (
    OmniAvatar_V2V_14B_Teacher,
    OmniAvatar_V2V_1_3B_FakeScore,
    CausalOmniAvatar_V2V_1_3B_Student,
)
from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataLoader, create_omniavatar_dataloader
from syncreward.config_sync_reward_sf import create_config as create_sync_reward_config

# ---- Paths (same as original config_sf.py) ----
OMNIAVATAR_ROOT = os.getenv("OMNIAVATAR_ROOT", "/home/work/.local/OmniAvatar")
DATA_ROOT = os.getenv("OMNIAVATAR_DATA_ROOT", "/home/work/stableavatar_data/v2v_training_data")


def create_config():
    config = create_sync_reward_config()

    # ---- Network configs (identical to original) ----
    config.model.net = CausalOmniAvatar_V2V_1_3B_Student
    config.model.net.total_num_frames = 21  # matches input_shape[1]
    config.model.teacher = OmniAvatar_V2V_14B_Teacher
    config.model.fake_score_net = OmniAvatar_V2V_1_3B_FakeScore

    # GAN disabled (pure VSD + sync reward)
    config.model.gan_loss_weight_gen = 0
    config.model.student_update_freq = 5

    # Student weights
    config.model.load_student_weights = False
    config.trainer.checkpointer.pretrained_ckpt_path = os.getenv(
        "OMNIAVATAR_DF_CKPT",
        "/home/work/.local/hyunbin/FastGen/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df/df_4gpu_bs16_lr1e5_10000iter_shift_5/checkpoints/0005000.pth",
    )
    config.trainer.checkpointer.pretrained_ckpt_key_map = {"net": "net"}

    # Timestep schedule — shift=5.0 matches OmniAvatar's inference scheduler
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 5.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Self-Forcing specific
    config.model.enable_gradient_in_rollout = True
    config.model.start_gradient_frame = 0
    config.model.same_step_across_blocks = True
    config.model.context_noise = 0.0

    # Learning rates and optimizer (matches original)
    config.model.net_optimizer.lr = 2e-6
    config.model.net_optimizer.betas = (0.0, 0.999)
    config.model.fake_score_optimizer.lr = 2e-6
    config.model.fake_score_optimizer.betas = (0.0, 0.999)

    # Multi-GPU: FSDP required
    config.trainer.fsdp = True
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"
    config.model.input_shape = [16, 21, 64, 64]
    config.model.fake_score_pred_type = "x0"
    config.model.guidance_scale = 4.5

    # ---- Sync Reward Forcing parameters ----
    config.model.sync_beta = float(os.getenv("SYNC_BETA", "1.0"))
    config.model.syncnet_ckpt_path = os.getenv(
        "SYNCNET_CKPT_PATH",
        "/home/work/.local/eval_metrics/checkpoints/auxiliary/syncnet_v2.model",
    )

    # ---- Training dataloader (same dataset as original) ----
    config.dataloader_train = L(OmniAvatarDataLoader)(
        data_list_path=f"{DATA_ROOT}/video_square_path.txt",
        latentsync_mask_path=os.getenv(
            "LATENTSYNC_MASK_PATH",
            "/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png",
        ),
        batch_size=8,
        num_workers=2,
        neg_text_emb_path=os.getenv("NEG_TEXT_EMB_PATH", None),
        use_ref_sequence=True,
    )

    # ---- Validation dataloader (same as original) ----
    VAL_LIST = os.getenv("OMNIAVATAR_VAL_LIST", f"{DATA_ROOT}/video_square_val10.txt")
    VAE_PATH = os.getenv(
        "OMNIAVATAR_VAE_PATH",
        os.path.join(OMNIAVATAR_ROOT, "pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
    )
    config.dataloader_val = L(create_omniavatar_dataloader)(
        data_list_path=VAL_LIST,
        latentsync_mask_path=os.getenv(
            "LATENTSYNC_MASK_PATH",
            "/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png",
        ),
        batch_size=1,
        num_workers=2,
        neg_text_emb_path=os.getenv("NEG_TEXT_EMB_PATH", None),
        use_ref_sequence=True,
        load_ode_path=False,
    )
    config.model.vae_path = VAE_PATH

    # ---- Training settings (same as original) ----
    config.trainer.grad_accum_rounds = 2
    config.model.grad_accum_rounds = 2
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 100
    config.trainer.validation_iter = 100
    config.trainer.skip_initial_validation = True

    # Wandb
    config.trainer.callbacks.wandb.sample_logging_iter = 100
    config.trainer.callbacks.wandb.fps = 25
    config.log_config.group = "omniavatar_sync_reward_sf"
    config.log_config.wandb_entity = "jhjangbot-korea-advanced-institute-of-science-and-technology"
    config.log_config.project = "omniavatar-sync-reward"

    return config
