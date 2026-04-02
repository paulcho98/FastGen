# Quick validation test: 6 iters (1 student update at iter 5), validate at iter 5, 2 val samples

import os
from fastgen.configs.experiments.OmniAvatar.config_sf import create_config as create_base_config
from omegaconf import DictConfig
from fastgen.utils import LazyCall as L
from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataLoader, create_omniavatar_dataloader
from fastgen.configs.callbacks import GradClip_CALLBACK, GPUStats_CALLBACK, ParamCount_CALLBACK, WANDB_CALLBACK
from fastgen.callbacks.stdout_logger import StdoutLoggerCallback
from fastgen.callbacks.wandb import WandbCallback

STDOUT_LOG_CALLBACK = {"stdout_logger": L(StdoutLoggerCallback)()}
DATA_ROOT = os.getenv("OMNIAVATAR_DATA_ROOT", "/home/work/stableavatar_data/v2v_training_data")
VAL_DATA = "/home/work/stableavatar_data/v2v_validation_data/recon"
MASK_PATH = "/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png"
NEG_TEXT_EMB = "/home/work/stableavatar_data/neg_text_emb.pt"


def create_config():
    config = create_base_config()

    # bs=1, no grad accum — fast iterations
    config.dataloader_train = L(OmniAvatarDataLoader)(
        data_list_path=os.path.join(DATA_ROOT, "video_square_path.txt"),
        latentsync_mask_path=MASK_PATH,
        batch_size=1,
        num_workers=0,
        use_ref_sequence=True,
        load_ode_path=False,
        neg_text_emb_path=NEG_TEXT_EMB,
    )

    # 2 val samples only
    config.dataloader_val = L(create_omniavatar_dataloader)(
        data_list_path=os.path.join(VAL_DATA, "video_square_path.txt"),
        latentsync_mask_path=MASK_PATH,
        batch_size=1,
        num_workers=0,
        use_ref_sequence=True,
        load_ode_path=False,
        neg_text_emb_path=NEG_TEXT_EMB,
    )

    config.trainer.callbacks = DictConfig({
        **GradClip_CALLBACK,
        **GPUStats_CALLBACK,
        **ParamCount_CALLBACK,
        **STDOUT_LOG_CALLBACK,
        "wandb": L(WandbCallback)(sample_logging_iter=999999, validation_logging_step=1),
    })

    config.trainer.fsdp = True
    config.model.precision_fsdp = "bfloat16"

    # 6 iters: 4 fake_score + 1 student update at iter 5 + 1 fake_score
    # Checkpoint + validate at iter 5 (after student update)
    config.trainer.max_iter = 7
    config.trainer.grad_accum_rounds = 1
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 5
    config.trainer.validation_iter = 5
    config.trainer.skip_initial_validation = True

    # Limit validation to 2 samples
    config.trainer.global_vars_val = [{"MAX_VAL_STEPS": 2}]

    config.log_config.project = "OmniAvatar-FastGen"
    config.log_config.group = "omniavatar_sf_valtest"
    config.log_config.name = "sf_valtest"
    config.log_config.wandb_mode = "online"

    return config
