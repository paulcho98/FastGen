# Quick memory test: 14B teacher + 14B student + 14B fake_score
# Just 6 iterations to measure peak GPU memory, no wandb

import os
from fastgen.configs.experiments.OmniAvatar.config_sf import (
    create_config as create_base_config,
    OmniAvatar_V2V_14B_Teacher,
    OMNIAVATAR_ROOT,
    TEACHER_CKPT,
)
from omegaconf import DictConfig
from fastgen.utils import LazyCall as L
from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataLoader
from fastgen.networks.OmniAvatar.network import OmniAvatarWan
from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
from fastgen.configs.callbacks import GradClip_CALLBACK, GPUStats_CALLBACK, ParamCount_CALLBACK
from fastgen.callbacks.stdout_logger import StdoutLoggerCallback

STDOUT_LOG_CALLBACK = {"stdout_logger": L(StdoutLoggerCallback)()}

# 14B causal student
CausalOmniAvatar_V2V_14B_Student: dict = L(CausalOmniAvatarWan)(
    model_size="14B",
    in_dim=65,
    mode="v2v",
    use_audio=True,
    audio_hidden_size=32,
    chunk_size=3,
    total_num_frames=21,
    base_model_paths=f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,"
                     f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,"
                     f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,"
                     f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,"
                     f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,"
                     f"{OMNIAVATAR_ROOT}/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
    omniavatar_ckpt_path=TEACHER_CKPT,
    net_pred_type="flow",
    schedule_type="rf",
)

# 14B bidirectional fake_score
OmniAvatar_V2V_14B_FakeScore: dict = L(OmniAvatarWan)(
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
    merge_lora=False,
    net_pred_type="flow",
    schedule_type="rf",
)


def create_config():
    config = create_base_config()

    # Override all three models to 14B
    config.model.net = CausalOmniAvatar_V2V_14B_Student
    config.model.net.total_num_frames = 21
    config.model.teacher = OmniAvatar_V2V_14B_Teacher
    config.model.fake_score_net = OmniAvatar_V2V_14B_FakeScore

    # Don't load DF checkpoint (no 14B DF checkpoint exists)
    config.trainer.checkpointer.pretrained_ckpt_path = None
    config.model.load_student_weights = False

    # bs=1 to minimize memory for profiling
    config.dataloader_train = L(OmniAvatarDataLoader)(
        data_list_path=os.path.join(
            os.getenv("OMNIAVATAR_DATA_ROOT", "/home/work/stableavatar_data/v2v_training_data"),
            "video_square_path.txt",
        ),
        latentsync_mask_path="/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png",
        batch_size=1,
        num_workers=0,
        use_ref_sequence=True,
        load_ode_path=False,
        neg_text_emb_path="/home/work/stableavatar_data/neg_text_emb.pt",
    )

    # Minimal callbacks — no wandb
    config.trainer.callbacks = DictConfig({
        **GradClip_CALLBACK,
        **GPUStats_CALLBACK,
        **ParamCount_CALLBACK,
        **STDOUT_LOG_CALLBACK,
    })

    config.trainer.fsdp = True
    config.model.precision_fsdp = "bfloat16"

    # Just 6 iterations: 4 critic + 1 student + 1 critic
    config.trainer.max_iter = 6
    config.trainer.grad_accum_rounds = 1
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 999999
    config.trainer.validation_iter = 999999
    config.trainer.skip_initial_validation = True

    config.log_config.group = "omniavatar_sf_14b_memtest"
    config.log_config.name = "14b_memtest"
    config.log_config.wandb_mode = "disabled"

    return config
