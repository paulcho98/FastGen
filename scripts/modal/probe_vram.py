"""Single-clip VRAM probe for FastGen on H100.

Usage:
    modal run scripts/modal/probe_vram.py::probe
    modal run scripts/modal/probe_vram.py::probe --use-taehv
"""
import os
import pathlib
import re
import subprocess

import modal

_LOCAL_FASTGEN_ROOT = pathlib.Path("/home/work/.local/hyunbin/FastGen-redmd")
_LOCAL_OMNIAVATAR_ROOT = pathlib.Path("/home/work/.local/OmniAvatar")

VOL = modal.Volume.from_name("fastgen-assets", create_if_missing=False)
ASSETS = "/assets"

CLIP = "WDA_BarackObama_000_cfr25.mp4"

_ignore = [
    ".git/**", "__pycache__/**", "*.pyc", "FASTGEN_OUTPUT/**", "logs/**",
    "checkpoints/**", "pretrained_models/**", "verification_data/**",
    "assets/**", "tests/**", "docs/**", "modal_out/**",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12",
    )
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0",
                 "build-essential", "g++", "clang")
    .pip_install("torch==2.10.0", "torchvision",
                 index_url="https://download.pytorch.org/whl/cu128")
    .run_commands(
        "pip install flash_attn_3 --find-links "
        "https://windreamer.github.io/flash-attention3-wheels/cu128_torch2100"
    )
    .pip_install(
        "diffusers==0.35.1", "transformers==4.49.0", "accelerate", "safetensors",
        "peft", "einops", "omegaconf", "hydra-core", "imageio", "imageio-ffmpeg",
        "librosa", "soundfile", "av", "opencv-python-headless", "onnxruntime",
        "insightface", "tqdm", "sentencepiece", "ftfy", "timm", "loguru",
        "wandb", "scipy", "numpy<2.0.0", "kornia",
    )
    .add_local_dir(str(_LOCAL_FASTGEN_ROOT), remote_path="/workspace/FastGen-redmd",
                   ignore=_ignore, copy=True)
    .add_local_dir(str(_LOCAL_OMNIAVATAR_ROOT / "OmniAvatar"),
                   remote_path="/workspace/OmniAvatar/OmniAvatar", copy=True)
)

app = modal.App("fastgen-vram-probe", image=image)


@app.function(gpu="H100", volumes={ASSETS: VOL}, timeout=60 * 15)
def probe_one(use_taehv: bool = False) -> str:
    env = os.environ.copy()
    env["OMNIAVATAR_ROOT"] = "/workspace/OmniAvatar"
    env["PYTHONPATH"] = (
        "/workspace/FastGen-redmd:/workspace/OmniAvatar:" + env.get("PYTHONPATH", "")
    )
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    insightface_dir = pathlib.Path.home() / ".insightface" / "models"
    insightface_dir.mkdir(parents=True, exist_ok=True)
    link = insightface_dir / "buffalo_l"
    if not link.exists():
        src = f"{ASSETS}/insightface/models/buffalo_l"
        if os.path.exists(src):
            link.symlink_to(src)

    work_dir = "/workspace/FastGen-redmd"
    out_dir = "/tmp/modal_out"
    os.makedirs(out_dir, exist_ok=True)
    stem = CLIP.replace("_cfr25.mp4", "")

    cmd = [
        "python", "scripts/inference/inference_causal_taehv.py",
        "--ckpt_path", f"{ASSETS}/sf_ckpts/0000600.pth",
        "--vae_path", f"{ASSETS}/wan_vae/Wan2.1_VAE.pth",
        "--wav2vec_path", f"{ASSETS}/wav2vec2-base-960h",
        "--mask_path", f"{ASSETS}/mask/mask.png",
        "--base_model_paths", f"{ASSETS}/wan_base/diffusion_pytorch_model.safetensors",
        "--omniavatar_ckpt_path", f"{ASSETS}/omniavatar/step-19500.pt",
        "--text_embeds_path", f"{ASSETS}/text_emb/text_emb.pt",
        "--video_path", f"{ASSETS}/hdtf/videos_batch/{CLIP}",
        "--output_path", f"{out_dir}/{stem}.mp4",
        "--t_list", "0.999", "0.833", "0.0",
        "--chunk_size", "3",
        "--local_attn_size", "7", "--sink_size", "1",
        "--use_dynamic_rope",
        "--latentsync",
        "--face_cache_dir", f"{ASSETS}/hdtf/face_cache_batch",
        "--num_latent_frames", "21", "--min_latent_frames", "21",
    ]
    if use_taehv:
        cmd += ["--taehv_ckpt", f"{ASSETS}/taehv/taew2_1.pth"]

    print(">>>", " ".join(cmd))
    result = subprocess.run(cmd, cwd=work_dir, env=env, capture_output=True, text=True)
    out = result.stdout + "\n" + result.stderr
    print(out[-3000:])
    m = re.search(r"\[VRAM\]\s*(.*)", out)
    return m.group(0) if m else "no VRAM line found"


@app.local_entrypoint()
def probe(use_taehv: bool = False):
    tag = "TAEHV" if use_taehv else "Wan VAE"
    print(f"[FastGen | {tag}] probing VRAM on H100 with clip {CLIP} ...")
    line = probe_one.remote(use_taehv)
    print(f"\nFastGen ({tag}): {line}")
