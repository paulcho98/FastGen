"""Run the streaming pipeline with Wan VAE decoder (cache-continuous) and
download the resulting video locally.
"""
import os
import pathlib
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

app = modal.App("fastgen-streaming-wan-video", image=image)


@app.function(gpu="H100", volumes={ASSETS: VOL}, timeout=60 * 15)
def produce_video(streamwise: bool = False) -> bytes:
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
    out_dir = "/tmp/streaming_wan_out"
    os.makedirs(out_dir, exist_ok=True)
    stem = CLIP.replace("_cfr25.mp4", "")
    suffix = "streamwise" if streamwise else "streaming"
    out_path = f"{out_dir}/{stem}_{suffix}_wan.mp4"

    cmd = [
        "python", "scripts/inference/inference_streaming.py",
        "--ckpt_path", f"{ASSETS}/sf_ckpts/0000600.pth",
        "--vae_path", f"{ASSETS}/wan_vae/Wan2.1_VAE.pth",
        "--wav2vec_path", f"{ASSETS}/wav2vec2-base-960h",
        "--mask_path", f"{ASSETS}/mask/mask.png",
        "--base_model_paths", f"{ASSETS}/wan_base/diffusion_pytorch_model.safetensors",
        "--omniavatar_ckpt_path", f"{ASSETS}/omniavatar/step-19500.pt",
        "--text_embeds_path", f"{ASSETS}/text_emb/text_emb.pt",
        "--video_path", f"{ASSETS}/hdtf/videos_batch/{CLIP}",
        "--output_path", out_path,
        "--t_list", "0.999", "0.833", "0.0",
        "--chunk_size", "3",
        "--local_attn_size", "7", "--sink_size", "1",
        "--use_dynamic_rope",
        "--latentsync",
        "--face_cache_dir", f"{ASSETS}/hdtf/face_cache_batch",
        "--num_latent_frames", "21", "--min_latent_frames", "21",
        "--streaming_decoder", "wan_vae",
        "--timing",
    ]
    if streamwise:
        cmd += ["--streamwise_encode"]
    print(">>>", " ".join(cmd))
    result = subprocess.run(cmd, cwd=work_dir, env=env, capture_output=True, text=True)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print("=== STDERR (tail) ===")
        print(result.stderr[-3000:])
        raise SystemExit(f"Inference failed (exit {result.returncode})")
    if not os.path.exists(out_path):
        raise SystemExit(f"No output file at {out_path}")
    print(f"Output: {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB)")
    return pathlib.Path(out_path).read_bytes()


@app.local_entrypoint()
def main(streamwise: bool = False):
    tag = "streamwise" if streamwise else "streaming"
    print(f"[{tag}-wan] producing video on H100 ...")
    data = produce_video.remote(streamwise=streamwise)
    out_path = _LOCAL_FASTGEN_ROOT / "modal_out" / f"WDA_BarackObama_000_{tag}_wan.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    print(f"[{tag}-wan] saved {out_path} ({len(data)/1e6:.2f} MB)")
