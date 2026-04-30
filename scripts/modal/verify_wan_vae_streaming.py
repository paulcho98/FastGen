"""Modal app: verify Wan VAE streaming decode against full decode on H100."""
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
    .pip_install(
        "diffusers==0.35.1", "transformers==4.49.0", "accelerate", "safetensors",
        "einops", "omegaconf", "imageio", "imageio-ffmpeg",
        "opencv-python-headless", "numpy<2.0.0",
    )
    .add_local_dir(str(_LOCAL_FASTGEN_ROOT), remote_path="/workspace/FastGen-redmd",
                   ignore=_ignore, copy=True)
    .add_local_dir(str(_LOCAL_OMNIAVATAR_ROOT / "OmniAvatar"),
                   remote_path="/workspace/OmniAvatar/OmniAvatar", copy=True)
)

app = modal.App("wan-vae-streaming-verify", image=image)


@app.function(gpu="H100", volumes={ASSETS: VOL}, timeout=60 * 10)
def verify():
    env = os.environ.copy()
    env["OMNIAVATAR_ROOT"] = "/workspace/OmniAvatar"
    env["PYTHONPATH"] = (
        "/workspace/FastGen-redmd:/workspace/OmniAvatar:" + env.get("PYTHONPATH", "")
    )

    cmd = [
        "python",
        "/workspace/FastGen-redmd/scripts/inference/verify_wan_vae_streaming_decode.py",
        "--video_path", f"{ASSETS}/hdtf/videos_batch/{CLIP}",
        "--vae_path", f"{ASSETS}/wan_vae/Wan2.1_VAE.pth",
        "--num_video_frames", "81",
        "--chunk_latents", "3",
        "--dtype", "bf16",
    ]
    print(">>>", " ".join(cmd))
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("=== STDOUT ===")
    print(result.stdout)
    if result.returncode != 0:
        print("=== STDERR (tail) ===")
        print(result.stderr[-3000:])
        raise SystemExit(f"Verification failed with exit code {result.returncode}")
    return result.stdout


@app.local_entrypoint()
def main():
    print("[verify] Wan VAE streaming-vs-full decode comparison on H100 ...")
    verify.remote()
    print("[verify] Done.")
