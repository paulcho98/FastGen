"""Modal app: single-clip inference smoke test on 1×H100.

Runs the existing scripts/inference/inference_causal.py or
inference_causal_taehv.py as a subprocess, against assets pre-uploaded to a
Modal Volume.

Usage (after you've run scripts/modal/upload.sh):

    # Full Wan VAE decoder
    modal run scripts/modal/app.py \\
        --ckpt-name 0000600 --video-name RD_Radio18_000

    # TAEHV tiny decoder
    modal run scripts/modal/app.py \\
        --ckpt-name 0000600 --video-name RD_Radio18_000 --use-taehv

Outputs land in /home/work/.local/hyunbin/FastGen-redmd/modal_out/.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import modal

# ---------------------------------------------------------------------------
# Volume: persistent storage for all heavyweight assets (~10 GB).
# Populate once via scripts/modal/upload.sh.
# ---------------------------------------------------------------------------
VOL = modal.Volume.from_name("fastgen-assets", create_if_missing=True)

# Mount point inside the container.
ASSETS = "/assets"

# ---------------------------------------------------------------------------
# Image: pytorch base + project pip deps. Heavy, but cached after first build.
# ---------------------------------------------------------------------------
_LOCAL_FASTGEN_ROOT = pathlib.Path("/home/work/.local/hyunbin/FastGen-redmd")
_LOCAL_OMNIAVATAR_ROOT = pathlib.Path("/home/work/.local/OmniAvatar")

_ignore = [
    ".git/**",
    "__pycache__/**",
    "*.pyc",
    "FASTGEN_OUTPUT/**",
    "logs/**",
    "checkpoints/**",        # uploaded to Volume instead
    "pretrained_models/**",  # uploaded to Volume instead
    "verification_data/**",
    "assets/**",
    "tests/**",
    "docs/**",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "build-essential", "g++", "clang")
    .pip_install(
        "torch==2.10.0",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands("pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch2100")
    .pip_install(
        "diffusers==0.35.1",
        "transformers==4.49.0",
        "accelerate",
        "safetensors",
        "peft",
        "einops",
        "omegaconf",
        "hydra-core",
        "imageio",
        "imageio-ffmpeg",
        "librosa",
        "soundfile",
        "av",
        "opencv-python-headless",
        "onnxruntime",
        "insightface",
        "tqdm",
        "sentencepiece",
        "ftfy",
        "timm",
        "loguru",
        "wandb",
        "scipy",
        "numpy<2.0.0",
        "kornia",
    )
    .add_local_dir(
        str(_LOCAL_FASTGEN_ROOT),
        remote_path="/workspace/FastGen-redmd",
        ignore=_ignore,
        copy=True,
    )
    .add_local_dir(
        str(_LOCAL_OMNIAVATAR_ROOT / "OmniAvatar"),
        remote_path="/workspace/OmniAvatar/OmniAvatar",
        copy=True,
    )
)

app = modal.App("fastgen-redmd-inference", image=image)


# ---------------------------------------------------------------------------
# Asset layout inside the Volume (populated by upload.sh).
# Keep this schema stable — upload.sh mirrors it.
# ---------------------------------------------------------------------------
#
#   /assets/
#     wan_vae/Wan2.1_VAE.pth
#     wan_base/diffusion_pytorch_model.safetensors
#     wav2vec2-base-960h/...
#     omniavatar/step-19500.pt
#     sf_ckpts/0000600.pth
#     sf_ckpts/0000600.net_model/*.distcp
#     taehv/taew2_1.pth
#     insightface/buffalo_l/*.onnx
#     mask/mask.png
#     text_emb/text_emb.pt
#     hdtf/videos/<name>.mp4
#     hdtf/face_cache/<name>_face_cache.pt
#
# ---------------------------------------------------------------------------

@app.function(
    gpu="H200",
    volumes={ASSETS: VOL},
    timeout=60 * 30,
    max_containers=4,
)
def run_inference(
    ckpt_name: str = "0000600",
    video_name: str = "RD_Radio18_000",
    use_taehv: bool = False,
    video_dir: str = "hdtf/videos",
    face_cache_dir: str = "hdtf/face_cache",
) -> bytes:
    """Run one inference_causal*.py invocation and return the output mp4 bytes."""
    env = os.environ.copy()
    env["OMNIAVATAR_ROOT"] = "/workspace/OmniAvatar"
    env["PYTHONPATH"] = (
        "/workspace/FastGen-redmd:/workspace/OmniAvatar:" + env.get("PYTHONPATH", "")
    )
    # insightface expects models at ~/.insightface/models/buffalo_l/
    insightface_dir = pathlib.Path.home() / ".insightface" / "models"
    insightface_dir.mkdir(parents=True, exist_ok=True)
    link = insightface_dir / "buffalo_l"
    if not link.exists():
        link.symlink_to(f"{ASSETS}/insightface/buffalo_l")

    work_dir = "/workspace/FastGen-redmd"
    out_dir = "/tmp/modal_out"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{video_name}.mp4"

    script = "inference_causal_taehv.py" if use_taehv else "inference_causal.py"
    cmd = [
        "python",
        f"scripts/inference/{script}",
        "--ckpt_path", f"{ASSETS}/sf_ckpts/{ckpt_name}.pth",
        "--vae_path", f"{ASSETS}/wan_vae/Wan2.1_VAE.pth",
        "--wav2vec_path", f"{ASSETS}/wav2vec2-base-960h",
        "--mask_path", f"{ASSETS}/mask/mask.png",
        "--base_model_paths", f"{ASSETS}/wan_base/diffusion_pytorch_model.safetensors",
        "--omniavatar_ckpt_path", f"{ASSETS}/omniavatar/step-19500.pt",
        "--text_embeds_path", f"{ASSETS}/text_emb/text_emb.pt",
        "--video_path", f"{ASSETS}/{video_dir}/{video_name}_cfr25.mp4",
        "--output_path", out_path,
        "--t_list", "0.999", "0.833", "0.0",
        "--chunk_size", "3",
        "--local_attn_size", "7",
        "--sink_size", "1",
        "--use_dynamic_rope",
        "--latentsync",
        "--face_cache_dir", f"{ASSETS}/{face_cache_dir}",
    ]
    if use_taehv:
        cmd += ["--taehv_ckpt", f"{ASSETS}/taehv/taew2_1.pth"]

    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=work_dir, env=env, check=True)

    return pathlib.Path(out_path).read_bytes()


@app.local_entrypoint()
def main(
    ckpt_name: str = "0000600",
    video_name: str = "RD_Radio18_000",
    use_taehv: bool = False,
    taehv_encode: bool = False,
    taehv_streaming: bool = False,
):
    # Run a single-clip timed invocation (reuses run_inference_timed so we can also check timing).
    mp4_bytes, csv_bytes = run_inference_timed.remote(
        ckpt_name, video_name, use_taehv, taehv_encode,
        "hdtf/videos_batch", "hdtf/face_cache_batch",
        taehv_streaming=taehv_streaming,
    )
    if taehv_encode:
        suffix = "_taehv_full"
    elif use_taehv:
        suffix = "_taehv"
    else:
        suffix = "_wan"
    out = _LOCAL_FASTGEN_ROOT / "modal_out" / f"{video_name}{suffix}.mp4"
    csv_out = out.with_suffix(".mp4.timing.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(mp4_bytes)
    csv_out.write_bytes(csv_bytes)
    print(f"Saved → {out}  ({len(mp4_bytes)/1e6:.1f} MB)  + timing → {csv_out}")


@app.function(
    gpu="H200",
    volumes={ASSETS: VOL},
    timeout=60 * 30,
    max_containers=4,
)
def run_inference_timed(
    ckpt_name: str = "0000600",
    video_name: str = "RD_Radio18_000",
    use_taehv: bool = False,
    taehv_encode: bool = False,
    video_dir: str = "hdtf/videos",
    face_cache_dir: str = "hdtf/face_cache",
    num_latent_frames: int = 21,
    min_latent_frames: int = 21,
    ckpt_subdir: str = "sf_ckpts",
    t_list: str = "0.999,0.833,0.0",
    local_attn_size: int = 7,
    sink_size: int = 1,
    tag: str = "",
    taehv_streaming: bool = False,
    no_latentsync: bool = False,
) -> tuple:
    """Same as run_inference but with --timing. Returns (mp4_bytes, csv_bytes)."""
    env = os.environ.copy()
    env["OMNIAVATAR_ROOT"] = "/workspace/OmniAvatar"
    env["PYTHONPATH"] = (
        "/workspace/FastGen-redmd:/workspace/OmniAvatar:" + env.get("PYTHONPATH", "")
    )
    insightface_dir = pathlib.Path.home() / ".insightface" / "models"
    insightface_dir.mkdir(parents=True, exist_ok=True)
    link = insightface_dir / "buffalo_l"
    if not link.exists():
        link.symlink_to(f"{ASSETS}/insightface/buffalo_l")

    work_dir = "/workspace/FastGen-redmd"
    out_dir = "/tmp/modal_out"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{video_name}.mp4"
    csv_path = f"{out_dir}/{video_name}.timing.csv"

    # Use the taehv-enabled script in both cases; it's a superset of the original.
    cmd = [
        "python",
        "scripts/inference/inference_causal_taehv.py",
        "--ckpt_path", f"{ASSETS}/{ckpt_subdir}/{ckpt_name}.pth",
        "--vae_path", f"{ASSETS}/wan_vae/Wan2.1_VAE.pth",
        "--wav2vec_path", f"{ASSETS}/wav2vec2-base-960h",
        "--mask_path", f"{ASSETS}/mask/mask.png",
        "--base_model_paths", f"{ASSETS}/wan_base/diffusion_pytorch_model.safetensors",
        "--omniavatar_ckpt_path", f"{ASSETS}/omniavatar/step-19500.pt",
        "--text_embeds_path", f"{ASSETS}/text_emb/text_emb.pt",
        "--video_path", f"{ASSETS}/{video_dir}/{video_name}_cfr25.mp4",
        "--output_path", out_path,
        "--t_list", *t_list.split(","),
        "--chunk_size", "3",
        "--use_dynamic_rope",
        "--timing",
        "--timing_csv", csv_path,
    ]
    if num_latent_frames > 0:
        cmd += ["--num_latent_frames", str(num_latent_frames)]
    if min_latent_frames > 0:
        cmd += ["--min_latent_frames", str(min_latent_frames)]
    if local_attn_size > 0:
        cmd += ["--local_attn_size", str(local_attn_size)]
    if sink_size > 0:
        cmd += ["--sink_size", str(sink_size)]
    if not no_latentsync:
        cmd += ["--latentsync", "--face_cache_dir", f"{ASSETS}/{face_cache_dir}"]
    if use_taehv:
        cmd += ["--taehv_ckpt", f"{ASSETS}/taehv/taew2_1.pth"]
    if taehv_encode:
        cmd += ["--taehv_encode"]
    if tag and "streaming" in tag:
        cmd += ["--taehv_streaming"]
    if taehv_streaming:
        cmd += ["--taehv_streaming"]

    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=work_dir, env=env, check=True)
    return pathlib.Path(out_path).read_bytes(), pathlib.Path(csv_path).read_bytes()


@app.local_entrypoint()
def timing_sweep(
    ckpt_name: str = "0000600",
    use_taehv: bool = False,
    taehv_encode: bool = False,
    video_dir: str = "hdtf/videos_batch",
    face_cache_dir: str = "hdtf/face_cache_batch",
    num_latent_frames: int = 21,
    min_latent_frames: int = 21,
    ckpt_subdir: str = "sf_ckpts",
    t_list: str = "0.999,0.833,0.0",
    local_attn_size: int = 7,
    sink_size: int = 1,
    tag: str = "",
    taehv_streaming: bool = False,
    no_latentsync: bool = False,
):
    """Timed sweep across all HDTF clips. Saves per-clip mp4 + timing CSV, writes aggregate CSV."""
    hdtf_local = pathlib.Path("/home/work/.local/HDTF/HDTF_original_testset_81frames/videos_cfr")
    video_names = sorted(p.stem.removesuffix("_cfr25") for p in hdtf_local.glob("*_cfr25.mp4"))
    print(f"Timed sweep: {len(video_names)} clips (use_taehv={use_taehv}, taehv_encode={taehv_encode}, ckpt={ckpt_name}, tag={tag})")

    if taehv_encode:
        suffix = "_taehv_full"
    elif use_taehv:
        suffix = "_taehv"
    else:
        suffix = "_wan"
    if tag:
        suffix = f"_{tag}{suffix}"
    out_dir = _LOCAL_FASTGEN_ROOT / "modal_out" / f"timing_{ckpt_name}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    timing_rows = []

    args = [(ckpt_name, vn, use_taehv, taehv_encode, video_dir, face_cache_dir, num_latent_frames, min_latent_frames, ckpt_subdir, t_list, local_attn_size, sink_size, tag, taehv_streaming, no_latentsync) for vn in video_names]
    for args_tuple, result in zip(
        args,
        run_inference_timed.starmap(args, return_exceptions=True),
    ):
        vn = args_tuple[1]
        if isinstance(result, Exception):
            print(f"  [FAIL] {vn}: {result}")
            continue
        mp4_bytes, csv_bytes = result
        (out_dir / f"{vn}.mp4").write_bytes(mp4_bytes)
        csv_text = csv_bytes.decode()
        (out_dir / f"{vn}.timing.csv").write_text(csv_text)
        # Parse the per-clip row (skip header + AVERAGE row which is itself).
        import csv as _csv
        reader = _csv.DictReader(csv_text.splitlines())
        for row in reader:
            if row.get("name") and row["name"] != "AVERAGE":
                timing_rows.append(row)
        print(f"  [OK]   {vn}  ({len(mp4_bytes)/1e6:.1f} MB)")

    # Aggregate CSV with all clips + AVERAGE.
    import csv as _csv
    agg_path = out_dir / "aggregate.csv"
    if timing_rows:
        fieldnames = list(timing_rows[0].keys())
        with open(agg_path, "w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in timing_rows:
                writer.writerow(r)
            avg = {"name": "AVERAGE"}
            for k in fieldnames:
                if k == "name":
                    continue
                vals = []
                for r in timing_rows:
                    try:
                        vals.append(float(r[k]))
                    except (TypeError, ValueError):
                        pass
                if vals:
                    avg[k] = f"{sum(vals)/len(vals):.6f}"
            writer.writerow(avg)
        print(f"\n[Timing] aggregate → {agg_path}")
        print(f"[Timing] averages over {len(timing_rows)} clips | FPS (num_video_frames/time):")
        nvf_vals = [float(r["num_video_frames"]) for r in timing_rows if r.get("num_video_frames") not in (None, "")]
        mean_nvf = (sum(nvf_vals) / len(nvf_vals)) if nvf_vals else 0.0
        print(f"  {'num_video_frames':20s} {mean_nvf:7.2f}")
        for k in fieldnames:
            if k in ("name", "num_video_frames"):
                continue
            vals = [float(r[k]) for r in timing_rows if r.get(k) not in (None, "")]
            if vals:
                mean_t = sum(vals) / len(vals)
                fps = (mean_nvf / mean_t) if mean_t > 0 else 0.0
                print(f"  {k:20s} {mean_t:7.4f} s   {fps:7.2f} fps")


@app.local_entrypoint()
def sweep(
    ckpt_name: str = "0000600",
    use_taehv: bool = False,
    video_dir: str = "hdtf/videos_batch",
    face_cache_dir: str = "hdtf/face_cache_batch",
):
    """Fan out inference across all HDTF clips in parallel (.map over H100 workers)."""
    # Discover clip names from local HDTF dir (stems without _cfr25.mp4).
    hdtf_local = pathlib.Path("/home/work/.local/HDTF/HDTF_original_testset_81frames/videos_cfr")
    video_names = sorted(p.stem.removesuffix("_cfr25") for p in hdtf_local.glob("*_cfr25.mp4"))
    print(f"Sweeping {len(video_names)} clips (use_taehv={use_taehv}, ckpt={ckpt_name})")

    suffix = "_taehv" if use_taehv else "_wan"
    out_dir = _LOCAL_FASTGEN_ROOT / "modal_out" / f"sweep_{ckpt_name}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [
        (ckpt_name, vn, use_taehv, video_dir, face_cache_dir)
        for vn in video_names
    ]
    succeeded = 0
    for (args_tuple, data) in zip(
        args,
        run_inference.starmap(args, return_exceptions=True),
    ):
        vn = args_tuple[1]
        if isinstance(data, Exception):
            print(f"  [FAIL] {vn}: {data}")
            continue
        (out_dir / f"{vn}.mp4").write_bytes(data)
        succeeded += 1
        print(f"  [OK]   {vn}  ({len(data)/1e6:.1f} MB)")

    print(f"\nDone: {succeeded}/{len(video_names)} clips saved → {out_dir}")
