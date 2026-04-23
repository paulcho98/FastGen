"""Modal benchmark: run our FastGen model on 10 HDTF clips with timing + FA3.

Usage:
    modal run scripts/modal/benchmark_10clips.py::sweep
    modal run scripts/modal/benchmark_10clips.py::sweep --use-taehv
    modal run scripts/modal/benchmark_10clips.py::sweep --use-taehv --taehv-encode
"""
import os
import pathlib
import subprocess
import modal

_LOCAL_FASTGEN_ROOT = pathlib.Path("/home/work/.local/hyunbin/FastGen-redmd")
_LOCAL_OMNIAVATAR_ROOT = pathlib.Path("/home/work/.local/OmniAvatar")

VOL = modal.Volume.from_name("v2v-benchmark", create_if_missing=True)
ASSETS = "/assets"

# First clip is warmup (absorbs Wav2Vec2 lazy-load); only remaining 10 are measured.
WARMUP_CLIP = "WDA_AndyLevin_000_cfr25.mp4"
CLIPS = [
    "RD_Radio18_000_cfr25.mp4", "WDA_BarackObama_000_cfr25.mp4",
    "WDA_DonnaShalala1_000_cfr25.mp4", "WDA_JerryNadler_000_cfr25.mp4",
    "WDA_KatherineClark_000_cfr25.mp4", "WDA_NancyPelosi0_000_cfr25.mp4",
    "WDA_TedLieu_000_cfr25.mp4", "WRA_AdamKinzinger1_000_cfr25.mp4",
    "WRA_EricCantor_000_cfr25.mp4", "WRA_MittRomney_000_cfr25.mp4",
]

_ignore = [
    ".git/**", "__pycache__/**", "*.pyc", "FASTGEN_OUTPUT/**", "logs/**",
    "checkpoints/**", "pretrained_models/**", "verification_data/**",
    "assets/**", "tests/**", "docs/**", "modal_out/**",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12",
    )
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "build-essential", "g++", "clang")
    .pip_install(
        "torch==2.10.0", "torchvision",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        "pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch2100"
    )
    .pip_install(
        "diffusers==0.35.1", "transformers==4.49.0", "accelerate", "safetensors",
        "peft", "einops", "omegaconf", "hydra-core", "imageio", "imageio-ffmpeg",
        "librosa", "soundfile", "av", "opencv-python-headless", "onnxruntime",
        "insightface", "tqdm", "sentencepiece", "ftfy", "timm", "loguru",
        "wandb", "scipy", "numpy<2.0.0", "kornia",
    )
    .add_local_dir(
        str(_LOCAL_FASTGEN_ROOT), remote_path="/workspace/FastGen-redmd",
        ignore=_ignore, copy=True,
    )
    .add_local_dir(
        str(_LOCAL_OMNIAVATAR_ROOT / "OmniAvatar"),
        remote_path="/workspace/OmniAvatar/OmniAvatar", copy=True,
    )
)

app = modal.App("fastgen-benchmark-10clips", image=image)

CKPTS = f"{ASSETS}/fastgen_ckpts"


@app.function(gpu="H200", volumes={ASSETS: VOL}, timeout=60 * 30, max_containers=1)
def run_benchmark(
    video_name: str,
    use_taehv: bool = False,
    taehv_encode: bool = False,
    streaming_pipeline: str = "",
    no_streaming_taehv: bool = False,
) -> str:
    env = os.environ.copy()
    env["OMNIAVATAR_ROOT"] = "/workspace/OmniAvatar"
    env["PYTHONPATH"] = "/workspace/FastGen-redmd:/workspace/OmniAvatar:" + env.get("PYTHONPATH", "")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    insightface_dir = pathlib.Path.home() / ".insightface" / "models"
    insightface_dir.mkdir(parents=True, exist_ok=True)
    link = insightface_dir / "buffalo_l"
    if not link.exists():
        # Reuse LatentSync's insightface models if available
        src = f"{ASSETS}/latentsync_ckpts/checkpoints/auxiliary/models/buffalo_l"
        if os.path.exists(src):
            link.symlink_to(src)

    work_dir = "/workspace/FastGen-redmd"
    out_dir = "/tmp/modal_out"
    os.makedirs(out_dir, exist_ok=True)

    stem = video_name.replace("_cfr25.mp4", "")
    out_path = f"{out_dir}/{stem}.mp4"
    csv_path = f"{out_dir}/{stem}.timing.csv"

    cmd = [
        "python", "scripts/inference/inference_causal_taehv.py",
        "--ckpt_path", f"{CKPTS}/sf_ckpts/0000600.pth",
        "--vae_path", f"{CKPTS}/wan_vae/Wan2.1_VAE.pth",
        "--wav2vec_path", f"{CKPTS}/wav2vec2-base-960h",
        "--mask_path", f"{CKPTS}/mask/mask.png",
        "--base_model_paths", f"{CKPTS}/wan_base/diffusion_pytorch_model.safetensors",
        "--omniavatar_ckpt_path", f"{CKPTS}/omniavatar/step-19500.pt",
        "--text_embeds_path", f"{CKPTS}/text_emb/text_emb.pt",
        "--video_path", f"{ASSETS}/test_clips/{video_name}",
        "--output_path", out_path,
        "--t_list", "0.999", "0.833", "0.0",
        "--chunk_size", "3",
        "--local_attn_size", "7",
        "--sink_size", "1",
        "--use_dynamic_rope",
        "--latentsync",
        "--face_cache_dir", f"{CKPTS}/face_cache",
        "--timing",
        "--timing_csv", csv_path,
        "--num_latent_frames", "21",
        "--min_latent_frames", "21",
    ]
    if use_taehv:
        cmd += ["--taehv_ckpt", f"{CKPTS}/taehv/taew2_1.pth"]
    if taehv_encode:
        cmd += ["--taehv_encode"]
    if streaming_pipeline:
        cmd += ["--streaming_pipeline", streaming_pipeline]
    if no_streaming_taehv:
        cmd += ["--no_streaming_taehv"]

    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=work_dir, env=env, check=True)

    # Copy output video to volume for persistence
    if os.path.exists(out_path):
        tag = "wan"
        if streaming_pipeline:
            if no_streaming_taehv:
                tag = "streaming_batch_taehv"
            elif not use_taehv:
                tag = "streaming_wan_dec"
            else:
                tag = "streaming_taehv"
        elif use_taehv:
            tag = "taehv"
        vol_video_dir = f"{ASSETS}/output_videos/{tag}"
        os.makedirs(vol_video_dir, exist_ok=True)
        import shutil
        shutil.copy2(out_path, f"{vol_video_dir}/{stem}.mp4")
        VOL.commit()

    if os.path.exists(csv_path):
        return pathlib.Path(csv_path).read_text()
    return ""


@app.local_entrypoint()
def sweep(
    use_taehv: bool = False,
    taehv_encode: bool = False,
    streaming_pipeline: str = "",
    no_streaming_taehv: bool = False,
):
    if streaming_pipeline:
        if no_streaming_taehv:
            suffix = "streaming_batch_taehv"
        elif not use_taehv:
            suffix = "streaming_wan_dec"
        else:
            suffix = f"streaming_{streaming_pipeline}"
    elif taehv_encode:
        suffix = "taehv_full"
    elif use_taehv:
        suffix = "taehv"
    else:
        suffix = "wan"
    tag = f"fastgen_{suffix}"

    print(f"Running warmup clip ({WARMUP_CLIP}) then {len(CLIPS)} measured clips ({tag})")
    print(f"Using max_containers=1 so all clips run on the same warm worker.")

    out_dir = _LOCAL_FASTGEN_ROOT / "modal_out" / f"benchmark_10clips_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warmup: run 1 clip first to absorb Wav2Vec2 lazy-load + CUDA kernel warmup
    all_clips = [WARMUP_CLIP] + CLIPS
    args = [(vn, use_taehv, taehv_encode, streaming_pipeline, no_streaming_taehv) for vn in all_clips]
    rows = []
    is_warmup = True
    for (vn, *_rest), csv_text in zip(
        args,
        run_benchmark.starmap(args, return_exceptions=True),
    ):
        stem = vn.replace("_cfr25.mp4", "")
        if is_warmup:
            is_warmup = False
            if isinstance(csv_text, Exception):
                print(f"  [WARMUP FAIL] {stem}: {csv_text}")
            else:
                print(f"  [WARMUP] {stem} (discarded)")
            continue
        if isinstance(csv_text, Exception):
            print(f"  [FAIL] {stem}: {csv_text}")
            continue
        if csv_text:
            (out_dir / f"{stem}.timing.csv").write_text(csv_text)
            lines = csv_text.strip().splitlines()
            if len(lines) >= 2:
                rows.append(lines)
            print(f"  [OK]   {stem}")
        else:
            print(f"  [FAIL] {stem}: no CSV")

    # Aggregate
    if rows:
        agg_path = out_dir / "aggregate.csv"
        with open(agg_path, "w") as f:
            f.write(rows[0][0] + "\n")  # header from first
            for r in rows:
                for line in r[1:]:
                    if line and not line.startswith("name") and "AVERAGE" not in line:
                        f.write(line + "\n")
        print(f"\nAggregate: {agg_path}")

    print(f"Done: {tag}")
