"""Modal H100 benchmark for inference_streaming.py: 1 warmup + 5 measured clips.

Records all 3 timing definitions (audio_to_decode, encode_to_decode,
pure_encode_to_decode) plus VRAM. max_containers=1 so the warmup actually
amortises kernel-compile cost on the same worker.

Usage:
    modal run scripts/modal/benchmark_h100_streaming_5clips.py::sweep
    modal run scripts/modal/benchmark_h100_streaming_5clips.py::sweep \
        --streaming-decoder wan_vae
    modal run scripts/modal/benchmark_h100_streaming_5clips.py::sweep \
        --streaming-decoder wan_vae --streamwise-encode
"""
import os
import pathlib
import subprocess
import modal

_LOCAL_FASTGEN_ROOT = pathlib.Path("/home/work/.local/hyunbin/FastGen-redmd")
_LOCAL_OMNIAVATAR_ROOT = pathlib.Path("/home/work/.local/OmniAvatar")

VOL = modal.Volume.from_name("fastgen-assets", create_if_missing=False)
ASSETS = "/assets"

# 1 warmup + 5 measured clips. Same set as the prior throughput benchmark.
WARMUP_CLIP = "WDA_AndyLevin_000_cfr25.mp4"
CLIPS = [
    "WDA_BarackObama_000_cfr25.mp4",
    "WDA_DonnaShalala1_000_cfr25.mp4",
    "WDA_NancyPelosi0_000_cfr25.mp4",
    "WRA_AdamKinzinger1_000_cfr25.mp4",
    "WRA_MittRomney_000_cfr25.mp4",
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

app = modal.App("fastgen-streaming-benchmark", image=image)


@app.function(gpu="H100", volumes={ASSETS: VOL}, timeout=60 * 30, max_containers=1)
def run_benchmark(
    video_name: str,
    streaming_decoder: str = "wan_vae",
    streamwise_encode: bool = False,
    defer_composite: bool = False,
    compile: bool = False,
) -> str:
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

    stem = video_name.replace("_cfr25.mp4", "")
    out_path = f"{out_dir}/{stem}.mp4"
    csv_path = f"{out_dir}/{stem}.timing.csv"

    cmd = [
        "python", "scripts/inference/inference_streaming.py",
        "--ckpt_path", f"{ASSETS}/sf_ckpts/0000600.pth",
        "--vae_path", f"{ASSETS}/wan_vae/Wan2.1_VAE.pth",
        "--wav2vec_path", f"{ASSETS}/wav2vec2-base-960h",
        "--mask_path", f"{ASSETS}/mask/mask.png",
        "--base_model_paths", f"{ASSETS}/wan_base/diffusion_pytorch_model.safetensors",
        "--omniavatar_ckpt_path", f"{ASSETS}/omniavatar/step-19500.pt",
        "--text_embeds_path", f"{ASSETS}/text_emb/text_emb.pt",
        "--video_path", f"{ASSETS}/hdtf/videos_batch/{video_name}",
        "--output_path", out_path,
        "--t_list", "0.999", "0.833", "0.0",
        "--chunk_size", "3",
        "--local_attn_size", "7", "--sink_size", "1",
        "--use_dynamic_rope",
        "--latentsync",
        "--face_cache_dir", f"{ASSETS}/hdtf/face_cache_batch",
        "--num_latent_frames", "21", "--min_latent_frames", "21",
        "--streaming_decoder", streaming_decoder,
        "--timing", "--timing_csv", csv_path,
    ]
    if streaming_decoder in ("streaming_taehv", "batch_taehv"):
        cmd += ["--taehv_ckpt", f"{ASSETS}/taehv/taew2_1.pth"]
    if streamwise_encode:
        cmd += ["--streamwise_encode"]
    if defer_composite:
        cmd += ["--defer_composite"]
    if compile:
        cmd += ["--compile"]

    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=work_dir, env=env, check=True)

    if os.path.exists(csv_path):
        return pathlib.Path(csv_path).read_text()
    return ""


@app.local_entrypoint()
def sweep(
    streaming_decoder: str = "wan_vae",
    streamwise_encode: bool = False,
    defer_composite: bool = False,
    compile: bool = False,
):
    suffix = streaming_decoder
    if streamwise_encode:
        suffix += "_streamwise"
    if defer_composite:
        suffix += "_deferred"
    if compile:
        suffix += "_compiled"
    tag = f"streaming_{suffix}"

    print(f"[{tag}] Warmup={WARMUP_CLIP}; measured={len(CLIPS)} clips on H100 "
          f"(max_containers=1).")

    out_dir = _LOCAL_FASTGEN_ROOT / "modal_out" / f"benchmark_h100_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_clips = [WARMUP_CLIP] + CLIPS
    args = [(vn, streaming_decoder, streamwise_encode, defer_composite, compile)
            for vn in all_clips]
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

    if rows:
        agg_path = out_dir / "aggregate.csv"
        with open(agg_path, "w") as f:
            f.write(rows[0][0] + "\n")
            for r in rows:
                for line in r[1:]:
                    if line and not line.startswith("name") and "AVERAGE" not in line:
                        f.write(line + "\n")
        print(f"\nAggregate: {agg_path}")
    print(f"Done: {tag}")
