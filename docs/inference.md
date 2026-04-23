# FastGen Re-DMD Inference Guide

How to run inference from a trained Self-Forcing checkpoint (this repo's output) on HDTF-format test data from a fresh machine.

---

## 1. What this bundle is

A **FastGen SF student (1.3B, causal V2V lip-sync)** trained via Re-DMD β=2 + TAEW decoder on top of a syncnet-trained DF initialization. The student consumes a reference video + driving audio and produces a lip-synced video.

**Run provenance** (see `scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh`):
- Student base: DF step-5000 from `train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh`
- Teacher: mouthweight 14B step-6000
- Fake-score init: syncnet-trained 1.3B adapter (`_fsmatched` variant)
- β=2, critic LR 3e-6, wandb run `1djswvuo`

---

## 2. Prerequisites on the target machine

### 2.1. Python environment

Same env as the training box: conda `hb_fastgen` (Python 3.12) or equivalent with the repo's `requirements.txt` / `pyproject.toml`. A CUDA 12.x GPU with ≥24 GB VRAM runs inference comfortably; 4 GPUs let you fan out across checkpoints.

### 2.2. Clone the repo

```bash
git clone https://github.com/paulcho98/FastGen.git
cd FastGen
git checkout feat/redmd-sync-c
pip install -e .
```

### 2.3. Base models (download / copy separately — not in this bundle)

| Artifact | Path on training box | Purpose |
|---|---|---|
| Wan2.1-T2V-1.3B (diffusion weights) | `/home/work/.local/OmniAvatar/pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors` | Student base architecture |
| Wan2.1 VAE | `/home/work/.local/OmniAvatar/pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` (484 MB) | Latent ↔ pixel |
| wav2vec2-base-960h | `/home/work/.local/OmniAvatar/pretrained_models/wav2vec2-base-960h/` | Audio conditioning |
| OmniAvatar V2V 1.3B adapter | `/home/work/output_omniavatar_v2v_1.3B_phase2/step-19500.pt` (339 MB) | V2V adapter weights; loaded via `--omniavatar_ckpt_path` |
| LatentSync mask | `/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png` (2 KB) | Face mask for compositing |
| Text embed (single prompt) | `/home/work/stableavatar_data/v2v_training_data/0010234f331f491ffacc538958094732_shot_001_000/text_emb.pt` | Static text cond (prompt = "a person talking") |

Wan and wav2vec are public HF downloads. The OmniAvatar adapter is internal; transfer `step-19500.pt` directly.

### 2.4. TAEW decoder (optional, for fast streaming inference)

```
pretrained_models/taew2_1.pth
```
Paper-referenced TAEHV tiny autoencoder. Only needed if you use `scripts/inference/inference_causal_taehv.py` or `inference_streaming.py`. The "main" path (`inference_causal.py`) uses the full Wan VAE and does NOT need TAEW.

---

## 3. Checkpoint format

Each SF training step is saved as a **sharded FSDP2 distributed checkpoint** alongside a small `.pth` metadata file:

```
checkpoints/
├── 0000100.pth                  # 4 KB — scheduler, grad_scaler, iteration
├── 0000100.net_model/           # ~5.3 GB — student model weights (distcp)
│   ├── __0_0.distcp
│   ├── __1_0.distcp
│   ├── __2_0.distcp
│   ├── __3_0.distcp
│   └── .metadata
├── 0000200.pth
└── 0000200.net_model/
```

**Only `.pth` + `.net_model/` are needed for inference.** The `.fake_score_model/`, `.net_optim/`, `.fake_score_optim/` subdirs are training-time only (critic + Adam state) and are **not** included in the inference bundle.

The loader (`scripts/inference/inference_causal.py:199-236`) auto-detects:
1. If `--ckpt_path <step>.pth` and adjacent `<step>.net_model/` exists → loads via `torch.distributed.checkpoint.FileSystemReader` (single-process is fine).
2. If `--ckpt_path` points at a `.net_model/` dir directly → same distcp path.
3. Else → falls back to `torch.load(.pth)` (only works for non-FSDP legacy checkpoints).

---

## 4. Input format expected by `inference_causal.py`

Generic contract (independent of HDTF):

| Argument | Accepted | Notes |
|---|---|---|
| `--video_path` | any `cv2.VideoCapture`-readable container (mp4/mov/mkv/avi) | alignment assumes **25 fps** (see `--fps`); training used `_cfr25.mp4` (H.264 CFR 25 fps) |
| `--audio_path` | any `librosa.load`-readable file (wav/mp3/flac/m4a/ogg) | resampled internally to **16 kHz mono** for wav2vec2; if omitted, extracted from the video via ffmpeg (`pcm_s16le -ar 16000 -ac 1`) |
| generation length | derived from audio | `num_video_frames = floor(audio_duration * 25)`, `num_latent_frames = num_video_frames // 4` (Wan VAE is 4× temporal). Override with `--num_latent_frames` (must satisfy chunk-size multiple) |
| `--text_embeds_path` | precomputed `.pt` from Wan's T5 encoder | alternative: `--text_encoder_path <dir>` + `--prompt "<str>"` to compute on the fly |
| `--mask_path` | 2 KB PNG defining mouth region | static file from LatentSync/StableAvatar |
| `--face_cache_dir` | dir containing `<video_stem>_face_cache.pt` per clip | required when `--latentsync` is set; auto-recomputes missing entries |

Output: 25 fps H.264 mp4 at `--output_path`. With `--latentsync`, mouth-only composite onto the original reference; without, full-face replacement. Optional `_aligned.mp4` sidecar shows the aligned face crop.

---

## 5. HDTF test-set layout (reference evaluation)

The above contract is what we supply at HDTF eval time via `HDTF_original_testset_81frames`. Layout:

```
HDTF_original_testset_81frames/
├── videos_cfr/                          # 33 clips, 81 frames each (3.24 s at 25 fps)
│   ├── RD_Radio18_000_cfr25.mp4
│   ├── WDA_AndyLevin_000_cfr25.mp4
│   └── ...
├── audios/                              # Driving audio candidates
│   ├── RD_Radio18_000.wav
│   └── ...
└── hdtf_video_audio_pairs.txt           # TSV: <video.mp4>\t<audio.wav>
```

- **Videos**: 81-frame constant-frame-rate 25 fps H.264 mp4 (`_cfr25` suffix). The video supplies the reference face + motion prior; only the face is altered.
- **Audios**: mono 16 kHz WAV is standard; wav2vec2 expects 16 kHz. File length is trimmed/padded to match video length during inference.
- **Pairs file**: tab-separated `video.mp4 \t audio.wav`. One pair per line. The driving audio is usually **a different speaker** than the reference video (cross-identity lip-sync test).

**Face cache** (~5.4 GB for 33 clips × mouth-only compositing) lives separately at `/home/work/.local/HDTF/face_cache/`. Either transfer it or let the inference script re-run face detection (slow). Pass the dir via `--face_cache_dir`; names expected: `<video_stem>_face_cache.pt` (e.g. `RD_Radio18_000_cfr25_face_cache.pt`).

---

## 6. Running inference

### 6.1. Single clip, single GPU

```bash
python scripts/inference/inference_causal.py \
    --ckpt_path /path/to/ckpts/0000200.pth \
    --vae_path /path/to/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
    --wav2vec_path /path/to/wav2vec2-base-960h \
    --mask_path /path/to/mask.png \
    --base_model_paths /path/to/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
    --omniavatar_ckpt_path /path/to/omniavatar_v2v_1.3B_phase2/step-19500.pt \
    --text_embeds_path /path/to/text_emb.pt \
    --video_path /path/to/HDTF/videos_cfr/RD_Radio18_000_cfr25.mp4 \
    --audio_path /path/to/HDTF/audios/WRA_MittRomney_000.wav \
    --output_path /tmp/out.mp4 \
    --t_list 0.999 0.833 0.0 \
    --local_attn_size 7 --sink_size 1 --use_dynamic_rope \
    --latentsync \
    --face_cache_dir /path/to/HDTF/face_cache
```

Key flags:
- `--t_list 0.999 0.833 0.0` — 3-step denoising schedule (matches training)
- `--local_attn_size 7 --sink_size 1 --use_dynamic_rope` — SF attention config (matches `sink1_window7` variant)
- `--latentsync` — use LatentSync-style face detection + mouth-only compositing (recommended)
- `--face_cache_dir` — reuse cached face detections; omit to recompute

### 6.2. Full HDTF eval across multiple checkpoints (parallel across GPUs)

A ready-made driver is in `scripts/infer_redmd_taew_audiofix_syncnet_mouthweight.sh` (and `_batched.sh` for the batched variant). Edit the hardcoded paths at the top for your machine, then:

```bash
CKPT_STEPS="100 200" GPUS="0 1" \
  bash scripts/infer_redmd_taew_audiofix_syncnet_mouthweight.sh
```

Outputs land in `$OUT_ROOT/step_0000XXX/<video_stem>.mp4`, one per pair in `hdtf_video_audio_pairs.txt`.

### 6.3. Streaming / TAEHV inference (optional, faster)

See `scripts/inference/inference_streaming.py` for per-chunk AR generation with first-frame-latency measurement. Requires `--taehv_ckpt path/to/taew2_1.pth`.

---

## 7. Evaluation

After inference produces `$OUT_ROOT/step_XXXX/*.mp4`, run:

```bash
bash scripts/eval_redmd_taew_audiofix_syncnet_mouthweight.sh
# or, for LMD only:
python scripts/eval/eval_lmd_only.py --pred_dir $OUT_ROOT/step_0000200 --gt_dir $HDTF/videos_cfr
```

Metrics covered: sync-C (syncnet confidence), LMD (landmark distance), FID (optional, separate pipeline).

---

## 8. Quick sanity check

After unzipping the checkpoint bundle:
```bash
ls checkpoints/
# Should show: 0000100.pth  0000100.net_model/  0000200.pth  0000200.net_model/

python -c "
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
sd = {}
dcp.load(sd, storage_reader=FileSystemReader('checkpoints/0000200.net_model'), no_dist=True)
print(f'Loaded {len(sd)} tensors; sample key: {next(iter(sd))}')
"
```

If that loads without error and prints a tensor count in the thousands, the bundle is intact.

---

## 9. Bundle contents (this transfer)

```
fastgen_redmd_fsmatched_lr3e6_ckpts.zip
└── checkpoints/
    ├── 0000100.pth
    ├── 0000100.net_model/
    ├── 0000200.pth
    └── 0000200.net_model/
```

Total ~11 GB uncompressed. Everything else (base models, adapters, HDTF data, face cache) is listed in §2 and §4 — transfer separately as needed.
