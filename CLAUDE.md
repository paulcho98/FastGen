# CLAUDE.md — FastGen (OmniAvatar Self-Forcing Distillation)

## Project Goal
Distill OmniAvatar's 14B V2V audio-driven lip sync model into a 1.3B student
using FastGen's Self-Forcing framework for fast few-step inference.
This repo also serves as the reference implementation for adapting FastGen to new
talking-face models (see `docs/fastgen-adaptation-guide.md`).

## Environment
This repo runs on two machines:
- **Dev machine** (original): `hb_fastgen` conda env, 4x H200, paths under `/home/work/`
- **Claude Code instance**: system Python 3.12 + pip-installed deps, 2x A100-80GB,
  paths under `/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/`

Key env vars (set before training):
```bash
export OMNIAVATAR_ROOT=".../OmniAvatar-Train"
export OMNIAVATAR_DATA_ROOT=".../datasets"
export OMNIAVATAR_STUDENT_CKPT=".../pretrained_models/step-1000.pt"
```

pip deps (install on fresh instance): `diffusers transformers safetensors accelerate
omegaconf attrs wandb loguru ftfy imageio webdataset boto3 hydra-core av kornia timm open_clip_torch`

**SAFETY**: Do NOT delete files outside this repo. Git commit every major change.
GitHub remote: `https://github.com/paulcho98/FastGen.git` (PAT in .git/config on NFS).

## Status (2026-03-24)
- Stage 1 training **verified end-to-end** on 3 real Hallo3 samples:
  - Diffusion Forcing: loss stable ~0.03-0.05, 2.5s/iter, 13.3GB peak
  - ODE KD: loss stable ~0.03-0.06, 2.5s/iter, 13.3GB peak
  - ODE trajectory generation: 14B teacher, 7.8min/sample, 31GB peak
- Stage 2 (Self-Forcing DMD): configs ready, not yet tested end-to-end
- Comprehensive code review completed — all critical bugs fixed
- LoRA merge moved to GPU (33s vs 40+ min on CPU for 14B)

## Architecture
- Teacher: 14B bidirectional `OmniAvatarWan(FastGenNetwork)` — frozen
- Student: 1.3B causal `CausalOmniAvatarWan(CausalFastGenNetwork)` — trainable
- Fake score: 1.3B bidirectional `OmniAvatarWan` — via `config.model.fake_score_net`
- V2V 65ch: noise(16)+ref(16)+mask(1)+masked_video(16)+ref_sequence(16)
- Causal: FlexAttention chunk-wise mask + KV cache, per-frame timesteps (auto-detected)
- Training dataloader: `OmniAvatarDataLoader` (infinite iterator + DistributedSampler)
- Stage 1 options: ODE KD (`config_kd.py`) or Diffusion Forcing (`config_df.py`)
- Causal forward auto-routing: `t.dim()==2` → full-sequence (KD), else `is_ar` flag (SF)

## Key Paths
On Claude Code instance:
- Pretrained models: `.../OmniAvatar-Train/pretrained_models/`
- Teacher ckpt: `.../pretrained_models/step-10500.pt` (14B, 1.2GB)
- Student ckpt: `.../pretrained_models/step-1000.pt` (1.3B, 339MB)
- Test data: `/data/karlo-research_715/workspace/kinemaar/datasets/sample_hallo3_latentsync/`
- TalkVid data: `/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid` (symlink)
- Mask: `.../OmniAvatar-Train/OmniAvatar/utils/latentsync/mask.png`

## Key Docs
- `docs/fastgen-adaptation-guide.md` — How to adapt FastGen for a new talking-face model
- `docs/code-review-findings.md` — Exhaustive comparison findings
- `docs/code-review-changes.md` — All fixes applied from code review
- `docs/implementation-notes.md` — Bug tracker (Bug 001-011)
- `docs/omniavatar-changes-summary.md` — File inventory and change summary

## Base FastGen Modifications (4 files)
- `config.py`: Added `fake_score_net: Optional[dict]` to `BaseModelConfig`
- `dmd2.py`: `build_model()` uses `fake_score_net` when set (separate arch from teacher)
- `methods/__init__.py`: Registered OmniAvatar model classes
- `noise_schedule.py`: Added `dtype` parameter to `sample_t_inhom`

## Test Configs
- `config_df_test.py` / `config_kd_test.py`: 20-iter smoke tests with stdout loss logging
- Run: `CUDA_VISIBLE_DEVICES=0 python train.py --config fastgen/configs/experiments/OmniAvatar/config_df_test.py`
- Uses `StdoutLoggerCallback` (no wandb needed)

## Config / Instantiation Gotchas
- `config.model_class._target_` (not `config.model._target_`) is the dispatch target; `config.model` is the attrs instance passed as `config=` kwarg.
- `config.log_config.{name,wandb_mode}`; `WANDB_MODE` env var is **ignored** — override via config.
- `config.model` is `attrs.define(slots=False)` but OmegaConf strips dynamic attrs on serialize. Declare typed `@attrs.define` classes for sub-configs (see `RewardConfig` in `config_omniavatar_sf.py`).
- `Model.__init__` calls `self.build_model()` during super chain — don't reassign attrs after `super().__init__()` or you'll clobber build_model state.
- `WanVideoVAE.decode` returns a stacked Tensor `[B,C,T,H,W]`, not a list.
- `data` dict flows intact from `single_train_step` → `_student_update_step` (`omniavatar_self_forcing.py:85`); dataset custom keys survive.
- Logger: `import fastgen.utils.logging_utils as logger` uses loguru + `@rank0_if_not_debug` — only rank 0 emits INFO. For all-rank debug: `print(..., file=sys.stderr, flush=True)`.
- Port 29500 needs ~30s TIME_WAIT cooldown between back-to-back torchrun launches.
- From a git worktree, `sys.path[0]=''` (CWD) wins over the editable install's MetaPathFinder — so imports resolve to the worktree correctly. Verify: `python -c "import fastgen; print(fastgen.__file__)"`.

## Re-DMD sync-C reward (branch `feat/redmd-sync-c`)
Full guide: `docs/redmd_sync_c.md`. Launch: `scripts/train_sf_sink1_window7_redmd.sh` (β=0.25 default). Smoke (4-GPU, 10 iters, MP4 dump): `scripts/smoke_test_redmd.sh`.
