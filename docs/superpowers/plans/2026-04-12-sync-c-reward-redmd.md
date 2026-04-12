# SyncNet-v2 Sync-C Reward for Re-DMD in FastGen OmniAvatar SF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Reward-Forcing-style reward-weighted distillation (Re-DMD) on top of FastGen's existing OmniAvatar self-forcing DMD training, using a SyncNet-v2 lip-sync confidence (sync-C) as the detached scalar reward.

**Architecture:** We subclass `OmniAvatarSelfForcing` → `OmniAvatarSelfForcingReDMD` and override only `_student_update_step` to scale the VSD loss by `exp(β · sync_c)`. The scorer (`SyncCScorer`) matches the `reward_from_frames(video_tensors, audio_tensors, prompts)` interface documented in Reward-Forcing's `docs/sync_c_scorer_design.md`. Raw audio waveform is plumbed through by adding a new `audio_waveform` key to the dataset batch (alongside the existing `audio_path`). Pixel decoding for the reward happens via the existing VAE wrapper under `torch.no_grad()`, separate from the main training path.

**Tech Stack:** PyTorch 2.x, FSDP2, torchaudio (MFCC), torchrun launch. SyncNet-v2 checkpoint from joonson/syncnet_python via LatentSync-1.5 distribution. Existing FastGen DMD2 / OmniAvatar SF training stack.

---

## Context and references

Read before starting:
- `/home/work/.local/hyunbin/Reward-Forcing/docs/sync_c_scorer_design.md` — the SyncCScorer design reference this plan operationalizes.
- `/home/work/.local/hyunbin/Reward-Forcing/docs/reward_forcing_implementation.md` — original Re-DMD writeup; §2 for the β convention, §7 for how to swap in a custom reward, §8 for empirical reward-scale numbers.
- `fastgen/methods/distribution_matching/dmd2.py:209-271` — base `_student_update_step` we'll override.
- `fastgen/methods/common_loss.py:63-103` — `variational_score_distillation_loss` (pseudo_target is built inside `torch.no_grad()`, so scaling the returned loss scales the generator gradient linearly).
- `fastgen/methods/omniavatar_self_forcing.py:40-93` — `single_train_step` override that fires both critic and student on iteration % 5 == 0.
- `fastgen/methods/omniavatar_self_forcing.py:210-260` — `_prepare_training_data`; this is where we'll add audio_waveform to the condition dict.
- `fastgen/datasets/omniavatar_dataloader.py:108-187` — dataset's `__getitem__`; the `audio_path` key already lands here but raw waveform is not loaded.

**Key formula (Re-DMD, outer-multiply variant):**
```
loss_gen_weighted = exp(β · sync_c_detached) · vsd_loss_unweighted
                  + gan_loss_weight_gen · gan_loss_gen
```
The `additional_scale` parameter already in `common_loss.py:67,93-94` is an alternative path (scales `w` inside VSD), but outer-multiply is chosen because:
  1. It literally matches the paper's formula in both gradient and logged loss value.
  2. It keeps `vsd_loss_unweighted` separately available for logging (parallel to Reward-Forcing's `dmd_mse_unweighted_mean`).

**Target baseline config:** `config_sf_sink1_window7_tscfg.py`, launched via `scripts/train_sf_sink1_window7.sh` with `torchrun --nproc_per_node=4`. `student_update_freq=5`, `batch_size=8`, `grad_accum_rounds=2`, 81-frame clips at 25 fps, FSDP with autocast around `single_train_step`.

---

## File structure

### New files
| File | Responsibility |
|---|---|
| `fastgen/methods/reward/__init__.py` | Package marker |
| `fastgen/methods/reward/syncnet_v2.py` | Vendored `S` module from joonson/syncnet_python (architecture only, no evaluation logic) |
| `fastgen/methods/reward/sync_c_scorer.py` | `SyncCScorer` class: face-aligned tensor in → sync-C scalar out, fully no_grad |
| `fastgen/methods/omniavatar_self_forcing_re_dmd.py` | `OmniAvatarSelfForcingReDMD` subclass overriding `_student_update_step` |
| `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd.py` | Config for the rewarded run |
| `scripts/train_sf_sink1_window7_redmd.sh` | Launch script |
| `tests/reward/__init__.py` | Test package marker |
| `tests/reward/test_sync_c_scorer.py` | Unit tests for `SyncCScorer` |
| `tests/reward/test_re_dmd_trainer.py` | Unit test for the rewarded `_student_update_step` with a mocked scorer |

### Modified files
| File | Change |
|---|---|
| `fastgen/datasets/omniavatar_dataloader.py:108-187` | Add `audio_waveform` key (raw 16 kHz float32 tensor) to `__getitem__` return dict |
| `fastgen/methods/omniavatar_self_forcing.py:210-260` (or subclass) | Thread `audio_waveform` into condition dict so the subclass can reach it |

### Not modified (verified)
- `fastgen/trainer.py` — generic; needs no changes.
- `fastgen/methods/distribution_matching/dmd2.py` — base class is fine; we subclass at the OmniAvatar level.
- `fastgen/methods/common_loss.py` — we deliberately do **not** use `additional_scale`; outer-multiply in the subclass.

---

## Task 0: Repo setup

**Files:**
- Create: `fastgen/methods/reward/__init__.py`
- Create: `tests/reward/__init__.py`

- [ ] **Step 1: Create empty package markers**

```bash
mkdir -p /home/work/.local/hyunbin/FastGen/fastgen/methods/reward
mkdir -p /home/work/.local/hyunbin/FastGen/tests/reward
touch /home/work/.local/hyunbin/FastGen/fastgen/methods/reward/__init__.py
touch /home/work/.local/hyunbin/FastGen/tests/reward/__init__.py
```

- [ ] **Step 2: Verify pytest runs at repo root**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/ -q --collect-only`
Expected: "no tests ran" or collects zero tests cleanly (no import errors).

- [ ] **Step 3: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add fastgen/methods/reward/__init__.py tests/reward/__init__.py
git commit -m "scaffold: reward/ package and tests/reward/ for Re-DMD sync reward"
```

---

## Task 1: Vendor SyncNet-v2 architecture

**Files:**
- Create: `fastgen/methods/reward/syncnet_v2.py`

- [ ] **Step 1: Copy the model architecture**

Copy the `S` class definition verbatim from `/home/work/.local/eval_metrics/eval/syncnet/syncnet.py` (lines 18-113) into `fastgen/methods/reward/syncnet_v2.py`. Rename `S` to `SyncNetV2` and remove the top-level `save`/`load` helpers. Final file content:

```python
# Vendored from https://github.com/joonson/syncnet_python/blob/master/SyncNetModel.py
# Used as a detached reward model under torch.no_grad() — no training.

import torch.nn as nn


class SyncNetV2(nn.Module):
    def __init__(self, num_layers_in_fc_layers: int = 1024):
        super().__init__()

        self.__nFeatures__ = 24
        self.__nChs__ = 32
        self.__midChs__ = 32

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

    def forward_aud(self, x):
        mid = self.netcnnaud(x)
        mid = mid.view((mid.size()[0], -1))
        out = self.netfcaud(mid)
        return out

    def forward_lip(self, x):
        mid = self.netcnnlip(x)
        mid = mid.view((mid.size()[0], -1))
        out = self.netfclip(mid)
        return out
```

- [ ] **Step 2: Write a loadability test**

Create `tests/reward/test_syncnet_v2.py`:

```python
import os
import pytest
import torch

from fastgen.methods.reward.syncnet_v2 import SyncNetV2

CKPT = "/home/work/.local/eval_metrics/eval/checkpoints/auxiliary/syncnet_v2.model"


def test_instantiates():
    m = SyncNetV2()
    assert sum(p.numel() for p in m.parameters()) > 10_000_000


def test_forward_shapes():
    m = SyncNetV2().eval()
    with torch.no_grad():
        lip = torch.randn(2, 3, 5, 224, 224)
        aud = torch.randn(2, 1, 13, 20)
        lip_emb = m.forward_lip(lip)
        aud_emb = m.forward_aud(aud)
    assert lip_emb.shape == (2, 1024)
    assert aud_emb.shape == (2, 1024)


@pytest.mark.skipif(not os.path.exists(CKPT), reason="SyncNet-v2 checkpoint not present locally")
def test_loads_checkpoint():
    m = SyncNetV2()
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    m.load_state_dict(state, strict=True)
```

- [ ] **Step 3: Run the tests**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_syncnet_v2.py -v`
Expected: `test_instantiates` and `test_forward_shapes` PASS. `test_loads_checkpoint` PASS if checkpoint is at `/home/work/.local/eval_metrics/eval/checkpoints/auxiliary/syncnet_v2.model`, otherwise SKIP.

- [ ] **Step 4: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add fastgen/methods/reward/syncnet_v2.py tests/reward/test_syncnet_v2.py
git commit -m "feat: vendor SyncNet-v2 architecture for Re-DMD sync reward"
```

---

## Task 2: SyncCScorer — video and audio preparation

**Files:**
- Create: `fastgen/methods/reward/sync_c_scorer.py`
- Test: `tests/reward/test_sync_c_scorer.py`

This task builds the scorer incrementally: prep helpers first, then scoring, each with its own test.

- [ ] **Step 1: Write failing test for `_prep_video`**

Create `tests/reward/test_sync_c_scorer.py`:

```python
import pytest
import torch

from fastgen.methods.reward.sync_c_scorer import SyncCScorer


@pytest.fixture
def scorer():
    # No-ckpt constructor for prep-only tests
    return SyncCScorer.__new__(SyncCScorer)


def test_prep_video_shape_dtype(scorer):
    scorer.device = "cpu"
    scorer.dtype = torch.float32
    scorer.face_crop_size = 224
    video = torch.randint(0, 256, (81, 3, 512, 512), dtype=torch.uint8)
    out = scorer._prep_video(video)
    assert out.shape == (1, 3, 81, 224, 224)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_sync_c_scorer.py::test_prep_video_shape_dtype -v`
Expected: FAIL with `ModuleNotFoundError` or `AttributeError` (scorer file doesn't exist yet).

- [ ] **Step 3: Create scorer stub with `_prep_video`**

Create `fastgen/methods/reward/sync_c_scorer.py`:

```python
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from fastgen.methods.reward.syncnet_v2 import SyncNetV2


class SyncCScorer(nn.Module):
    """SyncNet-v2 sync-C scorer. See Reward-Forcing docs/sync_c_scorer_design.md."""

    def __init__(
        self,
        checkpoint_path: str,
        input_fps: float = 25.0,
        audio_sample_rate: int = 16000,
        face_crop_size: int = 224,
        vshift: int = 15,
        mfcc_n: int = 13,
        mfcc_hop_ms: float = 10.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert input_fps == 25.0, "SyncNet-v2 is native 25 fps; resample upstream"
        self.net = SyncNetV2()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, nn.Module):
            state = state.state_dict()
        self.net.load_state_dict(state, strict=True)
        self.net.eval().to(device=device, dtype=dtype)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.audio_sample_rate = audio_sample_rate
        self.target_sample_rate = 16000
        self.face_crop_size = face_crop_size
        self.vshift = vshift
        self.device = device
        self.dtype = dtype
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.target_sample_rate,
            n_mfcc=mfcc_n,
            melkwargs={
                "n_fft": 512,
                "win_length": int(0.025 * self.target_sample_rate),
                "hop_length": int(mfcc_hop_ms / 1000 * self.target_sample_rate),
                "n_mels": 40,
                "center": False,
            },
        ).to(device)

    def _prep_video(self, video: torch.Tensor) -> torch.Tensor:
        # [F, 3, H, W] uint8 -> [1, 3, F, 224, 224] float in [0, 1]
        video = video.to(self.device).float() / 255.0
        video = F.interpolate(
            video, size=(self.face_crop_size, self.face_crop_size),
            mode="bilinear", align_corners=False,
        )
        return video.permute(1, 0, 2, 3).unsqueeze(0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_sync_c_scorer.py::test_prep_video_shape_dtype -v`
Expected: PASS.

- [ ] **Step 5: Add failing test for `_prep_audio`**

Append to `tests/reward/test_sync_c_scorer.py`:

```python
def test_prep_audio_shape_16k(scorer):
    scorer.device = "cpu"
    scorer.dtype = torch.float32
    scorer.audio_sample_rate = 16000
    scorer.target_sample_rate = 16000
    scorer.mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=13,
        melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160,
                   "n_mels": 40, "center": False},
    )
    # 3.24s of audio at 16 kHz
    audio = torch.randn(int(16000 * 3.24))
    out = scorer._prep_audio(audio)
    # Expected MFCC length ~ (L - win + hop) / hop = (51840 - 400 + 160) / 160 ~= 322
    assert out.shape[:3] == (1, 1, 13)
    assert 280 <= out.shape[-1] <= 340, f"MFCC length {out.shape[-1]} out of expected band"


def test_prep_audio_resamples_from_48k(scorer):
    scorer.device = "cpu"
    scorer.dtype = torch.float32
    scorer.audio_sample_rate = 48000
    scorer.target_sample_rate = 16000
    scorer.mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=13,
        melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160,
                   "n_mels": 40, "center": False},
    )
    audio = torch.randn(int(48000 * 3.24))
    out = scorer._prep_audio(audio)
    assert 280 <= out.shape[-1] <= 340
```

Also add `import torchaudio` to the top of the test file if not already there.

- [ ] **Step 6: Run to verify fails (not implemented)**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_sync_c_scorer.py::test_prep_audio_shape_16k -v`
Expected: FAIL with `AttributeError: 'SyncCScorer' object has no attribute '_prep_audio'`.

- [ ] **Step 7: Implement `_prep_audio`**

Append to `sync_c_scorer.py`:

```python
    def _prep_audio(self, audio: torch.Tensor) -> torch.Tensor:
        # [L] float -> [1, 1, 13, M]
        audio = audio.to(self.device).float()
        if self.audio_sample_rate != self.target_sample_rate:
            audio = torchaudio.functional.resample(
                audio, self.audio_sample_rate, self.target_sample_rate,
            )
        mfcc = self.mfcc(audio.unsqueeze(0))  # [1, 13, M]
        return mfcc.unsqueeze(1)  # [1, 1, 13, M]
```

- [ ] **Step 8: Run both audio tests to verify pass**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_sync_c_scorer.py -v -k "prep_audio"`
Expected: both PASS.

- [ ] **Step 9: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add fastgen/methods/reward/sync_c_scorer.py tests/reward/test_sync_c_scorer.py
git commit -m "feat: SyncCScorer video/audio preprocessing"
```

---

## Task 3: SyncCScorer — windowing and offset search

**Files:**
- Modify: `fastgen/methods/reward/sync_c_scorer.py`
- Modify: `tests/reward/test_sync_c_scorer.py`

- [ ] **Step 1: Add windowing tests**

Append to `tests/reward/test_sync_c_scorer.py`:

```python
def test_lip_windows_81_frames(scorer):
    scorer.device = "cpu"
    video = torch.zeros(1, 3, 81, 224, 224)
    out = scorer._lip_windows(video)
    assert out.shape == (77, 3, 5, 224, 224)


def test_aud_windows_length(scorer):
    scorer.device = "cpu"
    mfcc = torch.zeros(1, 1, 13, 324)
    out = scorer._aud_windows(mfcc)
    # stride 4, window 20: (324 - 20) / 4 + 1 = 77
    assert out.shape == (77, 1, 13, 20)


def test_lip_windows_rejects_short_clip(scorer):
    scorer.device = "cpu"
    with pytest.raises(ValueError, match="at least 5"):
        scorer._lip_windows(torch.zeros(1, 3, 4, 224, 224))
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_sync_c_scorer.py -v -k "windows"`
Expected: FAIL with `AttributeError` on `_lip_windows` / `_aud_windows`.

- [ ] **Step 3: Implement windowing helpers**

Append to `sync_c_scorer.py`:

```python
    def _lip_windows(self, video: torch.Tensor) -> torch.Tensor:
        # [1, 3, F, 224, 224] -> [F-4, 3, 5, 224, 224]
        F_ = video.shape[2]
        if F_ < 5:
            raise ValueError(f"Need at least 5 frames, got {F_}")
        w = video.unfold(2, 5, 1).squeeze(0)  # [3, N, 224, 224, 5]
        return w.permute(1, 0, 4, 2, 3).contiguous()

    def _aud_windows(self, mfcc: torch.Tensor) -> torch.Tensor:
        # [1, 1, 13, M] -> [N, 1, 13, 20], stride 4 (MFCC is 100 fps, video 25 fps)
        M = mfcc.shape[-1]
        if M < 20:
            raise ValueError(f"Need at least 20 MFCC frames, got {M}")
        w = mfcc.unfold(-1, 20, 4).squeeze(0).squeeze(0)  # [13, N, 20]
        return w.permute(1, 0, 2).unsqueeze(1)
```

- [ ] **Step 4: Run windowing tests to verify pass**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_sync_c_scorer.py -v -k "windows"`
Expected: all PASS.

- [ ] **Step 5: Add offset-search test**

Append to `tests/reward/test_sync_c_scorer.py`:

```python
def test_offset_search_returns_scalar(scorer):
    scorer.vshift = 15
    # Random orthogonal embeddings — expected median ~ sqrt(2) (unit vectors,
    # random direction), min should be slightly below it. Any positive
    # margin is acceptable.
    torch.manual_seed(0)
    lip = F.normalize(torch.randn(50, 1024), dim=-1)
    aud = F.normalize(torch.randn(50, 1024), dim=-1)
    conf = scorer._offset_search(lip, aud)
    assert conf.ndim == 0
    assert torch.isfinite(conf)


def test_offset_search_perfect_alignment_scores_high(scorer):
    scorer.vshift = 15
    torch.manual_seed(1)
    emb = F.normalize(torch.randn(50, 1024), dim=-1)
    # Perfect sync: lip == aud → min distance at shift 0 is exactly 0
    conf = scorer._offset_search(emb, emb)
    assert conf > 0.5, f"confidence should be clearly positive, got {conf}"
```

Add `import torch.nn.functional as F` to the test file if not already imported.

- [ ] **Step 6: Run test to verify fail**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_sync_c_scorer.py -v -k "offset_search"`
Expected: FAIL with `AttributeError: '_offset_search'`.

- [ ] **Step 7: Implement offset search**

Append to `sync_c_scorer.py`:

```python
    def _offset_search(self, lip_emb: torch.Tensor, aud_emb: torch.Tensor) -> torch.Tensor:
        # lip_emb, aud_emb: [N, 1024]
        # Returns scalar sync-C = median(mean_dists) - min(mean_dists) across shifts.
        N = min(lip_emb.shape[0], aud_emb.shape[0])
        lip_emb, aud_emb = lip_emb[:N], aud_emb[:N]
        dists = []
        for shift in range(-self.vshift, self.vshift + 1):
            if shift < 0:
                l, a = lip_emb[-shift:], aud_emb[:N + shift]
            elif shift > 0:
                l, a = lip_emb[:N - shift], aud_emb[shift:]
            else:
                l, a = lip_emb, aud_emb
            d = F.pairwise_distance(l, a).mean()
            dists.append(d)
        mean_dists = torch.stack(dists, dim=0)
        return mean_dists.median() - mean_dists.min()
```

- [ ] **Step 8: Run offset-search tests to verify pass**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_sync_c_scorer.py -v -k "offset_search"`
Expected: both PASS.

- [ ] **Step 9: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add fastgen/methods/reward/sync_c_scorer.py tests/reward/test_sync_c_scorer.py
git commit -m "feat: SyncCScorer windowing and offset search"
```

---

## Task 4: SyncCScorer — `reward_from_frames` public entry

**Files:**
- Modify: `fastgen/methods/reward/sync_c_scorer.py`
- Modify: `tests/reward/test_sync_c_scorer.py`

- [ ] **Step 1: Add end-to-end test with mocked SyncNet**

Append to `tests/reward/test_sync_c_scorer.py`:

```python
def test_reward_from_frames_returns_dict_with_MQ_alias(monkeypatch, scorer):
    scorer.device = "cpu"
    scorer.dtype = torch.float32
    scorer.audio_sample_rate = 16000
    scorer.target_sample_rate = 16000
    scorer.face_crop_size = 224
    scorer.vshift = 15
    scorer.mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=13,
        melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160,
                   "n_mels": 40, "center": False},
    )

    # Mock the SyncNet to return deterministic embeddings
    class _FakeNet:
        def forward_lip(self, x):
            return F.normalize(torch.randn(x.shape[0], 1024, generator=torch.Generator().manual_seed(0)), dim=-1)
        def forward_aud(self, x):
            return F.normalize(torch.randn(x.shape[0], 1024, generator=torch.Generator().manual_seed(1)), dim=-1)
    scorer.net = _FakeNet()

    video = torch.randint(0, 256, (81, 3, 128, 128), dtype=torch.uint8)
    audio = torch.randn(int(16000 * 3.24))

    out = scorer.reward_from_frames([video], [audio])
    assert set(out.keys()) >= {"sync_c", "MQ"}
    assert out["sync_c"].shape == (1,)
    assert torch.equal(out["sync_c"], out["MQ"])  # MQ is an alias


def test_reward_from_frames_batched(monkeypatch, scorer):
    scorer.device = "cpu"
    scorer.dtype = torch.float32
    scorer.audio_sample_rate = 16000
    scorer.target_sample_rate = 16000
    scorer.face_crop_size = 224
    scorer.vshift = 15
    scorer.mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=13,
        melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160,
                   "n_mels": 40, "center": False},
    )

    class _FakeNet:
        def forward_lip(self, x):
            return F.normalize(torch.randn(x.shape[0], 1024), dim=-1)
        def forward_aud(self, x):
            return F.normalize(torch.randn(x.shape[0], 1024), dim=-1)
    scorer.net = _FakeNet()

    videos = [torch.randint(0, 256, (81, 3, 128, 128), dtype=torch.uint8) for _ in range(4)]
    audios = [torch.randn(int(16000 * 3.24)) for _ in range(4)]

    out = scorer.reward_from_frames(videos, audios)
    assert out["sync_c"].shape == (4,)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_sync_c_scorer.py -v -k "reward_from_frames"`
Expected: FAIL with `AttributeError: 'reward_from_frames'`.

- [ ] **Step 3: Implement `reward_from_frames` and `_score_single`**

Append to `sync_c_scorer.py`:

```python
    @torch.no_grad()
    def reward_from_frames(
        self,
        video_tensors: List[torch.Tensor],
        audio_tensors: List[torch.Tensor],
        prompts: Optional[List[str]] = None,
        use_norm: bool = True,
    ) -> Dict[str, torch.Tensor]:
        assert len(video_tensors) == len(audio_tensors), "video/audio batch mismatch"
        confs = [self._score_single(v, a) for v, a in zip(video_tensors, audio_tensors)]
        sync_c = torch.stack(confs, dim=0)
        return {"sync_c": sync_c, "MQ": sync_c}

    def _score_single(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        video = self._prep_video(video)
        mfcc = self._prep_audio(audio)
        lip_windows = self._lip_windows(video)
        aud_windows = self._aud_windows(mfcc)
        lip_emb = self.net.forward_lip(lip_windows.to(self.dtype))
        aud_emb = self.net.forward_aud(aud_windows.to(self.dtype))
        return self._offset_search(lip_emb, aud_emb)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_sync_c_scorer.py -v`
Expected: all PASS. Skip on `test_loads_checkpoint` if ckpt not accessible is fine.

- [ ] **Step 5: Add a GPU integration test (skipped by default)**

Append:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not os.path.exists("/home/work/.local/eval_metrics/eval/checkpoints/auxiliary/syncnet_v2.model"),
    reason="SyncNet-v2 checkpoint not present",
)
def test_real_scorer_gpu_runs():
    s = SyncCScorer(
        checkpoint_path="/home/work/.local/eval_metrics/eval/checkpoints/auxiliary/syncnet_v2.model",
        device="cuda",
    )
    video = torch.randint(0, 256, (81, 3, 224, 224), dtype=torch.uint8)
    audio = torch.randn(int(16000 * 3.24))
    out = s.reward_from_frames([video], [audio])
    assert out["sync_c"].shape == (1,)
    assert torch.isfinite(out["sync_c"]).all()
```

Add `import os` to the test file if not already present.

- [ ] **Step 6: Run GPU test**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_sync_c_scorer.py::test_real_scorer_gpu_runs -v`
Expected: PASS if on a CUDA box with the checkpoint, otherwise SKIP.

- [ ] **Step 7: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add fastgen/methods/reward/sync_c_scorer.py tests/reward/test_sync_c_scorer.py
git commit -m "feat: SyncCScorer.reward_from_frames public entry (VideoVLMRewardInference-shaped)"
```

---

## Task 5: Plumb raw audio waveform through the dataset

**Files:**
- Modify: `fastgen/datasets/omniavatar_dataloader.py:108-187`
- Create: `tests/reward/test_dataset_audio_waveform.py`

- [ ] **Step 1: Write failing test**

Create `tests/reward/test_dataset_audio_waveform.py`:

```python
"""Smoke test: dataset emits raw 16 kHz mono waveform alongside audio_path."""
import os
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.path.exists("/home/work/stableavatar_data/v2v_training_data/video_square_path.txt"),
    reason="Training data not mounted",
)


def test_dataset_emits_audio_waveform():
    from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataset

    # Minimal init — match the keys the constructor actually needs.
    ds = OmniAvatarDataset(
        data_list_path="/home/work/stableavatar_data/v2v_training_data/video_square_path.txt",
        latentsync_mask_path="/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png",
        load_raw_audio=True,
        raw_audio_sample_rate=16000,
        raw_audio_num_frames=81,
        use_ref_sequence=True,
    )
    sample = ds[0]
    assert "audio_waveform" in sample
    wav = sample["audio_waveform"]
    assert wav.dtype == torch.float32
    # 81 frames at 25 fps = 3.24s; at 16 kHz = 51840 samples
    assert abs(wav.shape[0] - 51840) < 160, f"got {wav.shape[0]}"
    assert wav.abs().max() <= 1.0 + 1e-3  # normalized [-1, 1]
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_dataset_audio_waveform.py -v`
Expected: FAIL with `TypeError: unexpected keyword argument 'load_raw_audio'` (dataset doesn't have the knob yet).

- [ ] **Step 3: Inspect current dataset `__init__` and `__getitem__`**

Read `fastgen/datasets/omniavatar_dataloader.py:26-188`. Locate the `__init__` signature and the `__getitem__` return site (around line 154-155 where `audio_path` is added to the sample dict). Confirm the line numbers before editing.

- [ ] **Step 4: Add `load_raw_audio` constructor knobs**

Modify `OmniAvatarDataset.__init__` to accept:
  - `load_raw_audio: bool = False`
  - `raw_audio_sample_rate: int = 16000`
  - `raw_audio_num_frames: int = 81`
  - `raw_audio_fps: float = 25.0`

Store them on self. Compute `self.raw_audio_length = int(raw_audio_num_frames / raw_audio_fps * raw_audio_sample_rate)` (= 51840 for defaults).

- [ ] **Step 5: Add raw audio loading in `__getitem__`**

After the existing `audio_path` key assignment (around line 154-155), add a branch that loads the wav if `self.load_raw_audio` is True:

```python
if self.load_raw_audio and os.path.exists(sample["audio_path"]):
    import torchaudio
    wav, sr = torchaudio.load(sample["audio_path"])
    # Mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    # Resample to target rate
    if sr != self.raw_audio_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, self.raw_audio_sample_rate)
    # Pad or truncate to fixed length (matches 81 frames at 25 fps)
    L = self.raw_audio_length
    if wav.shape[0] < L:
        wav = torch.nn.functional.pad(wav, (0, L - wav.shape[0]))
    else:
        wav = wav[:L]
    sample["audio_waveform"] = wav.to(torch.float32)
```

Place the `import os` at file top if not already imported.

- [ ] **Step 6: Run test to verify pass**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_dataset_audio_waveform.py -v`
Expected: PASS.

- [ ] **Step 7: Verify non-breaking — default path still works**

```python
# tests/reward/test_dataset_audio_waveform.py — append:
def test_dataset_default_no_waveform_key():
    from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataset
    ds = OmniAvatarDataset(
        data_list_path="/home/work/stableavatar_data/v2v_training_data/video_square_path.txt",
        latentsync_mask_path="/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png",
        use_ref_sequence=True,
    )
    sample = ds[0]
    assert "audio_waveform" not in sample
```

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_dataset_audio_waveform.py -v`
Expected: both PASS.

- [ ] **Step 8: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add fastgen/datasets/omniavatar_dataloader.py tests/reward/test_dataset_audio_waveform.py
git commit -m "feat: optional raw audio waveform in OmniAvatarDataset for reward scoring"
```

---

## Task 6: `OmniAvatarSelfForcingReDMD` subclass — reward integration

**Files:**
- Create: `fastgen/methods/omniavatar_self_forcing_re_dmd.py`
- Create: `tests/reward/test_re_dmd_trainer.py`

- [ ] **Step 1: Write failing test with mocked scorer and VAE**

Create `tests/reward/test_re_dmd_trainer.py`:

```python
"""Test that the reward-weighted student loss equals exp(β·r) * vsd_loss_unweighted."""
import math
import pytest
import torch

from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD


class _FakeScorer:
    """Returns a configurable constant sync_c."""
    def __init__(self, sync_c: float):
        self.sync_c = sync_c

    def reward_from_frames(self, videos, audios, prompts=None, use_norm=True):
        c = torch.full((len(videos),), self.sync_c, dtype=torch.float32,
                       device=videos[0].device)
        return {"sync_c": c, "MQ": c}


def test_weighted_loss_equals_exp_beta_r_times_unweighted():
    model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    model.config = type("C", (), {})()
    model.config.reward_beta = 0.25
    model.config.center_reward = False
    model.config.clamp_reward = None
    model.reward_scorer = _FakeScorer(sync_c=4.0)

    # Fake video/audio tensors
    videos = [torch.randint(0, 256, (81, 3, 224, 224), dtype=torch.uint8)]
    audios = [torch.randn(51840)]

    vsd_loss = torch.tensor(1.5, requires_grad=False)
    weighted, log_map = model._apply_reward_weighting(vsd_loss, videos, audios)

    expected_weight = math.exp(0.25 * 4.0)
    assert abs(weighted.item() - expected_weight * 1.5) < 1e-4
    assert abs(log_map["reward_sync_c_mean"] - 4.0) < 1e-6
    assert abs(log_map["reward_weight_mean"] - expected_weight) < 1e-4
    assert abs(log_map["vsd_loss_unweighted"] - 1.5) < 1e-6
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_re_dmd_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create the subclass with `_apply_reward_weighting`**

Create `fastgen/methods/omniavatar_self_forcing_re_dmd.py`:

```python
"""OmniAvatar Self-Forcing with Re-DMD reward weighting.

Overrides _student_update_step to scale the VSD loss by exp(beta * sync_c_detached),
matching the Reward-Forcing paper formulation:

    L_gen = 0.5 * exp(beta * r) * MSE(gen_latent, (gen_latent - DMD_grad).detach())

The reward is a scalar sync-C from SyncNet-v2 (detached, no gradient). See
docs/superpowers/plans/2026-04-12-sync-c-reward-redmd.md for the plan.
"""
import logging
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from fastgen.methods.common_loss import variational_score_distillation_loss
from fastgen.methods.omniavatar_self_forcing import OmniAvatarSelfForcing
from fastgen.methods.reward.sync_c_scorer import SyncCScorer

logger = logging.getLogger(__name__)


class OmniAvatarSelfForcingReDMD(OmniAvatarSelfForcing):
    """Rewarded variant of OmniAvatar Self-Forcing. All Re-DMD logic lives here."""

    def __init__(self, config):
        super().__init__(config)
        self.reward_scorer: Optional[SyncCScorer] = None
        self._reward_running_mean: Optional[float] = None  # for center_reward

    def build_model(self):
        super().build_model()
        rcfg = getattr(self.config, "reward", None)
        if rcfg is None or not getattr(rcfg, "enabled", True):
            logger.info("Re-DMD reward disabled — running as vanilla OmniAvatar SF.")
            return
        device_str = f"cuda:{self.device}" if isinstance(self.device, int) else str(self.device)
        self.reward_scorer = SyncCScorer(
            checkpoint_path=rcfg.checkpoint_path,
            input_fps=rcfg.input_fps,
            audio_sample_rate=rcfg.audio_sample_rate,
            vshift=rcfg.vshift,
            device=device_str,
            dtype=torch.float32,
        )
        logger.info(
            f"SyncCScorer loaded: beta={rcfg.beta}, vshift={rcfg.vshift}, "
            f"ckpt={rcfg.checkpoint_path}"
        )

    def _apply_reward_weighting(
        self,
        vsd_loss: torch.Tensor,
        videos,
        audios,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute exp(beta * sync_c) and multiply vsd_loss by it.

        Returns (weighted_loss, log_map). log_map entries are python floats
        meant for the wandb loss_dict.
        """
        with torch.no_grad():
            reward = self.reward_scorer.reward_from_frames(videos, audios)
        sync_c = reward["sync_c"].detach().float()  # [B]

        beta = float(self.config.reward_beta)

        # Optional centering
        if getattr(self.config, "center_reward", False):
            ema_alpha = 0.9
            batch_mean = sync_c.mean().item()
            if self._reward_running_mean is None:
                self._reward_running_mean = batch_mean
            else:
                self._reward_running_mean = (
                    ema_alpha * self._reward_running_mean + (1 - ema_alpha) * batch_mean
                )
            sync_c = sync_c - self._reward_running_mean

        # Optional clamping
        clamp = getattr(self.config, "clamp_reward", None)
        if clamp is not None:
            sync_c = sync_c.clamp(clamp[0], clamp[1])

        weight = torch.exp(beta * sync_c)  # [B]
        mean_weight = weight.mean()

        weighted = mean_weight * vsd_loss

        log_map = {
            "reward_sync_c_mean": float(sync_c.mean().item()),
            "reward_sync_c_min": float(sync_c.min().item()),
            "reward_sync_c_max": float(sync_c.max().item()),
            "reward_weight_mean": float(mean_weight.item()),
            "reward_weight_min": float(weight.min().item()),
            "reward_weight_max": float(weight.max().item()),
            "vsd_loss_unweighted": float(vsd_loss.detach().item()),
            "vsd_loss_weighted": float(weighted.detach().item()),
        }
        return weighted, log_map
```

- [ ] **Step 4: Run test to verify pass**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_re_dmd_trainer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add fastgen/methods/omniavatar_self_forcing_re_dmd.py tests/reward/test_re_dmd_trainer.py
git commit -m "feat: OmniAvatarSelfForcingReDMD with _apply_reward_weighting"
```

---

## Task 7: Override `_student_update_step` with VAE-decode + reward

**Files:**
- Modify: `fastgen/methods/omniavatar_self_forcing_re_dmd.py`
- Modify: `tests/reward/test_re_dmd_trainer.py`

- [ ] **Step 1: Add test for the full override**

Append to `tests/reward/test_re_dmd_trainer.py`:

```python
def test_student_update_step_integrates_reward():
    """_student_update_step should (a) compute vsd_loss, (b) VAE-decode gen_data,
    (c) call reward_scorer with decoded pixels + audio_waveform from condition,
    (d) multiply vsd_loss by exp(beta * sync_c), (e) put reward/weight stats in
    loss_map.
    """
    model = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    model.config = type("C", (), {})()
    model.config.reward_beta = 0.25
    model.config.center_reward = False
    model.config.clamp_reward = None
    model.config.gan_loss_weight_gen = 0.0
    model.config.guidance_scale = None

    model.reward_scorer = _FakeScorer(sync_c=3.0)

    # Fake gen_data_from_net → returns a simple tensor with grad
    gen_latent = torch.randn(1, 16, 21, 8, 8, requires_grad=True)
    model.gen_data_from_net = lambda *a, **kw: gen_latent

    # Fake net + noise_scheduler + fake_score + teacher
    class _FakeSched:
        def forward_process(self, x, eps, t): return x + 0.1 * eps
    class _FakeNet:
        noise_scheduler = _FakeSched()
        vae = type("V", (), {"decode": lambda self, x: [torch.zeros(3, 81, 64, 64) for _ in x]})()
        def clear_caches(self): pass
    model.net = _FakeNet()

    model.fake_score = lambda x, t, condition, fwd_pred_type: torch.zeros_like(gen_latent)
    model._compute_teacher_prediction_gan_loss = lambda p, t, condition: (
        torch.zeros_like(gen_latent), torch.tensor(0.0),
    )

    data = {"audio_waveform": torch.randn(1, 51840)}
    condition = {}
    neg_condition = {}
    input_s = torch.randn_like(gen_latent)
    t_student = torch.tensor([0.5])
    t = torch.tensor([0.5])
    eps = torch.randn_like(gen_latent)

    loss_map, outputs = model._student_update_step(
        input_s, t_student, t, eps, data,
        condition=condition, neg_condition=neg_condition,
    )

    assert "reward_sync_c_mean" in loss_map
    assert "reward_weight_mean" in loss_map
    assert "vsd_loss_unweighted" in loss_map
    assert abs(float(loss_map["reward_sync_c_mean"]) - 3.0) < 1e-6
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_re_dmd_trainer.py::test_student_update_step_integrates_reward -v`
Expected: FAIL — `_student_update_step` is inherited from base, doesn't touch reward.

- [ ] **Step 3: Override `_student_update_step` in the subclass**

Append to `fastgen/methods/omniavatar_self_forcing_re_dmd.py`:

```python
    def _student_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        data: Dict[str, Any],
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ):
        """Re-DMD override: standard VSD loss, then scaled by exp(beta * sync_c)."""
        gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)
        perturbed = self.net.noise_scheduler.forward_process(gen_data, eps, t)

        with torch.no_grad():
            fake_score_x0 = self.fake_score(perturbed, t, condition=condition, fwd_pred_type="x0")

        teacher_x0, gan_loss_gen = self._compute_teacher_prediction_gan_loss(
            perturbed, t, condition=condition,
        )
        if self.config.guidance_scale is not None:
            teacher_x0 = self._apply_classifier_free_guidance(
                perturbed, t, teacher_x0, neg_condition=neg_condition,
            )

        vsd_loss = variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0)

        # --- Re-DMD reward weighting ---
        if self.reward_scorer is not None and "audio_waveform" in data:
            with torch.no_grad():
                # VAE decode (under no_grad — reward is detached)
                pixels = self._decode_gen_to_pixels(gen_data)  # [B, 3, F, H, W] in [-1, 1]
                videos_u8 = self._pixels_to_uint8_face_crop(pixels)  # list of [F, 3, H, W]
                audios = list(data["audio_waveform"].unbind(0))  # list of [L]

            weighted_vsd, reward_log = self._apply_reward_weighting(vsd_loss, videos_u8, audios)
        else:
            weighted_vsd = vsd_loss
            reward_log = {"vsd_loss_unweighted": float(vsd_loss.detach().item())}

        loss = weighted_vsd + self.config.gan_loss_weight_gen * gan_loss_gen

        loss_map = {
            "total_loss": loss,
            "vsd_loss": vsd_loss.detach(),
            "vsd_loss_weighted": weighted_vsd.detach(),
            "gan_loss_gen": gan_loss_gen.detach() if torch.is_tensor(gan_loss_gen) else torch.tensor(float(gan_loss_gen)),
            **reward_log,
        }
        outputs = self._get_outputs(gen_data, input_student, condition=condition)
        return loss_map, outputs

    def _decode_gen_to_pixels(self, gen_latent: torch.Tensor) -> torch.Tensor:
        """Decode [B, 16, T_lat, H_lat, W_lat] latents to [B, 3, T_pix, H_pix, W_pix]
        pixel tensor in [-1, 1]. Uses the VAE already loaded on self.net.vae
        (the visual-logging wrapper — safe to reuse, it's wrapped in no_grad).
        """
        if not hasattr(self.net, "vae") or self.net.vae is None:
            raise RuntimeError(
                "Re-DMD needs VAE for reward decode. Set config.vae_path in the "
                "rewarded config so _load_vae instantiates the VAEWrapper."
            )
        # VAEWrapper.decode expects a list of [C, T_lat, H_lat, W_lat] tensors
        # (one per batch item). Returns list of decoded pixels.
        decoded_list = self.net.vae.decode([gen_latent[b].float() for b in range(gen_latent.shape[0])])
        # decoded_list: list of [C, T_pix, H_pix, W_pix]
        return torch.stack(decoded_list, dim=0)

    def _pixels_to_uint8_face_crop(self, pixels: torch.Tensor):
        """[B, 3, T_pix, H_pix, W_pix] float in [-1, 1] -> list of B tensors each
        [T_pix, 3, H, W] uint8, ready for SyncCScorer.
        """
        pixels = pixels.clamp(-1.0, 1.0)
        u8 = ((pixels + 1.0) * 127.5).to(torch.uint8)  # [B, 3, T, H, W]
        # Transpose to [B, T, 3, H, W]
        u8 = u8.permute(0, 2, 1, 3, 4).contiguous()
        return [u8[b] for b in range(u8.shape[0])]
```

- [ ] **Step 4: Run test to verify pass**

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_re_dmd_trainer.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add fastgen/methods/omniavatar_self_forcing_re_dmd.py tests/reward/test_re_dmd_trainer.py
git commit -m "feat: _student_update_step with VAE decode + sync-C reward weighting"
```

---

## Task 8: Thread `audio_waveform` through `_prepare_training_data`

**Files:**
- Modify: `fastgen/methods/omniavatar_self_forcing_re_dmd.py`

The base class's `_prepare_training_data` (`omniavatar_self_forcing.py:210-260`) builds the condition dict but doesn't know about `audio_waveform`. We can't reach `data["audio_waveform"]` inside `_student_update_step` unless `single_train_step` keeps it on `data`. Good news: it does — `single_train_step` passes `data` through unchanged to `_student_update_step` (see `omniavatar_self_forcing.py:85-88`). So there's nothing to thread. **Verify.**

- [ ] **Step 1: Confirm `data` reaches `_student_update_step` intact**

Read `fastgen/methods/omniavatar_self_forcing.py:83-88`. Confirm that `data` (the full batch dict) is passed positionally to `_student_update_step`. If it is, skip steps 2-3 below.

- [ ] **Step 2: If not (edge case), add a `_prepare_training_data` override**

Only needed if the verification above fails. Otherwise move to Task 9.

```python
    def _prepare_training_data(self, data):
        real_data, condition, neg_condition = super()._prepare_training_data(data)
        # Attach raw waveform so _student_update_step can reach it
        if "audio_waveform" in data:
            condition["_audio_waveform"] = data["audio_waveform"]
        return real_data, condition, neg_condition
```

Then in `_student_update_step`, read from `condition["_audio_waveform"]` instead of `data["audio_waveform"]`.

- [ ] **Step 3: Commit if changed**

```bash
cd /home/work/.local/hyunbin/FastGen
git add fastgen/methods/omniavatar_self_forcing_re_dmd.py
git commit -m "fix: thread audio_waveform through _prepare_training_data"
```

---

## Task 9: Config file for the rewarded run

**Files:**
- Create: `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd.py`

- [ ] **Step 1: Read the base config**

Read `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_tscfg.py` end-to-end. Note what it imports from `config_sf.py`, what it overrides (`local_attn_size=7, sink_size=1, use_dynamic_rope=True`), and how it composes.

- [ ] **Step 2: Write the rewarded config**

Create `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd.py`:

```python
"""SF + Re-DMD (sync-C reward) on top of config_sf_sink1_window7_tscfg.

Key changes vs the baseline SF config:
  - model._target_ switches to OmniAvatarSelfForcingReDMD
  - reward.* section added (checkpoint path, beta, vshift, etc.)
  - dataloader_train passes load_raw_audio=True to the dataset
  - vae_path required so the model's VAE is available for reward decode
"""
from fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_tscfg import config as base_config
from copy import deepcopy

config = deepcopy(base_config)

# --- Switch the trainer class to the Re-DMD variant ---
config.model._target_ = "fastgen.methods.omniavatar_self_forcing_re_dmd.OmniAvatarSelfForcingReDMD"

# --- Reward config block ---
class _RewardCfg:
    enabled = True
    checkpoint_path = "/home/work/.local/eval_metrics/eval/checkpoints/auxiliary/syncnet_v2.model"
    input_fps = 25.0
    audio_sample_rate = 16000
    vshift = 15

config.model.reward = _RewardCfg()
config.model.reward_beta = 0.25       # suggested starting point for sync-C range
config.model.center_reward = False
config.model.clamp_reward = None       # e.g. (0.0, 15.0) to bound exp

# --- Require VAE for reward decode ---
# Point to the OmniAvatar VAE checkpoint used for logging; reused for reward decode.
assert getattr(config.model, "vae_path", "") != "", (
    "config.model.vae_path must be set for Re-DMD reward decode"
)

# --- Data: tell the dataset to load raw waveforms ---
config.dataloader_train.dataset.load_raw_audio = True
config.dataloader_train.dataset.raw_audio_sample_rate = 16000
config.dataloader_train.dataset.raw_audio_num_frames = 81
config.dataloader_train.dataset.raw_audio_fps = 25.0

# --- Logging knob (helpful to find this run later in wandb) ---
config.wandb.name = "sf_sink1_window7_redmd_syncc_beta0p25"
```

- [ ] **Step 3: Smoke-test the config parses**

Run:
```bash
cd /home/work/.local/hyunbin/FastGen
python -c "from fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd import config; print(config.model._target_); print(config.model.reward_beta)"
```
Expected stdout contains `OmniAvatarSelfForcingReDMD` and `0.25`.

- [ ] **Step 4: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd.py
git commit -m "config: SF + Re-DMD sync-C reward variant of sink1_window7"
```

---

## Task 10: Launch script

**Files:**
- Create: `scripts/train_sf_sink1_window7_redmd.sh`

- [ ] **Step 1: Copy the baseline script**

```bash
cp /home/work/.local/hyunbin/FastGen/scripts/train_sf_sink1_window7.sh \
   /home/work/.local/hyunbin/FastGen/scripts/train_sf_sink1_window7_redmd.sh
```

- [ ] **Step 2: Edit the config path**

In the new script, change the `--config-path` / `--config-name` arguments (or the direct python module path, matching whatever the baseline uses) so it points to `config_sf_sink1_window7_redmd` instead of `config_sf_sink1_window7_tscfg`. Also change the wandb run name if it's set in the script.

Read the baseline script first, then make the substitution. Do not paraphrase — preserve the exact torchrun / env / path pattern.

- [ ] **Step 3: Dry-run the script (no actual training)**

Run:
```bash
cd /home/work/.local/hyunbin/FastGen
bash -n scripts/train_sf_sink1_window7_redmd.sh
```
Expected: no syntax errors (return code 0).

- [ ] **Step 4: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add scripts/train_sf_sink1_window7_redmd.sh
git commit -m "script: launch SF + Re-DMD sync-C reward training"
```

---

## Task 11: Wandb logging of reward/weight stats

**Files:**
- Modify: `fastgen/methods/omniavatar_self_forcing_re_dmd.py` (verify logging path)
- Check: `fastgen/callbacks/wandb.py:580-592`

The wandb callback accumulates whatever is in `loss_map`. Our `_student_update_step` already puts reward/weight keys there. This task is mostly verification.

- [ ] **Step 1: Read the wandb callback**

Read `fastgen/callbacks/wandb.py:260-600` (roughly). Confirm that all keys in `loss_map` passed to `on_training_step_end` are logged to wandb under `train/*`.

- [ ] **Step 2: Add a cross-rank reduce for reward stats**

In `_apply_reward_weighting`, before building `log_map`, insert per-rank reduction so multi-GPU logs aren't rank-0-only:

```python
        def _reduce(x: torch.Tensor, op) -> torch.Tensor:
            y = x.detach().clone().float()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(y, op=op)
                if op == dist.ReduceOp.SUM:
                    y = y / dist.get_world_size()
            return y

        sync_c_mean_r = _reduce(sync_c.mean(), dist.ReduceOp.SUM if dist.is_initialized() else None)
        sync_c_min_r  = _reduce(sync_c.min(),  dist.ReduceOp.MIN if dist.is_initialized() else None)
        sync_c_max_r  = _reduce(sync_c.max(),  dist.ReduceOp.MAX if dist.is_initialized() else None)
        weight_mean_r = _reduce(weight.mean(), dist.ReduceOp.SUM if dist.is_initialized() else None)
        weight_min_r  = _reduce(weight.min(),  dist.ReduceOp.MIN if dist.is_initialized() else None)
        weight_max_r  = _reduce(weight.max(),  dist.ReduceOp.MAX if dist.is_initialized() else None)
```

And replace the `log_map` builder:

```python
        log_map = {
            "reward_sync_c_mean": float(sync_c_mean_r.item()),
            "reward_sync_c_min": float(sync_c_min_r.item()),
            "reward_sync_c_max": float(sync_c_max_r.item()),
            "reward_weight_mean": float(weight_mean_r.item()),
            "reward_weight_min": float(weight_min_r.item()),
            "reward_weight_max": float(weight_max_r.item()),
            "vsd_loss_unweighted": float(vsd_loss.detach().item()),
            "vsd_loss_weighted": float(weighted.detach().item()),
        }
```

- [ ] **Step 3: Update unit test to tolerate single-rank no-reduce path**

The existing test at `tests/reward/test_re_dmd_trainer.py::test_weighted_loss_equals_exp_beta_r_times_unweighted` already runs without distributed init, so it hits the no-op branch. Verify it still passes.

Run: `cd /home/work/.local/hyunbin/FastGen && python -m pytest tests/reward/test_re_dmd_trainer.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add fastgen/methods/omniavatar_self_forcing_re_dmd.py
git commit -m "feat: cross-rank reduce for Re-DMD reward/weight logging"
```

---

## Task 12: End-to-end 3-step smoke test

**Files:**
- Create: `scripts/smoke_test_redmd.sh`

This is a *real* training smoke test on 1 GPU with `max_iter=3`. It exercises the full stack: dataset emits audio_waveform, trainer calls VAE decode, reward scorer returns a finite value, loss is finite, and wandb logs the reward keys.

- [ ] **Step 1: Write the smoke-test script**

Create `scripts/smoke_test_redmd.sh`:

```bash
#!/usr/bin/env bash
# 3-step smoke test of the Re-DMD sync-C reward pipeline.
# Uses 1 GPU and overrides max_iter to 3 via hydra-style CLI args.
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline  # no network logging needed
export OMNIAVATAR_ROOT=/home/work/.local/hyunbin/OmniAvatar  # or wherever it lives

torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py \
    --config-module fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd \
    trainer.max_iter=3 \
    trainer.grad_accum_rounds=1 \
    dataloader_train.batch_size=1 \
    2>&1 | tee logs/smoke_test_redmd.log

# Verify reward keys were logged at least once
if ! grep -q "reward_sync_c_mean" logs/smoke_test_redmd.log; then
    echo "FAIL: reward_sync_c_mean never logged" >&2
    exit 1
fi
if grep -qiE "Traceback|Error" logs/smoke_test_redmd.log; then
    echo "FAIL: error in log" >&2
    exit 1
fi
echo "SMOKE TEST PASSED"
```

**Note:** The exact CLI override syntax depends on whether `train.py` uses hydra / our own config loader / argparse. Before running, read `train.py` and adapt the override style. If overrides aren't supported, add them to the rewarded config file instead (hardcode `max_iter=3` etc. temporarily, then revert).

- [ ] **Step 2: Make executable**

```bash
chmod +x /home/work/.local/hyunbin/FastGen/scripts/smoke_test_redmd.sh
```

- [ ] **Step 3: Run it**

```bash
mkdir -p /home/work/.local/hyunbin/FastGen/logs
bash /home/work/.local/hyunbin/FastGen/scripts/smoke_test_redmd.sh
```
Expected: script exits 0 with `SMOKE TEST PASSED`. Log should contain:
- `SyncCScorer loaded` (from the subclass `build_model`)
- `reward_sync_c_mean=<finite float>` at iteration 5 (or whatever the first student-step iteration is given `student_update_freq=5`)
- no `Traceback` / `Error`

If the first student step is at iter 5 but max_iter=3, bump max_iter to 6 in the override so at least one generator update actually fires.

- [ ] **Step 4: Inspect log for correctness**

Check the log for:
1. `reward_sync_c_mean` in approximately the range [0, 10]. Wildly out-of-range values (e.g. negative with big magnitude, NaN) indicate an issue.
2. `reward_weight_mean` should be `exp(0.25 * sync_c_mean)`. Spot-check numerically.
3. `vsd_loss_weighted` should equal `reward_weight_mean * vsd_loss_unweighted` within fp precision.

- [ ] **Step 5: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add scripts/smoke_test_redmd.sh
git commit -m "test: 3-step end-to-end smoke for Re-DMD sync-C reward"
```

---

## Task 13: Document the rewarded run

**Files:**
- Create: `docs/redmd_sync_c.md`

- [ ] **Step 1: Write the user-facing doc**

Create `docs/redmd_sync_c.md`:

```markdown
# Re-DMD with SyncNet-v2 Sync-C Reward

## Summary

Adds a rewarded-distillation variant of the OmniAvatar Self-Forcing DMD training:

    loss_gen = exp(β · sync_c_detached) · vsd_loss_unweighted
             + gan_loss_weight_gen · gan_loss_gen

where `sync_c` is SyncNet-v2's offset-margin confidence (higher = better lip sync),
computed under no_grad on the VAE-decoded generator output.

## When to use

- Baseline SF training has converged to reasonable visual quality; you now want
  to push it further on lip-sync specifically.
- You want to evaluate whether the Re-DMD reward-weighting mechanism transfers
  to a sync reward.

## Launch

```bash
bash scripts/train_sf_sink1_window7_redmd.sh
```

## Key config knobs

| Knob | Default | Notes |
|---|---|---|
| `model.reward.enabled` | `True` | Set False to disable (falls back to vanilla SF) |
| `model.reward_beta` | `0.25` | Paper-equivalent β=2 in this reward scale is catastrophic. Start at 0.25. |
| `model.reward.checkpoint_path` | eval_metrics path | Point at SyncNet-v2 `.model` file |
| `model.center_reward` | `False` | If True, subtract EMA(sync_c) before exp — keeps mean weight ≈ 1 |
| `model.clamp_reward` | `None` | e.g. `(0.0, 15.0)` to bound `exp(β·r)` |

## What to expect

Per the Reward-Forcing investigation (`/home/work/.local/hyunbin/Reward-Forcing/docs/reward_forcing_implementation.md §8`),
for a well-synced generator `sync_c` should sit around 3-8. At β=0.25 that
gives `exp(β·r)` in roughly [2, 7.5] — comparable to the "soft β" regime
in the reference investigation. If you see weight stats > 100× frequently,
reduce β or enable `center_reward`.

## Files

- `fastgen/methods/reward/sync_c_scorer.py` — the scorer
- `fastgen/methods/reward/syncnet_v2.py` — vendored SyncNet-v2 architecture
- `fastgen/methods/omniavatar_self_forcing_re_dmd.py` — trainer override
- `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd.py` — config

## See also

- `docs/superpowers/plans/2026-04-12-sync-c-reward-redmd.md` — the implementation plan this was built from
- `/home/work/.local/hyunbin/Reward-Forcing/docs/sync_c_scorer_design.md` — the design reference
- `/home/work/.local/hyunbin/Reward-Forcing/docs/reward_forcing_implementation.md` — original Re-DMD writeup
```

- [ ] **Step 2: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add docs/redmd_sync_c.md
git commit -m "docs: user-facing guide for Re-DMD sync-C reward run"
```

---

## Open questions to resolve during execution

1. **Checkpoint location.** Default path `/home/work/.local/eval_metrics/eval/checkpoints/auxiliary/syncnet_v2.model` is a guess based on LatentSync convention. Verify the file exists before Task 1; if it's elsewhere, update `config.model.reward.checkpoint_path` and the tests' `CKPT` constant.

2. **VAE decode shape.** Task 7's `_decode_gen_to_pixels` assumes the `VAEWrapper.decode` signature from `omniavatar_self_forcing.py:161-163` (list-of-tensors in, list-of-tensors out). If the decoded output has a different layout (e.g. already batched), update `_decode_gen_to_pixels` to match. Dry-run with `python -c` before the smoke test to confirm.

3. **Train.py config override syntax.** Task 12's smoke test uses hydra-style CLI overrides (`trainer.max_iter=3`). If `train.py` uses a different config loader, adjust the smoke test to temporarily modify the config file directly.

4. **Face alignment at decode time.** The design assumes VAE-decoded pixels are already face-aligned because training data is. Verify on the first smoke-test sample by saving one decoded frame to PNG (`torchvision.utils.save_image`) and eyeballing it. If the face is off-center, we'll need to add an InsightFace-based crop to `_pixels_to_uint8_face_crop` (not expected).

5. **Per-GPU batch memory.** VAE decode of `[8, 16, 21, 64, 64]` → `[8, 3, 81, 512, 512]` bf16 = 504 MB. Plus the reward scorer's lip window tensor: `[77, 3, 5, 224, 224]` float32 = 107 MB. Total ~600 MB per rank on top of the existing footprint. If OOM hits on the smoke test, move the VAE decode to fp16 or reduce batch size for reward scoring (score a subset of the batch).

6. **Autocast interaction.** `single_train_step` runs under autocast (per Agent 1 report). The reward block in `_student_update_step` does its own casting via `self.dtype=torch.float32`, but `_decode_gen_to_pixels` may get autocast'd inputs. Wrap the reward section in `with torch.autocast(device_type='cuda', enabled=False):` if the smoke test shows dtype warnings.

---

## Self-review notes (done during plan drafting)

- **Spec coverage.** All design-doc requirements covered: scorer class (Tasks 1-4), audio plumbing (Task 5), reward-weighted loss (Tasks 6-7), dataset integration (Task 5), config (Task 9), launch script (Task 10), logging (Task 11), smoke test (Task 12), docs (Task 13).
- **Placeholders.** No TBDs. Every code block is complete. Every command has expected output.
- **Type consistency.** `SyncCScorer.reward_from_frames` returns `{"sync_c": Tensor[B], "MQ": Tensor[B]}` — consistent from Task 4 to Task 6 to Task 7. `_apply_reward_weighting` signature consistent between Tasks 6 and 11. The config knob names (`reward_beta`, `center_reward`, `clamp_reward`) are consistent across Tasks 6, 9, and 13.
- **Known simplification.** We intentionally do NOT use `common_loss.py`'s `additional_scale` path; the outer-multiply approach gives us both the correct paper formulation and clean separation of `vsd_loss_unweighted` for logging.
