# TAEW Reward Decoder Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Swap the heavyweight Wan 2.1 VAE decode in the Re-DMD reward path for the ~10M-param TAEW tiny autoencoder, opt-in via config, to cut per-step overhead while preserving the 81-frame pixel output contract.

**Architecture:** TAEW (`taew2_1.pth`) already produces `(T_lat − 1) · 4 + 1` pixel frames via internal startup-frame trim, matching Wan 2.1 VAE's `1 + (T_lat − 1) · 4` mapping exactly. Integration is three pieces: (1) a vendored `TAEHV` nn.Module, (2) a `TAEHVDecoderWrapper` that mimics `WanVideoVAE.decode()`'s signature and `[-1, 1]`/NCTHW contract, (3) a `decoder_kind` flag on `RewardConfig` that the Re-DMD model checks in `build_model` and `_decode_gen_to_pixels`. Default remains Wan VAE — TAEW activates only when explicitly configured.

**Tech Stack:**
- PyTorch 2.x with FSDP2 (training)
- TAEHV model from [madebyollin/taehv](https://github.com/madebyollin/taehv), `taew2_1.pth` checkpoint
- pytest for unit tests (conda env `hb_fastgen`)
- Existing Re-DMD plumbing under `fastgen/methods/omniavatar_self_forcing_re_dmd.py` and `fastgen/methods/reward/`

---

## File Structure

**New files:**
- `fastgen/methods/reward/taehv.py` — vendored TAEHV model (copy of upstream taehv.py)
- `fastgen/methods/reward/taehv_decoder.py` — `TAEHVDecoderWrapper` class: NCTHW↔NTCHW permute + `[0,1]`→`[-1,1]` rescale + WanVideoVAE.decode-compatible signature
- `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_taew.py` — opt-in config preset
- `tests/reward/test_taehv_decoder.py` — unit tests for shape/dtype/range/equivalence
- `scripts/train_sf_sink1_window7_redmd_taew.sh` — 4-GPU launcher
- `scripts/smoke_test_redmd_taew.sh` — 10-iter smoke launcher

**Modified files:**
- `fastgen/configs/methods/config_omniavatar_sf.py` — extend `RewardConfig` with `decoder_kind` and `taew_checkpoint_path`
- `fastgen/methods/omniavatar_self_forcing_re_dmd.py` — conditional load in `build_model`, conditional branch in `_decode_gen_to_pixels`

**External artifact:**
- `/home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth` — downloaded TAEW weights (~10-50 MB)

**NOT touched:**
- The user's uncommitted `scripts/inference/taehv.py` and `scripts/inference/inference_causal_taehv.py` (their inference-side WIP).
- The running DF training job (the user confirmed the currently-running run is DF, not SF — so even VAE-related edits are safe).

---

## Pre-Flight Check

- [ ] **P1: Confirm clean tracked tree**

Run:
```bash
cd /home/work/.local/hyunbin/FastGen-redmd && git status --porcelain | grep -vE '^\?\?' | head
```
Expected: empty output (all tracked files committed).

- [ ] **P2: Confirm syncnet smoke artifacts still intact** (we'll reuse the same smoke config pattern)

Run:
```bash
ls /home/work/.local/eval_metrics/checkpoints/auxiliary/syncnet_v2.model
ls fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_smoke.py
```
Expected: both files present.

---

## Task 1: Download TAEW weights + verify they load

**Files:**
- Download to: `/home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth`

- [ ] **Step 1: Download checkpoint**

```bash
cd /home/work/.local/eval_metrics/checkpoints/auxiliary/ && \
  curl -fL -o taew2_1.pth \
    https://github.com/madebyollin/taehv/raw/main/taew2_1.pth && \
  ls -la taew2_1.pth
```
Expected: file present, size ~5-50MB (the Wan 2.1 tiny VAE is small).

- [ ] **Step 2: Verify checkpoint loads via upstream class (using the already-vendored copy)**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  /home/work/.local/miniconda3/envs/hb_fastgen/bin/python -c "
import sys; sys.path.insert(0, 'scripts/inference')
from taehv import TAEHV
m = TAEHV(checkpoint_path='/home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth')
print('latent_channels:', m.latent_channels)
print('patch_size:', m.patch_size)
print('t_upscale:', m.t_upscale)
print('frames_to_trim:', m.frames_to_trim)
print('param_count:', sum(p.numel() for p in m.parameters()))
"
```
Expected output:
```
latent_channels: 16
patch_size: 1
t_upscale: 4
frames_to_trim: 3
param_count: (some number in the 5-15M range)
```

- [ ] **Step 3: Verify decode frame count on a dummy 21-latent**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  /home/work/.local/miniconda3/envs/hb_fastgen/bin/python -c "
import sys, torch
sys.path.insert(0, 'scripts/inference')
from taehv import TAEHV
m = TAEHV(checkpoint_path='/home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth').to('cuda', torch.float16).eval()
lat = torch.randn(1, 21, 16, 64, 64, device='cuda', dtype=torch.float16)  # NTCHW
with torch.no_grad():
    vid = m.decode_video(lat, parallel=True, show_progress_bar=False)
print('output shape:', tuple(vid.shape))
print('output min/max:', vid.min().item(), vid.max().item())
"
```
Expected: shape `(1, 81, 3, 512, 512)` — confirms `(T_lat - 1) * 4 + 1 = 81` and `8× spatial upscale (64 → 512)`.

- [ ] **Step 4: Commit nothing yet** — artifact-only task, no code changes.

---

## Task 2: Vendor `taehv.py` into `fastgen/methods/reward/`

**Files:**
- Create: `fastgen/methods/reward/taehv.py` (copy from `scripts/inference/taehv.py`)
- Test: `tests/reward/test_taehv_vendored.py`

Motivation: the reward path is a training-time concern, so the model file belongs alongside `sync_c_scorer.py` and `syncnet_v2.py` under `fastgen/methods/reward/`. Keeps import path stable (`from fastgen.methods.reward.taehv import TAEHV`) regardless of future refactors in `scripts/`.

- [ ] **Step 1: Write a failing import test**

File: `tests/reward/test_taehv_vendored.py`

```python
"""Smoke test that the vendored TAEHV is importable and constructible."""
import os
import pytest
import torch

TAEW_CKPT = "/home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth"


def test_taehv_imports():
    from fastgen.methods.reward.taehv import TAEHV
    assert TAEHV is not None


@pytest.mark.skipif(not os.path.exists(TAEW_CKPT), reason="TAEW checkpoint missing")
def test_taehv_loads_and_reports_config():
    from fastgen.methods.reward.taehv import TAEHV
    m = TAEHV(checkpoint_path=TAEW_CKPT)
    assert m.latent_channels == 16
    assert m.patch_size == 1
    assert m.t_upscale == 4
    assert m.frames_to_trim == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  /home/work/.local/miniconda3/envs/hb_fastgen/bin/python -m pytest tests/reward/test_taehv_vendored.py -v
```
Expected: both tests FAIL with `ModuleNotFoundError: fastgen.methods.reward.taehv`.

- [ ] **Step 3: Copy the vendored file**

```bash
cp /home/work/.local/hyunbin/FastGen-redmd/scripts/inference/taehv.py \
   /home/work/.local/hyunbin/FastGen-redmd/fastgen/methods/reward/taehv.py
```

Prepend SPDX license header (taehv upstream is MIT; FastGen uses Apache-2.0 SPDX identifiers elsewhere). Edit the top of the new file to:

```python
# SPDX-FileCopyrightText: Copyright (c) 2024 madebyollin (Ollin Boer Bohan)
# SPDX-License-Identifier: MIT
#
# Vendored from https://github.com/madebyollin/taehv (MIT-licensed) for use
# as an opt-in TAEW decoder in the Re-DMD reward path.
#!/usr/bin/env python3
"""
Tiny AutoEncoder for Hunyuan Video
(DNN for encoding / decoding videos to Hunyuan Video's latent space)
"""
```

(Keep the rest of the file byte-for-byte identical to the upstream copy.)

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  /home/work/.local/miniconda3/envs/hb_fastgen/bin/python -m pytest tests/reward/test_taehv_vendored.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  git add fastgen/methods/reward/taehv.py tests/reward/test_taehv_vendored.py && \
  git commit -m "feat(redmd): vendor TAEHV tiny autoencoder for reward decoder

Copies madebyollin/taehv's taehv.py under fastgen/methods/reward/ so the
TAEW decoder can be imported from training code without reaching into
scripts/inference/. Source is MIT-licensed; SPDX header added.

Smoke test confirms the taew2_1.pth checkpoint loads with the expected
config (latent_channels=16, patch_size=1, t_upscale=4, frames_to_trim=3).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Write `TAEHVDecoderWrapper` (WanVideoVAE.decode-compatible shim)

**Files:**
- Create: `fastgen/methods/reward/taehv_decoder.py`
- Test: `tests/reward/test_taehv_decoder.py`

Design: match `WanVideoVAE.decode()`'s signature so `_decode_gen_to_pixels` can switch between them with a single attribute check. Input: list of `[C=16, T_lat, H, W]` tensors. Output: stacked `[N, 3, T_pix, H_pix, W_pix]` tensor in `[-1, 1]`, NCTHW.

**IMPORTANT: do NOT copy the double-trim pattern from `scripts/inference/inference_causal_taehv.py`.** That wrapper assumes an older vendored copy where trim was disabled. Our vendored `taehv.py` already applies `x[:, frames_to_trim:]` inside `decode_video` (line 275), so re-trimming in the wrapper would drop 3 extra frames off the front. Test #3 below explicitly catches this.

- [ ] **Step 1: Write failing tests**

File: `tests/reward/test_taehv_decoder.py`

```python
"""Unit tests for TAEHVDecoderWrapper — the WanVideoVAE.decode-compatible shim."""
import os
import pytest
import torch

TAEW_CKPT = "/home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(TAEW_CKPT),
    reason="CUDA and TAEW checkpoint required",
)


@pytest.fixture
def wrapper():
    from fastgen.methods.reward.taehv_decoder import TAEHVDecoderWrapper
    return TAEHVDecoderWrapper(checkpoint_path=TAEW_CKPT, device="cuda")


def test_decode_matches_wan_contract_shape(wrapper):
    # Wan 2.1 latent for an 81-frame clip at 512x512:
    #   C=16, T_lat=21, H=64, W=64
    # Expected pixel shape matching WanVideoVAE: [N=1, 3, 81, 512, 512]
    lat = torch.randn(16, 21, 64, 64, device="cuda", dtype=torch.float32)
    out = wrapper.decode([lat])
    assert out.shape == (1, 3, 81, 512, 512), f"got {tuple(out.shape)}"
    assert out.dtype == torch.float32


def test_decode_output_range_matches_wan(wrapper):
    # Wan VAE decode output is in [-1, 1]; wrapper must rescale from TAEHV's [0, 1]
    lat = torch.randn(16, 21, 64, 64, device="cuda", dtype=torch.float32)
    out = wrapper.decode([lat])
    assert out.min() >= -1.01, f"range underflow: {out.min().item()}"
    assert out.max() <= 1.01, f"range overflow: {out.max().item()}"


def test_decode_frame_count_no_double_trim(wrapper):
    # Regression: an older wrapper in scripts/inference/inference_causal_taehv.py
    # applies its own `vid[:, frames_to_trim:]` AFTER decode_video — but our
    # vendored taehv.py already trims inside decode_video. Double-trim would
    # produce 78 frames for a 21-latent input instead of 81.
    lat = torch.randn(16, 21, 64, 64, device="cuda", dtype=torch.float32)
    out = wrapper.decode([lat])
    assert out.shape[2] == 81, (
        f"frame count mismatch: expected 81, got {out.shape[2]}. "
        f"If 78, the wrapper is double-trimming — remove the manual "
        f"frames_to_trim slice (decode_video already trims)."
    )


def test_decode_batch_of_two(wrapper):
    # list of 2 latents → stacked output of shape [2, 3, 81, H, W]
    latents = [torch.randn(16, 21, 64, 64, device="cuda", dtype=torch.float32) for _ in range(2)]
    out = wrapper.decode(latents)
    assert out.shape == (2, 3, 81, 512, 512), f"got {tuple(out.shape)}"


def test_decode_runs_under_no_grad_even_if_called_in_grad_context(wrapper):
    # Re-DMD calls VAE decode inside torch.no_grad() upstream. Wrapper must
    # not break if accidentally called with a leaf latent that has requires_grad.
    lat = torch.randn(16, 21, 64, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    out = wrapper.decode([lat])
    assert out.requires_grad is False, "wrapper should produce a detached tensor"
```

- [ ] **Step 2: Run tests to verify they all fail**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  /home/work/.local/miniconda3/envs/hb_fastgen/bin/python -m pytest tests/reward/test_taehv_decoder.py -v
```
Expected: 5 FAILED with `ModuleNotFoundError: fastgen.methods.reward.taehv_decoder`.

- [ ] **Step 3: Implement the wrapper**

File: `fastgen/methods/reward/taehv_decoder.py`

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAEHV decoder wrapper that mimics WanVideoVAE.decode() for the Re-DMD reward path.

WanVideoVAE.decode contract (what _decode_gen_to_pixels consumes):
  input:  list of [C=16, T_lat, H, W] float tensors
  output: [N, 3, T_pix, H_pix, W_pix] float32 tensor in [-1, 1], NCTHW

TAEHV's decode_video contract:
  input:  NTCHW tensor in the raw diffusion latent space (no mean/std scaling)
  output: NTCHW RGB tensor in [0, 1], already trimmed to (T_lat - 1) * t_upscale + 1 frames
         via the built-in frames_to_trim slice at the end of decode_video.

Transformation applied here:
  1. stack list → [N, 16, T_lat, H, W] and permute to NTCHW: [N, T_lat, 16, H, W]
  2. run TAEHV.decode_video(parallel=True, show_progress_bar=False)
  3. rescale [0, 1] → [-1, 1]  via  x.mul(2).sub(1)
  4. permute back to NCTHW: [N, 3, T_pix, H_pix, W_pix]
  5. .float() for downstream compatibility (the scorer expects float32)
"""

from __future__ import annotations
from typing import List, Optional

import torch

from fastgen.methods.reward.taehv import TAEHV


class TAEHVDecoderWrapper:
    """Drop-in WanVideoVAE.decode replacement backed by TAEHV.

    Runs in fp16 internally for speed; returns fp32 to match WanVideoVAE's
    float contract with the sync-C scorer (which re-casts to float anyway).
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self._taehv = TAEHV(checkpoint_path=checkpoint_path).to(device, torch.float16).eval()
        for p in self._taehv.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def decode(self, latents_list: List[torch.Tensor], device: Optional[str] = None) -> torch.Tensor:
        target_device = device if device is not None else self.device
        # Stack list of [C, T, H, W] into [N, C, T, H, W], then NCTHW -> NTCHW.
        batched = torch.stack([lat.to(target_device, dtype=torch.float16) for lat in latents_list], dim=0)
        batched = batched.permute(0, 2, 1, 3, 4).contiguous()  # [N, T_lat, C, H, W]
        vid = self._taehv.decode_video(batched, parallel=True, show_progress_bar=False)
        # vid: [N, T_pix, 3, H_pix, W_pix] in [0, 1] — already trimmed by decode_video.
        vid = vid.mul(2.0).sub(1.0)  # [0, 1] -> [-1, 1], match WanVideoVAE contract
        return vid.permute(0, 2, 1, 3, 4).float().contiguous()  # [N, 3, T_pix, H_pix, W_pix]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  /home/work/.local/miniconda3/envs/hb_fastgen/bin/python -m pytest tests/reward/test_taehv_decoder.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  git add fastgen/methods/reward/taehv_decoder.py tests/reward/test_taehv_decoder.py && \
  git commit -m "feat(redmd): TAEHVDecoderWrapper for WanVideoVAE.decode parity

Adds a thin wrapper around the vendored TAEHV that exposes the same
decode() signature as WanVideoVAE: list of [C=16, T_lat, H, W] in, stacked
[N, 3, T_pix, H_pix, W_pix] out in [-1, 1] NCTHW. Lets _decode_gen_to_pixels
branch between the two with a single attribute check.

Tests cover: shape contract (1, 3, 81, 512, 512) for 21-latent input,
value range [-1, 1], no double-trim regression (the inference-side wrapper
in scripts/ double-trims because it was written for a taehv.py variant
that disabled internal trim — this wrapper trusts decode_video's trim),
batched decode, and detached output under grad-context.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Extend `RewardConfig` with decoder-kind fields

**Files:**
- Modify: `fastgen/configs/methods/config_omniavatar_sf.py`
- Test: `tests/reward/test_reward_config_decoder_kind.py`

- [ ] **Step 1: Read the current RewardConfig layout**

```bash
sed -n '25,45p' /home/work/.local/hyunbin/FastGen-redmd/fastgen/configs/methods/config_omniavatar_sf.py
```
Expected: `@attrs.define(slots=False) class RewardConfig: ...` with existing fields (`enabled`, `checkpoint_path`, `input_fps`, `audio_sample_rate`, `vshift`).

- [ ] **Step 2: Write failing tests**

File: `tests/reward/test_reward_config_decoder_kind.py`

```python
"""RewardConfig must carry a decoder_kind field that defaults to the Wan VAE."""
import pytest


def test_reward_config_has_decoder_kind_field():
    from fastgen.configs.methods.config_omniavatar_sf import RewardConfig
    c = RewardConfig(enabled=True, checkpoint_path="/fake/syncnet.model")
    assert hasattr(c, "decoder_kind")
    assert c.decoder_kind == "vae", (
        f"default must preserve existing Wan VAE behavior, got {c.decoder_kind!r}"
    )
    assert hasattr(c, "taew_checkpoint_path")
    assert c.taew_checkpoint_path == ""


def test_reward_config_accepts_taew_kind():
    from fastgen.configs.methods.config_omniavatar_sf import RewardConfig
    c = RewardConfig(
        enabled=True,
        checkpoint_path="/fake/syncnet.model",
        decoder_kind="taew",
        taew_checkpoint_path="/fake/taew.pth",
    )
    assert c.decoder_kind == "taew"
    assert c.taew_checkpoint_path == "/fake/taew.pth"


def test_reward_config_rejects_unknown_kind():
    # The attrs class shouldn't hard-reject at construction time (we validate
    # at build_model time instead, to give a clearer error message), so this
    # test just confirms the field accepts arbitrary strings. The Re-DMD model
    # handles the unknown-kind case.
    from fastgen.configs.methods.config_omniavatar_sf import RewardConfig
    c = RewardConfig(enabled=True, checkpoint_path="/fake", decoder_kind="nonsense")
    assert c.decoder_kind == "nonsense"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  /home/work/.local/miniconda3/envs/hb_fastgen/bin/python -m pytest tests/reward/test_reward_config_decoder_kind.py -v
```
Expected: FAIL with `AttributeError: ... has no attribute 'decoder_kind'` or similar.

- [ ] **Step 4: Add fields to RewardConfig**

Edit `fastgen/configs/methods/config_omniavatar_sf.py`. Find the `@attrs.define(slots=False) class RewardConfig` block and add two fields. Exact edit — locate the field list:

```python
@attrs.define(slots=False)
class RewardConfig:
    enabled: bool = False
    checkpoint_path: str = ""
    input_fps: float = 25.0
    audio_sample_rate: int = 16000
    vshift: int = 15
```

Replace with:

```python
@attrs.define(slots=False)
class RewardConfig:
    enabled: bool = False
    checkpoint_path: str = ""
    input_fps: float = 25.0
    audio_sample_rate: int = 16000
    vshift: int = 15
    # Opt-in TAEW decoder. Default "vae" preserves WanVideoVAE.decode behavior.
    # When "taew", the Re-DMD model loads a TAEHVDecoderWrapper from
    # taew_checkpoint_path and uses it in place of self.net.vae for the
    # reward-path pixel decode.
    decoder_kind: str = "vae"
    taew_checkpoint_path: str = ""
```

Do NOT change the exact names or defaults of the existing five fields — they're live in the currently-committed `config_sf_sink1_window7_redmd.py`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  /home/work/.local/miniconda3/envs/hb_fastgen/bin/python -m pytest tests/reward/test_reward_config_decoder_kind.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Also re-run the existing Re-DMD-related tests to confirm no regressions**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  /home/work/.local/miniconda3/envs/hb_fastgen/bin/python -m pytest tests/reward/ -v
```
Expected: all prior tests still pass (12 from `test_sync_c_scorer.py` + 2 from `test_taehv_vendored.py` + 5 from `test_taehv_decoder.py` + 3 new = 22 passed). If any test fails, fix before continuing.

- [ ] **Step 7: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  git add fastgen/configs/methods/config_omniavatar_sf.py tests/reward/test_reward_config_decoder_kind.py && \
  git commit -m "feat(redmd): add decoder_kind + taew_checkpoint_path to RewardConfig

Extends the attrs dataclass with two opt-in fields for selecting the
reward-path pixel decoder. decoder_kind defaults to 'vae' so all existing
configs preserve their WanVideoVAE.decode behavior unchanged.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Wire conditional load + conditional branch into `OmniAvatarSelfForcingReDMD`

**Files:**
- Modify: `fastgen/methods/omniavatar_self_forcing_re_dmd.py`
- Test: `tests/reward/test_re_dmd_decoder_branch.py`

- [ ] **Step 1: Read current build_model + _decode_gen_to_pixels**

```bash
grep -n "def build_model\|def _decode_gen_to_pixels\|self\._taew\|reward_scorer" /home/work/.local/hyunbin/FastGen-redmd/fastgen/methods/omniavatar_self_forcing_re_dmd.py | head -20
```
Expected: locate `build_model` definition line and `_decode_gen_to_pixels` definition line. The notes you have from the `project_sync_c_port_plan.md` memory confirm that `build_model` sets `self.reward_scorer`, and `_decode_gen_to_pixels` decodes via `self.net.vae`.

- [ ] **Step 2: Write failing tests**

File: `tests/reward/test_re_dmd_decoder_branch.py`

```python
"""Re-DMD model selects decoder based on config.model.reward.decoder_kind."""
import os
import types
import pytest
import torch
from unittest.mock import MagicMock


TAEW_CKPT = "/home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth"


def _make_minimal_config(decoder_kind="vae", taew_ckpt=""):
    """Build a minimal config object with just the bits the decoder-branch reads."""
    from fastgen.configs.methods.config_omniavatar_sf import RewardConfig
    reward = RewardConfig(
        enabled=False,  # skip actually loading SyncNet for these unit tests
        checkpoint_path="",
        decoder_kind=decoder_kind,
        taew_checkpoint_path=taew_ckpt,
    )
    # The class uses attrs.define(slots=False) on parent OmniAvatarModelConfig;
    # we mimic the same attr graph with SimpleNamespace-compatible layering.
    cfg_model = types.SimpleNamespace(
        reward=reward,
        reward_beta=0.25,
        center_reward=False,
        clamp_reward=None,
        vae_path="/fake/vae.pth",
        save_reward_debug_video=False,
        reward_debug_dir="",
    )
    return types.SimpleNamespace(model=cfg_model)


def test_build_model_skips_taew_when_decoder_kind_is_vae():
    # This test stubs out the heavy superclass build_model by using __new__
    # and calling only the TAEW-related branch logic directly.
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    m = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    m.config = _make_minimal_config(decoder_kind="vae")
    # Call only the decoder-setup helper (extracted in Step 3 below).
    m._maybe_init_taew_decoder()
    assert getattr(m, "_taew_decoder", None) is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(TAEW_CKPT),
    reason="CUDA and TAEW checkpoint required",
)
def test_build_model_loads_taew_when_decoder_kind_is_taew():
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    from fastgen.methods.reward.taehv_decoder import TAEHVDecoderWrapper
    m = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    m.config = _make_minimal_config(decoder_kind="taew", taew_ckpt=TAEW_CKPT)
    m._maybe_init_taew_decoder()
    assert isinstance(m._taew_decoder, TAEHVDecoderWrapper)


def test_taew_kind_requires_checkpoint_path():
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    m = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    m.config = _make_minimal_config(decoder_kind="taew", taew_ckpt="")
    with pytest.raises(ValueError, match="taew_checkpoint_path"):
        m._maybe_init_taew_decoder()


def test_unknown_decoder_kind_raises():
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    m = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    m.config = _make_minimal_config(decoder_kind="bogus")
    with pytest.raises(ValueError, match="decoder_kind"):
        m._maybe_init_taew_decoder()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(TAEW_CKPT),
    reason="CUDA and TAEW checkpoint required",
)
def test_decode_gen_to_pixels_uses_taew_when_configured():
    """With _taew_decoder set, _decode_gen_to_pixels must go through it and
    NOT touch self.net.vae."""
    from fastgen.methods.omniavatar_self_forcing_re_dmd import OmniAvatarSelfForcingReDMD
    from fastgen.methods.reward.taehv_decoder import TAEHVDecoderWrapper
    m = OmniAvatarSelfForcingReDMD.__new__(OmniAvatarSelfForcingReDMD)
    m._taew_decoder = TAEHVDecoderWrapper(checkpoint_path=TAEW_CKPT, device="cuda")
    # Give it a net with a .vae attribute that would blow up if touched.
    class _ExplodingVAE:
        def decode(self, *a, **k):
            raise AssertionError("vae.decode should NOT be called when _taew_decoder is set")
    m.net = types.SimpleNamespace(vae=_ExplodingVAE())
    lat = torch.randn(1, 16, 21, 64, 64, device="cuda", dtype=torch.float32)
    out = m._decode_gen_to_pixels(lat)
    assert out.shape == (1, 3, 81, 512, 512)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  /home/work/.local/miniconda3/envs/hb_fastgen/bin/python -m pytest tests/reward/test_re_dmd_decoder_branch.py -v
```
Expected: FAIL with `AttributeError: ... has no attribute '_maybe_init_taew_decoder'` for the non-GPU tests and branch mismatch for the GPU test.

- [ ] **Step 4: Add the helper method and the conditional branch**

Edit `fastgen/methods/omniavatar_self_forcing_re_dmd.py`. Two pieces:

(a) Add a new method `_maybe_init_taew_decoder` that runs inside `build_model` after `self.reward_scorer` is set up. It must handle the three cases:

```python
    def _maybe_init_taew_decoder(self):
        """If config.model.reward.decoder_kind == 'taew', construct a
        TAEHVDecoderWrapper and store on self._taew_decoder. Otherwise leave
        self._taew_decoder as None so _decode_gen_to_pixels falls back to
        self.net.vae."""
        kind = getattr(self.config.model.reward, "decoder_kind", "vae")
        if kind == "vae":
            self._taew_decoder = None
            return
        if kind == "taew":
            ckpt = getattr(self.config.model.reward, "taew_checkpoint_path", "")
            if not ckpt:
                raise ValueError(
                    "config.model.reward.decoder_kind='taew' requires "
                    "config.model.reward.taew_checkpoint_path to be set."
                )
            from fastgen.methods.reward.taehv_decoder import TAEHVDecoderWrapper
            # Device: rank-local GPU. build_model runs before FSDP wrap so
            # cuda:0 on every rank is correct — the FSDP wrap won't touch this
            # (it's not in fsdp_dict). Training loop always runs on current
            # cuda device, so "cuda" resolves correctly.
            self._taew_decoder = TAEHVDecoderWrapper(checkpoint_path=ckpt, device="cuda")
            return
        raise ValueError(
            f"unknown config.model.reward.decoder_kind={kind!r} "
            f"(expected 'vae' or 'taew')"
        )
```

Insert the call to `self._maybe_init_taew_decoder()` at the end of `build_model` (after all existing reward setup).

(b) Update `_decode_gen_to_pixels` to check `self._taew_decoder` first:

Locate the existing method (should look roughly like):

```python
    def _decode_gen_to_pixels(self, gen_latent: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.net, "vae") or self.net.vae is None:
            raise RuntimeError("Re-DMD needs VAE for reward decode...")
        decoded = self.net.vae.decode(
            [gen_latent[b].float() for b in range(gen_latent.shape[0])]
        )
        if isinstance(decoded, torch.Tensor):
            return decoded
        return torch.stack(decoded, dim=0)
```

Replace with:

```python
    def _decode_gen_to_pixels(self, gen_latent: torch.Tensor) -> torch.Tensor:
        # TAEW opt-in path — defined when config.model.reward.decoder_kind == "taew".
        if getattr(self, "_taew_decoder", None) is not None:
            return self._taew_decoder.decode(
                [gen_latent[b].float() for b in range(gen_latent.shape[0])]
            )
        # Default: Wan 2.1 full VAE.
        if not hasattr(self.net, "vae") or self.net.vae is None:
            raise RuntimeError(
                "Re-DMD needs a VAE for reward decode, but self.net.vae is unset. "
                "Either ensure the base OmniAvatar model loads a VAE, or set "
                "config.model.reward.decoder_kind='taew' + taew_checkpoint_path."
            )
        decoded = self.net.vae.decode(
            [gen_latent[b].float() for b in range(gen_latent.shape[0])]
        )
        if isinstance(decoded, torch.Tensor):
            return decoded
        return torch.stack(decoded, dim=0)
```

(c) In `__init__` of `OmniAvatarSelfForcingReDMD`, defensively initialize `self._taew_decoder = None` BEFORE calling `super().__init__(config)`. This handles the "base `Model.__init__` calls `build_model` via super chain" gotcha documented in CLAUDE.md — if the attribute isn't set before build_model runs, the decoder-branch check `getattr(self, "_taew_decoder", None)` would still work, but being explicit is safer. Actually, `_maybe_init_taew_decoder` itself writes `self._taew_decoder = None` in the VAE path, so an __init__ pre-set is redundant and would create the exact "clobber attrs set in build_model" bug the CLAUDE.md warns about. **Do NOT add anything to __init__.** The `getattr(..., None)` in `_decode_gen_to_pixels` covers the case where `_maybe_init_taew_decoder` somehow didn't run.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  /home/work/.local/miniconda3/envs/hb_fastgen/bin/python -m pytest tests/reward/test_re_dmd_decoder_branch.py -v
```
Expected: 5 passed (2 skipped if no CUDA/ckpt, but the full smoke box has both).

- [ ] **Step 6: Re-run all reward tests to catch regressions**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  /home/work/.local/miniconda3/envs/hb_fastgen/bin/python -m pytest tests/reward/ -v
```
Expected: all passed (22 + 5 new = 27). If any pre-existing test broke, STOP and inspect.

- [ ] **Step 7: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  git add fastgen/methods/omniavatar_self_forcing_re_dmd.py tests/reward/test_re_dmd_decoder_branch.py && \
  git commit -m "feat(redmd): conditional TAEW decoder in reward path

Adds _maybe_init_taew_decoder() called from build_model(), which reads
config.model.reward.decoder_kind and constructs a TAEHVDecoderWrapper
when 'taew' is set (otherwise leaves self._taew_decoder = None).

_decode_gen_to_pixels now branches: if _taew_decoder is set, route
through it; else fall back to self.net.vae.decode as before. No change
to default config behavior.

Unit tests cover all four branches: vae default, taew with checkpoint,
taew without checkpoint (raises), and unknown kind (raises). A GPU test
confirms self.net.vae.decode is NOT called when the TAEW wrapper is active.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Opt-in config preset + launch scripts

**Files:**
- Create: `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_taew.py`
- Create: `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_taew_smoke.py`
- Create: `scripts/train_sf_sink1_window7_redmd_taew.sh`
- Create: `scripts/smoke_test_redmd_taew.sh`

- [ ] **Step 1: Write the production TAEW config preset**

File: `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_taew.py`

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SF + Re-DMD (sync-C reward) with TAEW decoder for the reward path.

Differs from config_sf_sink1_window7_redmd.py only in:
  - config.model.reward.decoder_kind = "taew"
  - config.model.reward.taew_checkpoint_path = <taew2_1.pth>
  - config.log_config.name suffixed with "_taew"

Everything else (β=0.25, 2-step distillation, timestep-conditional CFG,
sliding-window attention, joonson-parity SyncCScorer preprocessing) is
inherited unchanged from the base Re-DMD config.
"""

import fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd as _redmd_base


TAEW_CKPT = "/home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth"


def create_config():
    config = _redmd_base.create_config()

    config.model.reward.decoder_kind = "taew"
    config.model.reward.taew_checkpoint_path = TAEW_CKPT

    config.log_config.name = "sf_sink1_window7_redmd_syncc_beta0p25_joonson_parity_taew"
    return config


config = create_config()
```

- [ ] **Step 2: Write the smoke variant of the TAEW config**

File: `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_taew_smoke.py`

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-sized Re-DMD + TAEW variant.

Inherits config_sf_sink1_window7_redmd_taew.py, narrows to 10 iters /
batch_size=1 / debug MP4 dump, and renames the run.
"""

from fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd_taew import (
    create_config as _taew_create_config,
)


def create_config():
    config = _taew_create_config()

    config.trainer.max_iter = 11
    config.trainer.grad_accum_rounds = 1
    config.dataloader_train.batch_size = 1

    config.model.save_reward_debug_video = True
    config.model.reward_debug_dir = "logs/redmd_smoke_debug_taew"

    config.log_config.name = "sf_sink1_window7_redmd_syncc_beta0p25_joonson_parity_taew_smoke"

    if hasattr(config.trainer, "eval_period"):
        config.trainer.eval_period = 999999
    return config


config = create_config()
```

- [ ] **Step 3: Write production launcher**

File: `scripts/train_sf_sink1_window7_redmd_taew.sh`

```bash
#!/bin/bash
# Re-DMD training with TAEW decoder in the reward path.
# Same architecture/reward config as train_sf_sink1_window7_redmd.sh;
# only the reward-path VAE decode is swapped for TAEW.
#
# Usage (inside tmux):
#   bash scripts/train_sf_sink1_window7_redmd_taew.sh 2>&1 | tee /tmp/train_sf_sink1_window7_redmd_taew.log
set -euo pipefail

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_SF_OUTPUT"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

RUN_NAME="sf_sink1_window7_redmd_syncc_beta0p25_joonson_parity_taew"

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_taew.py \
    - trainer.resume=False \
    log_config.group="omniavatar_sf" \
    log_config.name="${RUN_NAME}" \
    log_config.project="OmniAvatar-FastGen" \
    log_config.wandb_entity="paulhcho"
```

Make it executable:
```bash
chmod +x /home/work/.local/hyunbin/FastGen-redmd/scripts/train_sf_sink1_window7_redmd_taew.sh
```

- [ ] **Step 4: Write smoke launcher**

File: `scripts/smoke_test_redmd_taew.sh`

```bash
#!/usr/bin/env bash
# 4-GPU smoke for Re-DMD + TAEW decoder.
# Run: bash scripts/smoke_test_redmd_taew.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export OMNIAVATAR_ROOT="${OMNIAVATAR_ROOT:-/home/work/.local/OmniAvatar}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/tmp/FASTGEN_SF_OUTPUT"
export SKIP_GT_VAL_UPLOAD=1
export SKIP_EARLY_SAMPLE_LOG=1

mkdir -p logs logs/redmd_smoke_debug_taew

echo "=== Starting Re-DMD + TAEW smoke (10 iters, batch=1, 4 GPUs) ==="
echo "Log: logs/redmd_smoke_run_taew.log"

/home/work/.local/miniconda3/envs/hb_fastgen/bin/torchrun \
    --nproc_per_node=4 \
    train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_taew_smoke.py \
    - trainer.resume=False \
    log_config.group="omniavatar_sf_smoke" \
    log_config.name="sf_sink1_window7_redmd_syncc_beta0p25_joonson_parity_taew_smoke" \
    log_config.project="OmniAvatar-FastGen-Smoke" \
    log_config.wandb_entity="paulhcho" \
    2>&1 | tee logs/redmd_smoke_run_taew.log

echo
echo "=== Post-run checks ==="

if grep -qE "Traceback|Error " logs/redmd_smoke_run_taew.log; then
    echo "FAIL: Traceback or Error in log. First 30 hits:"
    grep -nE "Traceback|Error " logs/redmd_smoke_run_taew.log | head -30
    exit 1
fi

if ! grep -q "reward_sync_c_mean" logs/redmd_smoke_run_taew.log; then
    echo "FAIL: reward_sync_c_mean never appeared — reward path didn't fire."
    tail -40 logs/redmd_smoke_run_taew.log
    exit 1
fi

echo "OK: reward_sync_c_mean appeared"

debug_mp4s=$(ls logs/redmd_smoke_debug_taew/*.mp4 2>/dev/null | wc -l)
echo "Debug MP4s written: $debug_mp4s"

echo "SMOKE TEST PASSED"
```

Make it executable:
```bash
chmod +x /home/work/.local/hyunbin/FastGen-redmd/scripts/smoke_test_redmd_taew.sh
```

- [ ] **Step 5: Confirm configs import cleanly (no syntax errors)**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  /home/work/.local/miniconda3/envs/hb_fastgen/bin/python -c "
from fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd_taew import config as c_full
from fastgen.configs.experiments.OmniAvatar.config_sf_sink1_window7_redmd_taew_smoke import config as c_smoke
print('full:', c_full.model.reward.decoder_kind, c_full.model.reward.taew_checkpoint_path)
print('full name:', c_full.log_config.name)
print('smoke:', c_smoke.model.reward.decoder_kind, c_smoke.trainer.max_iter, c_smoke.dataloader_train.batch_size)
print('smoke name:', c_smoke.log_config.name)
"
```
Expected:
```
full: taew /home/work/.local/eval_metrics/checkpoints/auxiliary/taew2_1.pth
full name: sf_sink1_window7_redmd_syncc_beta0p25_joonson_parity_taew
smoke: taew 11 1
smoke name: sf_sink1_window7_redmd_syncc_beta0p25_joonson_parity_taew_smoke
```

- [ ] **Step 6: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  git add fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_taew.py \
         fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_taew_smoke.py \
         scripts/train_sf_sink1_window7_redmd_taew.sh \
         scripts/smoke_test_redmd_taew.sh && \
  git commit -m "feat(redmd): config presets + launchers for TAEW decoder variant

Two configs (production + smoke) inherit the existing Re-DMD config chain
and only override decoder_kind and taew_checkpoint_path, so the joonson-
parity syncnet preprocessing and all training hyperparams are identical
to the VAE baseline. Run names get a _taew suffix for easy A/B in wandb.

Launcher scripts mirror the VAE-variant scripts: same torchrun invocation,
same env vars, same post-run sanity checks. Smoke writes debug MP4s to a
separate directory (logs/redmd_smoke_debug_taew/) so it doesn't clobber
prior VAE-variant artifacts.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: 4-GPU smoke test + verify sync-C matches VAE baseline

**Files:** no code changes — verification only.

Expected outcome: with TAEW instead of the full Wan VAE, the reward-path decode is visually lossy but still in the same domain. sync-C should be in the same ballpark as the joonson-parity VAE smoke (which ran ~2.75 combined-iter mean / ~3.92 iter-10 mean in the previous session). A big drop (e.g., mean < 0.5) would indicate the TAEW output distribution is too far off for the SyncNet model.

- [ ] **Step 1: Run the smoke**

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  bash scripts/smoke_test_redmd_taew.sh
```
Expected: "SMOKE TEST PASSED" at the end, and per-rank reward_sync_c logged at iter 5 and iter 10.

- [ ] **Step 2: Extract sync-C values**

```bash
grep -E "avg_reward_sync_c_(mean|min|max|r[0-9])" \
  /home/work/.local/hyunbin/FastGen-redmd/logs/redmd_smoke_run_taew.log | tail -30
```
Expected: iter-5 and iter-10 sync_c_mean in the range [1.0, 6.0] (same order of magnitude as the VAE smoke). Values below 0.5 indicate a distribution mismatch worth investigating before launching a full run.

- [ ] **Step 3: Compare debug MP4s side-by-side**

Inspect both:
- `logs/redmd_smoke_debug/no255_bgr_audio/gen_iter000005.mp4` (VAE baseline from prior session)
- `logs/redmd_smoke_debug_taew/gen_iter000005.mp4` (TAEW)

Visually verify TAEW output is recognizable as a talking head (some blur is expected). If TAEW output is garbage, STOP — the wrapper contract is wrong.

- [ ] **Step 4: Measure wall-clock speedup**

```bash
grep -E "avg forward pass time|forward pass time" \
  /home/work/.local/hyunbin/FastGen-redmd/logs/redmd_smoke_run_taew.log | tail -5
```
Compare to the VAE smoke's forward-pass times (in `logs/redmd_smoke_run_no255_bgr_audio.log`). TAEW forward is expected to be noticeably faster.

- [ ] **Step 5: Commit nothing** (verification-only task) but archive smoke artifacts

```bash
cd /home/work/.local/hyunbin/FastGen-redmd && \
  cp logs/redmd_smoke_run_taew.log logs/redmd_smoke_run_no255_bgr_audio_taew.log
```
(Keeps the smoke log matched to the current fix set for future A/B reference.)

---

## Self-Review

### 1. Spec coverage

| Spec requirement | Task |
|---|---|
| Verify TAEW handles first-frame edge correctly | Task 1 step 3 (frame count 81 = `(21-1)*4+1`) |
| Minimize VAE-decode overhead via TAEW | Task 3 (wrapper) + Task 5 (branch) + Task 7 step 4 (speedup measurement) |
| Make it conditional (opt-in) | Task 4 (config field defaults `"vae"`) + Task 6 (separate config preset) |
| Preserve original functionality otherwise | Task 4 step 4 (new fields, existing five untouched) + Task 5 step 4 (`_decode_gen_to_pixels` default branch unchanged) |
| Verify at each step | TDD structure throughout: write test → fail → implement → pass → commit |

No gaps.

### 2. Placeholder scan

- No "TBD"/"TODO"/"implement later" strings.
- Every test has a full implementation snippet.
- Every code change has the exact new code, not a prose description.
- Every shell command has expected output.
- Task 1 step 1 gives an exact curl URL; Task 2 step 3 shows the exact cp command and SPDX header to add.

One soft spot: Task 1 says "file size ~5-50MB" without a precise expected value — acceptable because the repo doesn't publish a hash and we're validating by loading, not by checksum.

### 3. Type consistency

- `TAEHVDecoderWrapper(checkpoint_path, device)` signature used identically in: Task 3 Step 3 (definition), Task 5 Step 4 (in `_maybe_init_taew_decoder`), Task 5 Step 2 (tests). ✓
- `decoder_kind` field named consistently across Task 4 (added), Task 5 (read), Task 6 (set in configs). ✓
- `taew_checkpoint_path` field named consistently across Task 4 (added), Task 5 (read), Task 6 (set in configs). ✓
- `_taew_decoder` attribute named consistently across Task 5 (set, read, testing). ✓
- `self.config.model.reward.decoder_kind` access path matches the attrs dataclass structure in `config_omniavatar_sf.py` (verified the `.reward` sub-field exists). ✓

No inconsistencies.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-15-taew-reward-decoder.md`.

Per user's note ("we can begin making changes directly, verifying at each step"), execution mode is:

**Inline Execution** with per-task verification — run tests before committing each task, confirm expected outputs match, proceed to next.

If a subagent-driven approach is preferred instead, use `superpowers:subagent-driven-development` to dispatch one implementer + two reviewers (spec + code) per task.
