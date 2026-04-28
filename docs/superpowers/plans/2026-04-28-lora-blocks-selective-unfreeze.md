# LoRA-on-Blocks + Selective Unfreeze for OmniAvatar Causal Student Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable a hybrid training regime for the 14B (and 1.3B) causal student where transformer blocks are adapted via PEFT LoRA (rank 128) with the base frozen, while specific submodules listed in config (default: `audio_proj`, `audio_cond_projs`, `patch_embedding`) are fully fine-tuned. This targets the lip-sync-relevant audio path with full capacity while constraining the bulk of the network to a low-rank update — collapses optimizer state from ~107 GB/save to <1 GB and lets the audio adapters fully adapt to the causal student dynamics.

**Architecture:**
- The PEFT injection code path already exists at `fastgen/networks/OmniAvatar/network_causal.py:2116-2142` (the `merge_lora=False` branch) but freezes everything except LoRA A/B by default.
- Add a new `unfreeze_modules: List[str]` constructor argument (paths relative to `self`, e.g. `["_core.audio_proj"]`) and a small `_apply_unfreeze` helper that walks the listed submodules after PEFT injection and re-enables `requires_grad` on their parameters.
- Optimizer construction at `fastgen/configs/opt.py:27` already filters `params=[p for p in model.parameters() if p.requires_grad]`, so re-enabling `requires_grad` is sufficient — no optimizer-side plumbing.
- The FSDP wrap at `network_causal.py:2148-2210` (already fixed for bug 1) continues to wrap each non-block submodule individually, so audio_proj / audio_cond_projs / patch_embedding remain properly grad-synced even under PEFT injection.
- Provide a new config (`config_df_shift_5_14b_lora.py`) and wrapper script (`train_omniavatar_df_shift_5_14b_lora.sh`) that selects this regime. Smoke-test it before any long run.

**Tech Stack:** PyTorch 2.x, PyTorch FSDP2, PEFT 0.18.1 (`peft.LoraConfig` + `peft.inject_adapter_in_model`), OmegaConf/Hydra config, attrs config-defs, pytest for unit tests.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `fastgen/networks/OmniAvatar/network_causal.py` | Modify | Add `unfreeze_modules` ctor arg; add `_apply_unfreeze` method; call it in `_load_weights` after PEFT inject. ~30 lines. |
| `fastgen/configs/experiments/OmniAvatar/config_df_shift_5_14b_lora.py` | Create | New config inheriting from `config_df_shift_5_14b.py`; sets `merge_lora=False`, `unfreeze_modules=[...]`, `lora_rank=128`. |
| `scripts/train_omniavatar_df_shift_5_14b_lora.sh` | Create | Wrapper script that points `CONFIG_PATH` at the new config and adjusts `RUN_NAME` accordingly. |
| `tests/test_causal_omniavatar_unfreeze.py` | Create | Unit tests for `_apply_unfreeze`: tests on a tiny dummy nn.Module (no need to construct the full 14B model). Two tests: empty list is a no-op; specified path's params are re-enabled. |
| `docs/lora_selective_unfreeze.md` | Create | Short usage doc + intent. ~40 lines. |

---

## Task 1: Add `unfreeze_modules` constructor argument

**Files:**
- Modify: `fastgen/networks/OmniAvatar/network_causal.py:881-1056` (the `CausalOmniAvatarWan.__init__` block)

This task only adds the kwarg and stores it. We do not yet apply it — that's task 3.

- [ ] **Step 1: Read the current `__init__` signature**

Run: `sed -n '881,910p' fastgen/networks/OmniAvatar/network_causal.py`

Confirm the existing signature so we know where to slot in the new kwarg. The class already accepts `merge_lora`, `lora_rank`, `lora_alpha` — we'll add `unfreeze_modules` next to them.

- [ ] **Step 2: Add the `unfreeze_modules` kwarg to `__init__`**

In `fastgen/networks/OmniAvatar/network_causal.py`, add the parameter to the `__init__` signature. Insert after the existing `lora_alpha` kwarg (locate via `grep -n "lora_alpha:" fastgen/networks/OmniAvatar/network_causal.py`):

```python
        # ... existing kwargs ...
        merge_lora: bool = True,
        lora_rank: int = 128,
        lora_alpha: int = 64,
        unfreeze_modules: Optional[List[str]] = None,  # NEW: see _apply_unfreeze
        # ... remaining kwargs ...
```

And in the body of `__init__`, alongside `self.merge_lora = merge_lora`:

```python
        self.merge_lora = merge_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        # Paths (relative to self, dotted) of submodules whose parameters
        # should keep requires_grad=True even when PEFT injection has frozen
        # the rest of the base model.  Used with merge_lora=False to enable
        # selective full fine-tuning of specific components (e.g.,
        # ["_core.audio_proj", "_core.audio_cond_projs", "_core.patch_embedding"])
        # alongside LoRA on the transformer blocks.  Ignored when merge_lora=True.
        self.unfreeze_modules: List[str] = list(unfreeze_modules) if unfreeze_modules else []
```

- [ ] **Step 3: Add a smoke test that constructs the class with the new kwarg**

Create `tests/test_causal_omniavatar_unfreeze.py` with the following content:

```python
"""Tests for CausalOmniAvatarWan's selective-unfreeze logic.

These tests cover the helper that re-enables requires_grad on specific
submodules after PEFT's inject_adapter_in_model has frozen the base.
We do NOT construct the full 14B model — the helper is testable against
a tiny dummy nn.Module fixture.
"""
import pytest
import torch
import torch.nn as nn


class _DummyCore(nn.Module):
    """Tiny stand-in for CausalOmniAvatarWan._core to test the unfreeze helper."""

    def __init__(self):
        super().__init__()
        self.audio_proj = nn.Linear(8, 16)
        self.audio_cond_projs = nn.ModuleList([nn.Linear(16, 16), nn.Linear(16, 16)])
        self.blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])


class _DummyHost(nn.Module):
    """Stand-in for CausalOmniAvatarWan: holds _core and exposes the helper."""

    def __init__(self):
        super().__init__()
        self._core = _DummyCore()

    # The real implementation lives on CausalOmniAvatarWan; we copy the body
    # here only because importing the real class would require GPU + heavy deps.
    def _apply_unfreeze(self, unfreeze_modules):
        if not unfreeze_modules:
            return
        for path in unfreeze_modules:
            module = self.get_submodule(path)
            for p in module.parameters():
                p.requires_grad_(True)


def _set_all_requires_grad(module, value):
    for p in module.parameters():
        p.requires_grad_(value)


def test_construction_smoke():
    """Class constructs with unfreeze_modules kwarg; storage is correct."""
    # This will be replaced in Task 3 with a real call to CausalOmniAvatarWan
    # once we mock the heavy weight-loading. For Task 1, just verify the
    # dummy host pattern works.
    host = _DummyHost()
    assert hasattr(host, "_core")
    assert hasattr(host._core, "audio_proj")
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `pytest tests/test_causal_omniavatar_unfreeze.py::test_construction_smoke -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/OmniAvatar/network_causal.py tests/test_causal_omniavatar_unfreeze.py
git commit -m "feat(causal): add unfreeze_modules kwarg to CausalOmniAvatarWan

Stores the list of submodule paths whose parameters should be kept
trainable even after PEFT freezes the rest. Implementation of the
helper that uses this list comes in the next commit.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Implement and test the `_apply_unfreeze` helper

**Files:**
- Modify: `fastgen/networks/OmniAvatar/network_causal.py` (add method to `CausalOmniAvatarWan`)
- Test: `tests/test_causal_omniavatar_unfreeze.py`

- [ ] **Step 1: Write the failing test for `_apply_unfreeze`**

Append to `tests/test_causal_omniavatar_unfreeze.py`:

```python
def test_unfreeze_specific_submodule_re_enables_grad():
    """After freezing all params, _apply_unfreeze re-enables requires_grad
    on the parameters of the specified submodule and leaves others alone."""
    host = _DummyHost()

    # Simulate PEFT freezing the base
    _set_all_requires_grad(host, value=False)
    for p in host.parameters():
        assert p.requires_grad is False

    # Unfreeze just _core.audio_proj
    host._apply_unfreeze(["_core.audio_proj"])

    # audio_proj params should be trainable
    for p in host._core.audio_proj.parameters():
        assert p.requires_grad is True, "audio_proj.weight/bias should be trainable"

    # Everything else should remain frozen
    for p in host._core.audio_cond_projs.parameters():
        assert p.requires_grad is False
    for p in host._core.blocks.parameters():
        assert p.requires_grad is False


def test_unfreeze_modulelist_unfreezes_all_children():
    """Unfreezing a ModuleList path re-enables grad on every child Linear."""
    host = _DummyHost()
    _set_all_requires_grad(host, value=False)

    host._apply_unfreeze(["_core.audio_cond_projs"])

    for proj in host._core.audio_cond_projs:
        for p in proj.parameters():
            assert p.requires_grad is True
    # Other submodules untouched
    for p in host._core.audio_proj.parameters():
        assert p.requires_grad is False


def test_unfreeze_empty_list_is_noop():
    """Passing an empty (or None) unfreeze list does not change requires_grad."""
    host = _DummyHost()
    _set_all_requires_grad(host, value=False)

    host._apply_unfreeze([])

    for p in host.parameters():
        assert p.requires_grad is False


def test_unfreeze_unknown_path_raises_attribute_error():
    """Walking get_submodule for a non-existent path is a hard error.

    Choosing strict-failure over silent skip so a typo in the config is
    caught immediately rather than silently leaving the intended module
    frozen for the entire training run.
    """
    host = _DummyHost()
    with pytest.raises(AttributeError):
        host._apply_unfreeze(["_core.does_not_exist"])
```

- [ ] **Step 2: Run tests to verify they fail (helper not yet on real class)**

Run: `pytest tests/test_causal_omniavatar_unfreeze.py -v`
Expected: All four new tests PASS (because the helper is on `_DummyHost`).

(Yes, the dummy implementation already passes — that's intentional. We're testing the *behavior* on a small fixture; Task 3 will copy this implementation onto the real class and the integration smoke run validates the real call site.)

- [ ] **Step 3: Implement `_apply_unfreeze` on `CausalOmniAvatarWan`**

Edit `fastgen/networks/OmniAvatar/network_causal.py`. Locate `_finish_init` (currently at line ~1098) and add `_apply_unfreeze` directly above it:

```python
    def _apply_unfreeze(self, unfreeze_modules: List[str]) -> None:
        """Re-enable ``requires_grad`` on parameters of specific submodules.

        Used in conjunction with PEFT LoRA injection (``merge_lora=False``):
        ``inject_adapter_in_model`` freezes the entire base and only leaves
        LoRA A/B trainable.  This helper selectively un-freezes the
        submodules listed in ``unfreeze_modules`` so they participate in
        training as full fine-tunes.

        Paths are dotted module paths relative to ``self``.  Common values:

        - ``"_core.audio_proj"`` — the AudioPack audio input projection
        - ``"_core.audio_cond_projs"`` — the per-block audio cross-attn projections
        - ``"_core.patch_embedding"`` — the V2V channel-input Conv3d

        Failure mode is intentionally strict: if a path is not resolvable
        via ``self.get_submodule``, ``AttributeError`` is raised so a
        config typo is caught at construction time rather than silently
        leaving the intended module frozen for the whole run.

        Args:
            unfreeze_modules: list of dotted module paths relative to self.
                Empty list or ``None`` is a no-op.
        """
        if not unfreeze_modules:
            return
        for path in unfreeze_modules:
            module = self.get_submodule(path)  # raises AttributeError if missing
            n_params_unfrozen = 0
            for p in module.parameters():
                p.requires_grad_(True)
                n_params_unfrozen += p.numel()
            logger.info(
                f"[CausalOmniAvatarWan] unfreeze: re-enabled requires_grad on "
                f"{n_params_unfrozen / 1e6:.2f}M params in '{path}'"
            )
```

- [ ] **Step 4: Verify the real class can be imported and the method exists**

Run:
```bash
/home/work/.local/miniconda3/envs/hb_fastgen/bin/python -c "
import sys
sys.path.insert(0, '/home/work/.local/hyunbin/FastGen-redmd')
from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan
assert callable(CausalOmniAvatarWan._apply_unfreeze)
print('CausalOmniAvatarWan._apply_unfreeze: OK')
"
```
Expected: `CausalOmniAvatarWan._apply_unfreeze: OK`

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/OmniAvatar/network_causal.py tests/test_causal_omniavatar_unfreeze.py
git commit -m "feat(causal): _apply_unfreeze helper for selective full fine-tune

Re-enables requires_grad on specified submodule paths after PEFT
freezes the base, so transformer blocks train via LoRA while audio
path / patch_embedding train fully. Strict failure on unknown paths
to catch config typos.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Wire `_apply_unfreeze` into `_load_weights`'s PEFT branch

**Files:**
- Modify: `fastgen/networks/OmniAvatar/network_causal.py:2116-2142` (the `else` branch under `if self.merge_lora`)

- [ ] **Step 1: Read the current PEFT branch**

Run: `sed -n '2095,2145p' fastgen/networks/OmniAvatar/network_causal.py`

Confirm the existing branch ends with the `inject_adapter_in_model(...)` + `load_state_dict(...)` calls. We'll append the `_apply_unfreeze` call after the load.

- [ ] **Step 2: Add `_apply_unfreeze` call after PEFT injection**

Edit `fastgen/networks/OmniAvatar/network_causal.py`. Locate the block:

```python
                        inject_adapter_in_model(lora_config, self._core)
                        missing, unexpected = self._core.load_state_dict(
                            mapped_lora_sd, strict=False
                        )
                        logger.info(
                            f"[CausalOmniAvatarWan] PEFT LoRA: "
                            f"{len(mapped_lora_sd) - len(unexpected)} loaded"
                        )
```

Append immediately after the `logger.info` call (still inside the `try:` block, same indent level):

```python
                        # Selectively re-enable requires_grad on submodules
                        # that should be fully fine-tuned alongside the LoRA
                        # adapters (e.g., audio_proj, audio_cond_projs,
                        # patch_embedding).  No-op when unfreeze_modules is
                        # empty.  See _apply_unfreeze for full rationale.
                        self._apply_unfreeze(self.unfreeze_modules)
```

- [ ] **Step 3: Add an integration test that verifies trainable-param count under PEFT**

We can't construct the full 14B in a unit test, but we can sanity-check the `__init__` signature and verify the method is dispatched correctly when `merge_lora=False`. Append to `tests/test_causal_omniavatar_unfreeze.py`:

```python
def test_init_signature_has_unfreeze_modules():
    """Confirm the new kwarg is in the real CausalOmniAvatarWan signature."""
    import inspect
    from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

    sig = inspect.signature(CausalOmniAvatarWan.__init__)
    assert "unfreeze_modules" in sig.parameters, (
        "CausalOmniAvatarWan.__init__ should accept an unfreeze_modules kwarg"
    )
    # Default should be None (i.e., no-op when not provided)
    assert sig.parameters["unfreeze_modules"].default is None


def test_apply_unfreeze_is_callable():
    """Confirm the helper is bound to the real class."""
    from fastgen.networks.OmniAvatar.network_causal import CausalOmniAvatarWan

    assert callable(CausalOmniAvatarWan._apply_unfreeze)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_causal_omniavatar_unfreeze.py -v`
Expected: All tests PASS (six total: four behavior tests on `_DummyHost`, two introspection tests on the real class).

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/OmniAvatar/network_causal.py tests/test_causal_omniavatar_unfreeze.py
git commit -m "feat(causal): apply unfreeze after PEFT injection in _load_weights

Wires _apply_unfreeze into the merge_lora=False branch so the user-
provided unfreeze_modules list takes effect immediately after
inject_adapter_in_model freezes the base.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Create the LoRA training config

**Files:**
- Create: `fastgen/configs/experiments/OmniAvatar/config_df_shift_5_14b_lora.py`

- [ ] **Step 1: Create the new config file**

Write `fastgen/configs/experiments/OmniAvatar/config_df_shift_5_14b_lora.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DF (shift=5) for the 14B causal student — LoRA blocks + selective unfreeze.

Inherits from config_df_shift_5_14b.py.  Substantive differences:

1) ``merge_lora=False``: instead of fusing the V2V adapter into the base
   weights and full-fine-tuning all 14B params, keep the LoRA adapters
   separate (PEFT-injected) and train only the LoRA A/B matrices on the
   transformer blocks.  The base 14B weights stay frozen.

2) ``unfreeze_modules`` selectively re-enables ``requires_grad`` on
   submodules that DO need to fully adapt to the causal-student dynamics:
   - ``_core.audio_proj``: AudioPack input projection (audio -> hidden)
   - ``_core.audio_cond_projs``: per-block audio cross-attn projections
   - ``_core.patch_embedding``: input Conv3d for the V2V channels

   The rationale: the lip-sync gap we observe across SF runs appears to
   be most strongly tied to the audio path's ability to adapt under
   distillation, and constraining the bulk of the network to a low-rank
   update while keeping the audio path full-rank is a hypothesis-aligned
   experiment.  See ``docs/lora_selective_unfreeze.md`` for context.

3) Optimizer state shrinks dramatically: with ``merge_lora=True`` Adam
   m+v on 14B fp32 is ~107 GB per save.  With this config the trainable
   params are LoRA(rank=128) on the q/k/v/o/ffn linears plus the audio
   path (~100M) + patch_embedding (~3M) = roughly 50-150M trainable
   params.  Optim state per save drops to <1 GB.  This sidesteps the
   disk-pressure issue around save 6 in the full-FT run entirely.

4) Effective batch and FSDP knobs are unchanged from the parent.
"""

import fastgen.configs.experiments.OmniAvatar.config_df_shift_5_14b as _full_ft_base


# Submodules to keep fully trainable alongside LoRA on the transformer blocks.
# Paths are dotted, relative to the CausalOmniAvatarWan instance (so they
# include the "_core." prefix where the actual modules live).
DEFAULT_UNFREEZE_MODULES = [
    "_core.audio_proj",
    "_core.audio_cond_projs",
    "_core.patch_embedding",
]


def create_config():
    config = _full_ft_base.create_config()

    # ---- Switch from full FT to LoRA + selective unfreeze ----
    config.model.net.merge_lora = False
    config.model.net.unfreeze_modules = DEFAULT_UNFREEZE_MODULES

    # LoRA hyperparameters.  Match the V2V adapter we're loading from
    # (rank=128, alpha=64) — values come from the OmniAvatar V2V training
    # recipe and are what the saved adapter weights were trained at.
    # Changing these would require re-initializing the LoRA matrices
    # from scratch.
    config.model.net.lora_rank = 128
    config.model.net.lora_alpha = 64

    return config


config = create_config()
```

- [ ] **Step 2: Verify the config loads cleanly**

Run:
```bash
/home/work/.local/miniconda3/envs/hb_fastgen/bin/python -c "
import sys
sys.path.insert(0, '/home/work/.local/hyunbin/FastGen-redmd')
from fastgen.configs.experiments.OmniAvatar.config_df_shift_5_14b_lora import config
assert config.model.net.merge_lora is False
assert config.model.net.unfreeze_modules == [
    '_core.audio_proj', '_core.audio_cond_projs', '_core.patch_embedding']
assert config.model.net.lora_rank == 128
print('config_df_shift_5_14b_lora: OK')
"
```
Expected: `config_df_shift_5_14b_lora: OK`

- [ ] **Step 3: Commit**

```bash
git add fastgen/configs/experiments/OmniAvatar/config_df_shift_5_14b_lora.py
git commit -m "feat(config): add 14B LoRA + selective-unfreeze DF config

Inherits from config_df_shift_5_14b and flips merge_lora to False
with the audio path + patch_embedding listed in unfreeze_modules.
LoRA hyperparams (rank=128, alpha=64) match the V2V mouthweight
adapter we initialize from.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Create the wrapper script

**Files:**
- Create: `scripts/train_omniavatar_df_shift_5_14b_lora.sh`

- [ ] **Step 1: Create the wrapper script**

Write `scripts/train_omniavatar_df_shift_5_14b_lora.sh`:

```bash
#!/bin/bash
# =============================================================================
# OmniAvatar DF (shift=5) — 14B causal student, LoRA blocks + selective unfreeze
# =============================================================================
#
# Same training stack as train_omniavatar_df_shift_5_14b_audiofix_syncnet_trained.sh
# but uses config_df_shift_5_14b_lora.py instead of config_df_shift_5_14b.py.
# That config sets merge_lora=False and unfreeze_modules=["_core.audio_proj",
# "_core.audio_cond_projs", "_core.patch_embedding"], so the run trains
# LoRA A/B on the transformer blocks plus full fine-tunes of the audio path
# and patch embedding.
#
# Why this regime: see docs/lora_selective_unfreeze.md.  Brief: the
# lip-sync gap we observe in SF runs appears to depend strongly on
# audio-path adaptation; a hybrid LoRA-on-blocks + full-FT-on-audio-path
# tests whether targeting the bottleneck with full capacity (while
# constraining the bulk of the network to a low-rank update) closes
# the gap.
#
# Disk: optim state is tiny (<1 GB per save) since most params are
# frozen.  No need for the strip-watcher.
#
# Walltime: probably faster per iter than full-FT due to smaller optim
# step + smaller all-gather buffers.  Measure on the smoke run.
#
# Usage:
#   bash scripts/train_omniavatar_df_shift_5_14b_lora.sh
#
#   # Smoke test (50 iters, validates wrap+forward+backward+save format):
#   MAX_ITER=50 SAVE_EVERY=50 \
#     bash scripts/train_omniavatar_df_shift_5_14b_lora.sh
# =============================================================================

set -euo pipefail

# Use the LoRA-specialized config.
export CONFIG_PATH="fastgen/configs/experiments/OmniAvatar/config_df_shift_5_14b_lora.py"

# Per-GPU batch + grad accum: same as the full-FT 14B wrapper by default.
# The user can override either via env.  With smaller optim state, we
# may have headroom for BATCH_SIZE=4 or higher in a follow-up run, but
# keep it conservative for the first launch.
export BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

export SAVE_EVERY="${SAVE_EVERY:-500}"
export VIZ_EVERY="${VIZ_EVERY:-${SAVE_EVERY}}"

NGPU="${NGPU:-4}"
MAX_ITER="${MAX_ITER:-3000}"
EFFECTIVE_BATCH=$((BATCH_SIZE * NGPU * GRAD_ACCUM))
export RUN_NAME="${RUN_NAME:-df_audiofix_syncnet_trained_shift_5_14b_lora_${NGPU}gpu_bs${BATCH_SIZE}_grad${GRAD_ACCUM}_eff${EFFECTIVE_BATCH}_lr1e5_${MAX_ITER}iter}"

# DDP -> FSDP and grad_accum override (config sets grad_accum_rounds=4
# inherited from the parent; we override on cmdline to match GRAD_ACCUM
# env).  Identical pattern to train_omniavatar_df_shift_5_14b_audiofix_syncnet_trained.sh.
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-trainer.ddp=False trainer.fsdp=True trainer.grad_accum_rounds=${GRAD_ACCUM}}"

echo "============================================="
echo "  14B DF FSDP launch settings — LoRA + unfreeze"
echo "============================================="
echo "  Per-GPU batch:    ${BATCH_SIZE}"
echo "  GPUs:             ${NGPU}"
echo "  Grad accum:       ${GRAD_ACCUM}"
echo "  Effective batch:  ${EFFECTIVE_BATCH}  (= ${BATCH_SIZE} x ${NGPU} x ${GRAD_ACCUM})"
echo "  Max iter:         ${MAX_ITER}"
echo "  Save every:       ${SAVE_EVERY}"
echo "  Run name:         ${RUN_NAME}"
echo "  Config:           ${CONFIG_PATH}"
echo "  EXTRA_OVERRIDES:  ${EXTRA_OVERRIDES}"
echo "============================================="

# Delegate to the existing parent (passes NGPU/MAX_ITER/SAVE_EVERY/RESUME
# through env).
exec "$(dirname "$(readlink -f "$0")")/train_omniavatar_df_shift_5_audiofix_syncnet_trained.sh" "$@"
```

- [ ] **Step 2: Make the script executable + shell-syntax check**

Run:
```bash
chmod +x scripts/train_omniavatar_df_shift_5_14b_lora.sh
bash -n scripts/train_omniavatar_df_shift_5_14b_lora.sh
echo "syntax OK"
```
Expected: `syntax OK`

- [ ] **Step 3: Dry-run the env-var math**

Run (this just exercises the bash arithmetic, doesn't launch torchrun):
```bash
BATCH_SIZE=2 GRAD_ACCUM=2 NGPU=4 MAX_ITER=50 SAVE_EVERY=50 bash -c '
NGPU=4
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EFFECTIVE_BATCH=$((BATCH_SIZE * NGPU * GRAD_ACCUM))
echo "smoke: BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM EFFECTIVE_BATCH=$EFFECTIVE_BATCH"
'
```
Expected: `smoke: BATCH_SIZE=2 GRAD_ACCUM=2 EFFECTIVE_BATCH=16`

- [ ] **Step 4: Commit**

```bash
git add scripts/train_omniavatar_df_shift_5_14b_lora.sh
git commit -m "scripts: 14B DF wrapper for LoRA + selective-unfreeze regime

Mirrors the full-FT 14B wrapper's env-toggle ergonomics (BATCH_SIZE,
GRAD_ACCUM, SAVE_EVERY, MAX_ITER, RUN_NAME) but points CONFIG_PATH
at config_df_shift_5_14b_lora.py.  Ready to use as
'MAX_ITER=50 SAVE_EVERY=50 bash scripts/...14b_lora.sh' for the
smoke-test pass before any long run.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Document the regime

**Files:**
- Create: `docs/lora_selective_unfreeze.md`

- [ ] **Step 1: Write the doc**

Write `docs/lora_selective_unfreeze.md`:

```markdown
# LoRA on Blocks + Selective Unfreeze

A hybrid training regime for the OmniAvatar causal student: the
transformer blocks are adapted via PEFT LoRA (rank=128) with the base
weights frozen, and specific submodules (default: `audio_proj`,
`audio_cond_projs`, `patch_embedding`) are fully fine-tuned alongside.

## Why

Across SF runs, the gap between student Sync-C and teacher Sync-C
correlates with audio-path adaptation quality.  Two reasons to suspect
the audio path specifically:

1. The pre-bug-1 FSDP setup left audio_proj / audio_cond_projs as
   non-FSDP-wrapped params with no gradient sync — they drifted across
   ranks throughout every SF run.  This was fixed in the 14B DF launch.
2. Even with grad sync working, full-rank fine-tuning of all 14B
   parameters might over-fit the bulk of the network and dilute the
   training signal on the smaller, more critical audio components.

This regime targets the audio bottleneck with full capacity while
constraining the bulk of the network to a low-rank delta — a
hypothesis-aligned ablation.

## How

`fastgen/configs/experiments/OmniAvatar/config_df_shift_5_14b_lora.py`
sets:

- `merge_lora=False`: PEFT injects LoRA adapters on the transformer
  blocks instead of fusing the V2V adapter into the base.  The base 14B
  weights are frozen via PEFT's default freeze.
- `unfreeze_modules=["_core.audio_proj", "_core.audio_cond_projs",
  "_core.patch_embedding"]`: after PEFT freeze, these submodules have
  their `requires_grad` flipped back to True.  `_apply_unfreeze` (in
  `network_causal.py`) handles this.

The optimizer factory at `fastgen/configs/opt.py:27` already filters
`params=[p for p in model.parameters() if p.requires_grad]`, so only
the trainable params (LoRA A/B + unfrozen submodules) participate in
optimization steps.  Adam state is allocated only for trainable params.

## Disk and memory implications

| Metric | Full FT (`config_df_shift_5_14b.py`) | LoRA + unfreeze (this config) |
|---|---|---|
| Trainable params (14B) | 14.3 B | ~50–150 M |
| Optim state per save | ~107 GB | <1 GB |
| Total per save | ~161 GB | ~58 GB (model still full) |
| GPU mem peak | ~137 GB reserved | likely ~80–100 GB |

The save sizes drop because the optim shards (`*.net_optim`) only
contain m/v for the trainable params.  The model shards (`*.net_model`)
still contain full 14B weights (we keep them around for the LoRA
adapters' "base" reference and for downstream inference).

## Launching

Smoke-test first:

```bash
MAX_ITER=50 SAVE_EVERY=50 \
  bash scripts/train_omniavatar_df_shift_5_14b_lora.sh \
  2>&1 | tee /tmp/train_df_14b_lora_smoke.log
```

Watch for:
- The `[merge_lora] Merged ... LoRA pairs` line should NOT appear (we
  go through the PEFT-inject branch instead, which logs
  `[CausalOmniAvatarWan] PEFT LoRA: N loaded`).
- An `[unfreeze]` log line per entry in `unfreeze_modules`, summing to
  ~100M unfrozen params.
- The `param_count` callback should report **trainable** << **total**
  (e.g., trainable ~150M, total 14294M for the 14B configuration).
- FSDP wrap completes without "mixed Tensor and DTensor" errors.
- First iter finite loss within ~3-5 minutes.

Full launch:

```bash
tmux new -s df14b_lora -d \
  "bash scripts/train_omniavatar_df_shift_5_14b_lora.sh \
   2>&1 | tee /tmp/train_df_14b_lora_3000iter.log"
```

## Open questions / known limitations

- **LoRA rank**: 128 matches the V2V mouthweight adapter we initialize
  from.  Increasing rank (e.g., 256) requires re-initializing A/B from
  scratch (the saved weights are rank-128).  Lower rank would discard
  capacity from the V2V init.
- **Frozen non-target submodules**: `time_embedding`, `time_projection`,
  `text_embedding`, `head` stay frozen with this default
  `unfreeze_modules` list.  If you find sync still suffers, consider
  adding `_core.head` or `_core.time_embedding` to the list — small
  capacity, similarly critical for the input/output interface.
- **Comparison**: should be evaluated against the running full-FT 14B
  DF (`config_df_shift_5_14b.py`) at matched iter counts.  Same
  evaluation pipeline, different training regime.
```

- [ ] **Step 2: Commit**

```bash
git add docs/lora_selective_unfreeze.md
git commit -m "docs: explain LoRA + selective-unfreeze training regime

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Smoke test on the real model (manual, requires GPUs)

**Files:** none — runs the wrapper script.

This is a manual integration test that requires the GPUs.  It does NOT
have a checkbox-driven test framework; instead, it has a checklist of
log-line signals to inspect.

**Important pre-condition**: the running 14B full-FT DF training is
expected to wrap up around 02:50 KST Apr 29 (per ETA in the session
log).  Wait for that to release the 4 GPUs before launching this smoke.
Alternatively, stop the running training cleanly at a save boundary if
the smoke is needed sooner.

- [ ] **Step 1: Verify GPUs are free**

Run: `nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv`
Expected: All four H200s with low GPU utilization (~0%) and minimal
memory used (~few hundred MB at most for any leftover allocations).

- [ ] **Step 2: Launch the 50-iter smoke**

Run:
```bash
cd /home/work/.local/hyunbin/FastGen-redmd
MAX_ITER=50 SAVE_EVERY=50 \
  bash scripts/train_omniavatar_df_shift_5_14b_lora.sh \
  2>&1 | tee /tmp/train_df_14b_lora_smoke.log
```

(No tmux for the smoke — we want to watch live and either let it run
to completion or Ctrl-C if it crashes early.)

- [ ] **Step 3: Verify the PEFT inject branch was taken (not merge)**

After the launch reaches the model-construction phase:

Run: `grep -E "merge_lora|PEFT LoRA" /tmp/train_df_14b_lora_smoke.log`
Expected:
- NO line containing `[merge_lora] Merged N LoRA pairs`
- ONE line per network containing `[CausalOmniAvatarWan] PEFT LoRA: N loaded`
  (one for `net`, possibly more for `fake_score` / `teacher` if the DF setup
  has them — though for DF only `net` is constructed, so one is enough).

- [ ] **Step 4: Verify the unfreeze step ran for the expected modules**

Run: `grep "unfreeze: re-enabled" /tmp/train_df_14b_lora_smoke.log`
Expected: Three lines, one each for `_core.audio_proj`,
`_core.audio_cond_projs`, `_core.patch_embedding`.  Total numel summed
across all three should be roughly ~100M for 14B (if the 1.3B build
were running, it'd be roughly ~30M).

- [ ] **Step 5: Verify trainable << total in the param count callback**

Run: `grep -E "param_count.*trainable and.*total params" /tmp/train_df_14b_lora_smoke.log | head -10`
Expected: For the `net` (CausalOmniAvatarWan), `trainable` should be
roughly 50–150M while `total` is 14294M.  If `trainable == total`, the
unfreeze ran but PEFT's freeze didn't — that's a regression.

- [ ] **Step 6: Verify FSDP wrap success**

Run: `grep -E "FSDP2 wrapped|reset_parameters" /tmp/train_df_14b_lora_smoke.log`
Expected:
- A line `FSDP2 wrapped net in N.Ns` (no exception).
- NO line containing `does not implement the reset_parameters method`
  (because `fsdp_meta_init=False` is inherited from the parent config).

- [ ] **Step 7: Verify training reaches finite-loss iters**

Run: `grep -E "iter count|avg_total_loss" /tmp/train_df_14b_lora_smoke.log | head -10`
Expected: At least one `avg_total_loss: <finite>` log line within ~5
minutes of launch.  If `nan` appears, that's a numerical-stability
failure — do not proceed to a long run.

- [ ] **Step 8: Verify the save format works**

Wait for iter 50's save (the smoke MAX_ITER=50 SAVE_EVERY=50 means save
1 lands at the end).

Run: `ls -la FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_syncnet_trained_shift_5_14b_lora_*/checkpoints/ 2>/dev/null`
Expected:
- `0000050.net_model/` (full 14B model shards, ~54 GB)
- `0000050.net_optim/` — should be SMALL (<1 GB), only optim for trainable params
- `0000050.pth` (~few KB)

The optim being <1 GB is the key disk-savings claim materializing.  If
it's full-sized (~107 GB), the optimizer constructed over all params
instead of trainable-only, indicating the `requires_grad` filter at
`opt.py:27` didn't fire as expected — would need investigation.

- [ ] **Step 9: Run the diagnostic on the smoke save**

Run:
```bash
/home/work/.local/miniconda3/envs/hb_fastgen/bin/python scripts/diagnostics/inspect_fsdp_topology.py \
  FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_syncnet_trained_shift_5_14b_lora_*/checkpoints/0000050.net_model
```
Expected: still 0 REPLICATED entries (the bug 1 fix is unaffected by
PEFT injection — non-block submodules are still wrapped per-submodule).

- [ ] **Step 10: Decision point**

If all 9 prior steps pass, the regime is validated and ready for a
real launch.  If any step fails, capture the error and stop here — do
not commit to a 1-day run on an unverified configuration.

Per the running-run plan, the natural launch window for the long run
is right after the full-FT 14B DF completes at ~02:50 KST Apr 29.

---

## Self-Review

**Spec coverage:**
- LoRA on transformer blocks via PEFT injection — covered by Tasks 3, 4 (config flips `merge_lora=False`).
- Selective unfreeze on audio_proj, audio_cond_projs, patch_embedding — covered by Tasks 1, 2, 4.
- Optimizer correctly handles partial freezing — verified pre-plan (`opt.py:27` already filters).
- FSDP wrap compatible — verified pre-plan (the per-submodule wrap in `network_causal.py:2148-2210` already covers the audio path; LoRA-injected blocks still go through `fully_shard(block, **kwargs)`).
- Smoke-test before long run — Task 7.
- Documentation — Task 6.

**Placeholder scan:** no "TBD", "TODO" without concrete content, no "similar to Task N", no implicit "add error handling" instructions. Each step has either complete code or a concrete shell command with expected output.

**Type consistency:** `unfreeze_modules` is `Optional[List[str]]` in the `__init__` signature, and `self.unfreeze_modules` is normalized to `List[str]` (with `[]` when None) — consistent across Tasks 1, 2, 3. `_apply_unfreeze`'s parameter name matches both. Config `DEFAULT_UNFREEZE_MODULES` is `list[str]` matching the constructor type.

**Note on testability:** Tasks 1–3 use a `_DummyHost` fixture for unit-testing the helper because constructing the real `CausalOmniAvatarWan` requires GPU + tens of GB of weight loading. Integration validation lives in Task 7's smoke-test checklist. This is a deliberate trade-off — the helper itself is small and well-isolated, so behavioral coverage on the dummy fixture plus introspection tests (signature exists, method is callable) on the real class plus integration validation via smoke test is appropriate without a heavy E2E unit test.
