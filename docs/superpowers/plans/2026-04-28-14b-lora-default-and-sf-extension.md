# 14B LoRA Default + SF Extension Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish LoRA + selective-unfreeze as the standard regime for any 14B-trained component (DF student, future SF student, future SF fake_score), apply it to a new 14B SF training config, and modify the FSDP checkpointer so saves are storage-efficient by default for any partial-freeze setup.

**Context:** A 14B LoRA t769 DF run is currently in flight (started 2026-04-28 19:22 KST, expected ~36 h, ETA ~08:00 KST 2026-04-30). This plan captures the work to be done DURING the run (CPU-only — code changes, configs, doc, smoke-test infrastructure) and AFTER the run (post-hoc checkpoint conversion, launching the 14B SF run).

**Tech Stack:** PyTorch 2.x, PyTorch FSDP2, PEFT 0.18.1, OmegaConf/Hydra, attrs configs, pytest.

**Branch:** `feat/redmd-sync-c`

---

## Background — what's already been done today

- **Bug 1 fix** (commit `825237d`): per-submodule `fully_shard` wraps inside `CausalOmniAvatarWan.fully_shard` so non-block params (audio_proj, audio_cond_projs, patch_embedding, embeddings, head) get grad-sync under FSDP. Closes the silent ~4% rank drift for the causal student in any FSDP-trained run.
- **Bug 2 fix** (commit `0677584`): `apply_lora_freeze` recovery hook on `CausalOmniAvatarWan` + override of `OmniAvatarDiffusionForcingModel.build_model` and `init_optimizers` so the PEFT freeze survives `FastGenModel.build_model:260`'s `self.net.train().requires_grad_(True)` wipe. Closes the silent full-FT-when-LoRA-was-intended bug on the DF causal student.
- **14B LoRA DF configs** committed: `config_df_shift_5_14b_lora.py`, `config_df_shift_5_14b_lora_t769.py`, with matching wrappers.
- **CPU diagnostic** (`scripts/diagnostics/diag_lora_freeze.py`): reproduces the wipe + recovery cycle without needing GPUs.

These already cover the **DF** path. The work in this plan extends the same coverage to **SF** and to the checkpointer.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `fastgen/networks/OmniAvatar/network.py` | Modify | Add `apply_lora_freeze` to `OmniAvatarWan` (bidirectional). Symmetry with `CausalOmniAvatarWan`; defensive even though `OmniAvatarWan` instances aren't currently subject to the `model.py:260` wipe. |
| `fastgen/methods/omniavatar_self_forcing.py` | Modify | Override `build_model` and `init_optimizers` in `OmniAvatarSelfForcingModel` to call `self.net.apply_lora_freeze()` after super (mirrors `OmniAvatarDiffusionForcingModel`). Optionally also `self.fake_score.apply_lora_freeze()` if available. |
| `fastgen/utils/checkpointer.py` | Modify | In `FSDPCheckpointer.save`, filter `model.state_dict()` to `{k:v for k,v in sd.items() if v.requires_grad}` before `dcp.save`. Set `strict=False` in load `StateDictOptions`. Generic — applies to any partial-freeze training. |
| `fastgen/configs/experiments/OmniAvatar/config_sf_14b_lora_t769.py` | Create | 14B SF config: student (CausalOmniAvatarWan) + fake_score (OmniAvatarWan) both `model_size="14B"` + `merge_lora=False` + `unfreeze_modules=[...]` + LoRA hyperparams. Teacher stays 14B (frozen). t_list=[0.999,0.769,0.0]. BS=1 GA=4. |
| `scripts/train_sf_..._14b_lora_t769.sh` | Create | Wrapper mirroring `..._fsmatched_t769_fsdpfix.sh` but pointing CONFIG_PATH at the new SF config and adjusting batch + FSDP. |
| `tests/test_causal_omniavatar_unfreeze.py` | Modify | Add tests for `OmniAvatarWan.apply_lora_freeze` behavior (mirroring causal-class tests). |
| `tests/test_checkpointer_filter.py` | Create | Test that `FSDPCheckpointer.save` correctly filters to `requires_grad=True` keys when `merge_lora=False` is in effect. |
| `scripts/post_hoc_convert_lora_saves.py` | Create | One-off script: for each saved LoRA checkpoint dir, load via `build_model` + ckpt, save filtered state via the new (or local) trainable-only path, replace the original. ~5 min runtime for 10 saves. |
| `docs/lora_selective_unfreeze.md` | Modify | Update with: 14B-LoRA-default convention, SF support note, post-hoc conversion script reference. |
| `CLAUDE.md` | Modify | Document the convention: "14B trained components default to merge_lora=False + unfreeze_modules; full-FT 14B is reserved for ablation." |

---

## Task 1: `apply_lora_freeze` on `OmniAvatarWan` (bidirectional)

**Files:**
- Modify: `fastgen/networks/OmniAvatar/network.py`
- Test: `tests/test_causal_omniavatar_unfreeze.py`

Mirrors the implementation already on `CausalOmniAvatarWan`. The bidirectional class isn't currently subject to the `model.py:260` wipe (that only affects `self.net`, not `self.fake_score`), but adding the method ensures defensive coverage and lets us call it uniformly from the SF method class regardless of where the network is bound.

- [ ] **Step 1: Confirm `OmniAvatarWan` has the same `unfreeze_modules` storage shape as `CausalOmniAvatarWan`**

Run: `grep -nE "unfreeze_modules|merge_lora" fastgen/networks/OmniAvatar/network.py | head -10`

If `unfreeze_modules` kwarg + `self.unfreeze_modules` storage exists: skip ahead. If not: add them mirroring `CausalOmniAvatarWan.__init__` (commit `30250f0`'s pattern).

- [ ] **Step 2: Add `_apply_unfreeze` and `apply_lora_freeze` methods**

Copy the implementation from `network_causal.py` lines `_apply_unfreeze` and `apply_lora_freeze`, paste into `OmniAvatarWan` (place above `_load_weights` or wherever consistent with the class's existing method order). The bodies are identical.

- [ ] **Step 3: Add a unit test**

Append to `tests/test_causal_omniavatar_unfreeze.py`:

```python
def test_omniavatarwan_apply_lora_freeze_is_callable():
    from fastgen.networks.OmniAvatar.network import OmniAvatarWan
    assert callable(OmniAvatarWan.apply_lora_freeze)
```

- [ ] **Step 4: Run tests**

Run: `/home/work/.local/miniconda3/envs/hb_fastgen/bin/pytest tests/test_causal_omniavatar_unfreeze.py -v`

Expected: 13 tests pass (12 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/OmniAvatar/network.py tests/test_causal_omniavatar_unfreeze.py
git commit -m "feat(omniavatar): apply_lora_freeze on bidirectional OmniAvatarWan

Mirrors the existing implementation on CausalOmniAvatarWan. Lets the
SF method class call apply_lora_freeze uniformly on both student (causal)
and fake_score (bidirectional) without dispatching on type. Defensive
coverage even though OmniAvatarWan isn't currently subject to the
model.py:260 wipe.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: SF method class freeze recovery overrides

**Files:**
- Modify: `fastgen/methods/omniavatar_self_forcing.py`

`OmniAvatarSelfForcingModel.build_model` already overrides to instantiate `fake_score` from a separate config and to load the VAE. We extend it to call `apply_lora_freeze` after super.

- [ ] **Step 1: Read the current `OmniAvatarSelfForcingModel.build_model`**

Run: `sed -n '130,160p' fastgen/methods/omniavatar_self_forcing.py`

- [ ] **Step 2: Add `apply_lora_freeze` calls after super, plus override `init_optimizers`**

Append after the existing `super().build_model()` line (and after the fake_score re-instantiation block):

```python
        # Restore PEFT-applied freeze after FastGenModel.build_model:260's
        # `self.net.train().requires_grad_(True)` wipe.  Same recovery hook
        # used in OmniAvatarDiffusionForcingModel.  Defensive on fake_score
        # too — it's not subject to the wipe (which only touches self.net),
        # but if a future config sets merge_lora=False on fake_score with
        # selective unfreeze, this catches drift from any later mutation.
        if hasattr(self.net, "apply_lora_freeze"):
            self.net.apply_lora_freeze()
        if hasattr(self, "fake_score") and hasattr(self.fake_score, "apply_lora_freeze"):
            self.fake_score.apply_lora_freeze()
```

Then add a sibling override:

```python
    def init_optimizers(self):
        """Defensive LoRA freeze re-apply right before optimizer construction."""
        if hasattr(self.net, "apply_lora_freeze"):
            self.net.apply_lora_freeze()
        if hasattr(self, "fake_score") and hasattr(self.fake_score, "apply_lora_freeze"):
            self.fake_score.apply_lora_freeze()
        super().init_optimizers()
```

- [ ] **Step 3: Run existing tests to make sure nothing broke**

Run: `/home/work/.local/miniconda3/envs/hb_fastgen/bin/pytest tests/ -v -k "omniavatar"`

- [ ] **Step 4: Commit**

```bash
git add fastgen/methods/omniavatar_self_forcing.py
git commit -m "feat(sf): apply_lora_freeze recovery in OmniAvatarSelfForcingModel

Restore the LoRA freeze on net (and defensively on fake_score) after
super().build_model() and right before optimizer construction. Mirrors
the recovery already in OmniAvatarDiffusionForcingModel for DF runs.

Past 1.3B SF runs were unaffected because (a) the SF student uses
merge_lora=True (full FT, no freeze to wipe) and (b) fake_score uses
merge_lora=False but the wipe at model.py:260 only touches self.net,
not self.fake_score. This change is preparation for 14B SF where the
student switches to merge_lora=False + selective unfreeze.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Modify the FSDP checkpointer to save trainable-only

**Files:**
- Modify: `fastgen/utils/checkpointer.py`
- Test: `tests/test_checkpointer_filter.py` (new)

This is Route B from the original plan. Generic across all training methods.

- [ ] **Step 1: Read the current save path**

Run: `sed -n '300,330p' fastgen/utils/checkpointer.py`

- [ ] **Step 2: Filter the state dict before `dcp.save`**

Locate the loop in `FSDPCheckpointer.save` that does `model_state_dict = ModelWrapper(model=v).state_dict()` followed by `dcp.save(model_state_dict, storage_writer=...)`. Replace with:

```python
        for k, v in model_dict.items():
            model_state_dict = ModelWrapper(model=v).state_dict()
            # Filter to trainable-only when partial-freeze is in effect.
            # No-op for full-FT (where every param has requires_grad=True);
            # collapses ~14B base + ~620M LoRA save to just the LoRA + unfreeze
            # portion (~5 GB instead of 56 GB) for 14B LoRA runs.
            params_dict = dict(v.named_parameters())
            filtered = {
                key: tensor for key, tensor in model_state_dict.items()
                if key not in params_dict or params_dict[key].requires_grad
            }
            storage_writer = self.get_storage_writer(checkpoint_path=f"{path}.{k}_model")
            dcp.save(filtered, storage_writer=storage_writer)
```

The `key not in params_dict` clause keeps non-parameter state (e.g., buffers like RoPE freqs, batchnorm running stats) — only params are filtered.

- [ ] **Step 3: Set `strict=False` in load options**

Locate `ModelWrapper.__init__` (around line 210). Change the default options to use `strict=False`:

```python
class ModelWrapper(Stateful):
    def __init__(self, model: torch.nn.Module, options: StateDictOptions | None = None):
        self.model = model
        if options is None:
            options = StateDictOptions(strict=False)
        elif options.strict:
            # Caller passed strict=True; warn but honor it.
            logger.warning(
                "ModelWrapper received options with strict=True; trainable-only "
                "saves will fail to load. Consider strict=False unless you have "
                "a reason."
            )
        self.options = options
```

- [ ] **Step 4: Write the test**

Create `tests/test_checkpointer_filter.py`:

```python
"""Tests for FSDPCheckpointer's trainable-only state-dict filter."""
import torch
import torch.nn as nn

from fastgen.utils.checkpointer import ModelWrapper


def test_state_dict_filters_to_trainable():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    # Freeze the second layer
    for p in model[1].parameters():
        p.requires_grad_(False)

    sd = ModelWrapper(model).state_dict()
    # ModelWrapper itself doesn't filter — that happens in
    # FSDPCheckpointer.save's local logic. We test the filter logic directly.
    params_dict = dict(model.named_parameters())
    filtered = {
        k: v for k, v in sd.items()
        if k not in params_dict or params_dict[k].requires_grad
    }

    # Layer 0 params present, layer 1 params absent
    assert "0.weight" in filtered
    assert "0.bias" in filtered
    assert "1.weight" not in filtered
    assert "1.bias" not in filtered


def test_filter_noop_when_all_trainable():
    """Full-FT case: filter is a no-op."""
    model = nn.Sequential(nn.Linear(4, 4))
    sd = ModelWrapper(model).state_dict()
    params_dict = dict(model.named_parameters())
    filtered = {
        k: v for k, v in sd.items()
        if k not in params_dict or params_dict[k].requires_grad
    }
    assert filtered.keys() == sd.keys()
```

- [ ] **Step 5: Run the new tests**

Run: `/home/work/.local/miniconda3/envs/hb_fastgen/bin/pytest tests/test_checkpointer_filter.py -v`

- [ ] **Step 6: Commit**

```bash
git add fastgen/utils/checkpointer.py tests/test_checkpointer_filter.py
git commit -m "feat(checkpointer): filter saved state_dict to trainable params

In FSDPCheckpointer.save, save only params with requires_grad=True
(buffers and other non-parameter state are preserved). Generic across
all training methods. For 14B LoRA + selective-unfreeze runs, this
collapses save size from 56 GB (full base) to ~5 GB (LoRA + unfreeze).
For full-FT runs, every param has requires_grad=True so the filter is
a no-op.

ModelWrapper now defaults to StateDictOptions(strict=False) so loads
of trainable-only saves silently leave frozen base params at their
construction-time values (loaded from base safetensors + V2V mouthweight
ckpt during build_model). Caller can still pass strict=True explicitly
if they want.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Create `config_sf_14b_lora_t769.py`

**Files:**
- Create: `fastgen/configs/experiments/OmniAvatar/config_sf_14b_lora_t769.py`

Inherits from `config_sf_sink1_window7_redmd_beta2_taew.py`, switches:

- Student (`config.model.net`): `model_size="14B"`, `merge_lora=False`, `unfreeze_modules=["_core.audio_proj", "_core.audio_cond_projs", "_core.patch_embedding"]`, `lora_rank=128`, `lora_alpha=64`
- Fake_score (`config.model.fake_score_net`): `model_size="14B"`, `merge_lora=False`, `unfreeze_modules=["_core.audio_proj", ...]` (same paths but inside `OmniAvatarWan`'s `_core` namespace — verify the dotted paths align)
- Teacher (`config.model.teacher`): stays 14B, `merge_lora=True` (frozen, full)
- Batch / accum: `config.dataloader_train.batch_size = 1`, `config.trainer.grad_accum_rounds = 4` → effective batch 16
- Schedule: `config.model.sample_t_cfg.t_list = [0.999, 0.769, 0.0]`, `config.model.student_sample_steps = 2`
- FSDP knobs: same as the 14B DF LoRA config (`fsdp_min_num_params=1e8`, `fsdp_meta_init=False`, `precision=bfloat16`, `precision_fsdp=float32`)

- [ ] **Step 1: Verify path alignment between causal and bidirectional `_core`**

`CausalOmniAvatarWan._core` has `audio_proj`, `audio_cond_projs`, `patch_embedding`. `OmniAvatarWan` uses `self.model` (not `self._core`) per `network.py:395`. So the unfreeze paths for fake_score should be `model.audio_proj`, `model.audio_cond_projs`, `model.patch_embedding` — different prefix.

Verify by:

```bash
/home/work/.local/miniconda3/envs/hb_fastgen/bin/python -c "
import sys; sys.path.insert(0, '/home/work/.local/hyunbin/FastGen-redmd')
from fastgen.networks.OmniAvatar.network import OmniAvatarWan
m = OmniAvatarWan(model_size='1.3B', in_dim=49, mode='v2v', use_audio=True, base_model_paths=None, omniavatar_ckpt_path=None, merge_lora=False)
for n, _ in m.named_modules():
    if 'audio' in n or 'patch_embedding' in n:
        print(n)
" 2>&1 | head -20
```

Confirm the actual prefix used (likely `model.audio_proj` etc.) and use that in the SF config's fake_score `unfreeze_modules`.

- [ ] **Step 2: Write the config file**

Following the pattern of `config_df_shift_5_14b_lora_t769.py`. Inherit from `config_sf_sink1_window7_redmd_beta2_taew.py`. Override the specific fields.

(Code is too long to paste here but the implementer should follow the same structure; ~100 lines.)

- [ ] **Step 3: Verify the config loads**

Run:

```bash
/home/work/.local/miniconda3/envs/hb_fastgen/bin/python -c "
import sys; sys.path.insert(0, '/home/work/.local/hyunbin/FastGen-redmd')
from fastgen.configs.experiments.OmniAvatar.config_sf_14b_lora_t769 import config
assert config.model.net.model_size == '14B'
assert config.model.net.merge_lora is False
assert config.model.fake_score_net.model_size == '14B'
assert config.model.fake_score_net.merge_lora is False
assert config.model.sample_t_cfg.t_list == [0.999, 0.769, 0.0]
assert config.dataloader_train.batch_size == 1
assert config.trainer.grad_accum_rounds == 4
print('config_sf_14b_lora_t769: OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add fastgen/configs/experiments/OmniAvatar/config_sf_14b_lora_t769.py
git commit -m "feat(config): 14B SF with LoRA student + LoRA fake_score, t769 schedule

Both student (causal) and fake_score (bidirectional) at 14B with
merge_lora=False + selective unfreeze on the audio path + patch
embedding. Teacher stays 14B frozen.  BS=1 GA=4 -> effective batch 16
(matches the DF run's effective batch).

This is the first 14B SF config; full-FT 14B SF would not fit memory
(LoRA on both networks is mandatory).  Pairs with
train_sf_..._14b_lora_t769.sh wrapper.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Create the 14B SF wrapper

**Files:**
- Create: `scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_t769_14b_lora.sh`

Mirrors `..._fsmatched_t769_fsdpfix.sh` but:

- `CONFIG_PATH` env override to point at `config_sf_14b_lora_t769.py` (or override the underlying config file in EXTRA_OVERRIDES)
- `BATCH_SIZE=1`, GA=4 (set via EXTRA_OVERRIDES)
- `EXTRA_OVERRIDES` includes FSDP enabling
- `OMNIAVATAR_DF_CKPT` points at the 14B LoRA DF run's eventual checkpoint (placeholder until the run finishes; default path: `FASTGEN_OUTPUT/.../df_audiofix_syncnet_trained_shift_5_14b_lora_t769_4gpu_bs4_grad1_eff16_lr1e5_5000iter/checkpoints/0005000.pth`)
- `RUN_NAME` with `_14b_lora` suffix

- [ ] **Steps 1–5**: Mirror the structure of `..._fsmatched_t769_fsdpfix.sh` with the substitutions above. Make executable, shell-syntax check, dry-run env arithmetic, commit.

---

## Task 6: Smoke test 14B SF

**Files:** none — runs the wrapper.

- [ ] **Step 1: Wait for 14B DF LoRA run to complete or have a usable checkpoint at iter ≥1500.**
- [ ] **Step 2: Launch 50-iter smoke** with the standard checklist:
  - [merge_lora] log line for **both** net and fake_score should NOT appear
  - 2× `[CausalOmniAvatarWan] PEFT LoRA: N loaded` (or 1× for student + 1× for fake_score using whatever class)
  - `apply_lora_freeze` log fires for both net and fake_score
  - param_count: trainable << total for both
  - FSDP wrap completes for all 3 networks (net, fake_score, teacher)
  - First iter finite loss (with the SF rollout / fake_score updates this is more complex than DF)
  - Save format: `*.net_optim` AND `*.fake_score_optim` should both be small (~1 GB each)
- [ ] **Step 3: Decision point.** If smoke green, full launch. If issues, escalate.

---

## Task 7: Post-hoc convert the running 14B DF LoRA run's saves

**Files:**
- Create: `scripts/post_hoc_convert_lora_saves.py`

After the running 14B DF LoRA run finishes (~08:00 KST 2026-04-30), convert all 10 saved `.net_model` dirs from full-state to trainable-only format.

- [ ] **Step 1: Write the script**

Loads each saved checkpoint via the standard `build_model` + checkpointer path, then re-saves using the modified `FSDPCheckpointer.save` (which now filters to trainable). Replaces the original.

Pseudocode:
```python
for step in [500, 1000, 1500, ..., 5000]:
    src = f"{run_dir}/checkpoints/{step:07d}"
    # Load
    model = build_model(config)
    checkpointer.load(model, path=f"{src}.pth")
    # Save filtered
    tmp = f"{src}.net_model_stripped"
    checkpointer.save_one_network(model.net, path=tmp)
    # Replace
    os.rename(f"{src}.net_model", f"{src}.net_model_FULL_BACKUP")
    os.rename(tmp, f"{src}.net_model")
    os.rename(f"{src}.net_model_FULL_BACKUP", trash)  # or rm -rf
```

- [ ] **Step 2: Run on the saves**

~5 min runtime per save, ~50 min total. Frees ~510 GB.

- [ ] **Step 3: Verify each save loads correctly via the standard inference path** (resume / inference smoke).

- [ ] **Step 4: Commit the script**.

---

## Task 8: Update docs

**Files:**
- Modify: `docs/lora_selective_unfreeze.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: docs/lora_selective_unfreeze.md** — add sections:
  - "SF support" — note that `OmniAvatarSelfForcingModel.build_model` calls `apply_lora_freeze` on net + fake_score
  - "Storage-efficient saves" — explain the filter in `FSDPCheckpointer.save`
  - "Post-hoc conversion" — point at `scripts/post_hoc_convert_lora_saves.py`

- [ ] **Step 2: CLAUDE.md** — add a `## Convention: 14B = LoRA default` section:
  - Any 14B trained component should use `merge_lora=False + unfreeze_modules`
  - Reference `config_df_shift_5_14b_lora.py` and `config_sf_14b_lora_t769.py` as canonical examples
  - Note that full-FT 14B is reserved for ablations

- [ ] **Step 3: Commit.**

---

## Self-Review

**Spec coverage:**
- 14B SF support: Tasks 4, 5, 6.
- Storage efficiency for 14B LoRA runs: Task 3, plus post-hoc Task 7.
- SF freeze recovery: Tasks 1, 2.
- Documentation: Task 8.

**Placeholder scan:** No "TBD" or "implement later". Code in Task 3 fully specified. Task 5 references the existing pattern explicitly.

**Type consistency:** `unfreeze_modules` paths differ between `CausalOmniAvatarWan` (rooted at `_core.X`) and `OmniAvatarWan` (rooted at `model.X`). Task 4 Step 1 explicitly verifies this before writing the config.

**Note on order**: Tasks 1–3 are pure code/test changes that can run during the 14B DF LoRA run. Task 4–5 are config/wrapper writes (no GPU). Task 6 needs free GPUs (after 14B DF finishes). Task 7 needs free GPUs and Task 3 done. Task 8 can land anytime.
