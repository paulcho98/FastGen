# Reward clamp + centering follow-up (post per-sample-coupling fix)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement. Steps use checkbox (`- [ ]`) syntax.

**Status:** Not yet scheduled. Written 2026-04-19 immediately after the per-sample reward-loss coupling fix landed on `feat/redmd-sync-c`. This plan covers the remaining Re-DMD reward-path corrections that were deferred for future work.

**Context:** The previously-landed fix changed the reward-weighted loss from

```
L = mean_i(exp(beta·r_i)) · mean_i(L_i)        # broken: batch-mean collapse
```

to

```
L = mean_i(exp(beta·r_i) · L_i)                # fixed: per-sample coupling
```

by having `variational_score_distillation_loss` support `reduction="none"` and having `_apply_reward_weighting` accept a per-sample `[B]`-shaped `vsd_loss`. That fix alone is algorithmically complete for **uncentered, unclamped** reward weighting. Two optional knobs in the same codepath were also found to be either misleading or subtly broken under the new regime:

1. `config.center_reward = True` — was originally a "scale normalizer" disguised as baseline subtraction; only becomes a meaningful advantage baseline under per-sample coupling, AND its EMA is computed on a per-rank local batch mean (not the global mean).
2. `config.clamp_reward = (lo, hi)` — matters *more* under per-sample coupling than it did under the old batch-mean formulation, because a single high-reward outlier can now dominate `mean_i(exp(β·r_i) · L_i)`.

Both knobs are currently defaulted OFF (`center_reward=False`, `clamp_reward=None`) in every production config, so nothing is actively misbehaving. This plan lands the fixes so the knobs are honest/safe before anyone flips them.

**Goal:** Make `clamp_reward` the recommended safety rail under per-sample coupling, and make `center_reward` a correct cross-rank advantage baseline.

**Architecture:** Three targeted changes to `fastgen/methods/omniavatar_self_forcing_re_dmd.py` plus config-level guidance. No new files. Tests added to the existing `tests/reward/test_re_dmd_trainer.py`.

**Tech Stack:** PyTorch, `torch.distributed`, existing pytest reward suite.

---

## Background: why these interact with the per-sample-coupling fix

Grep confirmed (2026-04-19) that **neither centering nor clamping existed in the original Reward-Forcing repo** — they were only added in this port:

- Original: `model/re_dmd.py:190-204` computes `weight = torch.exp(beta * mq); rl_dmd_loss = 0.5 * weight * mse`. No preprocessing of `mq`. (`clamp` calls in that file are for pixels + timesteps, unrelated.)
- Original `docs/reward_forcing_implementation.md:262-271` lists centering/clamping/weight-normalization under "**Consider adding safety rails**" — explicit future-work suggestions, not implemented.
- The FastGen-redmd port (`_apply_reward_weighting`, introduced by commit in `feat/redmd-sync-c`) added both as opt-in config knobs, with defaults OFF.

So the interaction between these knobs and per-sample coupling is genuinely novel to this codebase; nothing to crib from upstream.

### Centering

Under the old (broken) batch-mean collapse:
- Uncentered: `mean(exp(β·r)) * mean(MSE)` — the rewards acted as a single global scale factor on the batch-mean loss. Scale drifted with the absolute reward level.
- Centered: `mean(exp(β·(r − r̄_EMA))) * mean(MSE) ≈ 1 * mean(MSE)` (Jensen's inequality, with small positive variance-penalty). **The reward signal was essentially annihilated** — a scale regularizer dressed up as a baseline subtraction.

Under the new per-sample coupling:
- Uncentered: `mean(exp(β·r_i) · L_i)` — correct RL-weighted regression form, but absolute loss scale still drifts with reward level.
- Centered: `mean(exp(β·(r_i − r̄_EMA)) · L_i)` — **now actually an advantage baseline**. Above-average samples get weight >1 (amplified), below-average get weight <1 (damped). Loss scale roughly preserved.

So centering becomes meaningful only under the fix. Before merging centering-on for any serious run, one latent bug has to go: the EMA update uses `sync_c.mean().item()`, which is the **per-rank local** batch mean (no `all_reduce`). With batch_size=8/rank × 4 GPUs, each rank ends up EMA-ing its own 8-sample slice and the running means drift apart. The fix is to `all_reduce` the batch sum before computing the mean used for the EMA update.

### Clamping

Clamping bounds `exp(β·r)` to `[exp(β·clamp_lo), exp(β·clamp_hi)]`. Under per-sample coupling it matters **more**: a single sample at sync_c=10 with β=2 produces weight ≈ 5×10⁸ — one term dominates `mean_i(w_i · L_i)` and washes out the other seven batch members. Under the old batch-mean formulation the mean-reduce already dampened this.

Practical recommendation: β=2 configs should default to something like `clamp_reward=(0.0, 8.0)` → per-sample weight capped at exp(16) ≈ 8.9M (still large, but bounded).

---

## File structure

Changes concentrate in two files; no new files:

- **`fastgen/methods/omniavatar_self_forcing_re_dmd.py`** — the `_apply_reward_weighting` method and the containing class.
  - Fix the EMA centering so it uses a cross-rank batch mean.
  - (No structural change for clamping — the existing per-sample `sync_c.clamp(...)` line is already correct under per-sample coupling. Only config-level guidance changes.)

- **`fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_beta2.py`** and siblings — update `clamp_reward` defaults + comments.
  - Set `clamp_reward = (0.0, 8.0)` in β=2 presets.
  - Leave β=0.25 preset with `clamp_reward = None` (low-β exponent dynamic range is already bounded).

- **`tests/reward/test_re_dmd_trainer.py`** — extend.
  - Add `test_centering_uses_cross_rank_mean` (single-rank smoke + distributed smoke via `torch.distributed` spawn).
  - Add `test_clamping_guards_outlier_under_per_sample_coupling`.

The file that holds today's centering code:

```python
# omniavatar_self_forcing_re_dmd.py, current
if getattr(self.config, "center_reward", False):
    ema_alpha = 0.9
    batch_mean = sync_c.mean().item()        # ← per-rank local mean (bug)
    if self._reward_running_mean is None:
        self._reward_running_mean = batch_mean
    else:
        self._reward_running_mean = (
            ema_alpha * self._reward_running_mean
            + (1.0 - ema_alpha) * batch_mean
        )
    sync_c = sync_c - self._reward_running_mean
```

---

## Tasks

### Task 1: Cross-rank batch mean for EMA centering

**Files:**
- Modify: `fastgen/methods/omniavatar_self_forcing_re_dmd.py` (inside `_apply_reward_weighting`, the `if center_reward:` block)
- Test: `tests/reward/test_re_dmd_trainer.py`

- [ ] **Step 1: Write the failing test (single-rank path)**

In the single-rank path, the EMA mean should equal the local batch mean (no cross-rank op is a no-op, which is correct behavior). This test guards against accidentally breaking the no-distributed-available path.

```python
def test_centering_uses_local_mean_without_distributed():
    import math, torch
    model = _make_model(beta=0.25, center=True)
    model.reward_scorer = _VaryingScorer([0.0, 2.0, 4.0, 6.0])  # mean = 3.0
    videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8) for _ in range(4)]
    audios = [torch.randn(51840) for _ in range(4)]
    vsd = torch.ones(4)

    _, log_map = model._apply_reward_weighting(vsd, videos, audios)
    # First call → EMA seeded with batch mean 3.0; post-centering sync_c = [-3,-1,1,3]
    # weight_mean = mean(exp(0.25 * [-3,-1,1,3]))
    expected = (math.exp(-0.75) + math.exp(-0.25) + math.exp(0.25) + math.exp(0.75)) / 4
    assert abs(log_map["reward_weight_mean"] - expected) < 1e-4
```

Run: `pytest tests/reward/test_re_dmd_trainer.py::test_centering_uses_local_mean_without_distributed -v`
Expected: PASS (current implementation already does this — this test is a regression guard).

- [ ] **Step 2: Write the failing test (cross-rank path)**

This one requires spawning a distributed group. Use `torch.multiprocessing.spawn` with a gloo backend (no CUDA required for the math check).

```python
def _centering_cross_rank_worker(rank, world_size, local_mean_per_rank, seed_out):
    import os, torch, torch.distributed as dist
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    from tests.reward.test_re_dmd_trainer import _make_model, _VaryingScorer
    import torch

    # Each rank gets a distinct local batch mean. Global mean = avg of the per-rank means.
    per_rank_sync_c = local_mean_per_rank[rank]  # a [B] tensor, e.g. [1.0, 3.0]
    model = _make_model(beta=0.25, center=True)
    model.reward_scorer = _VaryingScorer(per_rank_sync_c.tolist())
    videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8) for _ in per_rank_sync_c]
    audios = [torch.randn(51840) for _ in per_rank_sync_c]
    vsd = torch.ones(len(per_rank_sync_c))

    _, log_map = model._apply_reward_weighting(vsd, videos, audios)
    # Pass the rank's seeded running mean back out via a file / pipe
    seed_out.put((rank, model._reward_running_mean))
    dist.destroy_process_group()


def test_centering_uses_cross_rank_mean():
    import torch, torch.multiprocessing as mp
    # Rank 0 batch: [1, 3] → local mean 2; Rank 1 batch: [5, 7] → local mean 6
    # Global mean = 4.
    per_rank = [torch.tensor([1.0, 3.0]), torch.tensor([5.0, 7.0])]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_centering_cross_rank_worker, args=(r, 2, per_rank, q)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join(timeout=30)
    results = [q.get(timeout=5) for _ in range(2)]
    results.sort()  # by rank
    # Under the fix, both ranks should have EMA seeded with the GLOBAL mean (4.0), not their local (2 vs 6).
    assert abs(results[0][1] - 4.0) < 1e-4, f"rank 0 EMA = {results[0][1]}"
    assert abs(results[1][1] - 4.0) < 1e-4, f"rank 1 EMA = {results[1][1]}"
```

Run: `pytest tests/reward/test_re_dmd_trainer.py::test_centering_uses_cross_rank_mean -v`
Expected: FAIL with current code — rank 0 EMA will be 2.0, rank 1 will be 6.0.

- [ ] **Step 3: Implement the cross-rank fix**

In `_apply_reward_weighting`, replace the centering block:

```python
if getattr(self.config, "center_reward", False):
    ema_alpha = 0.9
    if dist.is_available() and dist.is_initialized():
        # All-reduce sum of local sync_c values, then divide by global count
        local_sum = sync_c.sum()
        local_count = torch.tensor(float(sync_c.numel()), device=sync_c.device)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        batch_mean = (local_sum / local_count).item()
    else:
        batch_mean = sync_c.mean().item()
    if self._reward_running_mean is None:
        self._reward_running_mean = batch_mean
    else:
        self._reward_running_mean = (
            ema_alpha * self._reward_running_mean
            + (1.0 - ema_alpha) * batch_mean
        )
    sync_c = sync_c - self._reward_running_mean
```

Notes:
- `local_count` is a float-valued 0-dim tensor so we avoid int-tensor all-reduce quirks on gloo.
- `dist.ReduceOp.SUM` + divide-by-count is exactly `ReduceOp.AVG`-equivalent and works on gloo (AVG is NCCL-only on older PyTorch).
- Place BEFORE `sync_c = sync_c - self._reward_running_mean` so the subtraction uses the fresh running mean.

- [ ] **Step 4: Run both tests**

Run: `pytest tests/reward/test_re_dmd_trainer.py::test_centering_uses_local_mean_without_distributed tests/reward/test_re_dmd_trainer.py::test_centering_uses_cross_rank_mean -v`
Expected: both PASS.

- [ ] **Step 5: Run the whole reward suite**

Run: `pytest tests/reward/ -v`
Expected: all 45 existing tests + 2 new = 47 PASS. No regressions.

- [ ] **Step 6: Commit**

```bash
git add fastgen/methods/omniavatar_self_forcing_re_dmd.py tests/reward/test_re_dmd_trainer.py
git commit -m "fix(redmd): use cross-rank batch mean for reward-centering EMA

Per-rank local mean caused each rank's running mean to drift independently
in multi-GPU training, breaking the 'mean weight ≈ 1' invariant that
centering is supposed to maintain. Now that per-sample coupling gives
centering a meaningful interpretation as advantage-baseline subtraction,
correct it to use a globally-reduced batch mean.

No effect on production runs: center_reward=False in all current configs."
```

---

### Task 2: Raise `clamp_reward` default for β=2 presets

**Files:**
- Modify: `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_beta2.py`
- Modify: `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_beta2_taew.py` (inherits, only override if needed)
- Test: `tests/reward/test_re_dmd_trainer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_clamp_prevents_single_sample_domination_under_per_sample_coupling():
    """With β=2, one high-reward outlier can dominate mean(w_i · L_i). Clamp to (0, 8) caps it."""
    import math, torch
    model = _make_model(beta=2.0, clamp=(0.0, 8.0))
    model.reward_scorer = _VaryingScorer([0.0, 0.0, 0.0, 20.0])  # one r=20 outlier
    videos = [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8) for _ in range(4)]
    audios = [torch.randn(51840) for _ in range(4)]
    vsd = torch.ones(4)

    weighted, log_map = model._apply_reward_weighting(vsd, videos, audios)
    # After clamp: sync_c = [0, 0, 0, 8]; weight = [1, 1, 1, exp(16)]
    # weighted = mean(1 + 1 + 1 + exp(16)) ≈ exp(16)/4 + 3/4
    expected = (1 + 1 + 1 + math.exp(16)) / 4
    assert abs(weighted.item() - expected) < 1e-2 * expected
    assert log_map["reward_sync_c_max"] == 8.0
```

Run: `pytest tests/reward/test_re_dmd_trainer.py::test_clamp_prevents_single_sample_domination_under_per_sample_coupling -v`
Expected: PASS (current clamp implementation already does this under the post-fix coupling — this test documents the guarantee).

- [ ] **Step 2: Update β=2 config defaults**

In `config_sf_sink1_window7_redmd_beta2.py`:

```python
# was: config.model.clamp_reward = None
config.model.clamp_reward = [0.0, 8.0]  # β=2 safety rail; see plan 2026-04-19-reward-clamp-centering-followup.md
```

Update the docstring comment at the top of the config to document the change.

The taew variant inherits from `_beta2_base` so it automatically picks up the new default. Verify no explicit override exists.

- [ ] **Step 3: Verify tests still pass**

Run: `pytest tests/reward/ -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_beta2.py tests/reward/test_re_dmd_trainer.py
git commit -m "feat(redmd): enable (0, 8) clamp_reward for β=2 configs

Per-sample reward-loss coupling means a single high-sync_c outlier can
dominate mean(exp(β·r_i)·L_i). At β=2, an unclamped r=10 produces weight
~5e8 — one sample washes out the other seven. (0, 8) caps per-sample
weight at exp(16)≈9M, still generous but bounded.

β=0.25 presets keep clamp=None; exp(β·r) dynamic range is already small."
```

---

## Non-goals (explicitly out of scope)

- Weight normalization (paper's Z(c) term): mentioned in original `docs/reward_forcing_implementation.md` as a third safety rail. Not implemented here; would be a follow-up after we have data on whether unclamped per-sample coupling causes training instability.
- A proper unit test of centering under per-sample coupling interacting with non-uniform per-sample loss. The centering tests above only assert the EMA running-mean value; the algebraic "centering + per-sample coupling ⇒ advantage baseline" claim is a consequence, not independently tested. If we ever want to gate a training run on the centering path, add a numerical assertion for it.

## Self-review checklist

- [x] Both knobs are currently OFF in production; this plan is pre-emptive, not reactive.
- [x] No changes to the reduction-kwarg API of `variational_score_distillation_loss` — that contract landed with the per-sample-coupling fix and is treated as stable.
- [x] Types/signatures consistent: `_apply_reward_weighting` still takes per-sample `[B]` vsd_loss, still returns `(scalar, dict)`. The centering change is internal.
- [x] Every task includes concrete code, failing test, expected PASS/FAIL, and a single commit. No placeholders.
- [x] Grep-verified that neither centering nor clamping existed in the original Reward-Forcing repo (so no upstream to match — these are safety rails introduced in this port).

## Execution handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — controller dispatches a fresh subagent per task with review checkpoints.
2. **Inline Execution** — batch execution with checkpoints in the same session.

**Which approach?** (Leave until the per-sample-coupling fix has run for a while in production and we have a concrete reason — e.g., we want to try a centered experiment, or we've observed outlier-driven training instability — before picking this up.)
