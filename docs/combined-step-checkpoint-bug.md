# Combined Fake_Score + Student Step: Gradient Checkpointing Bug

## Goal

Match the original Self-Forcing training loop where the critic (fake_score) updates **every** step, including on the student (generator) step. This gives a true 1:5 ratio (5 critic updates per student update in a 5-step cycle).

FastGen's native DMD2 training loop uses an **exclusive** if/else: either student OR fake_score per step, never both. With `student_update_freq=5`, this gives 1:4 (4 critic, 1 student per cycle).

## What Was Tried

### Approach: Override `single_train_step` in `omniavatar_self_forcing.py`

On student steps (iter % 5 == 0), run both updates sequentially:

```python
def single_train_step(self, data, iteration):
    if iteration % self.config.student_update_freq != 0:
        return super().single_train_step(data, iteration)  # fake_score only

    # Step 1: Fake score update (manual backward + step)
    self.net.requires_grad_(False)
    self.fake_score.train().requires_grad_(True)
    real_data, condition, neg_condition = self._prepare_training_data(data)
    input_student, t_student, t, eps = self._generate_noise_and_time(real_data)
    fake_loss_map, _ = self._fake_score_discriminator_update_step(...)
    self.fake_score_optimizer.zero_grad()
    fake_loss_map["total_loss"].backward()
    clip_grad_norm_fsdp(self.fake_score.parameters(), max_norm=10.0)
    self.fake_score_optimizer.step()

    # Step 2: Student update (returned for trainer's backward)
    self.net.clear_caches()
    self.fake_score.eval().requires_grad_(False)
    self.net.train().requires_grad_(True)
    input_student, t_student, t, eps = self._generate_noise_and_time(real_data)
    student_loss_map, student_outputs = self._student_update_step(...)
    student_loss_map["fake_score_loss"] = fake_score_loss_val
    return student_loss_map, student_outputs
```

### Result: `CheckpointError: 94 tensors saved during forward, 80 during recomputation`

Every attempt crashed at the student update's backward pass with this exact tensor count mismatch (94 vs 80, always 14 tensor difference).

## Root Cause Analysis

### The Gradient Checkpointing + CrossAttention Cache Problem

The `CausalOmniAvatarWan` network has a **CrossAttention module** with a mutable `is_init` cache:

```python
# network_causal.py, CrossAttention.forward()
if crossattn_cache is not None:
    if not crossattn_cache["is_init"]:      # First call: compute K,V
        crossattn_cache["is_init"] = True   # ← MUTATES flag
        k = self.norm_k(self.k(context))    # ← saves ~5 tensors
        v = self.v(context)
        crossattn_cache["k"] = k
        crossattn_cache["v"] = v
    else:                                    # Subsequent calls: read cache
        k = crossattn_cache["k"]            # ← skips computation, fewer tensors
        v = crossattn_cache["v"]
```

With **gradient checkpointing** (`torch.utils.checkpoint`), PyTorch runs the forward pass once (saving tensor count), then re-runs during backward for recomputation. If `is_init` changed between forward and recomputation, different code paths execute → different tensor counts → `CheckpointError`.

### Why It Works Without the Combined Step

In the **exclusive** pattern (1:4 ratio), the student step runs alone. The `rollout_with_gradient` in `self_forcing.py` does:

1. For each chunk: denoise (no_grad steps) → exit step (with grad, checkpointed) → cache update (no_grad, `store_kv=True`)
2. The `store_kv=True` call sets `crossattn_cache["is_init"] = True`
3. The next chunk's exit step sees `is_init=True` at entry

We have a fix in `_forward_ar` that:
- Snapshots `is_init` before each checkpointed block
- Restores it in the checkpoint wrapper for recomputation

This works because within a single student step, the cache state is predictable.

### Why It Breaks With the Combined Step

The **combined** step runs two rollouts sequentially:

1. **Fake score rollout** (Step 1): `_fake_score_discriminator_update_step` → `gen_data_from_net` → `rollout_with_gradient(enable_gradient=False)` → runs full rollout under `no_grad`, populates all KV and crossattn caches, sets `is_init=True` across all blocks
2. **`self.net.clear_caches()`** — resets caches to `None`
3. **Student rollout** (Step 2): `_student_update_step` → `gen_data_from_net` → `rollout_with_gradient(enable_gradient=True)` → re-allocates caches, runs with gradient checkpointing

The problem: even after `clear_caches()`, the student's rollout re-creates the caches fresh (`is_init=False`). But the rollout's internal structure creates a subtle interaction:

- Chunk 0 exit step (with grad): `is_init=False` → computes K,V (94 tensors)
- Chunk 0 post-exit `store_kv=True` (no_grad): sets `is_init=True`
- Chunk 1 exit step (with grad): our `_forward_ar` entry resets `is_init=False` (the fix we added)
- But during backward, PyTorch recomputes Chunk 0's exit step. The checkpoint wrapper restores `is_init=False` (our per-block fix). However, some other state has changed between forward and recompute.

The 14-tensor difference (94-80) suggests ~3 blocks worth of crossattn K,V computations are being skipped during recomputation. This indicates that despite our fixes, some cache state (possibly the actual K,V tensors stored in the cache dict, or the KV cache's `global_end_index`/`local_end_index`) is mutated in a way that affects the recomputation path.

### The Original FastGen Wan Causal Network Avoids This

The original `fastgen/networks/Wan/network_causal.py` handles this differently:
- Uses `functools.partial` to freeze cache offsets before checkpointing
- Creates **isolated cache snapshots** with immutable length values
- The crossattn cache is managed differently (no `is_init` pattern)

The OmniAvatar causal network introduced the `is_init` pattern which is fundamentally incompatible with gradient checkpointing across multiple rollout chunks.

## What Was Tried and Why It Failed

| Attempt | What | Result |
|---------|------|--------|
| 1. `clear_caches()` between steps | Reset all caches to None before student rollout | Same error (94 vs 80) — caches re-created but checkpoint still sees mutation |
| 2. Reset `is_init=False` at `_forward_ar` entry | Force fresh K,V computation when checkpointing active | Same error — the mutation happens during rollout between chunks, not between steps |
| 3. Disable gradient checkpointing on student step | Skip checkpointing entirely during combined step | OOM at 138 GB — bs=8 requires checkpointing for the 7-chunk rollout |
| 4. Per-block `is_init` snapshot in checkpoint wrapper | Save/restore `is_init` around each checkpointed block call | Works for exclusive steps but not combined — the inter-chunk `store_kv` creates additional state that the snapshot doesn't capture |

## Proper Fix (Not Yet Implemented)

To truly fix this, the OmniAvatar causal network needs to adopt the original Wan causal network's approach:

1. **Remove the `is_init` pattern** from CrossAttention — always compute K,V fresh, or cache them outside the checkpointed region
2. **Use `functools.partial`** to freeze all mutable cache state before checkpointing
3. **Create isolated cache snapshots** with immutable metadata, following the pattern at `fastgen/networks/Wan/network_causal.py:865-913`

This is a significant refactor of `network_causal.py`'s attention and cache management — roughly 100-200 lines of changes across `OmniAvatarWanAttention`, `CrossAttention`, and `_forward_ar`.

## Current State

Using FastGen's native exclusive pattern with `student_update_freq=5` (1:4 ratio). This works correctly and has been verified end-to-end with training, validation, checkpointing, and wandb logging.

## Files Involved

- `fastgen/methods/omniavatar_self_forcing.py` — the `single_train_step` override (reverted)
- `fastgen/networks/OmniAvatar/network_causal.py` — `_forward_ar`, `CrossAttention`, `clear_caches`, `is_init` handling
- `fastgen/methods/distribution_matching/self_forcing.py` — `rollout_with_gradient` (shared, not modified)
- `fastgen/methods/distribution_matching/dmd2.py` — `single_train_step`, `_setup_grad_requirements`
- `/home/work/.local/reference/Self-Forcing/trainer/gan.py:356-398` — original SF training loop for reference
