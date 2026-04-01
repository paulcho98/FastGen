# Session Summary — 2026-04-01: ODE Trajectory Extraction & Self-Forcing Debugging

## Overview

This session focused on two main goals:
1. Extracting full ODE trajectories from the OmniAvatar teacher for analysis
2. Getting Self-Forcing (Stage 2) training running on 4× H200 GPUs

Both tasks exposed critical bugs in the codebase that were diagnosed and fixed.

---

## 1. Full ODE Trajectory Extraction

### Goal
Extract all 50 timestep states (x_t and x0_pred) from the OmniAvatar teacher's ODE solve, for both the 14B and 1.3B models, on 10 validation samples.

### Scripts Created
- `scripts/generate_omniavatar_ode_pairs_full.py` — Main extraction script
- `scripts/run_ode_full_trajectory.sh` — Launch script (handles precomputation + extraction)
- `scripts/verify_ode_trajectory.py` — VAE-decodes final x0 for visual verification

### Key Design Decisions
- **Timestep shift = 5.0**: Matches OmniAvatar's inference scheduler (`pipe.scheduler.set_timesteps(N, shift=5.0)`)
- **Schedule formula**: `t_shifted = 5.0 * t / (1 + 4.0 * t)` applied to `linspace(0.999, 0, 51)`
- **Data**: Uses `vae_latents_mask_all.pt` (all frames masked including frame 0) per the inference script convention
- **65-channel input**: noise(16) + ref(16) + mask(1) + masked_video(16) + ref_sequence(16)

### Checkpoints Used
- **14B teacher**: `/home/work/output_omniavatar_v2v_phase2/step-10500.pt`
- **1.3B model**: `/home/work/output_omniavatar_v2v_1.3B_phase2/step-19500.pt`

### Bug Found: Negative Text Embedding
**Symptom**: Mouth region completely disfigured in generated output.

**Root cause**: The negative text embedding for CFG was `torch.zeros(1, 512, 4096)` instead of the proper T5 encoding of the empty string `""`. The T5 encoding has range [-0.75, 0.50] — not zero. With guidance_scale=4.5, this difference is amplified massively, corrupting the CFG direction specifically in the generation region (mouth/mask area).

**Fix**: Generated proper negative text embedding via `pipe.encode_prompt('', positive=False)` and saved to `/home/work/stableavatar_data/neg_text_emb.pt`. The extraction script's `--neg_text_emb_path` argument passes this, with a warning if not provided.

**Note**: This was NOT a bug in OmniAvatar's own code (which always encodes `""` through T5). It was in our extraction script's default of using zeros when no neg_text_emb_path was provided.

### Output
- `/home/work/ode_full_trajectories/1.3B/` — 10 samples × 102 files each
- `/home/work/ode_full_trajectories/14B/` — 10 samples × 102 files each
- `/home/work/ode_full_trajectories/{1.3B,14B}_verify/` — Decoded verification videos
- Per sample: `ode_schedule.json`, `input_latents.pt`, `step_{000..049}_{xt,x0}.pt`

### Precomputation
The `vae_latents_mask_all.pt` files were precomputed for the 10 recon samples using the existing OmniAvatar script `precompute_vae_latents_masked.py`. This applies the LatentSync spatial mask to frame 0 in pixel space, VAE-encodes it, and saves alongside the already-masked frames 1+.

---

## 2. Self-Forcing (Stage 2) Training Setup

### Config Updates (`config_sf.py`)
- **Shift**: Changed from 3.0 to **5.0** (matches OmniAvatar inference)
- **t_list**: Updated to `[0.999, 0.937, 0.833, 0.624, 0.0]` (shift=5.0 derived)
- **Teacher checkpoint**: Phase2 14B `step-10500.pt`
- **Student checkpoint**: Phase2 1.3B `step-19500.pt`
- **DF pretrained checkpoint**: `FASTGEN_OUTPUT/.../checkpoints/0005000.pth` (DF shift=5 at 5000 steps)
- **neg_text_emb_path**: `/home/work/stableavatar_data/neg_text_emb.pt`

### Test config (`config_sf_test.py`)
- Updated paths to local machine
- Uses recon validation data
- 11 iterations (covers 2 student updates at freq=5)
- FSDP enabled, wandb disabled

---

## 3. Bugs Found and Fixed During SF Testing

### Bug 1: Teacher OOM — Missing `torch.no_grad()` (Pre-existing in FastGen)

**File**: `fastgen/methods/distribution_matching/dmd2.py`, `_compute_teacher_prediction_gan_loss()`

**Symptom**: 14B teacher forward consumed 138 GB/GPU, OOM on H200 140GB.

**Root cause**: When GAN is disabled (`gan_loss_weight_gen=0`), the `else` branch at line 156 calls `self.teacher(perturbed_data, ...)` WITHOUT `torch.no_grad()`. Since `perturbed_data` has gradients (from student rollout), PyTorch builds a full autograd graph through all 40 teacher layers, storing ~4 GB of activations per layer (40 × 4 GB = 160 GB).

The `if` branch (GAN enabled) intentionally lacks `no_grad` because `fake_feat` must flow through the discriminator. The `else` branch has no such need — `teacher_x0.detach()` is called at line 164, but this only prevents backward, not forward graph construction.

**Fix**: Wrapped the `else` branch in `with torch.no_grad():`.

**Impact**: Peak memory dropped from 138 GB to **23.6 GB** per GPU.

**Why not caught before**: Original FastGen configs always use GAN (`gan_loss_weight_gen=0.003`), so the `else` branch was never exercised. With same-architecture 1.3B teacher, the memory waste was small enough to not OOM even without `no_grad`.

**Memory profile (per-layer logging)**:
```
Before fix: block 0=24GB, block 10=64GB, block 20=103GB, block 30=142GB → OOM
After fix:  block 0=20GB, block 10=20GB, block 20=20GB, block 39=20GB (flat!)
```

### Bug 2: Gradient Checkpointing Error — CrossAttention Cache Mutation (OmniAvatar-specific)

**File**: `fastgen/networks/OmniAvatar/network_causal.py`, `_forward_ar()`

**Symptom**: `CheckpointError: A different number of tensors was saved during forward (80) and recomputation (70)`.

**Root cause**: The `CrossAttention` module has an `is_init` flag in its cache dict. During the first forward call, it computes K,V projections and sets `is_init=True` (10 extra tensors from the linear ops). During gradient checkpointing recomputation, `is_init` is already True, so it reads cached K,V (skipping the projections) → 10 fewer tensors → mismatch.

The original FastGen Wan causal network avoids this by using `functools.partial` to freeze cache state and creating isolated cache snapshots before checkpointing.

**Fix**: Snapshot `is_init` before the checkpoint call and restore it in the checkpointed forward wrapper:
```python
saved_is_init = crossattn_cache_i["is_init"]
def make_ckpt_forward(module, cache_ref, init_flag):
    def fn(*inputs, **kw):
        if cache_ref is not None:
            cache_ref["is_init"] = init_flag
        return module(*inputs, **kw)
    return fn
```

### Bug 3: DTensor/Tensor Mix in Gradient Clipping (Pre-existing in FastGen)

**File**: `fastgen/callbacks/grad_clip.py`

**Symptom**: `RuntimeError: got mixed torch.Tensor and DTensor` during gradient clipping after student update.

**Root cause**: The grad clip callback uses `torch.nn.utils.clip_grad_norm_(..., foreach=True)` for non-CPU-offload FSDP. With FSDP2, sharded params have DTensor gradients, but non-sharded params (OmniAvatar audio modules) have regular Tensor gradients. `foreach=True` can't batch these.

The custom `clip_grad_norm_fsdp()` function already handles this mix correctly (extracting `._local_tensor` from DTensors), but was only used for CPU-offload mode.

**Fix**: Changed the condition from "DTensor grads on CPU" to "any DTensor grads present":
```python
has_dtensor_grads = any(isinstance(p.grad, DTensor) for p in model.parameters() if p.grad is not None)
```

**Why not caught before**: Original Wan models don't have non-FSDP audio modules, so all grads are DTensors (homogeneous). The mix only occurs with OmniAvatar's audio_proj and audio_cond_projs.

---

## 4. Memory Profile (Final, All Fixes Applied)

**Setup**: 4× H200 140GB, FSDP2, bf16, batch_size=1

| Stage | Allocated/GPU | Peak/GPU |
|-------|--------------|----------|
| Fake score update (iters 1-4) | 15 GB | 17 GB |
| Student update START | 12.6 GB | — |
| After rollout (7 chunks, grad ckpt) | 18.8 GB | 19.0 GB |
| After fake_score (no_grad) | 18.8 GB | 20.5 GB |
| After teacher (no_grad, 14B) | 19.3 GB | 23.8 GB |
| After CFG teacher (no_grad) | 19.3 GB | 23.8 GB |
| After VSD loss | 19.3 GB | 23.8 GB |
| **Second student update (iter 10)** | 20.8 GB | **25.3 GB** |

Peak: **25.3 GB/GPU** — well within 140 GB H200 capacity.

### Why Memory Is Low
1. **Gradient checkpointing** on student: stores only block inputs (30 × 9.4 MB = 282 MB per rollout exit), not full intermediates
2. **`torch.no_grad()`** on teacher: FSDP unshards one layer at a time (~700 MB), frees immediately after each layer
3. **FSDP2 sharding**: 14B teacher = ~7 GB/GPU, 1.3B models = ~0.65 GB/GPU each
4. **KV cache**: pre-allocated for 21 frames = ~4 GB
5. **Chunk processing**: student processes 3 frames per rollout step (3072 tokens), not all 21

---

## 5. Remaining Work / Open Questions

### Immediate
- [ ] **Remove temp memory logging** from `dmd2.py` and `wan_model.py`
- [ ] **Commit all fixes** and push
- [x] **Memory comparison with original SF**: Ran successfully using `sf` conda env (torch 2.8, diffusers 0.31). Results confirm our numbers are correct:
  - Original SF (1.3B teacher, 32K tokens): peak **19.8 GB/GPU**
  - **Original SF (14B teacher, 32K tokens): peak 32.7 GB/GPU**
  - Ours (14B teacher, 21K tokens): peak **25.3 GB/GPU**
  - The ~7 GB difference vs original 14B is sequence length (32K vs 21K tokens → more rollout activations + KV cache) and FSDP1 vs FSDP2 overhead.
  - Reference SF (pristine repo at `/home/work/.local/reference/Self-Forcing`) with 14B teacher: **85-98 GB/GPU** on H200.
  - The 60+ GB gap vs our 25 GB is explained by: (1) FSDP1 fp32 master weights ~21 GB, (2) FSDP1 unsharded optimizer states ~15 GB, (3) 11B T5 text encoder loaded on GPU ~11 GB, (4) EMA shadow copy ~5 GB. All legitimate architectural differences, not bugs.
  - Reference SF's bs=1 on H100 80GB IS actually memory-constrained (85 GB barely fits). Our FastGen setup has genuine headroom thanks to FSDP2 + pre-computed embeddings.

### Self-Forcing Training
- [ ] **Create SF launch script** (`scripts/train_omniavatar_sf.sh`)
- [ ] **Determine optimal batch size** — with 25 GB peak on 140 GB GPUs, bs=4+ may be possible
- [ ] **Test DDP mode** — with the no_grad fix, DDP might fit now (avoids FSDP complexity)
- [ ] **Full training run** — 5000 iterations with grad accumulation

### Bug 4: Head Module bs>1 Broadcasting (OmniAvatar porting omission)
**File**: `fastgen/networks/OmniAvatar/wan_model.py`, `Head.forward()`

The original OmniAvatar `Head.forward()` has an `unsqueeze(1)` for 2D `t_mod` input:
```python
if t_mod.dim() == 2:
    t_mod = t_mod.unsqueeze(1)
```
This was dropped during porting. Without it, `[B, dim]` + `[1, 2, dim]` only works for bs=1 (broadcasts accidentally) but fails for bs>1.

### Memory Comparison: FastGen vs Reference Self-Forcing

**Why FastGen uses ~25 GB/GPU vs Reference SF's ~85-98 GB/GPU (both 14B teacher, 4 GPUs):**

| Difference | Reference SF | FastGen | Memory Impact/GPU |
|-----------|-------------|---------|------------------|
| FSDP version | FSDP1 hybrid_full (fp32 master weights) | FSDP2 fully_shard (native bf16) | ~21 GB |
| Optimizer sharding | Unsharded per-GPU | Fully sharded | ~15 GB |
| Text encoder | 11B T5 on GPU | Pre-computed embeddings | ~11 GB |
| EMA | Full fp32 shadow copy | In-place within FSDP | ~5 GB |
| **Total difference** | | | **~52 GB** |

These are **infrastructure differences**, not algorithmic ones. Same training dynamics — FSDP2 is just more memory-efficient.

### Config Decisions Needed
- **Batch size**: Testing bs=4 with grad_accum=4 (same effective bs=64, 4× faster). Also plan to test bs=8 with grad_accum=2.
- **GAN**: Currently disabled. Enable later for quality? Would need the no_grad-free teacher path.
- **Gradient accumulation**: Target effective bs 64 to match original SF.

### Other
- [ ] **DF training**: Currently at 5000/10000 steps. Continue or use current checkpoint?
- [ ] **Inference pipeline**: Need causal inference script for evaluating SF-trained student
- [ ] **Causal inference** scripts were added by remote commits — review and test

---

## 6. Files Modified

### New files (committed + pushed)
- `scripts/generate_omniavatar_ode_pairs_full.py`
- `scripts/run_ode_full_trajectory.sh`
- `scripts/verify_ode_trajectory.py`

### Modified (not yet committed)
- `fastgen/methods/distribution_matching/dmd2.py` — teacher no_grad fix + temp memory logging
- `fastgen/networks/OmniAvatar/network_causal.py` — crossattn cache checkpoint fix
- `fastgen/callbacks/grad_clip.py` — DTensor mix fix
- `fastgen/networks/OmniAvatar/wan_model.py` — temp memory logging (to remove)
- `fastgen/configs/experiments/OmniAvatar/config_sf.py` — shift=5.0, checkpoint paths
- `fastgen/configs/experiments/OmniAvatar/config_sf_test.py` — local paths, neg_text_emb

### Modified externally (temp, should revert)
- `/home/work/.local/Self-Forcing/model/dmd.py` — temp memory logging for comparison
