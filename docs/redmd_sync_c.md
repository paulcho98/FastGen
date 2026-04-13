# Re-DMD with SyncNet-v2 Sync-C Reward

Reward-weighted distillation on top of OmniAvatar Self-Forcing DMD. Scales the student VSD loss by `exp(β · sync_c)` where `sync_c` is SyncNet-v2's offset-margin confidence (higher = more confident lip-sync). Matches the Reward-Forcing paper's outer-multiply formulation:

```
L_gen = exp(β · sync_c_detached) · vsd_loss_unweighted
      + gan_loss_weight_gen · gan_loss_gen
```

Branch: `feat/redmd-sync-c`. All changes vs. `main` are confined to `fastgen/methods/reward/`, `fastgen/methods/omniavatar_self_forcing_re_dmd.py`, and the rewarded config + launch script — the vanilla SF path is untouched.

Design reference (external): `/home/work/.local/hyunbin/Reward-Forcing/docs/sync_c_scorer_design.md`.

---

## Quick launch

**Full training (4 GPUs, long run, project `OmniAvatar-FastGen`):**
```bash
cd /home/work/.local/hyunbin/FastGen-redmd
nohup bash scripts/train_sf_sink1_window7_redmd.sh > /tmp/train_redmd.log 2>&1 &
```
Wandb: `paulhcho/OmniAvatar-FastGen/runs/<name>`, run name `sf_sink1_window7_redmd_syncc_beta0p25`.

**4-GPU smoke (10 iters, MP4 dump, project `OmniAvatar-FastGen-Smoke`):**
```bash
bash scripts/smoke_test_redmd.sh
```
Writes `logs/redmd_smoke_debug/gen_iter{N}.mp4` and the reward-path debug output.

**Disable reward (run as vanilla SF with same subclass):**
Set `config.model.reward.enabled = False` in the config file.

---

## What lands in wandb

Per-generator-step (every 5 iterations with `student_update_freq=5`):

| Key | Source |
|---|---|
| `reward_sync_c_{mean,min,max}` | `dist.all_reduce` across ranks on per-rank batch stats |
| `reward_sync_c_r{0..3}` | `dist.all_gather` of per-rank first-sample sync_c |
| `reward_weight_{mean,min,max,r0..r3}` | `exp(β · sync_c)` reduced same ways |
| `vsd_loss_unweighted` | The original VSD loss before multiplication (rank-local) |
| `vsd_loss_weighted` | `reward_weight_mean × vsd_loss_unweighted` (rank-local) |
| `total_loss` | `weighted_vsd + gan_loss_weight_gen × gan_loss_gen` |

All get averaged across `grad_accum_rounds` by the trainer's `_LossDictRecord` (same semantics as `vsd_loss`). Non-student iterations emit no `reward_*` keys — only `total_loss`, `fake_score_loss`, `gan_loss_disc`.

---

## Files touched

### New files
| File | Purpose |
|---|---|
| `fastgen/methods/reward/syncnet_v2.py` | Vendored `SyncNetV2` from joonson/syncnet_python |
| `fastgen/methods/reward/sync_c_scorer.py` | `SyncCScorer` — tensor-in/scalar-out reward model |
| `fastgen/methods/omniavatar_self_forcing_re_dmd.py` | `OmniAvatarSelfForcingReDMD` subclass with reward weighting |
| `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd.py` | Full-training rewarded config |
| `fastgen/configs/experiments/OmniAvatar/config_sf_sink1_window7_redmd_smoke.py` | 10-iter 4-GPU smoke variant with MP4 dump |
| `scripts/train_sf_sink1_window7_redmd.sh` | Full-training launcher |
| `scripts/smoke_test_redmd.sh` | Smoke launcher with post-run verification |
| `tests/reward/test_*.py` | 28 unit tests (scorer, windowing, offset search, reward weighting, dataset audio plumbing, trainer with mocked scorer) |

### Modified files
| File | Change |
|---|---|
| `fastgen/configs/methods/config_omniavatar_sf.py` | Added `RewardConfig` attrs class; declared `reward`, `reward_beta`, `center_reward`, `clamp_reward`, `save_reward_debug_video`, `reward_debug_dir` as typed fields on `OmniAvatarModelConfig` |
| `fastgen/datasets/omniavatar_dataloader.py` | Optional `load_raw_audio=True` kwarg emits `audio_waveform` (mono, 16 kHz, padded/truncated to 81×25 = 51840 samples) alongside the existing `audio_path` key |
| `fastgen/callbacks/wandb.py` | `_LossDictRecord.add` tolerates python-float loss entries alongside tensors (needed because our `log_map` mixes both) |

---

## How it's wired together

1. **`OmniAvatarDataset`** with `load_raw_audio=True` reads the `.wav` for each sample, resamples to 16 kHz mono, pads/truncates to 51840 samples, and emits `sample["audio_waveform"]: Tensor[51840] float32`.
2. **`OmniAvatarSelfForcingReDMD.build_model`** loads the SyncNet-v2 checkpoint into a `SyncCScorer` (all params frozen, `torch.no_grad` everywhere).
3. **`OmniAvatarSelfForcingReDMD._student_update_step`** (runs every 5 iters):
   - Computes `vsd_loss` inline the same way `DMD2Model._student_update_step` does (generator latent → fake_score + teacher_x0 → `variational_score_distillation_loss`).
   - VAE-decodes `gen_data` under `torch.no_grad`.
   - Converts pixels → uint8, unbinds batch.
   - Extracts `data["audio_waveform"]` → per-sample list.
   - Calls `SyncCScorer.reward_from_frames(videos, audios)` → scalar `sync_c` per sample.
   - `weight = exp(β · sync_c)`; `weighted_vsd = weight.mean() × vsd_loss`.
   - Returns `loss_map` with the 14 reward/weight/loss keys (python floats post-reduce).
4. **Rank-0 MP4 save** (only when `save_reward_debug_video=True`): `logs/redmd_smoke_debug/gen_iter{N:06d}.mp4` — 81 frames × 512² RGB @ 25 fps, silent.

---

## β calibration

Starting point is `β=0.25`. Rationale and adjustment guide:

| Observed `reward_sync_c_mean` | Observed `reward_weight_mean` at β=0.25 | Guidance |
|---|---|---|
| < 0.5 | ≈ 1.0–1.1 | Reward is barely modulating loss (our smoke regime). Either let training progress, or increase β (up to ~1.0) once sync-C starts rising. |
| 2–4 | ≈ 1.6–2.7 | Healthy mid-training reward signal. Keep β fixed. |
| 5–8 | ≈ 3.5–7.4 | Strong. Keep β. |
| 8+ | ≈ 7.4+ | Excellent sync. Watch for `reward_weight_max` tail — if it exceeds 20–30 consistently, enable `center_reward=True` to keep mean weight near 1. |

**Safety rails available** (also see `model/re_dmd.py` §7.3 in Reward-Forcing design doc):
- `center_reward=True`: subtracts an EMA of `sync_c` before exp — keeps the mean weight ≈ 1 and re-scales the dynamic range rather than the absolute magnitude.
- `clamp_reward=(lo, hi)`: clamps the sync_c tensor before exp. E.g. `(0.0, 10.0)` caps the maximum weight at `exp(β × 10) = 12.2` with β=0.25.

---

## Empirical findings from the smoke test

4 GPUs × batch_size=1 × grad_accum=1, 10 iterations, starting from `df_4gpu_bs16_stochastic_attn_shift5/0010000.pth` (student generator):

### Iter 5 (first and only student step in the smoke)
```
reward_sync_c_r0 = 0.1618
reward_sync_c_r1 = 0.3212   ← max
reward_sync_c_r2 = 0.0270   ← min
reward_sync_c_r3 = 0.0463
reward_sync_c_mean = 0.1391
reward_weight_mean = 1.0358  # ≈ exp(0.25 × 0.1391)
vsd_loss_unweighted = 0.6138
vsd_loss_weighted  = 0.6352  # = 1.0358 × 0.6138 ✓
```

### What this tells us
- **Reward pipeline is correct end-to-end.** The math works (weight × unweighted ≈ weighted to fp precision).
- **Sync-C values are small** (0.03–0.32) vs. the "5–10 for good sync" range in the design doc. Most likely explanation: step-10000 of the df pretrained checkpoint doesn't yet produce lip motion that correlates tightly with the audio, so SyncNet's offset-distance curve is flat and `median − min` is tiny. Expected to grow as training progresses.
- **Weights are near 1.0** (1.0068–1.0836), meaning the reward is barely modulating loss at this point. This is fine for a smoke test; the mechanism is validated even if the signal is weak.

---

## Debugging / inspection guide

**Reward scorer didn't load?**
Check logs for `SyncCScorer loaded: beta=..., vshift=..., ckpt=...` during build_model. Absence indicates the reward is disabled or the config didn't propagate. Likely causes: `config.model.reward.enabled=False`, or the reward sub-config wasn't declared on the attrs class (see "Config / Instantiation Gotchas" in `CLAUDE.md`).

**`vsd_loss_weighted == vsd_loss_unweighted`?**
Reward path fell back to the else branch. Two conditions must both hold for the reward path: `self.reward_scorer is not None AND "audio_waveform" in data`. Verify:
- Scorer check: re-read SyncCScorer log during startup.
- Audio check: confirm `config.dataloader_train.load_raw_audio=True` and that `sample["audio_path"]` points to a real `.wav` file.

**Visual inspection:**
Enable `config.model.save_reward_debug_video=True` in the config. Each generator step saves rank-0's first-sample decoded video to `{reward_debug_dir}/gen_iter{N:06d}.mp4` (default `logs/redmd_debug/`). Default in full training is **off** to avoid thousands of MP4s.

**MP4s have no audio by default.** SyncNet doesn't need it (operates on tensors directly), but for human listening you'd need to mux the audio afterward — `torchvision.io.write_video` supports `audio_array`/`audio_codec` kwargs; patch is in `_maybe_save_debug_video` if needed.

---

## Checkpoints used

| Role | Path | Frozen? |
|---|---|---|
| Teacher (14B) | `/home/work/output_omniavatar_v2v_phase2/step-10500.pt` | Yes |
| Fake score (1.3B) | base Wan2.1-T2V-1.3B + LoRA from `output_omniavatar_v2v_1.3B_phase2/step-19500.pt` | No (critic) |
| Student (1.3B) | same base + LoRA, then resume from `/home/work/.local/hyunbin/checkpoints/df_4gpu_bs16_stochastic_attn_shift5/0010000.pth` at iter 10000 | No (trained) |
| VAE (for reward decode) | `/home/work/.local/OmniAvatar/pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | Yes |
| SyncNet-v2 (reward model) | `/home/work/.local/eval_metrics/checkpoints/auxiliary/syncnet_v2.model` | Yes |

The student checkpoint (`df_...step-10000`) is what the MP4 dumps visualize. Sync quality ceiling depends on this starting point.

---

## Test coverage (`tests/reward/`, 28 tests)

| File | Tests |
|---|---|
| `test_syncnet_v2.py` | 3 — arch instantiation, forward shapes, real checkpoint load |
| `test_sync_c_scorer.py` | 12 — video/audio prep, windowing, offset search, `reward_from_frames` with mock net, GPU-path with real ckpt |
| `test_dataset_audio_waveform.py` | 4 — unit tests for `_load_raw_waveform_for_reward` + dataset smoke tests gated on training data presence |
| `test_re_dmd_trainer.py` | 9 — `_apply_reward_weighting` math (β·r, centering, clamping, batched), `_student_update_step` integrates reward, fallback when scorer is None, per-rank logging, debug-video save path |

All pass in the `hb_fastgen` env (`/home/work/.local/miniconda3/envs/hb_fastgen/bin/python -m pytest tests/reward/ -v`).

---

## Commit history (on `feat/redmd-sync-c`)

```
docs: Re-DMD documentation + CLAUDE.md gotchas   ← this commit
test(redmd): 4-GPU smoke test config + launch script
fix(redmd): post-smoke-test production fixes
fix(redmd): __init__ no longer clobbers reward_scorer to None
feat(redmd): per-rank logging + optional MP4 debug save
feat(redmd): cross-rank reduce for reward/weight logging
script: launch Re-DMD sync-C variant of sink1_window7
config(redmd): SF + sync-C reward variant of sink1_window7_tscfg
feat(redmd): _student_update_step with VAE decode + sync-C reward
feat(redmd): OmniAvatarSelfForcingReDMD subclass + _apply_reward_weighting
feat(dataset): optional raw audio waveform for Re-DMD sync reward
feat(reward): SyncCScorer.reward_from_frames public entry
feat(reward): SyncCScorer windowing and offset search
fix(reward): address code review for SyncCScorer preprocessing
feat: SyncCScorer video/audio preprocessing
feat: vendor SyncNet-v2 architecture for Re-DMD sync reward
scaffold: reward/ package and tests/reward/ for Re-DMD sync reward
docs: Re-DMD sync-C reward implementation plan
```

---

## Related docs

- `docs/superpowers/plans/2026-04-12-sync-c-reward-redmd.md` — The implementation plan this was built from (13 tasks, all complete).
- `/home/work/.local/hyunbin/Reward-Forcing/docs/sync_c_scorer_design.md` — The scorer design reference (pre-port).
- `/home/work/.local/hyunbin/Reward-Forcing/docs/reward_forcing_implementation.md` — Original Re-DMD study; §2 for β convention, §7 for the "swap in a custom reward" interface, §8 for empirical reward-scale data.
- `CLAUDE.md` — "Config / Instantiation Gotchas" section has the patterns that cost time during this port.
