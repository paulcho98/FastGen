# 2026-04-30 Session Summary

20 commits on branch `feat/redmd-sync-c`, ranging from `69badd5` to `eb1879c`. Two
classes of work: **bug fixes** that change training dynamics, and **infrastructure
+ refactor** that doesn't.

## Bugs fixed (training dynamics)

### 1. 14B SF cgroup-OOM at on_train_begin
**Symptom:** SF launched with `OMNIAVATAR_DF_CKPT` set → rank-0 SIGKILL during
on_train_begin's bf16→fp32 cast. Three OOM-kills before diagnosis.

**Root cause:** `fastgen/utils/scripts.py:97-100` force-disabled
`fsdp_meta_init` whenever `pretrained_ckpt_path` was set ("TODO: not
implemented yet"). With meta-init silently off, all 4 ranks materialized
3 × 14B nets (~84 GB each) and the cast pushed the cgroup past its 704 GB
cap.

**Commits:**
- `35d6695` skip dcp.load on meta-tensor ranks in FSDPCheckpointer.load
- `ce9581f` move meta-params check before state_dict() (state_dict()
  itself was materializing meta tensors as a side effect)
- `a2bb3a5` ranks 1-3 still call dcp.load with empty state_dict so
  DCP's collective planner doesn't deadlock
- `6c9b885` skip pre-FSDP cast on meta-tensor ranks in on_train_begin
- `b45ca93` meta-skip ranks must still call synchronize() barrier
- `5c0cdc8` remove the force-disable; the new code path makes
  pretrained_ckpt_path compatible with meta-init

### 2. fake_score requires_grad wipes corrupted save filter
**Symptom:** 14B SF iter-100 save had `fake_score_model = 56 GB` (full 14B
fp32) instead of expected ~2.4 GB.

**Root cause:** `omniavatar_self_forcing.py:67` and
`dmd2._setup_grad_requirements` both did
`self.fake_score.train().requires_grad_(True)` per iter, flipping every
fake_score param to `requires_grad=True` and making the trainable-only
save filter keep all 1937 entries.

Training dynamics were correct (optim was built once at init with only
the LoRA + audio + patch params, ~613M for 14B), but the *save format*
was bloated.

**Commits:**
- `00e0e0c` drop the requires_grad wipe in the SF combined step
- `76f853e` override `_setup_grad_requirements` for the critic-only step
- `f049693` gate both fixes on `unfreeze_modules` non-empty so 1.3B
  full-FT runs (which rely on the wipes) are unaffected

### 3. 1.3B SF asymmetric trainable capacity (legacy bug, newly diagnosed)
**Symptom:** Across all legacy 1.3B SF runs, the student-vs-teacher Sync-C
gap never closed.

**Root cause:** `config_sf.py:61` set `OmniAvatar_V2V_1_3B_FakeScore` with
`merge_lora=False` while the student inherited `merge_lora=True`
(constructor default). At `init_optimizers` time:
- Student: V2V LoRA fused at construction → no PEFT → all 1421M params
  trainable in optim.
- Fake_score: V2V LoRA stayed as PEFT → only LoRA + audio + patch
  trainable in optim (~175M).

Per-iter `requires_grad_(True)` wipes flipped fake_score's grad flags
to True but the optimizer was already built — only the 175M actually
evolved. **8× critic-capacity asymmetry**: critic could not track the
student's distribution, biasing VSD gradients.

The legacy parent `..._fsmatched.sh` also hardcoded
`fake_score_optimizer.lr = 3e-6` (1.5× student) as a half-fix layered on
top of the asymmetry — incorrect once the asymmetry is removed.

**Commits:**
- `2695819` new symmetric configs (`config_sf_full_ft_t769.py`,
  `config_sf_full_ft_t769_no_reward.py`); clean parent
  (`train_sf_parent.sh`); 21 legacy scripts moved to
  `scripts/deprecated_asymmetric/` with DEPRECATED headers + README
- `eb1879c` 1.3B LoRA counterparts (`config_sf_lora_t769.py`,
  `config_df_shift_5_lora_t769.py`) + wrappers, mirroring the 14B LoRA
  recipe symmetrically

## Other fixes (no training-dynamics impact for in-flight runs)

- `69badd5` FSDPCheckpointer.load detects FSDP-saved .pth via sibling
  layout (was KeyError'ing on the metadata stub format)
- `3eff882`, `a2b1010` reset_parameters + PEFT-structure-on-meta inject
  for the bidirectional + causal classes (unblocked meta-init at 14B)
- `a064e8f` 14B SF wrapper forwards MAX_ITER + SAVE_EVERY to trainer

## Infrastructure additions

- `8f7a828` 14B inference script variants (`inference_causal_14b.py`,
  `inference_causal_taehv_14b.py`) supporting `--model_size`, PEFT-aware
  loading, post-load LoRA merge
- `92bbacb` HDTF launcher for 14B SF inference
- `d1d6b70` `scripts/post_hoc_convert_lora_saves.py` — converts the
  14B DF run's bloated 56 GB DCP saves to ~5 GB flat .pth (originals
  preserved; manual cleanup printed per save)
- `5874c88` ablation wrapper (now superseded by `train_sf_full_ft_t769_no_reward.sh`)
- `e1497e8` default outputs to `/home/work/.local` NFS for noredmd

## What this means for runs

- **In-flight 14B SF run** (now killed at iter 800): training dynamics
  were correct; iter-100 save was bloated but iter-200+ saves are clean
  (LoRA-only filter). Resumable from any iter-200+ save with no quality
  loss.
- **All legacy 1.3B SF runs**: had the 8× critic asymmetry. Sync-C
  results from those runs are biased and should be re-evaluated against
  the new symmetric `train_sf_full_ft_t769.sh` baseline.
- **14B SF launches going forward**: use the existing 14B wrapper
  (`train_sf_..._t769_14b_lora.sh`); it now points at the new clean
  parent, inherits matched 2e-6 LRs from `config_sf.py`, and the
  apply_lora_freeze gating ensures regime correctness.

## Files moved

`scripts/deprecated_asymmetric/` (21 files + README.md) — see that
directory's README for the asymmetry diagnosis. Do not launch new runs
from there.
