# Deprecated Asymmetric SF Training Scripts

These scripts launch 1.3B Self-Forcing training with an **asymmetric
trainable-capacity bug** — the student trains as full-FT (~1421M
trainable) while the fake_score (critic) trains as LoRA-only (~175M
trainable, 8× smaller).

## How the asymmetry happens

The base SF config (`fastgen/configs/experiments/OmniAvatar/config_sf.py`)
sets:
- `CausalOmniAvatar_V2V_1_3B_Student`: no explicit `merge_lora` →
  defaults to **True** → V2V LoRA fuses into base at construction → no
  PEFT layers → **all 1.3B params trainable** via the per-iter
  `requires_grad_(True)` wipes.
- `OmniAvatar_V2V_1_3B_FakeScore` (line 61): explicit
  `merge_lora=False` → V2V LoRA stays as PEFT layers → PEFT default
  freezes base + only enables LoRA → init_optimizers builds Adam state
  for **only the LoRA params (~175M)**.

The per-iter wipes flip `requires_grad=True` on every fake_score
param, but the optimizer is already built — those wipes only inflate
the save filter's "trainable count," not what actually trains.

## Consequences

- 8× capacity mismatch between student and critic
- Critic falls behind every step the student moves → biased VSD
  gradients
- Likely contributor to the persistent student-vs-teacher Sync-C gap
  observed across all of these legacy 1.3B SF runs
- Saved fake_score_model files are bloated (full 1.3B written despite
  only 175M actually evolving)
- The legacy parent
  `train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched.sh`
  also hardcoded `model.fake_score_optimizer.lr=3e-6` (1.5× student
  2e-6) as a half-fix for the asymmetry, which is incorrect once the
  asymmetry is removed.

## Replacement

For new 1.3B SF runs, use the symmetric scripts in `scripts/`:

| Old (deprecated) | New (in scripts/) |
| --- | --- |
| `..._fsmatched_t769_fsdpfix.sh` (asymmetric redmd) | `train_sf_full_ft_t769.sh` |
| `..._fsmatched_t769_fsdpfix_noredmd.sh` (asymmetric noredmd) | `train_sf_full_ft_t769_no_reward.sh` |

The new scripts use `config_sf_full_ft_t769.py` which:
- Forces `fake_score_net.merge_lora=True` → full-FT critic (~1421M)
- Sets matched LRs (both 2e-6)
- Asserts coherence at config-load time

## Why kept here

These scripts are preserved for reproducibility — past runs that
trained against this asymmetric regime are still on disk and may need
the corresponding launch script for resume / re-eval. **Do NOT launch
new runs from this directory.**

The 14B SF scripts (in `scripts/`) are NOT affected — they use
`config_sf_14b_lora_t769.py` which sets `merge_lora=False` +
`unfreeze_modules` symmetrically on both student and fake_score, and
`apply_lora_freeze` (gated on `unfreeze_modules` non-empty per commit
`f049693`) freezes both networks identically.
