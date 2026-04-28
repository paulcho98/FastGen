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
