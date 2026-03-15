# CLAUDE.md — FastGen (OmniAvatar Self-Forcing Distillation)

## Project Goal
Distill OmniAvatar's 14B V2V audio-driven lip sync model into a 1.3B student
using FastGen's Self-Forcing framework for fast few-step inference.

## Environment
- `conda activate omniavatar` (primary env for all work)
- GPUs: 4x H200 (150GB). GPU 2 for testing (`CUDA_VISIBLE_DEVICES=2`).
- Write scope: `/home/work/.local/hyunbin/FastGen/` and `/home/work/.local/OmniAvatar/`
- **SAFETY**: Do NOT delete files outside these repos. Git commit every major change.

## Implementation Status
All phases implemented and verified:
- **Phase 1A-D**: Network wrappers (wan_model, audio_pack, network, network_causal) — BIT-IDENTICAL
- **Phase 2**: Dataset adapter (29K samples, precomputed .pt files)
- **Phase 3**: Method subclasses (OmniAvatarSelfForcingModel, OmniAvatarKDModel)
- **Phase 4**: Experiment configs (SF + KD)
- **Phase 5**: ODE trajectory generation script (tested with 1.3B)
- **Phase 6**: End-to-end integration verified (teacher + student AR + VSD loss)

## Created Files
```
fastgen/networks/OmniAvatar/
  __init__.py, wan_model.py (390L), audio_pack.py (39L),
  network.py (739L), network_causal.py (1525L)
fastgen/methods/
  omniavatar_self_forcing.py, omniavatar_kd.py
fastgen/datasets/
  omniavatar_dataloader.py (205L)
fastgen/configs/experiments/OmniAvatar/
  config_sf.py, config_kd.py
fastgen/configs/methods/
  config_omniavatar_sf.py, config_omniavatar_kd.py
scripts/generate_omniavatar_ode_pairs.py (584L)
```

## Architecture
- Teacher: 14B bidirectional `OmniAvatarWan(FastGenNetwork)` — frozen
- Student: 1.3B causal `CausalOmniAvatarWan(CausalFastGenNetwork)` — trainable
- Fake score: 1.3B bidirectional `OmniAvatarWan` — trained on DSM loss
- Custom WanModel (not diffusers) with args singleton removed
- Audio: Wav2Vec2→AudioPack→per-layer additive residuals (identical everywhere)
- V2V: 49ch (noise+ref+mask+masked_video) or 65ch (+ref_sequence)
- Causal: FlexAttention block mask + KV cache, audio sliced per chunk

## Key Paths
- Teacher ckpt: `/home/work/output_omniavatar_v2v_maskall_refseq_new_data_loss_weights_mouth_weights/step-1500.pt`
- 1.3B base: `pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors`
- Training data: `/home/work/stableavatar_data/v2v_training_data/video_square_path.txt`
- Mask: `/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png`
- Verification data: `verification_data/sample_{0,1,2}_{inputs,output,output_merged}.pt`
- Bugs/notes: `docs/implementation-notes.md`
- Full plan: `docs/omniavatar-self-forcing-plan.md`

## Known Issues
- sinusoidal_embedding_1d returns float32, cast to model dtype (Bug 001)
- LoRA merge in bf16 causes ~0.19 max_diff vs live LoRA (Note 001, acceptable)
- KV cache writes must use current_start for idempotency (Bug 003, fixed)
- text_embeds has extra dim from .pt file, squeeze in _prepare_training_data
