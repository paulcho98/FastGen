# CLAUDE.md — FastGen (OmniAvatar Self-Forcing Distillation)

## Project Goal
Adapt FastGen's Self-Forcing distillation framework to distill OmniAvatar's 14B V2V
audio-driven lip sync model into a 1.3B student for fast few-step inference.

## Environment
- `conda activate fastgen` (or omniavatar for OmniAvatar-side work)
- GPUs: 4x H200 (150GB each). Use `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Write scope: Full read/write to `/home/work/.local/hyunbin/FastGen/`
- **SAFETY**: Do NOT delete files outside this repo or `/home/work/.local/OmniAvatar/`

## Implementation Plan
See `docs/omniavatar-self-forcing-plan.md` for the full 7-phase plan.

## Key Architecture Facts
- FastGen's Self-Forcing: `FastGenModel → DMD2Model → CausVidModel → SelfForcingModel`
- OmniAvatar uses a CUSTOM WanModel (NOT diffusers' WanTransformer3DModel)
- Must port OmniAvatar's DiT into `fastgen/networks/OmniAvatar/` with args singleton removed
- Audio: Wav2Vec2 [B,81,10752] → AudioPack [B,32,21,1,1] → per-layer projections → additive residuals
- V2V conditioning (65ch): 16 noise + 16 ref_repeated + 1 mask + 16 masked_video + 16 ref_sequence
- Teacher: 14B bidirectional (frozen). Student: 1.3B causal (trainable). Fake score: 1.3B bidirectional (trained on DSM).
- Reference causal impl: `/home/work/.local/Self-Forcing-OmniAvatar/Self-Forcing/wan/modules/causal_model.py`
- Audio mixin pattern: `/home/work/.local/Self-Forcing-OmniAvatar/Self-Forcing/wan/modules/audio_mixin.py`
- Audio conditioning is IDENTICAL everywhere — teacher, student, fake_score, ODE extraction, inference.
- Testing: Use GPU 3 (`CUDA_VISIBLE_DEVICES=3`) with real data samples to verify numerical correctness.

## Files to Create (starting from scratch)
```
fastgen/networks/OmniAvatar/{__init__,wan_model,audio_pack,network,network_causal}.py
fastgen/methods/{omniavatar_self_forcing,omniavatar_kd}.py
fastgen/datasets/omniavatar_dataloader.py
fastgen/configs/experiments/OmniAvatar/{__init__,config_sf,config_kd}.py
fastgen/configs/methods/{config_omniavatar_sf,config_omniavatar_kd}.py
scripts/generate_omniavatar_ode_pairs.py
```

## ODE Trajectory Generation
- Existing code: `scripts/generate_ode_trajectories.py` (verified correct for T2V)
- KD method: `fastgen/methods/knowledge_distillation/KD.py` (KDModel + CausalKDModel)
- ODE format: `path.pth` [4, C, T, H, W] (noisy states) + `latent.pth` [C, T, H, W] (clean)
- t_list: [0.999, 0.937, 0.833, 0.624, 0.0] — 4 noisy + 1 clean
- For OmniAvatar: adapt to include audio+V2V conditioning, save as `ode_path.pt` per sample dir
- Teacher for ODE is BIDIRECTIONAL (full sequence at once, no chunking)
- 14B teacher ckpt: `/home/work/output_omniavatar_v2v_maskall_refseq_new_data_loss_weights_mouth_weights/step-1500.pt`

## Critical Gotchas
1. Audio slicing in causal mode: audio is [B,81,10752] in VIDEO frame space but chunks
   operate in LATENT frame space (21 frames). Must map latent→video frames correctly.
2. Patch embedding expansion: 16ch (base Wan) → 33ch (I2V OmniAvatar) → 65ch (V2V+refseq).
   smart_load_weights copies existing channels, zero-inits new ones.
3. LoRA key mapping: OmniAvatar uses `lora_A.weight`, PEFT uses `lora_A.default.weight`.
4. The global `args` singleton in OmniAvatar's DiT must be completely removed in the port.
5. Wav2Vec2 must stay float32 (CNN feature extractor fails on bf16).
6. Noise schedule: OmniAvatar uses rectified flow ("rf"). Verify FastGen's RF matches exactly.

## OmniAvatar Repo
Located at `/home/work/.local/OmniAvatar/`. See its CLAUDE.md for full documentation.
Key files: `OmniAvatar/models/wan_video_dit.py`, `scripts/train_v2v.py`, `scripts/inference_v2v.py`.

## Training Data
- V2V training: `/home/work/stableavatar_data/v2v_training_data/`
- Path list: `video_square_path_combined.txt` (36K+ samples)
- Per sample: `vae_latents_mask_all.pt`, `audio_emb_omniavatar.pt`, `text_emb.pt`, `ref_latents.pt`
- Validation: `/home/work/stableavatar_data/v2v_validation_data/{recon,mixed}/`
- LatentSync mask: `/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png`
