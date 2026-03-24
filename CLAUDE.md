# CLAUDE.md — FastGen (OmniAvatar Self-Forcing Distillation)

## Project Goal
Distill OmniAvatar's 14B V2V audio-driven lip sync model into a 1.3B student
using FastGen's Self-Forcing framework for fast few-step inference.

## Environment
- Python: `/home/work/.local/miniconda3/envs/hb_fastgen/bin/python` (hb_fastgen env)
- GPUs: 4x H200 (150GB). GPU 2 for testing (`CUDA_VISIBLE_DEVICES=2`).
- Write scope: `/home/work/.local/hyunbin/FastGen/` and `/home/work/.local/OmniAvatar/`
- **SAFETY**: Do NOT delete files outside these repos. Git commit every major change.

## Status
All OmniAvatar integration code implemented and verified (BIT-IDENTICAL to original).
Pending: 1.3B refseq training, ODE pair generation with 14B, actual KD/SF training runs.

## Architecture
- Teacher: 14B bidirectional `OmniAvatarWan(FastGenNetwork)` — frozen
- Student: 1.3B causal `CausalOmniAvatarWan(CausalFastGenNetwork)` — trainable
- Fake score: 1.3B bidirectional `OmniAvatarWan` — via `config.model.fake_score_net` (NOT teacher_config)
- V2V 65ch: noise(16)+ref(16)+mask(1)+masked_video(16)+ref_sequence(16)
- Causal: FlexAttention + KV cache (detach/cat), dynamic RoPE, local attn window, attention sink
- Training dataloader: `OmniAvatarDataLoader` (infinite iterator + DistributedSampler)
- Stage 1 options: ODE KD (`config_kd.py`) or Diffusion Forcing (`config_df.py`)

## Key Paths & Docs
- Teacher ckpt: `/home/work/output_omniavatar_v2v_maskall_refseq_new_data_loss_weights_mouth_weights/step-1500.pt`
- Training data: `/home/work/stableavatar_data/v2v_training_data/video_square_path.txt`
- Mask: `/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png`
- Reference impl: `/home/work/.local/FastGen/` (initial attempt, has bugs — see comparison notes)
- Detailed docs: `docs/omniavatar-changes-summary.md`, `docs/implementation-notes.md`, `docs/omniavatar-self-forcing-plan.md`

## Base FastGen Modifications
- `config.py`: Added `fake_score_net: Optional[dict]` to `BaseModelConfig`
- `dmd2.py`: `build_model()` uses `fake_score_net` when set (separate arch from teacher)
- `methods/__init__.py`: Registered OmniAvatar model classes
