"""Generate reference outputs from the ORIGINAL OmniAvatar 1.3B model.

Produces deterministic verification data (inputs + DiT outputs) that can be used
to numerically verify ported code across all phases.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/create_verification_samples.py

Outputs saved to: verification_data/
    sample_{i}_inputs.pt   — all inputs to DiT forward pass
    sample_{i}_output.pt   — DiT output tensor
    sample_{i}_metadata.pt — sample path, model config, etc.
"""

import os
import sys
import argparse
import time

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ============================================================================
# 0. Environment setup
# ============================================================================

OMNIAVATAR_ROOT = "/home/work/.local/OmniAvatar"
DATA_LIST_PATH = "/home/work/stableavatar_data/v2v_training_data/video_square_path.txt"
MASK_PATH = "/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png"
BASE_WEIGHTS = os.path.join(OMNIAVATAR_ROOT, "pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
V2V_CKPT = "/home/work/output_omniavatar_v2v_1.3B_maskall/step-2500.pt"
OUTPUT_DIR = "/home/work/.local/hyunbin/FastGen/verification_data"
NUM_SAMPLES = 3
SEED = 42
TIMESTEP_VALUE = 500.0
NUM_FRAMES = 81  # video frames -> 21 latent frames
IN_DIM = 49
AUDIO_HIDDEN_SIZE = 32
DTYPE = torch.bfloat16

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. Set up the global args singleton BEFORE any OmniAvatar imports
# ============================================================================

sys.path.insert(0, OMNIAVATAR_ROOT)
import OmniAvatar.utils.args_config as args_module

args_obj = argparse.Namespace(
    use_audio=True,
    sp_size=1,
    model_config={"in_dim": IN_DIM, "audio_hidden_size": AUDIO_HIDDEN_SIZE},
    i2v=True,
    random_prefix_frames=True,
)
args_module.args = args_obj

# ============================================================================
# 2. Now safe to import OmniAvatar modules
# ============================================================================

from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.utils.io_utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model


def find_valid_samples(data_list_path, num_needed):
    """Find first N samples that have all required precomputed files."""
    required_files = [
        "vae_latents_mask_all.pt",
        "audio_emb_omniavatar.pt",
        "text_emb.pt",
        "ref_latents.pt",
    ]
    valid = []
    with open(data_list_path, "r") as f:
        for line in f:
            d = line.strip()
            if not d:
                continue
            if all(os.path.isfile(os.path.join(d, fn)) for fn in required_files):
                valid.append(d)
                if len(valid) >= num_needed:
                    break
    return valid


def load_mask(mask_path, latent_h=64, latent_w=64):
    """Load LatentSync mask, resize to latent resolution, return as float tensor.

    LatentSync convention: 255=keep upper face, 0=mask mouth.
    Returns: [H_lat, W_lat] float tensor, 1=keep, 0=mask (LatentSync convention).
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_img, dtype=np.float32) / 255.0  # [H, W], 0-1
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    mask_resized = F.interpolate(mask_t, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
    mask_resized = (mask_resized > 0.5).float().squeeze(0).squeeze(0)  # [H_lat, W_lat]
    return mask_resized


def prepare_v2v_y(ref_latent, masked_video_latents, latent_mask, mask_all_frames=True):
    """Prepare y conditioning tensor for 49ch V2V input.

    y = ref_repeated(16ch) + mask(1ch) + masked_video(16ch) = 33ch
    The DiT forward concatenates x(16ch) + y(33ch) = 49ch for patch_embedding.

    Args:
        ref_latent: [1, 16, 1, H, W] — reference frame latent
        masked_video_latents: [1, 16, T, H, W] — masked video latents
        latent_mask: [H, W] — LatentSync mask (1=keep, 0=mask)
        mask_all_frames: if True, all frames get spatial mask; if False, frame 0 is unmasked

    Returns:
        y: [1, 33, T, H, W]
    """
    T_lat = masked_video_latents.shape[2]
    H_lat, W_lat = masked_video_latents.shape[3], masked_video_latents.shape[4]
    device = masked_video_latents.device
    dtype = masked_video_latents.dtype

    # Repeat reference across time
    ref_repeated = ref_latent.repeat(1, 1, T_lat, 1, 1)  # [1, 16, T, H, W]

    # OmniAvatar mask: 0=keep, 1=generate. Invert LatentSync (1=keep -> 0, 0=mask -> 1)
    inverted = 1.0 - latent_mask  # [H, W]
    mask_ch = torch.zeros(1, 1, T_lat, H_lat, W_lat, device=device, dtype=dtype)
    if mask_all_frames:
        mask_ch[:, :, :] = inverted[None, None, None]
    else:
        mask_ch[:, :, 0] = 0
        mask_ch[:, :, 1:] = inverted[None, None, None]

    y = torch.cat([ref_repeated, mask_ch, masked_video_latents], dim=1)  # [1, 33, T, H, W]
    return y


def build_model(device):
    """Build the 1.3B DiT model with base weights + LoRA + V2V checkpoint.

    Follows the same procedure as train_v2v.py:
    1. Load base Wan 2.1 weights via ModelManager (with in_dim=49 via args singleton)
    2. Inject PEFT LoRA adapters
    3. Load trained V2V checkpoint (LoRA + audio + patch_embedding)
    """
    print("=" * 70)
    print("Building OmniAvatar 1.3B DiT model")
    print("=" * 70)

    # Step 1: Load base weights via ModelManager
    # infer=False triggers xavier init + smart_load_weights overlay
    print("\n[1/3] Loading base Wan 2.1 1.3B weights...")
    model_manager = ModelManager(device="cpu", infer=False)
    model_manager.load_models(
        [[BASE_WEIGHTS]],
        torch_dtype=DTYPE,
        device="cpu",
    )

    # Extract the DiT model
    dit = model_manager.fetch_model("wan_video_dit")
    assert dit is not None, "Failed to load DiT from ModelManager"
    print(f"  DiT loaded: patch_embedding.weight shape = {dit.patch_embedding.weight.shape}")
    print(f"  use_audio = {dit.use_audio}")
    print(f"  num_layers = {len(dit.blocks)}")
    print(f"  audio_cond_projs count = {len(dit.audio_cond_projs)}")

    # Step 2: Inject LoRA (matching train_v2v.py config)
    print("\n[2/3] Injecting LoRA adapters...")
    lora_config = LoraConfig(
        r=128,
        lora_alpha=64,
        init_lora_weights=True,
        target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"],
    )
    dit = inject_adapter_in_model(lora_config, dit)
    # Cast LoRA params to bf16
    for param in dit.parameters():
        if param.requires_grad:
            param.data = param.to(DTYPE)

    # Step 3: Load V2V checkpoint
    print(f"\n[3/3] Loading V2V checkpoint: {V2V_CKPT}")
    ckpt_sd = load_state_dict(V2V_CKPT)

    # The V2V checkpoint already has lora_A.default.weight format (PEFT style)
    # But check and remap if needed (the OmniAvatar-1.3B pretrained has lora_A.weight format)
    mapped_sd = {}
    for k, v in ckpt_sd.items():
        new_k = k
        if "lora_A.weight" in k and "default" not in k:
            new_k = k.replace("lora_A.weight", "lora_A.default.weight")
        if "lora_B.weight" in k and "default" not in k:
            new_k = k.replace("lora_B.weight", "lora_B.default.weight")
        mapped_sd[new_k] = v

    # Handle patch_embedding shape mismatch
    pe_key = "patch_embedding.weight"
    if pe_key in mapped_sd:
        model_pe = dit.patch_embedding.weight
        if mapped_sd[pe_key].shape != model_pe.shape:
            old_shape = mapped_sd[pe_key].shape
            print(f"  Expanding patch_embedding: {old_shape} -> {model_pe.shape}")
            new_pe = torch.zeros_like(model_pe.data)
            slices = tuple(slice(0, s) for s in old_shape)
            new_pe[slices] = mapped_sd[pe_key]
            mapped_sd[pe_key] = new_pe

    missing, unexpected = dit.load_state_dict(mapped_sd, strict=False)
    loaded = len(ckpt_sd) - len(unexpected)
    print(f"  Loaded {loaded} params, {len(missing)} missing, {len(unexpected)} unexpected")
    if unexpected:
        print(f"  Unexpected keys (first 5): {unexpected[:5]}")
    if missing:
        # Filter out base model keys (expected to be missing from the checkpoint)
        non_base_missing = [k for k in missing if "lora" in k or "audio" in k]
        if non_base_missing:
            print(f"  WARNING: Missing non-base keys: {non_base_missing[:10]}")

    # Move to GPU and eval mode
    dit = dit.to(device=device, dtype=DTYPE)
    dit.eval()

    print(f"\n  Model on device: {device}")
    print(f"  Total parameters: {sum(p.numel() for p in dit.parameters()):,}")
    print(f"  Trainable params: {sum(p.numel() for p in dit.parameters() if p.requires_grad):,}")

    return dit


def main():
    device = torch.device("cuda")
    t_start = time.time()

    # ======================================================================
    # Find valid samples
    # ======================================================================
    print(f"\nSearching for {NUM_SAMPLES} valid samples in {DATA_LIST_PATH}...")
    sample_dirs = find_valid_samples(DATA_LIST_PATH, NUM_SAMPLES)
    assert len(sample_dirs) == NUM_SAMPLES, (
        f"Only found {len(sample_dirs)} valid samples, need {NUM_SAMPLES}"
    )
    for i, d in enumerate(sample_dirs):
        print(f"  Sample {i}: {os.path.basename(d)}")

    # ======================================================================
    # Load mask
    # ======================================================================
    print(f"\nLoading LatentSync mask from {MASK_PATH}...")
    latent_mask = load_mask(MASK_PATH, latent_h=64, latent_w=64)
    print(f"  Mask shape: {latent_mask.shape}, keep ratio: {latent_mask.mean():.3f}")

    # ======================================================================
    # Build model
    # ======================================================================
    dit = build_model(device)

    # ======================================================================
    # Process each sample
    # ======================================================================
    print("\n" + "=" * 70)
    print("Running forward passes")
    print("=" * 70)

    for sample_idx, sample_dir in enumerate(sample_dirs):
        print(f"\n--- Sample {sample_idx}: {os.path.basename(sample_dir)} ---")

        # Load precomputed data
        vae_data = torch.load(
            os.path.join(sample_dir, "vae_latents_mask_all.pt"),
            map_location="cpu", weights_only=True,
        )
        input_latents = vae_data["input_latents"]  # [16, 21, 64, 64]
        masked_latents = vae_data["masked_latents"]  # [16, 21, 64, 64]
        print(f"  input_latents: {input_latents.shape} {input_latents.dtype}")
        print(f"  masked_latents: {masked_latents.shape} {masked_latents.dtype}")

        audio_data = torch.load(
            os.path.join(sample_dir, "audio_emb_omniavatar.pt"),
            map_location="cpu", weights_only=True,
        )
        audio_emb_full = audio_data["audio_emb"]  # [N, 10752]
        # Slice to first 81 frames
        audio_emb = audio_emb_full[:NUM_FRAMES]  # [81, 10752]
        print(f"  audio_emb: {audio_emb_full.shape} -> sliced to {audio_emb.shape}")

        text_emb = torch.load(
            os.path.join(sample_dir, "text_emb.pt"),
            map_location="cpu", weights_only=True,
        )  # [1, 512, 4096]
        print(f"  text_emb: {text_emb.shape} {text_emb.dtype}")

        ref_data = torch.load(
            os.path.join(sample_dir, "ref_latents.pt"),
            map_location="cpu", weights_only=True,
        )
        ref_sequence_latents = ref_data["ref_sequence_latents"]  # [16, 21, 64, 64]
        print(f"  ref_sequence_latents: {ref_sequence_latents.shape}")

        # Prepare inputs
        # Add batch dimension: [C, T, H, W] -> [1, C, T, H, W]
        input_latents_5d = input_latents.unsqueeze(0).to(dtype=DTYPE, device=device)
        masked_latents_5d = masked_latents.unsqueeze(0).to(dtype=DTYPE, device=device)
        text_emb_dev = text_emb.to(dtype=DTYPE, device=device)
        if text_emb_dev.dim() == 2:
            text_emb_dev = text_emb_dev.unsqueeze(0)
        audio_emb_dev = audio_emb.unsqueeze(0).to(dtype=DTYPE, device=device)  # [1, 81, 10752]

        # Reference latent = first frame
        ref_latent = input_latents_5d[:, :, :1]  # [1, 16, 1, 64, 64]

        # Prepare y (33ch conditioning)
        mask_dev = latent_mask.to(device=device)
        y = prepare_v2v_y(
            ref_latent, masked_latents_5d, mask_dev, mask_all_frames=True,
        )  # [1, 33, 21, 64, 64]
        print(f"  y shape: {y.shape}")

        # Generate deterministic noise
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        noise = torch.randn(1, 16, 21, 64, 64, device=device, dtype=DTYPE)
        print(f"  noise shape: {noise.shape}")

        # Timestep
        timestep = torch.tensor([TIMESTEP_VALUE], device=device, dtype=DTYPE)
        print(f"  timestep: {timestep.item()}")

        # Run forward pass
        print("  Running DiT forward pass...")
        with torch.no_grad():
            output = dit(
                x=noise,
                timestep=timestep,
                context=text_emb_dev,
                y=y,
                audio_emb=audio_emb_dev,
            )
        print(f"  output shape: {output.shape} {output.dtype}")
        print(f"  output stats: mean={output.float().mean():.6f}, std={output.float().std():.6f}, "
              f"min={output.float().min():.6f}, max={output.float().max():.6f}")

        # Save inputs
        inputs_dict = {
            "noise": noise.cpu(),
            "timestep": timestep.cpu(),
            "context": text_emb_dev.cpu(),
            "audio_emb": audio_emb_dev.cpu(),
            "y": y.cpu(),
            "input_latents": input_latents,       # original [16, 21, 64, 64]
            "masked_latents": masked_latents,      # original [16, 21, 64, 64]
            "ref_sequence_latents": ref_sequence_latents,  # [16, 21, 64, 64]
            "latent_mask": latent_mask,            # [64, 64]
        }
        inputs_path = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}_inputs.pt")
        torch.save(inputs_dict, inputs_path)
        print(f"  Saved inputs -> {inputs_path}")

        # Save output
        output_path = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}_output.pt")
        torch.save(output.cpu(), output_path)
        print(f"  Saved output -> {output_path}")

        # Save metadata
        metadata = {
            "sample_dir": sample_dir,
            "sample_name": os.path.basename(sample_dir),
            "seed": SEED,
            "timestep": TIMESTEP_VALUE,
            "dtype": str(DTYPE),
            "in_dim": IN_DIM,
            "audio_hidden_size": AUDIO_HIDDEN_SIZE,
            "num_frames": NUM_FRAMES,
            "mask_all_frames": True,
            "model_config": {
                "dim": 1536,
                "ffn_dim": 8960,
                "num_heads": 12,
                "num_layers": 30,
                "in_dim": IN_DIM,
                "out_dim": 16,
                "text_dim": 4096,
                "freq_dim": 256,
                "eps": 1e-6,
                "patch_size": [1, 2, 2],
                "has_image_input": False,
                "audio_hidden_size": AUDIO_HIDDEN_SIZE,
            },
            "base_weights_path": BASE_WEIGHTS,
            "v2v_checkpoint_path": V2V_CKPT,
            "lora_rank": 128,
            "lora_alpha": 64,
            "lora_target_modules": ["q", "k", "v", "o", "ffn.0", "ffn.2"],
            "mask_path": MASK_PATH,
            "output_shape": list(output.shape),
            "output_mean": output.float().mean().item(),
            "output_std": output.float().std().item(),
        }
        metadata_path = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}_metadata.pt")
        torch.save(metadata, metadata_path)
        print(f"  Saved metadata -> {metadata_path}")

    # ======================================================================
    # Summary
    # ======================================================================
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"DONE: Generated {NUM_SAMPLES} verification samples in {elapsed:.1f}s")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)
    print("\nFiles created:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"  {f} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
