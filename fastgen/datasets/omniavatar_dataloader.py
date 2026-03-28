# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch Dataset for OmniAvatar V2V precomputed training data.

Each sample directory contains precomputed .pt files:
    - vae_latents_mask_all.pt: {input_latents [16,21,64,64], masked_latents [16,21,64,64]}
    - audio_emb_omniavatar.pt: {audio_emb [N,10752]} where N >= 81
    - text_emb.pt: tensor [1,512,4096]
    - ref_latents.pt: {ref_sequence_latents [16,21,64,64], metadata}
    - ode_path.pt: [4, 16, 21, 64, 64] (ODE trajectories for KD, optional)
"""

import os
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class OmniAvatarDataset(Dataset):
    """
    Dataset for OmniAvatar V2V training data with precomputed tensors.

    Returns dict with:
        real: [16, 21, 64, 64] — clean video latents (bf16)
        masked_video: [16, 21, 64, 64] — mouth-masked video latents (bf16)
        audio_emb: [81, 10752] — Wav2Vec2 audio features (bf16)
        text_embeds: [1, 512, 4096] — T5 text embedding (bf16)
        ref_sequence: [16, 21, 64, 64] — reference sequence latents (bf16, optional)
        mask: [64, 64] — spatial mask, LatentSync convention: 1=keep, 0=mask (float32)
        neg_text_embeds: [1, 512, 4096] — negative text embedding for CFG (bf16)
        path: [4, 16, 21, 64, 64] — ODE trajectory (bf16, optional, for KD training)
    """

    def __init__(
        self,
        data_list_path: str,
        latentsync_mask_path: str,
        neg_text_emb_path: str = None,
        use_ref_sequence: bool = True,
        load_ode_path: bool = False,
        num_video_frames: int = 81,
        latent_h: int = 64,
        latent_w: int = 64,
    ):
        self.use_ref_sequence = use_ref_sequence
        self.load_ode_path = load_ode_path
        self.num_video_frames = num_video_frames
        self.latent_h = latent_h
        self.latent_w = latent_w

        # Read sample directories from text file
        with open(data_list_path) as f:
            all_dirs = [line.strip() for line in f if line.strip()]

        # Filter out samples missing required files
        self.dirs = []
        required_files = ["vae_latents_mask_all.pt", "audio_emb_omniavatar.pt", "text_emb.pt"]
        for d in all_dirs:
            missing = [fn for fn in required_files if not os.path.exists(os.path.join(d, fn))]
            if missing:
                warnings.warn(f"Skipping {d}: missing {missing}")
            else:
                self.dirs.append(d)

        if len(self.dirs) < len(all_dirs):
            print(
                f"[OmniAvatarDataset] Kept {len(self.dirs)}/{len(all_dirs)} samples "
                f"({len(all_dirs) - len(self.dirs)} skipped due to missing files)"
            )

        # Load spatial mask once: PNG (256x256 RGB) -> single channel -> resize to latent res -> threshold
        # LatentSync convention: 1=keep (upper face), 0=mask (mouth region)
        mask_img = Image.open(latentsync_mask_path)
        mask_arr = np.array(mask_img, dtype=np.float32)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]  # take first channel
        mask_arr = mask_arr / 255.0  # normalize to [0, 1]
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        mask_resized = F.interpolate(
            mask_tensor, size=(latent_h, latent_w), mode="bilinear", align_corners=False
        )
        self.mask = (mask_resized.squeeze() > 0.5).float()  # [64, 64], float32

        # Load negative text embedding (for CFG)
        if neg_text_emb_path is not None and os.path.exists(neg_text_emb_path):
            neg_emb = torch.load(neg_text_emb_path, map_location="cpu", weights_only=False)
            if isinstance(neg_emb, dict):
                # Handle dict format if needed
                neg_emb = next(v for v in neg_emb.values() if isinstance(v, torch.Tensor))
            self.neg_text_embeds = neg_emb.to(torch.bfloat16)
        else:
            self.neg_text_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)

        # Ensure correct shape
        if self.neg_text_embeds.dim() == 2:
            self.neg_text_embeds = self.neg_text_embeds.unsqueeze(0)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx) -> dict:
        sample_dir = self.dirs[idx]

        try:
            # --- VAE latents ---
            vae_data = torch.load(
                os.path.join(sample_dir, "vae_latents_mask_all.pt"),
                map_location="cpu",
                weights_only=False,
            )
            real = vae_data["input_latents"].to(torch.bfloat16)  # [16, 21, 64, 64]
            masked_video = vae_data["masked_latents"].to(torch.bfloat16)  # [16, 21, 64, 64]

            # --- Audio embeddings ---
            audio_data = torch.load(
                os.path.join(sample_dir, "audio_emb_omniavatar.pt"),
                map_location="cpu",
                weights_only=False,
            )
            audio_emb = audio_data["audio_emb"][: self.num_video_frames]  # [81, 10752]
            audio_emb = audio_emb.to(torch.bfloat16)

            # --- Text embedding ---
            text_emb = torch.load(
                os.path.join(sample_dir, "text_emb.pt"),
                map_location="cpu",
                weights_only=False,
            )
            if isinstance(text_emb, dict):
                text_emb = next(v for v in text_emb.values() if isinstance(v, torch.Tensor))
            text_emb = text_emb.to(torch.bfloat16)
            # Ensure shape [1, 512, 4096]
            if text_emb.dim() == 2:
                text_emb = text_emb.unsqueeze(0)

            result = {
                "real": real,
                "masked_video": masked_video,
                "audio_emb": audio_emb,
                "text_embeds": text_emb,
                "mask": self.mask,  # shared across all samples, float32
                "neg_text_embeds": self.neg_text_embeds.clone(),
            }

            # Audio file path for wandb video logging with audio muxing.
            # Always include key so default_collate doesn't fail on mixed-key batches.
            audio_wav_path = os.path.join(sample_dir, "audio.wav")
            result["audio_path"] = audio_wav_path if os.path.exists(audio_wav_path) else ""

            # --- Reference sequence latents (optional) ---
            if self.use_ref_sequence:
                ref_path = os.path.join(sample_dir, "ref_latents.pt")
                if os.path.exists(ref_path):
                    ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
                    result["ref_sequence"] = ref_data["ref_sequence_latents"].to(torch.bfloat16)
                else:
                    # Fallback: zeros with same shape as real latents
                    result["ref_sequence"] = torch.zeros_like(real)

            # --- ODE trajectory (optional, for KD training) ---
            if self.load_ode_path:
                ode_path_file = os.path.join(sample_dir, "ode_path.pt")
                if os.path.exists(ode_path_file):
                    result["path"] = torch.load(ode_path_file, map_location="cpu", weights_only=False).to(
                        torch.bfloat16
                    )
                else:
                    # Also check path.pth (alternative filename)
                    alt_path_file = os.path.join(sample_dir, "path.pth")
                    if os.path.exists(alt_path_file):
                        result["path"] = torch.load(alt_path_file, map_location="cpu", weights_only=False).to(
                            torch.bfloat16
                        )

            return result

        except Exception as e:
            warnings.warn(f"Error loading sample {sample_dir}: {e}")
            # Return None; collate_fn should filter these out
            return None


class OmniAvatarDataLoader:
    """Infinite-iterator DataLoader wrapper with DistributedSampler support.

    FastGen's trainer expects an infinite iterator. This class wraps a standard
    DataLoader and yields batches indefinitely, cycling through the dataset.
    """

    def __init__(
        self,
        data_list_path: str = None,
        datatags: list = None,
        latentsync_mask_path: str = None,
        batch_size: int = 1,
        num_workers: int = 4,
        load_ode_path: bool = False,
        **kwargs,
    ):
        # Support both data_list_path and datatags (list of paths)
        if data_list_path is None and datatags is not None:
            data_list_path = datatags[0] if isinstance(datatags, list) else datatags
        assert data_list_path is not None, "Must provide data_list_path or datatags"
        assert latentsync_mask_path is not None, "Must provide latentsync_mask_path"

        self.dataset = OmniAvatarDataset(
            data_list_path=data_list_path,
            latentsync_mask_path=latentsync_mask_path,
            load_ode_path=load_ode_path,
            **kwargs,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Use DistributedSampler for multi-GPU training
        if dist.is_initialized():
            self._sampler = DistributedSampler(self.dataset, shuffle=True)
            shuffle = False
        else:
            self._sampler = None
            shuffle = True

        def collate_fn(batch):
            """Filter out None samples from failed loads."""
            valid = [s for s in batch if s is not None]
            if not valid:
                return {}
            return torch.utils.data.default_collate(valid)

        self._dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=self._sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def __iter__(self):
        """Infinite iterator — cycles through the dataset."""
        epoch = 0
        while True:
            if self._sampler is not None:
                self._sampler.set_epoch(epoch)
            yield from self._dataloader
            epoch += 1

    def __len__(self):
        return len(self.dataset)


# Keep the simple function for backward compatibility
def create_omniavatar_dataloader(
    data_list_path: str,
    latentsync_mask_path: str,
    batch_size: int = 1,
    num_workers: int = 4,
    load_ode_path: bool = False,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for OmniAvatar training data (non-infinite, no DDP support).

    For training, prefer OmniAvatarDataLoader which provides infinite iteration
    and DistributedSampler support.
    """
    dataset = OmniAvatarDataset(
        data_list_path=data_list_path,
        latentsync_mask_path=latentsync_mask_path,
        load_ode_path=load_ode_path,
        **kwargs,
    )

    def collate_fn(batch):
        """Filter out None samples from failed loads."""
        valid = [s for s in batch if s is not None]
        if not valid:
            return {}
        return torch.utils.data.default_collate(valid)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
