# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OmniAvatar Self-Forcing model for V2V lip sync distillation.

Overrides _prepare_training_data to build OmniAvatar-specific condition dicts
with audio, reference frames, spatial mask, masked video, and reference sequence.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

import torch

from fastgen.methods.distribution_matching.self_forcing import SelfForcingModel
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.methods.config_self_forcing import ModelConfig


class OmniAvatarSelfForcingModel(SelfForcingModel):
    """Self-Forcing distillation for OmniAvatar V2V audio-driven lip sync.

    Inherits the full Self-Forcing training loop (rollout_with_gradient, VSD loss,
    fake_score/discriminator updates). Only overrides data preparation to handle
    OmniAvatar's condition dict format.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._latentsync_mask = None

    def _prepare_training_data(self, data: Dict[str, Any]) -> tuple[torch.Tensor, Any, Any]:
        """Build OmniAvatar condition and neg_condition dicts from dataset output.

        The OmniAvatar dataset returns:
            real: [B, 16, 21, 64, 64] — clean video latents
            masked_video: [B, 16, 21, 64, 64] — mouth-masked video latents
            audio_emb: [B, 81, 10752] — Wav2Vec2 audio features
            text_embeds: [B, 1, 512, 4096] — T5 text embedding
            ref_sequence: [B, 16, 21, 64, 64] — reference sequence latents
            mask: [B, 64, 64] — spatial mask (LatentSync convention: 1=keep, 0=generate)
            neg_text_embeds: [B, 1, 512, 4096] — negative text embedding

        Returns:
            real_data: [B, 16, 21, 64, 64]
            condition: dict with all V2V conditioning
            neg_condition: dict with null audio + negative text
        """
        real_data = data["real"]
        B = real_data.shape[0]

        # Reference latent: first frame of clean video
        ref_latent = real_data[:, :, :1, :, :]  # [B, 16, 1, H, W]

        # Spatial mask — use first sample's mask (same across batch)
        mask = data["mask"]
        if mask.dim() == 3:  # [B, H, W] from DataLoader batching
            mask = mask[0]  # [H, W] — same for all samples

        # Positive condition
        condition = {
            "text_embeds": data["text_embeds"].squeeze(1) if data["text_embeds"].dim() == 4 else data["text_embeds"],
            "audio_emb": data["audio_emb"],
            "ref_latent": ref_latent,
            "mask": mask,
            "masked_video": data["masked_video"],
        }
        if "ref_sequence" in data:
            condition["ref_sequence"] = data["ref_sequence"]

        # Negative condition: null audio, negative text, same spatial conditioning
        neg_condition = {
            "text_embeds": data["neg_text_embeds"].squeeze(1) if data["neg_text_embeds"].dim() == 4 else data["neg_text_embeds"],
            "audio_emb": torch.zeros_like(data["audio_emb"]),
            "ref_latent": ref_latent,
            "mask": mask,
            "masked_video": data["masked_video"],
        }
        if "ref_sequence" in data:
            neg_condition["ref_sequence"] = data["ref_sequence"]

        return real_data, condition, neg_condition
