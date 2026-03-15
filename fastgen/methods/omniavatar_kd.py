# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OmniAvatar Causal KD model for Stage 1 (ODE initialization).

Overrides single_train_step to build OmniAvatar-specific condition dicts
from the precomputed ODE trajectory data.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING, Callable
from functools import partial

import torch
import torch.nn.functional as F

from fastgen.methods.knowledge_distillation.KD import CausalKDModel
from fastgen.methods.distribution_matching.causvid import CausVidModel
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.config import BaseModelConfig as ModelConfig


class OmniAvatarKDModel(CausalKDModel):
    """Causal KD for OmniAvatar — Stage 1 of Self-Forcing pipeline.

    Trains the causal 1.3B student to match ODE trajectories from the 14B teacher.
    Overrides single_train_step to handle OmniAvatar's condition dict format
    (audio, reference frames, spatial mask, masked video, reference sequence).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def _build_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build OmniAvatar condition dict from data batch.

        Args:
            data: Batch from OmniAvatarDataset with ODE paths.

        Returns:
            Condition dict for OmniAvatar networks.
        """
        real_data = data["real"]  # [B, 16, 21, 64, 64]
        ref_latent = real_data[:, :, :1, :, :]  # [B, 16, 1, H, W]

        mask = data["mask"]
        if mask.dim() == 3:
            mask = mask[0]

        condition = {
            "text_embeds": data["text_embeds"].squeeze(1) if data["text_embeds"].dim() == 4 else data["text_embeds"],
            "audio_emb": data["audio_emb"],
            "ref_latent": ref_latent,
            "mask": mask,
            "masked_video": data["masked_video"],
        }
        if "ref_sequence" in data:
            condition["ref_sequence"] = data["ref_sequence"]

        return condition

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        noise = torch.randn_like(gen_data, dtype=self.precision)
        context_noise = getattr(self.config, "context_noise", 0)
        gen_rand_func = partial(
            CausVidModel.generator_fn,
            net=self.net_inference,
            noise=noise,
            condition=condition,
            student_sample_steps=self.config.student_sample_steps,
            t_list=self.config.sample_t_cfg.t_list,
            context_noise=context_noise,
            precision_amp=self.precision_amp_infer,
        )
        return {"gen_rand": gen_rand_func, "input_rand": noise, "gen_rand_train": gen_data}

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """Single training step for OmniAvatar causal KD.

        Builds the OmniAvatar condition dict before running the standard
        CausalKDModel training logic.
        """
        denoise_path = data["path"]  # [B, num_steps, 16, 21, 64, 64]
        denoised_data = data["real"]  # [B, 16, 21, 64, 64]
        condition = self._build_condition(data)

        batch_size, num_frames = denoise_path.shape[0], denoise_path.shape[3]
        chunk_size = self.net.chunk_size

        # Sample inhomogeneous timesteps for causal training
        t_inhom, ids = self.net.noise_scheduler.sample_t_inhom(
            batch_size,
            num_frames,
            chunk_size,
            sample_steps=self.config.student_sample_steps,
            t_list=self.config.sample_t_cfg.t_list,
            device=self.device,
            dtype=denoise_path.dtype,
        )

        # Gather noisy data from path at sampled timesteps
        expand_shape = [ids.shape[0], 1, 1, ids.shape[1]] + [1] * max(0, denoise_path.ndim - 4)
        ids = ids.view(expand_shape).expand(-1, -1, *denoise_path.shape[2:])
        denoise_path_all = torch.cat([denoise_path, denoised_data.unsqueeze(1)], dim=1)
        noisy_data = torch.gather(denoise_path_all, 1, ids).squeeze(1)

        # Student forward
        gen_data = self.gen_data_from_net(noisy_data, t_inhom, condition=condition)

        # L2 loss
        loss = 0.5 * F.mse_loss(gen_data, denoised_data, reduction="mean")

        loss_map = {"total_loss": loss, "recon_loss": loss}
        outputs = self._get_outputs(gen_data, condition=condition)

        return loss_map, outputs
