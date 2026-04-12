# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""OmniAvatar Self-Forcing with Re-DMD reward weighting.

Overrides `_student_update_step` (in Task 7) to scale the VSD loss by
`exp(beta * sync_c_detached)`, matching the Reward-Forcing paper formulation:

    L_gen = 0.5 * exp(beta * r) * MSE(gen_latent, (gen_latent - DMD_grad).detach())

where r is a scalar sync-C from SyncNet-v2 (detached, no gradient). See
`docs/superpowers/plans/2026-04-12-sync-c-reward-redmd.md` for the plan and
`/home/work/.local/hyunbin/Reward-Forcing/docs/sync_c_scorer_design.md` for the
scorer contract.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from fastgen.methods.omniavatar_self_forcing import OmniAvatarSelfForcingModel
from fastgen.methods.reward.sync_c_scorer import SyncCScorer

logger = logging.getLogger(__name__)


class OmniAvatarSelfForcingReDMD(OmniAvatarSelfForcingModel):
    """Rewarded variant of OmniAvatar Self-Forcing.

    All Re-DMD-specific logic (reward scorer instantiation, reward-weighting of
    the VSD loss) lives in this subclass. The base class stays unchanged.
    """

    def __init__(self, config):
        super().__init__(config)
        self.reward_scorer: Optional[SyncCScorer] = None
        # EMA running mean of sync_c, used only when config.center_reward is True.
        self._reward_running_mean: Optional[float] = None

    def build_model(self):
        """Build base-class components, then load the SyncCScorer."""
        super().build_model()

        rcfg = getattr(self.config, "reward", None)
        if rcfg is None or not getattr(rcfg, "enabled", True):
            logger.info("Re-DMD reward disabled — running as vanilla OmniAvatar SF.")
            return

        device_str = (
            f"cuda:{self.device}" if isinstance(self.device, int) else str(self.device)
        )
        self.reward_scorer = SyncCScorer(
            checkpoint_path=rcfg.checkpoint_path,
            input_fps=getattr(rcfg, "input_fps", 25.0),
            audio_sample_rate=getattr(rcfg, "audio_sample_rate", 16000),
            vshift=getattr(rcfg, "vshift", 15),
            device=device_str,
            dtype=torch.float32,
        )
        logger.info(
            f"SyncCScorer loaded: beta={self.config.reward_beta}, "
            f"vshift={getattr(rcfg, 'vshift', 15)}, ckpt={rcfg.checkpoint_path}"
        )

    def _apply_reward_weighting(
        self,
        vsd_loss: torch.Tensor,
        videos: Any,
        audios: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute `exp(beta * sync_c)` from the reward scorer and multiply vsd_loss.

        Returns:
            (weighted_loss, log_map)

            `log_map` entries are python floats intended for the trainer's
            wandb loss dict (parallel to Reward-Forcing's reward_MQ_* and
            reward_weight_* keys).
        """
        with torch.no_grad():
            reward = self.reward_scorer.reward_from_frames(videos, audios)
        sync_c = reward["sync_c"].detach().float()  # [B]

        beta = float(self.config.reward_beta)

        # Optional centering (EMA subtraction — keeps mean weight ≈ 1)
        if getattr(self.config, "center_reward", False):
            ema_alpha = 0.9
            batch_mean = sync_c.mean().item()
            if self._reward_running_mean is None:
                self._reward_running_mean = batch_mean
            else:
                self._reward_running_mean = (
                    ema_alpha * self._reward_running_mean
                    + (1.0 - ema_alpha) * batch_mean
                )
            sync_c = sync_c - self._reward_running_mean

        # Optional clamping (bounds exp(beta * r))
        clamp = getattr(self.config, "clamp_reward", None)
        if clamp is not None:
            sync_c = sync_c.clamp(clamp[0], clamp[1])

        weight = torch.exp(beta * sync_c)  # [B]
        mean_weight = weight.mean()
        weighted = mean_weight * vsd_loss

        log_map = {
            "reward_sync_c_mean": float(sync_c.mean().item()),
            "reward_sync_c_min": float(sync_c.min().item()),
            "reward_sync_c_max": float(sync_c.max().item()),
            "reward_weight_mean": float(mean_weight.item()),
            "reward_weight_min": float(weight.min().item()),
            "reward_weight_max": float(weight.max().item()),
            "vsd_loss_unweighted": float(vsd_loss.detach().item()),
            "vsd_loss_weighted": float(weighted.detach().item()),
        }
        return weighted, log_map
