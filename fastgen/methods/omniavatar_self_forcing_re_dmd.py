# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""OmniAvatar Self-Forcing with Re-DMD reward weighting (Task 7: _student_update_step).

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
import torch.distributed as dist

from fastgen.methods.omniavatar_self_forcing import OmniAvatarSelfForcingModel
from fastgen.methods.reward.sync_c_scorer import SyncCScorer

logger = logging.getLogger(__name__)


def _reduce(x: torch.Tensor, op) -> torch.Tensor:
    """Reduce a 0-d tensor across ranks, returning a 0-d tensor.

    No-op when torch.distributed is not initialized (single-rank / unit test
    path). For SUM, divides by world_size to get a mean; for MIN/MAX, leaves
    the value as-is.
    """
    y = x.detach().clone().float().reshape(())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(y, op=op)
        if op == dist.ReduceOp.SUM:
            y = y / dist.get_world_size()
    return y


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

        sync_c_mean_r = _reduce(sync_c.mean(), dist.ReduceOp.SUM)
        sync_c_min_r  = _reduce(sync_c.min(),  dist.ReduceOp.MIN)
        sync_c_max_r  = _reduce(sync_c.max(),  dist.ReduceOp.MAX)
        weight_mean_r = _reduce(weight.mean(), dist.ReduceOp.SUM)
        weight_min_r  = _reduce(weight.min(),  dist.ReduceOp.MIN)
        weight_max_r  = _reduce(weight.max(),  dist.ReduceOp.MAX)

        log_map = {
            "reward_sync_c_mean": float(sync_c_mean_r.item()),
            "reward_sync_c_min": float(sync_c_min_r.item()),
            "reward_sync_c_max": float(sync_c_max_r.item()),
            "reward_weight_mean": float(weight_mean_r.item()),
            "reward_weight_min": float(weight_min_r.item()),
            "reward_weight_max": float(weight_max_r.item()),
            "vsd_loss_unweighted": float(vsd_loss.detach().item()),
            "vsd_loss_weighted": float(weighted.detach().item()),
        }
        return weighted, log_map

    # ------------------------------------------------------------------
    # Task 7: _student_update_step with VAE decode + sync-C reward
    # ------------------------------------------------------------------

    def _student_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        data: Dict[str, Any],
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ):
        """Re-DMD override: standard VSD loss, then scaled by exp(beta * sync_c).

        When ``self.reward_scorer is None`` (reward disabled), falls back to the
        base DMD2 behavior (no VAE decode, no reward keys in loss_map).

        Reproduces DMD2Model._student_update_step inline so we can intervene
        between vsd_loss and total_loss without fragile post-hoc rescaling.

        NOTE: per-rank EMA on ``self._reward_running_mean`` means multi-rank runs
        will have independent running means. For the typical OmniAvatar setup
        (batch_size=1 per GPU, 4 GPUs) this is close enough to a batch mean.
        """
        from fastgen.methods.common_loss import variational_score_distillation_loss

        # ---- reproduce DMD2Model._student_update_step lines 234-258 ----
        gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)

        perturbed_data = self.net.noise_scheduler.forward_process(gen_data, eps, t)

        with torch.no_grad():
            fake_score_x0 = self.fake_score(
                perturbed_data, t, condition=condition, fwd_pred_type="x0"
            )

        teacher_x0, gan_loss_gen = self._compute_teacher_prediction_gan_loss(
            perturbed_data, t, condition=condition
        )

        if getattr(self.config, "guidance_scale", None) is not None:
            teacher_x0 = self._apply_classifier_free_guidance(
                perturbed_data, t, teacher_x0, neg_condition=neg_condition
            )

        vsd_loss = variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0)

        # ---- Re-DMD reward weighting (intervenes here) ----
        reward_log: Dict[str, Any] = {}
        if self.reward_scorer is not None and "audio_waveform" in data:
            with torch.no_grad():
                pixels = self._decode_gen_to_pixels(gen_data)           # [B, 3, T_pix, H, W] in [-1, 1]
                videos_u8 = self._pixels_to_uint8_face_crop(pixels)     # list of B [T_pix, 3, H, W] uint8
                audios = list(data["audio_waveform"].unbind(0))         # list of B [L]
            weighted_vsd, reward_log = self._apply_reward_weighting(vsd_loss, videos_u8, audios)
        else:
            weighted_vsd = vsd_loss
            reward_log = {"vsd_loss_unweighted": float(vsd_loss.detach().item())}

        loss = weighted_vsd + self.config.gan_loss_weight_gen * gan_loss_gen

        loss_map: Dict[str, Any] = {
            "total_loss": loss,
            "vsd_loss": vsd_loss.detach(),
            "vsd_loss_weighted": weighted_vsd.detach(),
            "gan_loss_gen": (
                gan_loss_gen.detach()
                if torch.is_tensor(gan_loss_gen)
                else torch.tensor(float(gan_loss_gen))
            ),
            **reward_log,
        }
        outputs = self._get_outputs(gen_data, input_student, condition=condition)
        return loss_map, outputs

    def _decode_gen_to_pixels(self, gen_latent: torch.Tensor) -> torch.Tensor:
        """Decode ``[B, 16, T_lat, H_lat, W_lat]`` latents to ``[B, 3, T_pix, H_pix, W_pix]``.

        Uses the VAEWrapper loaded by the base class's ``_load_vae``. That wrapper's
        ``decode`` takes a list of per-sample tensors and returns a list of decoded
        pixel tensors in ``[-1, 1]`` range (WanVideoVAE native output range).

        Raises a clear RuntimeError when ``self.net.vae`` is missing, so the user
        knows to set ``config.vae_path`` in the rewarded config.
        """
        if not hasattr(self.net, "vae") or self.net.vae is None:
            raise RuntimeError(
                "Re-DMD needs VAE for reward decode. Set config.vae_path in the "
                "rewarded config so _load_vae instantiates the VAEWrapper."
            )
        decoded_list = self.net.vae.decode(
            [gen_latent[b].float() for b in range(gen_latent.shape[0])]
        )
        return torch.stack(decoded_list, dim=0)

    def _pixels_to_uint8_face_crop(self, pixels: torch.Tensor) -> list:
        """``[B, 3, T_pix, H_pix, W_pix]`` float in ``[-1, 1]`` -> list of B tensors
        each ``[T_pix, 3, H, W]`` uint8.

        Face alignment is already done upstream (pre-aligned training data); we
        only range-map and permute to match the ``[F, 3, H, W]`` uint8 format
        expected by ``SyncCScorer.reward_from_frames``.
        """
        pixels = pixels.clamp(-1.0, 1.0)
        u8 = ((pixels + 1.0) * 127.5).to(torch.uint8)          # [B, 3, T, H, W]
        u8 = u8.permute(0, 2, 1, 3, 4).contiguous()            # [B, T, 3, H, W]
        return [u8[b] for b in range(u8.shape[0])]
