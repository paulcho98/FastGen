"""OmniAvatar Self-Forcing model with SyncNet Reward Forcing.

Extends OmniAvatarSelfForcingModel to weight the VSD loss by exp(beta * sync_c),
following the Reward Forcing (Re-DMD) framework from arxiv.org/abs/2512.04678.

The only behavioral change is in _student_update_step: after generating video and
computing teacher/fake_score predictions, it decodes the generated latents, scores
them with SyncNet, and passes exp(beta * sync_c) as additional_scale to the
existing variational_score_distillation_loss function.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

from fastgen.methods.omniavatar_self_forcing import OmniAvatarSelfForcingModel
from fastgen.methods.common_loss import variational_score_distillation_loss
import fastgen.utils.logging_utils as logger


class SyncNetReward:
    """Wrapper around SyncNet for computing Sync-C scores during training.

    Decodes video frames + audio into a temp .mp4 and runs SyncNet evaluation.
    OmniAvatar videos are already face-cropped 512x512, so face detection is skipped.
    """

    def __init__(self, syncnet_ckpt_path: str, device: str = "cpu"):
        import sys
        sys.path.insert(0, "/home/work/.local/eval_metrics")
        from eval.syncnet import SyncNetEval

        self.syncnet = SyncNetEval(device=device)
        self.syncnet.loadParameters(syncnet_ckpt_path)
        self.syncnet.eval()
        self.device = device
        logger.info(f"[SyncNetReward] Loaded SyncNet from {syncnet_ckpt_path}")

    @torch.no_grad()
    def compute_sync_c(self, video_frames: torch.Tensor, audio_path: str) -> float:
        """Compute Sync-C score from video frames tensor and audio file.

        Args:
            video_frames: [T, 3, H, W] float tensor in [-1, 1] or [0, 1]
            audio_path: Path to .wav audio file

        Returns:
            Sync-C confidence score (higher = better sync). Returns 0.0 on failure.
        """
        tmp_dir = tempfile.mkdtemp(prefix="syncreward_")
        tmp_video_noaudio = os.path.join(tmp_dir, "video_noaudio.mp4")
        tmp_video = os.path.join(tmp_dir, "video.mp4")
        tmp_eval = os.path.join(tmp_dir, "eval")

        try:
            frames = video_frames.detach().cpu().float()
            if frames.min() < 0:
                frames = (frames + 1) / 2  # [-1, 1] → [0, 1]
            frames = (frames.clamp(0, 1) * 255).byte()
            # [T, 3, H, W] → [T, H, W, 3]
            frames = frames.permute(0, 2, 3, 1).numpy()

            T, H, W, C = frames.shape

            # Write video frames (SyncNet expects 25fps)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_video_noaudio, fourcc, 25, (W, H))
            for i in range(T):
                writer.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
            writer.release()

            # Mux with audio
            subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-nostdin", "-y",
                    "-i", tmp_video_noaudio,
                    "-i", audio_path,
                    "-c:v", "copy", "-c:a", "aac", "-shortest",
                    tmp_video,
                ],
                check=True,
                capture_output=True,
            )

            # Run SyncNet (skip face detection — OmniAvatar is already face-cropped)
            _, sync_d, sync_c = self.syncnet.evaluate(
                video_path=tmp_video, temp_dir=tmp_eval
            )
            return sync_c

        except Exception as e:
            logger.warning(f"[SyncNetReward] Failed to compute Sync-C: {e}")
            return 0.0
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class OmniAvatarSyncRewardSFModel(OmniAvatarSelfForcingModel):
    """Self-Forcing model with SyncNet Reward Forcing.

    On student update steps, decodes the generated video, computes Sync-C
    with the ground-truth audio, and weights the VSD loss by exp(beta * sync_c).
    """

    def __init__(self, config):
        super().__init__(config)

        self.sync_beta = getattr(config, "sync_beta", 0.5)
        self._sync_reward: Optional[SyncNetReward] = None
        self._syncnet_ckpt_path = getattr(
            config, "syncnet_ckpt_path",
            "/home/work/.local/eval_metrics/checkpoints/auxiliary/syncnet_v2.model",
        )

    def _get_sync_reward(self) -> SyncNetReward:
        if self._sync_reward is None:
            self._sync_reward = SyncNetReward(
                self._syncnet_ckpt_path, device="cpu"
            )
        return self._sync_reward

    def _student_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        data: Dict[str, Any],
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
        iteration: int = 0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Student update with SyncNet reward weighting on VSD loss."""

        # --- Standard Self-Forcing student step (same as parent) ---
        gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)
        perturbed_data = self.net.noise_scheduler.forward_process(gen_data, eps, t)

        with torch.no_grad():
            fake_score_x0 = self.fake_score(perturbed_data, t, condition=condition, fwd_pred_type="x0")

        assert perturbed_data.dtype == data["real"].dtype == input_student.dtype
        assert t.dtype == t_student.dtype == self.net.noise_scheduler.t_precision

        teacher_x0, gan_loss_gen = self._compute_teacher_prediction_gan_loss(
            perturbed_data, t, condition=condition
        )

        if self.config.guidance_scale is not None:
            teacher_x0 = self._apply_classifier_free_guidance(
                perturbed_data, t, teacher_x0, neg_condition=neg_condition
            )

        # --- Sync Reward Computation ---
        additional_scale = None
        sync_c_val = 0.0

        audio_path = data.get("audio_path", [""])[0] if isinstance(data.get("audio_path"), list) else data.get("audio_path", "")
        if audio_path and os.path.exists(audio_path) and hasattr(self.net, "vae") and self.net.vae is not None:
            try:
                with torch.inference_mode():
                    gen_latents = gen_data[:1].detach().clone().float()
                    decoded = self.net.vae.decode(gen_latents)
                    # decoded: [B, 3, T_video, H, W] or list — handle both
                    if isinstance(decoded, (list, tuple)):
                        decoded = decoded[0]
                    if decoded.dim() == 5:
                        video_pixels = decoded[0]  # [3, T, H, W]
                    else:
                        video_pixels = decoded  # [3, T, H, W]
                    video_pixels = video_pixels.permute(1, 0, 2, 3)  # [T, 3, H, W]

                sync_c_val = self._get_sync_reward().compute_sync_c(video_pixels, audio_path)

                reward_weight = math.exp(self.sync_beta * sync_c_val)
                additional_scale = torch.tensor(
                    [reward_weight], dtype=gen_data.dtype, device=gen_data.device
                )
                if iteration <= 5 or iteration % 100 == 0:
                    logger.info(
                        f"[SyncReward] iter {iteration}: sync_c={sync_c_val:.4f}, "
                        f"weight=exp({self.sync_beta}*{sync_c_val:.4f})={reward_weight:.4f}"
                    )
            except Exception as e:
                logger.warning(f"[SyncReward] iter {iteration}: failed: {e}")

        # --- VSD loss with reward weighting ---
        vsd_loss = variational_score_distillation_loss(
            gen_data, teacher_x0, fake_score_x0, additional_scale=additional_scale
        )

        loss = vsd_loss + self.config.gan_loss_weight_gen * gan_loss_gen

        loss_map = {
            "total_loss": loss,
            "vsd_loss": vsd_loss,
            "gan_loss_gen": gan_loss_gen,
            "sync_c": torch.tensor(sync_c_val),
        }
        if additional_scale is not None:
            loss_map["sync_reward_weight"] = additional_scale.squeeze()

        outputs = self._get_outputs(gen_data, input_student, condition=condition)
        return loss_map, outputs
