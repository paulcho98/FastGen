# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OmniAvatar Self-Forcing model for V2V lip sync distillation.

Overrides _prepare_training_data to build OmniAvatar-specific condition dicts
with audio, reference frames, spatial mask, masked video, and reference sequence.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

import os
import sys
import torch

from fastgen.methods.distribution_matching.self_forcing import SelfForcingModel
from fastgen.utils import instantiate
from fastgen.utils.distributed import synchronize, is_rank0
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.methods.config_self_forcing import ModelConfig


class OmniAvatarSelfForcingModel(SelfForcingModel):
    """Self-Forcing distillation for OmniAvatar V2V audio-driven lip sync.

    Inherits the full Self-Forcing training loop (rollout_with_gradient, VSD loss,
    fake_score/discriminator updates). Only overrides data preparation to handle
    OmniAvatar's condition dict format, and build_model to support a separate
    fake_score architecture (1.3B) from the teacher (14B).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def single_train_step(self, data: Dict[str, Any], iteration: int):
        """Combined fake_score + student update on student steps (1:5 ratio).

        Matches the original Self-Forcing training loop where the critic updates
        EVERY step, including on the generator (student) step. This gives a true
        1:5 ratio (5 critic updates per student update in a 5-step cycle).

        On non-student steps (iter % freq != 0): delegate to base class (fake_score only).
        On student steps (iter % freq == 0): run fake_score backward manually
        (freeing its graph to save memory), then return student loss for the
        trainer's backward. Both sets of gradients accumulate across grad_accum
        rounds; the trainer steps both optimizers at the end.
        """
        if iteration % self.config.student_update_freq != 0:
            # Critic-only step — unchanged from base class
            return super().single_train_step(data, iteration)

        # === Combined step: fake_score + student ===

        real_data, condition, neg_condition = self._prepare_training_data(data)
        grad_accum_rounds = getattr(self.config, "grad_accum_rounds", None) or 1

        # --- Step 1: Fake score forward + manual backward (frees graph) ---
        # Keep self.net.requires_grad_ unchanged (True) — same as the exclusive
        # pattern. The no_grad() inside _fake_score_discriminator_update_step is
        # sufficient. Toggling requires_grad on FSDP2 DTensors leaves stale
        # internal state that breaks gradient checkpointing recomputation.
        self.fake_score.train().requires_grad_(True)

        input_fs, t_student_fs, t_fs, eps_fs = self._generate_noise_and_time(real_data)

        fake_loss_map, _ = self._fake_score_discriminator_update_step(
            input_fs, t_student_fs, t_fs, eps_fs, real_data, condition=condition,
        )

        # Manual backward with same scaling the trainer uses for the student loss.
        # This ensures fake_score grads accumulate correctly across grad_accum rounds.
        # The graph is freed here so it doesn't overlap with the student forward.
        (fake_loss_map["total_loss"] / grad_accum_rounds).backward()

        # --- Step 2: Student forward (returned for trainer's backward) ---
        self.net.clear_caches()

        input_student, t_student, t, eps = self._generate_noise_and_time(real_data)

        student_loss_map, student_outputs = self._student_update_step(
            input_student, t_student, t, eps, data,
            condition=condition, neg_condition=neg_condition,
        )

        # Attach fake_score loss for logging (detached — no gradient)
        student_loss_map["fake_score_loss"] = fake_loss_map["total_loss"].detach()

        return student_loss_map, student_outputs

    def get_optimizers(self, iteration: int) -> list:
        """On student steps, return BOTH optimizers (trainer steps both)."""
        if iteration % self.config.student_update_freq == 0:
            return [self.net_optimizer, self.fake_score_optimizer]
        else:
            if self.config.gan_loss_weight_gen > 0:
                return [self.fake_score_optimizer, self.discriminator_optimizer]
            else:
                return [self.fake_score_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list:
        """On student steps, return BOTH schedulers (trainer steps both)."""
        if iteration % self.config.student_update_freq == 0:
            return [self.net_lr_scheduler, self.fake_score_lr_scheduler]
        else:
            if self.config.gan_loss_weight_gen > 0:
                return [self.fake_score_lr_scheduler, self.discriminator_lr_scheduler]
            else:
                return [self.fake_score_lr_scheduler]

    def build_model(self):
        """Override to instantiate fake_score from config.fake_score if provided.

        The base DMD2Model.build_model() always creates fake_score from
        self.teacher_config (= config.teacher), which is 14B. When
        config.fake_score is set, we use that instead (1.3B bidirectional).
        """
        super().build_model()

        fake_score_config = getattr(self.config, "fake_score", None)
        if fake_score_config is not None:
            logger.info("Re-instantiating fake_score from config.fake_score (1.3B)")
            with self._get_meta_init_context():
                self.fake_score = instantiate(fake_score_config)
            synchronize()

        # Load VAE for wandb visual logging (same logic as OmniAvatarDiffusionForcing)
        vae_path = getattr(self.config, "vae_path", "") or ""
        if vae_path and os.path.exists(vae_path):
            self._load_vae(vae_path)

    def _load_vae(self, vae_path: str):
        """Load WanVideoVAE for visual logging in wandb callback."""
        omni_root = os.environ.get("OMNIAVATAR_ROOT", "")
        if omni_root:
            for p in [os.path.join(omni_root, "OmniAvatar"), omni_root]:
                if p not in sys.path:
                    sys.path.insert(0, p)
        try:
            from models.wan_video_vae import WanVideoVAE
        except ImportError:
            logger.warning("Could not import WanVideoVAE — visual logging disabled.")
            return

        raw_vae = WanVideoVAE(z_dim=16)
        vae_state = torch.load(vae_path, map_location="cpu", weights_only=False)
        if any(k.startswith("encoder.") for k in vae_state):
            vae_state = {f"model.{k}": v for k, v in vae_state.items()}
        raw_vae.load_state_dict(vae_state)
        device_str = f"cuda:{self.device}" if isinstance(self.device, int) else str(self.device)
        raw_vae = raw_vae.to(device_str).eval()

        class VAEWrapper:
            def __init__(self, vae, device):
                self._vae = vae
                self._device = device
            def decode(self, x):
                with torch.no_grad():
                    return self._vae.decode([xi.float() for xi in x], self._device)
            def to(self, *args, **kwargs):
                return self

        self.net.vae = VAEWrapper(raw_vae, device_str)
        logger.info(f"Loaded WanVideoVAE from {vae_path} for visual logging")

    def validation_step(self, data: Dict[str, Any], iteration: int) -> tuple[dict, dict]:
        """Validation using CausVid's causal AR inference (chunk-by-chunk with KV cache).

        Uses CausVidModel._student_sample_loop which does proper AR inference:
        chunk-by-chunk denoising with KV cache updates, matching inference behavior.
        No teacher, no fake_score — just the student generating video.
        """
        import time
        from fastgen.methods.distribution_matching.causvid import CausVidModel

        t0 = time.time()

        real_data, condition, neg_condition = self._prepare_training_data(data)
        B, C, T, H, W = real_data.shape

        logger.info(f"[val] Starting CausVid AR inference (B={B}, T={T}, steps={self.config.student_sample_steps})")

        noise = torch.randn_like(real_data)
        context_noise = getattr(self.config, "context_noise", 0)

        with torch.no_grad():
            gen_data = CausVidModel.generator_fn(
                net=self.net,
                noise=noise,
                condition=condition,
                student_sample_steps=self.config.student_sample_steps,
                student_sample_type=self.config.student_sample_type,
                t_list=self.config.sample_t_cfg.t_list,
                context_noise=context_noise,
                precision_amp=self.precision_amp_infer,
            )

        t_gen = time.time() - t0
        logger.info(f"[val] AR inference done in {t_gen:.1f}s")

        loss_map = {"total_loss": torch.tensor(0.0, device=self.device)}
        outputs = {"gen_rand": gen_data}  # Already generated, not a callable

        return loss_map, outputs

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
