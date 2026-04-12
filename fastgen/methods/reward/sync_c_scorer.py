"""SyncNet-v2 sync-C reward scorer. Design reference:
/home/work/.local/hyunbin/Reward-Forcing/docs/sync_c_scorer_design.md
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from fastgen.methods.reward.syncnet_v2 import SyncNetV2


class SyncCScorer(nn.Module):
    """Drop-in replacement for VideoVLMRewardInference.

    Scores talking-head lip-sync using SyncNet-v2's offset-margin confidence,
    as a detached scalar multiplier on the DMD MSE term.

    Pre-conditions:
      - input video frames are face-aligned (centered); no detection here
      - audio waveforms are the driving audio, temporally aligned with frame 0
      - generator emits 25 fps; scorer does NO FPS conversion
    """

    def __init__(
        self,
        checkpoint_path: str,
        input_fps: float = 25.0,
        audio_sample_rate: int = 16000,
        face_crop_size: int = 224,
        vshift: int = 15,
        mfcc_n: int = 13,
        mfcc_hop_ms: float = 10.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert input_fps == 25.0, "SyncNet-v2 is native 25 fps; resample upstream"

        self.net = SyncNetV2()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, nn.Module):
            state = state.state_dict()
        self.net.load_state_dict(state, strict=True)
        self.net.eval().to(device=device, dtype=dtype)
        for p in self.net.parameters():
            p.requires_grad_(False)

        self.audio_sample_rate = audio_sample_rate
        self.target_sample_rate = 16000
        self.face_crop_size = face_crop_size
        self.vshift = vshift
        self.device = device
        self.dtype = dtype

        # torchaudio MFCC: 13 coeffs, 25 ms win / 10 ms hop at 16 kHz
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.target_sample_rate,
            n_mfcc=mfcc_n,
            melkwargs={
                "n_fft": 512,
                "win_length": int(0.025 * self.target_sample_rate),
                "hop_length": int(mfcc_hop_ms / 1000 * self.target_sample_rate),
                "n_mels": 40,
                "center": False,
            },
        ).to(device)

    def _prep_video(self, video: torch.Tensor) -> torch.Tensor:
        """[F, 3, H, W] uint8 -> [1, 3, F, 224, 224] float in [0, 1]."""
        video = video.to(self.device).float() / 255.0
        video = F.interpolate(
            video, size=(self.face_crop_size, self.face_crop_size),
            mode="bilinear", align_corners=False,
        )
        return video.permute(1, 0, 2, 3).unsqueeze(0)

    def _prep_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """[L] float waveform -> [1, 1, 13, M] MFCC."""
        audio = audio.to(self.device).float()
        if self.audio_sample_rate != self.target_sample_rate:
            audio = torchaudio.functional.resample(
                audio, self.audio_sample_rate, self.target_sample_rate,
            )
        mfcc = self.mfcc(audio.unsqueeze(0))  # [1, 13, M]
        return mfcc.unsqueeze(1)              # [1, 1, 13, M]
