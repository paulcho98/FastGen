import pytest
import torch
import torch.nn as nn
import torchaudio
from unittest.mock import patch

from fastgen.methods.reward.sync_c_scorer import SyncCScorer


@pytest.fixture
def scorer():
    s = SyncCScorer.__new__(SyncCScorer)
    nn.Module.__init__(s)  # Initialize the Module base class
    # Stub net parameters so _device() / _dtype() work
    stub_net = torch.nn.Linear(1, 1)  # arbitrary — just needs one parameter on CPU
    object.__setattr__(s, "net", stub_net)
    s.face_crop_size = 224
    s.audio_sample_rate = 16000
    s.target_sample_rate = 16000
    s.mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=13,
        melkwargs={
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "n_mels": 40,
            "center": False,
        },
    )
    s.vshift = 15
    return s


def test_prep_video_shape_dtype(scorer):
    video = torch.randint(0, 256, (10, 3, 64, 64), dtype=torch.uint8)
    out = scorer._prep_video(video)
    assert out.shape == (1, 3, 10, 224, 224)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_prep_audio_shape_16k(scorer):
    # 3.24 s at 16 kHz
    audio = torch.randn(int(16000 * 3.24))
    out = scorer._prep_audio(audio)
    # Expected MFCC length: (L - win) // hop + 1 = (51840 - 400) // 160 + 1 = 322
    assert out.shape[:3] == (1, 1, 13)
    assert 280 <= out.shape[-1] <= 340, f"MFCC length {out.shape[-1]} out of expected band"


def test_prep_audio_resamples_from_48k(scorer):
    # Override audio_sample_rate for this test
    scorer.audio_sample_rate = 48000
    audio = torch.randn(int(48000 * 3.24))
    out = scorer._prep_audio(audio)
    assert 280 <= out.shape[-1] <= 340
