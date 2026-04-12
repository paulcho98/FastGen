import pytest
import torch
import torchaudio
from unittest.mock import patch

from fastgen.methods.reward.sync_c_scorer import SyncCScorer


@pytest.fixture
def scorer():
    # Mock checkpoint loading and SyncNetV2 to skip actual model loading
    with patch('fastgen.methods.reward.sync_c_scorer.SyncNetV2'), \
         patch('torch.load', return_value={}):
        return SyncCScorer(
            checkpoint_path='/dev/null',
            device='cpu',
            dtype=torch.float32,
        )


def test_prep_video_shape_dtype(scorer):
    video = torch.randint(0, 256, (81, 3, 512, 512), dtype=torch.uint8)
    out = scorer._prep_video(video)
    assert out.shape == (1, 3, 81, 224, 224)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_prep_audio_shape_16k(scorer):
    # 3.24 s at 16 kHz
    audio = torch.randn(int(16000 * 3.24))
    out = scorer._prep_audio(audio)
    # Expected MFCC length ≈ (51840 - 400) / 160 + 1 = 323
    assert out.shape[:3] == (1, 1, 13)
    assert 280 <= out.shape[-1] <= 340, f"MFCC length {out.shape[-1]} out of expected band"


def test_prep_audio_resamples_from_48k(scorer):
    # Override audio_sample_rate for this test
    scorer.audio_sample_rate = 48000
    audio = torch.randn(int(48000 * 3.24))
    out = scorer._prep_audio(audio)
    assert 280 <= out.shape[-1] <= 340
