import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def test_lip_windows_81_frames(scorer):
    video = torch.zeros(1, 3, 81, 224, 224)
    out = scorer._lip_windows(video)
    assert out.shape == (77, 3, 5, 224, 224)


def test_aud_windows_length(scorer):
    mfcc = torch.zeros(1, 1, 13, 324)
    out = scorer._aud_windows(mfcc)
    # stride 4, window 20: (324 - 20) / 4 + 1 = 77
    assert out.shape == (77, 1, 13, 20)


def test_lip_windows_rejects_short_clip(scorer):
    with pytest.raises(ValueError, match="at least 5"):
        scorer._lip_windows(torch.zeros(1, 3, 4, 224, 224))


def test_offset_search_returns_scalar(scorer):
    scorer.vshift = 15
    torch.manual_seed(0)
    lip = F.normalize(torch.randn(50, 1024), dim=-1)
    aud = F.normalize(torch.randn(50, 1024), dim=-1)
    conf = scorer._offset_search(lip, aud)
    assert conf.ndim == 0
    assert torch.isfinite(conf)


def test_offset_search_perfect_alignment_scores_high(scorer):
    scorer.vshift = 15
    torch.manual_seed(1)
    emb = F.normalize(torch.randn(50, 1024), dim=-1)
    # Perfect sync: lip == aud → min distance at shift 0 is exactly 0
    conf = scorer._offset_search(emb, emb)
    assert conf > 0.5, f"confidence should be clearly positive, got {conf}"
