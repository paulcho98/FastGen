import os
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
    s.mfcc_n = 13
    s.vshift = 15
    return s


def test_prep_video_shape_dtype(scorer):
    video = torch.randint(0, 256, (10, 3, 64, 64), dtype=torch.uint8)
    out = scorer._prep_video(video)
    assert out.shape == (1, 3, 10, 224, 224)
    assert out.dtype == torch.float32
    # Updated for joonson-parity: float in [0, 255], not [0, 1]
    assert out.min() >= 0.0 and out.max() <= 255.0


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


def test_reward_from_frames_returns_dict_with_MQ_alias(scorer):
    scorer.face_crop_size = 224
    scorer.audio_sample_rate = 16000
    scorer.target_sample_rate = 16000
    scorer.vshift = 15
    scorer.mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=13,
        melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160,
                   "n_mels": 40, "center": False},
    )

    # Stub out the heavy SyncNet-v2 forward — deterministic random embeddings
    class _FakeNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._anchor = torch.nn.Linear(1, 1)  # for _device()/_dtype() helpers
        def forward_lip(self, x):
            g = torch.Generator().manual_seed(0)
            return F.normalize(torch.randn(x.shape[0], 1024, generator=g), dim=-1)
        def forward_aud(self, x):
            g = torch.Generator().manual_seed(1)
            return F.normalize(torch.randn(x.shape[0], 1024, generator=g), dim=-1)
    object.__setattr__(scorer, "net", _FakeNet())

    video = torch.randint(0, 256, (81, 3, 128, 128), dtype=torch.uint8)
    audio = torch.randn(int(16000 * 3.24))
    out = scorer.reward_from_frames([video], [audio])

    assert set(out.keys()) >= {"sync_c", "MQ"}
    assert out["sync_c"].shape == (1,)
    assert torch.equal(out["sync_c"], out["MQ"])  # MQ is an alias


def test_reward_from_frames_batched(scorer):
    scorer.face_crop_size = 224
    scorer.audio_sample_rate = 16000
    scorer.target_sample_rate = 16000
    scorer.vshift = 15
    scorer.mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=13,
        melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160,
                   "n_mels": 40, "center": False},
    )

    class _FakeNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._anchor = torch.nn.Linear(1, 1)
        def forward_lip(self, x):
            return F.normalize(torch.randn(x.shape[0], 1024), dim=-1)
        def forward_aud(self, x):
            return F.normalize(torch.randn(x.shape[0], 1024), dim=-1)
    object.__setattr__(scorer, "net", _FakeNet())

    videos = [torch.randint(0, 256, (81, 3, 128, 128), dtype=torch.uint8) for _ in range(4)]
    audios = [torch.randn(int(16000 * 3.24)) for _ in range(4)]
    out = scorer.reward_from_frames(videos, audios)
    assert out["sync_c"].shape == (4,)


def test_reward_from_frames_mismatched_batch_raises(scorer):
    with pytest.raises(AssertionError, match="video/audio batch mismatch"):
        scorer.reward_from_frames(
            [torch.randint(0, 256, (81, 3, 64, 64), dtype=torch.uint8)],
            [torch.randn(16000), torch.randn(16000)],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not os.path.exists("/home/work/.local/eval_metrics/checkpoints/auxiliary/syncnet_v2.model"),
    reason="SyncNet-v2 checkpoint not present",
)
def test_real_scorer_gpu_runs():
    from fastgen.methods.reward.sync_c_scorer import SyncCScorer
    s = SyncCScorer(
        checkpoint_path="/home/work/.local/eval_metrics/checkpoints/auxiliary/syncnet_v2.model",
        device="cuda",
    )
    video = torch.randint(0, 256, (81, 3, 224, 224), dtype=torch.uint8)
    audio = torch.randn(int(16000 * 3.24))
    out = s.reward_from_frames([video], [audio])
    assert out["sync_c"].shape == (1,)
    assert torch.isfinite(out["sync_c"]).all()
