"""Tests for the optional raw-audio-waveform path in OmniAvatarDataset."""
import os
import pytest
import tempfile
import wave

import torch
import numpy as np


def _save_wav_file(path, audio_data, sample_rate):
    """Manually save audio data to a WAV file (no torchcodec needed)."""
    # audio_data: [channels, num_samples] tensor or numpy array in [-1, 1]
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.cpu().numpy()

    # Convert to int16
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # Handle shape
    if audio_int16.ndim == 2:
        # [channels, samples] -> [samples, channels]
        audio_int16 = audio_int16.T
        num_channels = audio_data.shape[0]
    else:
        num_channels = 1

    with wave.open(path, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


# ---- Unit test with synthetic wav (no real training data required) ----

def test_load_and_pad_raw_audio_unit():
    """Direct test of the waveform-load logic: 3.0 s of 48 kHz audio at a tmp path
    should be resampled, mono'd, and padded/truncated to exactly 51840 samples
    at 16 kHz (= 81 frames at 25 fps)."""
    from fastgen.datasets.omniavatar_dataloader import _load_raw_waveform_for_reward

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        # 3.0 s of stereo 48 kHz noise
        audio_data = torch.randn(2, int(3.0 * 48000))
        _save_wav_file(wav_path, audio_data, 48000)

        out = _load_raw_waveform_for_reward(
            wav_path, target_sample_rate=16000, target_length=51840,
        )

        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
        assert out.ndim == 1
        assert out.shape[0] == 51840


def test_load_and_pad_raw_audio_pads_short_clip():
    from fastgen.datasets.omniavatar_dataloader import _load_raw_waveform_for_reward

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        # 1.0 s of mono 16 kHz (shorter than 81 frames of audio)
        audio_data = torch.randn(1, 16000)
        _save_wav_file(wav_path, audio_data, 16000)

        out = _load_raw_waveform_for_reward(
            wav_path, target_sample_rate=16000, target_length=51840,
        )

        assert out.shape[0] == 51840
        # Last 35840 samples should be exact zero (pad)
        assert (out[16000:] == 0).all()


# ---- Smoke test with real training data (skipped if data unmounted) ----

_VIDEO_LIST = "/home/work/stableavatar_data/v2v_training_data/video_square_path.txt"
_MASK = "/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png"


@pytest.mark.skipif(
    not (os.path.exists(_VIDEO_LIST) and os.path.exists(_MASK)),
    reason="OmniAvatar training data not mounted",
)
def test_dataset_emits_audio_waveform_when_opted_in():
    from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataset

    # Minimal init — match the keys the constructor actually needs.
    # If your constructor has required args not listed here, add them.
    ds = OmniAvatarDataset(
        data_list_path=_VIDEO_LIST,
        latentsync_mask_path=_MASK,
        use_ref_sequence=True,
        load_raw_audio=True,
        raw_audio_sample_rate=16000,
        raw_audio_num_frames=81,
        raw_audio_fps=25.0,
    )
    sample = ds[0]
    assert "audio_waveform" in sample
    wav = sample["audio_waveform"]
    assert wav.dtype == torch.float32
    assert wav.shape[0] == 51840, f"got {wav.shape[0]}"
    assert wav.abs().max() <= 1.0 + 1e-3


@pytest.mark.skipif(
    not (os.path.exists(_VIDEO_LIST) and os.path.exists(_MASK)),
    reason="OmniAvatar training data not mounted",
)
def test_dataset_default_no_waveform_key():
    """Default path (load_raw_audio not set) does NOT add the key — no regression."""
    from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataset

    ds = OmniAvatarDataset(
        data_list_path=_VIDEO_LIST,
        latentsync_mask_path=_MASK,
        use_ref_sequence=True,
    )
    sample = ds[0]
    assert "audio_waveform" not in sample
