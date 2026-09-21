import numpy as np
import pytest

from blanci.audio import load_audio, resample
from tests.conftest import write_wav


def tone(freq: float, sr: int, duration_s: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def peak_frequency(x: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / sr)[spectrum.argmax()])


def test_same_rate_is_identity():
    x = tone(1000, 32000)
    y = resample(x, 32000, 32000)
    assert y.dtype == np.float32
    np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("sr, target", [(48000, 32000), (32000, 16000), (16000, 32000), (44100, 32000)])
def test_length_dtype_and_frequency_preserved(sr, target):
    y = resample(tone(4750, sr), sr, target)
    assert y.dtype == np.float32
    assert abs(len(y) - target) <= 1
    assert abs(peak_frequency(y, target) - 4750) < 5


def test_anti_aliasing_removes_content_above_new_nyquist():
    y = resample(tone(20000, 48000), 48000, 32000)
    assert np.sqrt(np.mean(y**2)) < 0.01


def test_load_audio_downmixes_to_mono_float32(tmp_path):
    path = write_wav(tmp_path / "stereo.wav", sr=16000, duration_s=0.5, channels=2)
    wav, sr = load_audio(path)
    assert sr == 16000
    assert wav.ndim == 1 and wav.dtype == np.float32 and len(wav) == 8000
