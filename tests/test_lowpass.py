"""Contrôle passe-bas (§2)."""

import numpy as np

from blanci.encoders.base import BaseEncoder
from blanci.encoders.lowpass import LowpassEncoder, lowpass

SR = 48_000


def tone(hz, seconds=1.0):
    return np.sin(2 * np.pi * hz * np.arange(int(SR * seconds)) / SR).astype(np.float32)


def test_lowpass_keeps_the_note_and_removes_the_highs():
    note, high = tone(5000), tone(12_000)
    assert np.std(lowpass(note, SR, 8000)) > 0.99 * np.std(note)
    assert np.std(lowpass(high, SR, 8000)) < 0.01 * np.std(high)


def test_lowpass_above_nyquist_is_a_no_op():
    x = tone(5000)
    assert np.array_equal(lowpass(x, 16_000, 8000), x)


class HighEnergy(BaseEncoder):
    """Encodeur factice : puissance au-dessus de 9 kHz, par FFT."""

    name, version, sample_rate, window_s, dim, has_tokens = "fake", "1", SR, 1.0, 1, False

    def _forward(self, batch):
        spectrum = np.abs(np.fft.rfft(batch, axis=1)) ** 2
        freqs = np.fft.rfftfreq(batch.shape[1], 1 / SR)
        return spectrum[:, freqs > 9000].sum(axis=1, keepdims=True)


def test_wrapped_encoder_is_blind_above_the_cutoff():
    inner = HighEnergy()
    control = LowpassEncoder(inner, 8000)
    assert control.name == "fake_lp8k" and control.dim == 1 and control.window_s == 1.0
    x = (tone(5000) + tone(12_000))[None, :]
    assert control.embed(x, SR)[0, 0] < 0.01 * inner.embed(x, SR)[0, 0]
    assert control.embed_tokens(x, SR) is None
