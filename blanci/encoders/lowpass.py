"""Contrôle passe-bas (§2) : un encodeur qui n'entend que ce qui est sous `cutoff_hz`.

Le tuteur juge les modèles à 16 kHz (beats, naturebeats : rien au-dessus de 8 kHz) « très
limite pour les paysages sonores ». La note d'A. blanci est à 4,4–5,5 kHz : l'objection ne
vaut pour ce chant que si les fréquences hautes aident à le reconnaître. Mesure : AP(birdmae)
sur l'audio natif contre AP(birdmae) sur l'audio filtré à 8 kHz, mêmes plis. Si le contrôle
égale birdmae natif, la f_e de 16 kHz n'est pas un défaut ici.

Le filtre est appliqué à la f_e d'origine, avant le rééchantillonnage de l'encodeur.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from blanci.encoders.base import Encoder


def lowpass(wav: np.ndarray, sr: int, cutoff_hz: float, order: int = 8) -> np.ndarray:
    """Butterworth passe-bas à phase nulle ; sans effet si `cutoff_hz` ≥ Nyquist."""
    if cutoff_hz >= sr / 2:
        return np.asarray(wav, dtype=np.float32)
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    return sosfiltfilt(sos, np.asarray(wav, dtype=np.float64), axis=-1).astype(np.float32)


class LowpassEncoder:
    """Enveloppe d'un encodeur : filtre, puis délègue. Nom : <encodeur>_lp<kHz>k."""

    def __init__(self, inner: Encoder, cutoff_hz: float):
        self.inner = inner
        self.cutoff_hz = float(cutoff_hz)
        self.name = f"{inner.name}_lp{self.cutoff_hz / 1000:g}k"
        self.version = inner.version
        self.sample_rate = inner.sample_rate
        self.window_s = inner.window_s
        self.dim = inner.dim
        self.has_tokens = inner.has_tokens

    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        return self.inner.embed(lowpass(wav, sr, self.cutoff_hz), sr)

    def embed_tokens(self, wav: np.ndarray, sr: int) -> np.ndarray | None:
        return self.inner.embed_tokens(lowpass(wav, sr, self.cutoff_hz), sr)
