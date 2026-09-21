"""Décodage (forme d'onde float32 mono, f_e native) et rééchantillonnage.

Le spectrogramme n'est jamais calculé ici : chaque encodeur calcule le sien (§4).
"""

from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(source: Path | BinaryIO) -> tuple[np.ndarray, int]:
    """Lit un fichier audio entier ; les canaux sont moyennés."""
    wav, sr = sf.read(source, dtype="float32", always_2d=True)
    return wav.mean(axis=1), sr


def resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Rééchantillonnage polyphase avec filtre anti-repliement (scipy)."""
    if sr == target_sr:
        return np.asarray(wav, dtype=np.float32)
    g = gcd(sr, target_sr)
    return resample_poly(wav, target_sr // g, sr // g).astype(np.float32)


def cut_windows(wav: np.ndarray, sr: int, windows: list[tuple[float, float]]) -> np.ndarray:
    """Découpe les fenêtres (offset_s, dur_s) de même durée ; complète la fin par des zéros."""
    durations = {round(dur, 6) for _, dur in windows}
    if len(durations) != 1:
        raise ValueError(f"fenêtres de durées différentes : {sorted(durations)}")
    n = round(windows[0][1] * sr)
    out = np.zeros((len(windows), n), dtype=np.float32)
    for i, (offset, _) in enumerate(windows):
        start = round(offset * sr)
        chunk = wav[start : start + n]
        out[i, : len(chunk)] = chunk
    return out
