"""Contrôle qualité par enregistrement : pluie, saturation, micro dans sac, silence.

Indices simples (numpy/scipy) ; les seuils de config/default.yaml sont provisoires et se
calibrent sur les enregistrements étiquetés `rain` et `artefact_in_bag`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import welch


def qc_indices(wav: np.ndarray, sr: int) -> dict[str, float]:
    x = wav - wav.mean()
    rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
    freqs, psd = welch(x, fs=sr, nperseg=min(2048, len(x)))
    total = psd[freqs >= 100].sum()
    hf = psd[freqs >= 2000].sum()
    mid = psd[(freqs >= 1000) & (freqs <= min(10000, sr / 2))]
    flatness = float(np.exp(np.mean(np.log(mid + 1e-20))) / (mid.mean() + 1e-20)) if len(mid) else 0.0
    return {
        "rms_dbfs": 20 * float(np.log10(max(rms, 1e-10))),
        "peak": float(np.abs(wav).max()) if len(wav) else 0.0,
        "clip_fraction": float(np.mean(np.abs(wav) >= 0.999)) if len(wav) else 0.0,
        "hf_ratio": float(hf / total) if total > 0 else 0.0,
        "flatness_1_10k": flatness,
    }


def qc_flags(indices: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    silent = indices["rms_dbfs"] < thresholds["silent_dbfs"]
    return {
        "silent": silent,
        "saturation": indices["clip_fraction"] > thresholds["clip_fraction"],
        "in_bag": not silent and indices["hf_ratio"] < thresholds["in_bag_hf_ratio"],
        "rain": indices["flatness_1_10k"] > thresholds["rain_flatness"]
        and indices["rms_dbfs"] > thresholds["rain_min_dbfs"],
        "indices": {k: round(v, 6) for k, v in indices.items()},
    }
