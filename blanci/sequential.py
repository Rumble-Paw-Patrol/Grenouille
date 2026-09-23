"""Module séquentiel (§3) : descripteurs calculés depuis l'audio et les scores, hors encodeur.

- Rythme : onsets de notes en bande (4,4–5,5 kHz), intervalles entre notes (IOI). A. blanci :
  note de 0,090–0,103 s, IOI 1,200–1,906 s (moyenne 1,414 s).
- Persistance : fraction de fenêtres positives, plus longue série, détections isolées.
- Solo contre chœur [HYPOTHÈSE H21] : IOI courts (< 0,6 s) et densité d'onsets élevée
  signalent plusieurs chanteurs.
Le score séquentiel module la décision (fusion), il ne met jamais de veto.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


def band_envelope_db(
    wav: np.ndarray, sr: int, band: tuple[int, int], smooth_s: float = 0.01
) -> np.ndarray:
    """Enveloppe en dB du signal filtré dans la bande, lissée sur `smooth_s`."""
    high = min(band[1], 0.99 * sr / 2)
    sos = butter(4, [band[0], high], btype="bandpass", fs=sr, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, wav)))
    n = max(1, round(smooth_s * sr))
    env = np.convolve(env, np.ones(n) / n, mode="same")
    return 20 * np.log10(env + 1e-10)


def detect_onsets(
    wav: np.ndarray,
    sr: int,
    band: tuple[int, int] = (4400, 5500),
    note_dur_s: tuple[float, float] = (0.07, 0.13),
    k_mad: float = 4.0,
) -> np.ndarray:
    """Instants (s) de début des événements en bande dont la durée est celle d'une note.

    Seuil adaptatif : médiane + k × MAD de l'enveloppe en dB (robuste au fond de chaque
    enregistrement). Sert au module séquentiel, jamais de filtre amont (bande saturée, §3).
    """
    env = band_envelope_db(np.asarray(wav, dtype=np.float64), sr, band)
    median = np.median(env)
    mad = np.median(np.abs(env - median)) + 1e-6
    above = np.concatenate([[False], env > median + k_mad * mad, [False]])
    edges = np.flatnonzero(np.diff(above.astype(np.int8)))
    starts, ends = edges[::2], edges[1::2]
    durations = (ends - starts) / sr
    keep = (durations >= note_dur_s[0]) & (durations <= note_dur_s[1])
    return starts[keep] / sr


def rhythm_features(
    onsets: np.ndarray, duration_s: float, ioi_range_s: tuple[float, float] = (1.2, 1.906)
) -> dict[str, float]:
    iois = np.diff(np.sort(onsets))
    out = {
        "onset_rate_hz": len(onsets) / duration_s if duration_s > 0 else 0.0,
        "ioi_median_s": float(np.median(iois)) if len(iois) else float("nan"),
        "ioi_cv": float(iois.std() / iois.mean()) if len(iois) > 1 else float("nan"),
        "frac_ioi_blanci": float(np.mean((iois >= ioi_range_s[0]) & (iois <= ioi_range_s[1])))
        if len(iois)
        else 0.0,
        "frac_ioi_short": float(np.mean(iois < 0.6)) if len(iois) else 0.0,  # chœur ? (H21)
    }
    return out


def persistence_features(
    window_scores: np.ndarray, threshold: float, offsets_s: np.ndarray | None = None
) -> dict[str, float]:
    """Fraction de fenêtres positives, plus longue série, nombre de séries isolées, étendue."""
    positive = np.asarray(window_scores) >= threshold
    padded = np.concatenate([[False], positive, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    runs = edges[1::2] - edges[::2]
    if offsets_s is not None and positive.any():
        span = float(np.ptp(np.asarray(offsets_s)[positive]))
    else:
        span = float(np.ptp(np.flatnonzero(positive))) if positive.any() else 0.0
    return {
        "frac_windows": float(positive.mean()) if len(positive) else 0.0,
        "n_positive": int(positive.sum()),
        "longest_run": int(runs.max()) if len(runs) else 0,
        "n_isolated": int((runs == 1).sum()),
        "span": span,
    }


def sequential_features(
    onsets: np.ndarray,
    window_scores: np.ndarray,
    threshold: float = 0.5,
    duration_s: float = 120.0,
    offsets_s: np.ndarray | None = None,
    ioi_range_s: tuple[float, float] = (1.2, 1.906),
) -> dict[str, float]:
    """Descripteurs d'un enregistrement : rythme (audio) + persistance (scores de fenêtres)."""
    return rhythm_features(onsets, duration_s, ioi_range_s) | persistence_features(
        window_scores, threshold, offsets_s
    )


def note_snr_db(
    wav: np.ndarray,
    sr: int,
    band: tuple[int, int] = (4400, 5500),
    note_dur_s: tuple[float, float] = (0.07, 0.13),
    k_mad: float = 4.0,
    neighbour_s: float = 0.5,
) -> float:
    """RSB estimé du chant (§6) : énergie en bande pendant les notes détectées, contre les
    `neighbour_s` voisines de part et d'autre (notes exclues). NaN sans note détectée.

    Sert à ventiler le rappel par RSB : un détecteur qui ne rate que les chants à < 6 dB
    n'a pas le même défaut qu'un détecteur qui rate des chants nets.
    """
    wav = np.asarray(wav, dtype=np.float64)
    onsets = detect_onsets(wav, sr, band, note_dur_s, k_mad)
    if not len(onsets):
        return float("nan")
    env = 10 ** (band_envelope_db(wav, sr, band) / 10)  # puissance d'enveloppe
    note_n = round(note_dur_s[1] * sr)
    near_n = round(neighbour_s * sr)
    in_note = np.zeros(len(env), dtype=bool)
    near = np.zeros(len(env), dtype=bool)
    for t in onsets:
        i = round(t * sr)
        in_note[i : i + note_n] = True
        near[max(0, i - near_n) : i + note_n + near_n] = True
    near &= ~in_note
    if not near.any():
        return float("nan")
    return float(10 * np.log10(env[in_note].mean() / env[near].mean()))
