"""Détecteurs audio → score (DECISIONS n° 97) : emplacements de la distillation et du modèle
fait maison, et un détecteur de référence déjà branché.

`get_detector(nom, cfg)` : `distilled`, `homemade` (emplacements, lèvent NotImplementedError
tant qu'ils ne sont pas écrits), `band_contrast` (contraste de bande, sans apprentissage : il
montre que le banc d'essai, le stock hors-pli et le benchmark complet sont prêts).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from blanci.detectors.base import Detector, TrainableDetector, evaluate_detector

__all__ = ["Detector", "TrainableDetector", "evaluate_detector", "get_detector"]


class BandContrastDetector:
    """Référence sans apprentissage : contraste bande de la note / bandes voisines (§3)."""

    name = "band_contrast"
    version = "1"

    def __init__(self, cfg: dict[str, Any]):
        self.signal = cfg["signal"]

    def score(self, windows: list[np.ndarray], sr: int) -> np.ndarray:
        from blanci.baselines import window_features

        return np.array(
            [window_features(w, sr, self.signal).fixed["band_contrast"] for w in windows]
        )


def get_detector(name: str, cfg: dict[str, Any]) -> Detector:
    if name == "distilled":
        from blanci.detectors.distilled import DistilledDetector

        return DistilledDetector(cfg)
    if name == "homemade":
        from blanci.detectors.homemade import HomemadeDetector

        return HomemadeDetector(cfg)
    if name == "band_contrast":
        return BandContrastDetector(cfg)
    raise ValueError(f"détecteur inconnu : {name!r} (distilled, homemade, band_contrast)")
