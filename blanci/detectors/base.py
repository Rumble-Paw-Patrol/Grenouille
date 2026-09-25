"""Contrat des détecteurs (DECISIONS n° 97) : audio → score, sans encodeur de fondation.

Un détecteur reçoit des fenêtres de son brut et rend un score par fenêtre (plus haut = plus
probablement A. blanci). C'est la forme du modèle de distillation et du modèle fait maison
(§3) : un petit réseau qui tourne seul sur l'i5, sans encodeur ni tête.

- `Detector` : `score(fenêtres, f_e)` ;
- `TrainableDetector` : en plus `fit(fenêtres, f_e, labels)`, qui rend un détecteur appris. Le
  banc d'essai l'entraîne pli par pli (plis communs, `dataset.folds_for`) : ses scores sont
  hors-pli, comparables à ceux des encodeurs, des baselines et des ensembles.

`evaluate_detector` juge un détecteur sur les fenêtres des baselines (annotations de 3 s et
négatifs appariés, audio lu à la demande, jamais écrit) et range ses scores hors-pli dans le
stock commun (`detector/<nom>`), d'où le benchmark complet et les ensembles le reprennent.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Detector(Protocol):
    name: str
    version: str

    def score(self, windows: list[np.ndarray], sr: int) -> np.ndarray:  # (n,)
        ...


@runtime_checkable
class TrainableDetector(Detector, Protocol):
    def fit(self, windows: list[np.ndarray], sr: int, y: np.ndarray) -> Detector: ...


def read_evaluation_audio(
    con: sqlite3.Connection, cfg: dict
) -> tuple[pd.DataFrame, list[np.ndarray], int]:
    """(fenêtres des baselines, forme d'onde de chacune au canal de la config, f_e).

    Toutes les fenêtres doivent avoir la même f_e (Song Meter à 48 kHz) : sinon on rééchantillonne
    vers la plus fréquente."""
    from blanci.audio import resample
    from blanci.baselines import evaluation_windows, read_windows
    from blanci.config import config_path

    windows = evaluation_windows(con, cfg).reset_index(drop=True)
    channel = cfg["audio"]["channel"]
    channels = (0, 1) if channel == "mean" else (int(channel),)
    waves: list[np.ndarray | None] = [None] * len(windows)
    rates = np.zeros(len(windows), dtype=int)
    for i, parts, sr in read_windows(windows, config_path(cfg, "raw"), channels):
        waves[i] = np.mean([parts[c] for c in channels], axis=0).astype(np.float32)
        rates[i] = sr
    target = int(pd.Series(rates).mode().iloc[0]) if len(rates) else 0
    waves = [
        w if r == target else resample(w, int(r), target) for w, r in zip(waves, rates, strict=True)
    ]
    return windows, waves, target


def evaluate_detector(
    con: sqlite3.Connection,
    cfg: dict,
    detector: Detector,
    audio: tuple[pd.DataFrame, list[np.ndarray], int] | None = None,
) -> dict[str, Any]:
    """AP et rappel du détecteur, hors-pli s'il apprend ; scores rangés sous
    `detector/<nom>`. `audio` : fenêtres déjà lues (tests, plusieurs détecteurs)."""
    from blanci.dataset import folds_for
    from blanci.evaluate import evaluate, grouped_folds
    from blanci.oof import labels_fingerprint, oof_frame, save_oof

    windows, waves, sr = audio or read_evaluation_audio(con, cfg)
    y = windows["y"].to_numpy(dtype=int)
    groups = windows["point"].to_numpy()
    assignment = folds_for(con, cfg)
    scores = np.full(len(windows), np.nan)
    if isinstance(detector, TrainableDetector):
        for train, test in grouped_folds(y, groups, assignment=assignment):
            fitted = detector.fit([waves[i] for i in train], sr, y[train])
            scores[test] = fitted.score([waves[i] for i in test], sr)
    else:
        scores = np.asarray(detector.score(waves, sr), dtype=float)
    source = f"detector/{detector.name}"
    save_oof(
        cfg,
        oof_frame(source, "detector", windows, scores, assignment, labels_fingerprint(con, cfg)),
    )
    bench, seed = cfg["benchmark"], cfg["head"]["seed"]
    rows = [
        {
            "detector": detector.name,
            **evaluate(
                scores,
                y,
                windows["recording_id"].to_numpy(),
                level,
                tuple(bench["precisions"]),
                bench["n_boot"],
                seed,
            ),
        }
        for level in ("window", "recording")
    ]
    return {"source": source, "table": pd.DataFrame(rows), "scores": scores}
