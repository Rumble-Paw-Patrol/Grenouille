"""Agrégation fenêtre → enregistrement → point (§1, unité de décision).

- Enregistrement : fraction de fenêtres au-dessus du seuil. Positif si assez de fenêtres,
  étalées dans le temps (chant continu, H20) ; détection isolée → « suspect » (à vérifier).
- Point : classé par force ; « présence confirmée » seulement après validation humaine ;
  sinon « non détecté », jamais « absent ».
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RECORDING_STATUSES = ("positive", "suspect", "negative", "verified_positive", "verified_negative")
POINT_CONFIRMED, POINT_TO_CHECK, POINT_NOT_DETECTED = (
    "présence confirmée",
    "à vérifier",
    "non détecté",
)


def aggregate_recording(
    window_scores: np.ndarray,
    threshold: float,
    offsets_s: np.ndarray | None = None,
    min_positive_windows: int = 3,
    min_span_s: float = 10.0,
) -> tuple[float, str]:
    """(fraction de fenêtres positives, statut) d'un enregistrement."""
    positive = np.asarray(window_scores) >= threshold
    fraction = float(positive.mean()) if len(positive) else 0.0
    n = int(positive.sum())
    if n == 0:
        return fraction, "negative"
    if n < min_positive_windows:
        return fraction, "suspect"
    if offsets_s is not None and np.ptp(np.asarray(offsets_s)[positive]) < min_span_s:
        return fraction, "suspect"
    return fraction, "positive"


def rank_points(decisions: pd.DataFrame) -> pd.DataFrame:
    """Classement des points (dataset, site, mic_id) à partir des décisions par enregistrement.

    Colonnes attendues : dataset, site, mic_id, recording_id, start_utc, fraction, status.
    Force = nombre d'enregistrements positifs, puis de jours, puis fraction maximale.
    """
    df = decisions.copy()
    df["day"] = df["start_utc"].astype(str).str[:10]
    df["is_pos"] = df["status"].isin(["positive", "verified_positive"])
    df["is_flag"] = df["status"].isin(["positive", "suspect"])
    df["is_verified"] = df["status"] == "verified_positive"
    points = (
        df.groupby(["dataset", "site", "mic_id"])
        .agg(
            n_recordings=("recording_id", "nunique"),
            n_positive=("is_pos", "sum"),
            n_suspect=("status", lambda s: int((s == "suspect").sum())),
            n_verified=("is_verified", "sum"),
            n_to_check=("is_flag", "sum"),
            n_days_positive=("day", lambda d: d[df.loc[d.index, "is_pos"]].nunique()),
            max_fraction=("fraction", "max"),
        )
        .reset_index()
    )
    points["status"] = np.select(
        [points["n_verified"] > 0, points["n_to_check"] > 0],
        [POINT_CONFIRMED, POINT_TO_CHECK],
        default=POINT_NOT_DETECTED,
    )
    order = {POINT_CONFIRMED: 0, POINT_TO_CHECK: 1, POINT_NOT_DETECTED: 2}
    points["_order"] = points["status"].map(order)
    points = points.sort_values(
        ["_order", "n_positive", "n_days_positive", "max_fraction"],
        ascending=[True, False, False, False],
    )
    return points.drop(columns=["_order", "n_to_check"]).reset_index(drop=True)
