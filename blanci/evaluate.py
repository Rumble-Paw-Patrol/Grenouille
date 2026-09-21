"""Évaluation (§6) : plis groupés, AP, rappel à précision fixée, bootstrap, Wilson.

Règle (§13.7) : toute évaluation passe par ici avec des groupes explicites ; un découpage
aléatoire est une erreur. Métriques rejetées : exactitude, F1 au seuil 0,5, kappa.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold

Metric = Callable[[np.ndarray, np.ndarray], float]


def grouped_folds(
    y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plis groupés (par micro ou par site) et stratifiés ; jamais deux plis pour un même groupe."""
    if groups is None or len(groups) != len(y):
        raise ValueError("groupes explicites obligatoires, un par exemple")
    y, groups = np.asarray(y), np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("au moins deux groupes sont nécessaires pour une validation groupée")
    cv = StratifiedGroupKFold(n_splits=min(n_splits, n_groups), shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, scores))


def recall_at_precision(y: np.ndarray, scores: np.ndarray, min_precision: float) -> tuple[float, float]:
    """(rappel, seuil) : rappel maximal parmi les seuils gardant la précision ≥ min_precision.

    Rappel 0 et seuil +inf si aucun seuil n'atteint la précision plancher.
    """
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan"), float("nan")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    ok = precision[:-1] >= min_precision  # le dernier point (rappel 0) n'a pas de seuil
    if not ok.any():
        return 0.0, float("inf")
    best = np.flatnonzero(ok)[np.argmax(recall[:-1][ok])]
    return float(recall[best]), float(thresholds[best])


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de Wilson d'une proportion k/n (rappel compté en enregistrements)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def _unit_index(units: np.ndarray) -> list[np.ndarray]:
    codes, uniques = pd.factorize(np.asarray(units))
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    return [order[bounds[i] : bounds[i + 1]] for i in range(len(uniques))]


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    units: np.ndarray,
    metric: Metric = average_precision,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Intervalle percentile, en rééchantillonnant des unités entières (enregistrements)."""
    y, scores = np.asarray(y), np.asarray(scores)
    blocks = _unit_index(units)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        values.append(metric(y[idx], scores[idx]))
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if not len(values):
        return float("nan"), float("nan")
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def paired_bootstrap(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    units: np.ndarray,
    metric: Metric = average_precision,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Différence metric(A) − metric(B) sur les mêmes rééchantillonnages.

    « A meilleur que B » seulement si l'intervalle exclut zéro (§6).
    """
    y, a, b = np.asarray(y), np.asarray(scores_a), np.asarray(scores_b)
    blocks = _unit_index(units)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "diff": metric(y, a) - metric(y, b),
        "lo": float(lo),
        "hi": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }


def to_recordings(
    scores: np.ndarray, labels: np.ndarray, recordings: np.ndarray, how: str = "max"
) -> pd.DataFrame:
    """Score par enregistrement (max ou moyenne des 3 meilleures fenêtres) ; label = max."""
    df = pd.DataFrame({"recording_id": recordings, "score": scores, "y": labels})
    if how == "max":
        score = df.groupby("recording_id")["score"].max()
    elif how == "top3":
        score = df.groupby("recording_id")["score"].apply(lambda s: s.nlargest(3).mean())
    else:
        raise ValueError(f"agrégation inconnue : {how}")
    return pd.DataFrame({"score": score, "y": df.groupby("recording_id")["y"].max()}).reset_index()


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    level: Literal["window", "recording"],
    precisions: tuple[float, ...] = (0.1, 0.5),
    n_boot: int = 1000,
    seed: int = 0,
    how: str = "max",
) -> dict[str, float]:
    """Métriques sur des scores hors-pli. `groups` = identifiant d'enregistrement de chaque
    fenêtre : unité du bootstrap, et unité d'agrégation au niveau « recording »."""
    scores, labels, groups = np.asarray(scores), np.asarray(labels), np.asarray(groups)
    if level == "recording":
        rec = to_recordings(scores, labels, groups, how)
        scores, labels, groups = rec["score"].to_numpy(), rec["y"].to_numpy(), rec["recording_id"].to_numpy()
    ap = average_precision(labels, scores)
    lo, hi = bootstrap_ci(labels, scores, groups, average_precision, n_boot, seed)
    out = {
        "level": level,
        "n_pos": int(labels.sum()),
        "n_neg": int(len(labels) - labels.sum()),
        "ap": ap,
        "ap_lo": lo,
        "ap_hi": hi,
    }
    for p in precisions:
        recall, threshold = recall_at_precision(labels, scores, p)
        k = int(((scores >= threshold) & (labels == 1)).sum()) if np.isfinite(threshold) else 0
        w_lo, w_hi = wilson_interval(k, int(labels.sum()))
        out |= {
            f"recall@p{p}": recall,
            f"recall@p{p}_lo": w_lo,
            f"recall@p{p}_hi": w_hi,
            f"threshold@p{p}": threshold,
        }
    return out
