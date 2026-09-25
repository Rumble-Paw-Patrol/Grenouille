"""Évaluation (§6) : plis groupés, AP, rappel à précision fixée, bootstrap, Wilson.

Règle (§13.7) : toute évaluation passe par ici avec des groupes explicites ; un découpage
aléatoire est une erreur. Métriques rejetées : exactitude, F1 au seuil 0,5, kappa.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold

Metric = Callable[[np.ndarray, np.ndarray], float]


def grouped_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    assignment: dict[str, int] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plis groupés (par micro ou par site) et stratifiés ; jamais deux plis pour un même groupe.

    Avec `assignment` ({groupe: pli}, `fold_assignment`), les plis sont ceux-là, quels que soient
    les exemples : tous les modèles d'un benchmark sont alors jugés sur exactement les mêmes plis
    (DECISIONS n° 91). Sans, ils dépendent des exemples (StratifiedGroupKFold).
    """
    if groups is None or len(groups) != len(y):
        raise ValueError("groupes explicites obligatoires, un par exemple")
    y, groups = np.asarray(y), np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("au moins deux groupes sont nécessaires pour une validation groupée")
    if assignment is not None:
        fold_of = np.array([assignment.get(str(g), -1) for g in groups])
        if (fold_of < 0).any():
            missing = sorted({str(g) for g in groups[fold_of < 0]})
            raise ValueError(f"groupes sans pli dans l'affectation : {missing[:5]}")
        return [
            (np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f))
            for f in sorted(set(fold_of.tolist()))
        ]
    cv = StratifiedGroupKFold(n_splits=min(n_splits, n_groups), shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def fold_assignment(
    groups: np.ndarray, has_positive: np.ndarray, n_splits: int = 5, seed: int = 0
) -> dict[str, int]:
    """Pli de chaque groupe (micro), calculé une fois sur les **enregistrements** annotés :
    une ligne par enregistrement, `has_positive` = il contient A. blanci.

    Indépendant de la grille et des négatifs présumés de chaque encodeur : tous les modèles
    partagent ces plis (DECISIONS n° 91). Stratifié sur les enregistrements positifs.
    """
    groups = np.asarray(groups).astype(str)
    y = np.asarray(has_positive).astype(int)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("au moins deux groupes sont nécessaires pour une validation groupée")
    k = min(n_splits, len(unique))
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    out: dict[str, int] = {}
    with warnings.catch_warnings():  # classe rare : moins de positifs que de plis
        warnings.simplefilter("ignore", UserWarning)
        for fold, (_, test) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            out.update(dict.fromkeys(np.unique(groups[test]).tolist(), fold))
    return out


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, scores))


def recall_at_precision(
    y: np.ndarray, scores: np.ndarray, min_precision: float
) -> tuple[float, float]:
    """(rappel, seuil) : rappel maximal parmi les seuils gardant la précision ≥ min_precision.

    À rappel égal, le seuil le plus élevé (§6) : même rappel, moins de candidats dans la file
    de vérification. Rappel 0 et seuil +inf si aucun seuil n'atteint la précision plancher.
    """
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan"), float("nan")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    ok = precision[:-1] >= min_precision  # le dernier point (rappel 0) n'a pas de seuil
    if not ok.any():
        return 0.0, float("inf")
    candidates = np.flatnonzero(ok)
    best_recall = recall[:-1][candidates].max()
    best = candidates[recall[:-1][candidates] == best_recall][-1]  # dernier = seuil le plus haut
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
    if not len(diffs):  # métrique indéfinie partout (une seule classe) : pas de comparaison
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "significant": False}
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
) -> dict[str, Any]:
    """Métriques sur des scores hors-pli. `groups` = identifiant d'enregistrement de chaque
    fenêtre : unité du bootstrap, et unité d'agrégation au niveau « recording ».

    Valeurs flottantes, sauf `level` (str) et `n_pos` / `n_neg` (int).
    """
    scores, labels, groups = np.asarray(scores), np.asarray(labels), np.asarray(groups)
    if level == "recording":
        rec = to_recordings(scores, labels, groups, how)
        scores, labels, groups = (
            rec["score"].to_numpy(),
            rec["y"].to_numpy(),
            rec["recording_id"].to_numpy(),
        )
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


# --- Rappel par strate et fausses alarmes (§6) ------------------------------------------------


def recall_by_group(
    scores: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Rappel au seuil de décision, positif par positif, dans chaque strate (qualité A/B/C,
    tranche de RSB, site…), avec intervalle de Wilson. Les négatifs sont ignorés : la strate
    décrit un chant. Une strate manquante (qualité non renseignée) apparaît comme « ? »."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(int)
    strata = pd.Series(strata).fillna("?").astype(str).to_numpy()
    rows = []
    for stratum in sorted(set(strata[labels == 1])):
        hit = (scores >= threshold)[(labels == 1) & (strata == stratum)]
        lo, hi = wilson_interval(int(hit.sum()), len(hit))
        rows.append(
            {
                "stratum": stratum,
                "n_pos": len(hit),
                "recall": float(hit.mean()),
                "recall_lo": lo,
                "recall_hi": hi,
            }
        )
    return pd.DataFrame(rows, columns=["stratum", "n_pos", "recall", "recall_lo", "recall_hi"])


def false_alarms_per_hour(
    scores: np.ndarray, labels: np.ndarray, threshold: float, audio_hours: float
) -> float:
    """Négatifs au-dessus du seuil par heure d'audio examinée (§6).

    N'a de sens que sur un ensemble **exhaustif** (enregistrements écoutés en entier, audit,
    jeu gelé) : sur des négatifs tirés, le dénominateur ne représente pas l'audio réel.
    """
    if audio_hours <= 0:
        return float("nan")
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(int)
    return float(((scores >= threshold) & (labels == 0)).sum() / audio_hours)


def snr_bins(snr_db: np.ndarray, edges: tuple[float, ...] = (6.0, 12.0)) -> np.ndarray:
    """Tranches de RSB lisibles : « <6 dB », « 6–12 dB », « ≥12 dB » ; « ? » si inconnu."""
    snr_db = np.asarray(snr_db, dtype=float)
    names = [f"<{edges[0]:g} dB"]
    names += [f"{a:g}–{b:g} dB" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    names += [f"≥{edges[-1]:g} dB"]
    out = np.array(
        [
            names[int(np.searchsorted(edges, v, side="right"))]
            for v in np.nan_to_num(snr_db, nan=-np.inf)
        ],
        dtype=object,
    )
    out[np.isnan(snr_db)] = "?"
    return out
