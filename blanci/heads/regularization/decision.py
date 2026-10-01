"""R73 : courbe d'exploitation, seuil pour la précision du stock, calibration de Platt, changement
de prévalence ; recalibration de chaque pli avant la mise bout à bout (`fold_platt`, n° 135)."""

from __future__ import annotations

import numpy as np

# --- R73 : le seuil pour la précision du stock, pas celle du benchmark ---------------------------


def operating_curve(y: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray]:
    """Pour chaque seuil possible (scores distincts, du plus haut au plus bas) : rappel (TPR),
    taux de fausses alertes (FPR) et précision **dans cet échantillon**. TPR et FPR ne dépendent
    pas de la proportion de positifs ; la précision, si (`precision_at_prevalence`)."""
    y, scores = np.asarray(y).astype(int), np.asarray(scores, dtype=float)
    thresholds = np.unique(scores)[::-1]
    pos, neg = np.sort(scores[y == 1]), np.sort(scores[y == 0])
    tp = len(pos) - np.searchsorted(pos, thresholds, side="left")
    fp = len(neg) - np.searchsorted(neg, thresholds, side="left")
    tpr = tp / max(len(pos), 1)
    fpr = fp / max(len(neg), 1)
    precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
    return {"threshold": thresholds, "tpr": tpr, "fpr": fpr, "precision": precision}


def precision_at_prevalence(tpr, fpr, prevalence: float):
    """Précision attendue là où la part de positifs vaut `prevalence` :
    π·TPR / (π·TPR + (1 − π)·FPR). Hypothèse forte : les négatifs jugés ressemblent à ceux du
    stock (le benchmark tire ses négatifs près des positifs : mêmes micros, mêmes heures)."""
    tpr, fpr = np.asarray(tpr, dtype=float), np.asarray(fpr, dtype=float)
    num = prevalence * tpr
    den = num + (1.0 - prevalence) * fpr
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def threshold_at_prevalence(
    y: np.ndarray, scores: np.ndarray, min_precision: float, prevalence: float
) -> dict[str, float]:
    """R73 : le seuil au plus grand rappel dont la précision **à la prévalence du stock**
    atteint `min_precision` (à rappel égal, le seuil le plus haut). Rappel 0 et seuil +inf si
    aucun seuil n'y arrive."""
    curve = operating_curve(y, scores)
    precision = precision_at_prevalence(curve["tpr"], curve["fpr"], prevalence)
    ok = np.flatnonzero(np.nan_to_num(precision) >= min_precision)
    if not len(ok):
        return {"recall": 0.0, "threshold": float("inf"), "precision": float("nan")}
    # Seuils décroissants : le premier maximum du rappel est le seuil le plus haut.
    best = ok[np.argmax(curve["tpr"][ok])]
    return {
        "recall": float(curve["tpr"][best]),
        "threshold": float(curve["threshold"][best]),
        "precision": float(precision[best]),
    }


def platt(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """R73, calibration de Platt : (a, b) tels que σ(a·score + b) soit la probabilité d'être un
    positif **à la proportion de positifs de l'échantillon** (à corriger par `prior_shift`).
    Deux paramètres : stable avec peu de positifs, à la différence de l'isotonique."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.asarray(scores, dtype=float)[:, None], np.asarray(y).astype(int)
    )
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def prior_shift(p, source_prevalence: float, target_prevalence: float):
    """Probabilité ramenée d'une proportion de positifs à une autre (règle de Bayes sur les
    cotes : cote × [π'/(1 − π')] / [π/(1 − π)])."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    ratio = (target_prevalence / (1 - target_prevalence)) / (
        source_prevalence / (1 - source_prevalence)
    )
    odds = p / (1 - p) * ratio
    return odds / (1 + odds)


# --- Recalibration par pli (DECISIONS n° 133, 135) ----------------------------------------------


def fold_platt(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(a, b) : a·score + b est la cote (logit) « A. blanci contre fond » à **classes
    équilibrées**, donc indépendante de la part de positifs, apprise sur des scores hors-pli.
    Sert à mettre les plis sur une même échelle avant de les mettre bout à bout
    (`head.calibrated_fold_scores`, `fusion.fusion_model_oof`).

    Scores centrés-réduits d'abord, puis logistique à une variable faiblement pénalisée (C =
    100) : stable même quand les scores séparent parfaitement les classes. Sans les deux
    classes, ou si la pente sort négative (scores à l'envers), repli sur le seul centrage-
    réduction : l'échelle reste commune, le classement du pli n'est jamais retourné."""
    from sklearn.linear_model import LogisticRegression

    scores, y = np.asarray(scores, dtype=float), np.asarray(y).astype(int)
    mean = float(scores.mean()) if len(scores) else 0.0
    std = float(scores.std()) if len(scores) else 0.0
    std = std if std > 0 else 1.0
    fallback = (1.0 / std, -mean / std)
    if len(np.unique(y)) < 2:
        return fallback
    z = ((scores - mean) / std)[:, None]
    model = LogisticRegression(C=100.0, class_weight="balanced", max_iter=1000).fit(z, y)
    slope, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
    if not (np.isfinite(slope) and slope > 0):
        return fallback
    return slope / std, intercept - slope * mean / std
