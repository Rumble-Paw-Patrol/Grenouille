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
    n_splits: int | str = 5,
    seed: int = 0,
    assignment: dict[str, int] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plis groupés (par micro ou par site) et stratifiés ; jamais deux plis pour un même groupe.

    Avec `assignment` ({groupe: pli}, `fold_assignment`), les plis sont ceux-là, quels que soient
    les exemples : tous les modèles d'un benchmark sont alors jugés sur exactement les mêmes plis
    (DECISIONS n° 91). Sans, ils dépendent des exemples (StratifiedGroupKFold). `n_splits`
    = "lomo" : un pli par micro positif (R77, `lomo_assignment`).
    """
    if groups is None or len(groups) != len(y):
        raise ValueError("groupes explicites obligatoires, un par exemple")
    y, groups = np.asarray(y), np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("au moins deux groupes sont nécessaires pour une validation groupée")
    if assignment is None and n_splits == LOMO:
        assignment = lomo_assignment(groups, y, seed)
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


LOMO = "lomo"  # R77 : un pli par micro (leave-one-micro-out), `head.n_splits: lomo`


def lomo_assignment(groups: np.ndarray, y: np.ndarray, seed: int = 0) -> dict[str, int]:
    """R77 : un pli par groupe (micro) qui a des positifs ; les groupes sans positif sont
    répartis entre ces plis au hasard (graine fixe), à parts égales. Chaque micro positif est
    ainsi jugé seul, par un modèle appris sur tous les autres."""
    groups = np.asarray(groups).astype(str)
    y = np.asarray(y).astype(int)
    positive = sorted(set(groups[y == 1].tolist()))
    others = sorted(set(groups.tolist()) - set(positive))
    if len(positive) < 2:
        raise ValueError("leave-one-micro-out : il faut au moins deux micros avec des positifs")
    out = {g: i for i, g in enumerate(positive)}
    order = np.random.default_rng(seed).permutation(len(others))
    for rank, i in enumerate(order):
        out[others[i]] = rank % len(positive)
    return out


def fold_assignment(
    groups: np.ndarray, has_positive: np.ndarray, n_splits: int | str = 5, seed: int = 0
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
    if n_splits == LOMO:
        return lomo_assignment(groups, y, seed)
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


def fold_mean_ap(y: np.ndarray, scores: np.ndarray, folds: np.ndarray) -> tuple[float, int]:
    """(AP moyenne des plis, nombre de plis comptés) : l'AP de chaque pli, ses scores entre eux,
    puis la moyenne sur les plis qui ont les deux classes (DECISIONS n° 133, 135).

    L'AP « poolée » met bout à bout les scores de modèles différents, un par pli : si les plis
    retiennent des réglages différents (C, σ, époques), leurs scores ne sont pas sur la même
    échelle et le classement commun se dégrade, sans que le classement dans chaque pli change.
    L'AP par pli ne compare jamais deux modèles : elle ne voit que le classement. Plus bruitée
    (chaque pli a peu de positifs), elle se lit à côté de l'AP poolée, pas à sa place."""
    y, scores, folds = np.asarray(y).astype(int), np.asarray(scores), np.asarray(folds)
    values = [
        average_precision(y[folds == f], scores[folds == f]) for f in np.unique(folds[folds >= 0])
    ]
    values = [v for v in values if np.isfinite(v)]
    return (float(np.mean(values)) if values else float("nan")), len(values)


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


def cross_fitted_recall(
    y: np.ndarray, scores: np.ndarray, folds: np.ndarray, min_precision: float
) -> dict[str, Any]:
    """Rappel à précision plancher, le seuil de chaque pli choisi sur les scores hors-pli des
    **autres** plis puis appliqué à ce pli (R74, DECISIONS n° 140). Le seuil choisi et jugé sur
    les mêmes scores (`recall_at_precision`) est un oracle : il connaît les labels qu'il
    mesure, son rappel est optimiste. Les fenêtres sans pli (`folds` < 0) sont ignorées.

    Renvoie `recall`, `k` (positifs retrouvés), `n_pos`, `precision` obtenue, `thresholds`
    ({pli: seuil}) et `decided` (chaque exemple au-dessus du seuil de son pli)."""
    y, scores = np.asarray(y).astype(int), np.asarray(scores, dtype=float)
    folds = np.asarray(folds)
    known = folds >= 0
    decided = np.zeros(len(y), dtype=bool)
    thresholds: dict[int, float] = {}
    for f in np.unique(folds[known]):
        other, this = known & (folds != f), folds == f
        _, t = recall_at_precision(y[other], scores[other], min_precision)
        thresholds[int(f)] = float(t)
        if np.isfinite(t):  # NaN : aucun positif ailleurs ; +inf : plancher jamais atteint
            decided[this] = scores[this] >= t
    n_pos = int((y[known] == 1).sum())
    k = int((decided & (y == 1)).sum())
    return {
        "recall": k / n_pos if n_pos else float("nan"),
        "k": k,
        "n_pos": n_pos,
        "precision": k / int(decided.sum()) if decided.any() else float("nan"),
        "thresholds": thresholds,
        "decided": decided,
    }


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
    """Intervalle percentile, en rééchantillonnant des unités entières : enregistrements, ou
    micros quand ils sont connus (`evaluate(clusters=…)`, DECISIONS n° 139)."""
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

    « A meilleur que B » seulement si l'intervalle exclut zéro (§6). `units` : l'unité tirée,
    enregistrement ou micro (DECISIONS n° 139). `p` : p-valeur bilatérale du bootstrap,
    2 × (min(k(Δ ≤ 0), k(Δ ≥ 0)) + 1) / (n + 1) (Davison & Hinkley), à corriger par `with_holm`
    quand un tableau aligne plusieurs comparaisons (n° 139). Jamais nulle : la plus petite p
    atteignable est 2/(n + 1), donc avec Holm sur M comparaisons il faut n_boot ≳ 2M/α pour
    qu'une différence puisse rester significative.
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
        return {
            "diff": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "significant": False,
            "p": float("nan"),
        }
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = bootstrap_p(diffs)
    return {
        "diff": metric(y, a) - metric(y, b),
        "lo": float(lo),
        "hi": float(hi),
        "significant": bool(lo > 0 or hi < 0),
        "p": p,
    }


def bootstrap_p(diffs: np.ndarray) -> float:
    """p-valeur bilatérale d'un bootstrap : min(1, 2 × (min(k(Δ ≤ 0), k(Δ ≥ 0)) + 1) / (n + 1)),
    n tirages valides. Jamais nulle (plus petite valeur : 2/(n + 1))."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    return min(1.0, 2 * (min(int((d <= 0).sum()), int((d >= 0).sum())) + 1) / (len(d) + 1))


def holm(p_values: np.ndarray) -> np.ndarray:
    """p-valeurs ajustées de Holm (risque de se tromper au moins une fois sur la famille) ;
    les NaN restent NaN et ne comptent pas dans la famille."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        out[i] = running
    return out


def with_holm(comparisons: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Ajoute `p_holm` et `significant_holm` à un tableau de comparaisons appariées (colonne
    `p`) : vingt têtes comparées à 5 % donnent une « victoire » par hasard (DECISIONS n° 139).
    `significant` (l'intervalle de chaque comparaison, prise seule) reste pour mémoire."""
    if comparisons.empty or "p" not in comparisons:
        return comparisons
    adjusted = holm(comparisons["p"].to_numpy())
    return comparisons.assign(p_holm=adjusted, significant_holm=adjusted < alpha)


def cluster_units(
    units: np.ndarray, recordings: np.ndarray, clusters: np.ndarray | None
) -> np.ndarray:
    """Unités du bootstrap : le groupe (micro) de chaque enregistrement de `units` quand
    `clusters` (un par fenêtre, aligné sur `recordings`) est donné, sinon `units` tel quel."""
    if clusters is None:
        return np.asarray(units)
    cluster_of = pd.Series(np.asarray(clusters), index=np.asarray(recordings))
    cluster_of = cluster_of.groupby(level=0).first()
    return cluster_of.loc[np.asarray(units)].to_numpy()


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
    folds: np.ndarray | None = None,
    clusters: np.ndarray | None = None,
) -> dict[str, Any]:
    """Métriques sur des scores hors-pli. `groups` = identifiant d'enregistrement de chaque
    fenêtre : unité d'agrégation au niveau « recording », et unité du bootstrap sans
    `clusters`. `clusters` (micro de chaque fenêtre) : unité du bootstrap ; les enregistrements
    d'un même micro partagent fond et faune, les tirer un à un rend l'intervalle trop étroit
    (DECISIONS n° 139). `folds` (pli de test de chaque fenêtre, `regularization.fold_ids`) :
    ajoute `ap_fold_mean`, l'AP moyenne par pli (`fold_mean_ap`), et `n_folds_ap` ; le rappel
    à précision plancher est alors jugé avec un seuil choisi sur les autres plis
    (`cross_fitted_recall`, n° 140), l'oracle restant en `recall@p…_oracle`. `threshold@p…` :
    le seuil choisi sur tous les scores, celui qu'on déploierait.

    Valeurs flottantes, sauf `level` (str) et `n_pos` / `n_neg` / `n_folds_ap` (int).
    """
    scores, labels, groups = np.asarray(scores), np.asarray(labels), np.asarray(groups)
    if folds is not None:  # un enregistrement est d'un seul micro, donc d'un seul pli
        fold_of = pd.Series(np.asarray(folds), index=groups).groupby(level=0).first()
    units = groups if clusters is None else np.asarray(clusters)
    if level == "recording":
        windows_recordings = groups
        rec = to_recordings(scores, labels, groups, how)
        scores, labels, groups = (
            rec["score"].to_numpy(),
            rec["y"].to_numpy(),
            rec["recording_id"].to_numpy(),
        )
        if folds is not None:
            folds = fold_of.loc[groups].to_numpy()
        units = cluster_units(groups, windows_recordings, clusters)
    ap = average_precision(labels, scores)
    lo, hi = bootstrap_ci(labels, scores, units, average_precision, n_boot, seed)
    out = {
        "level": level,
        "n_pos": int(labels.sum()),
        "n_neg": int(len(labels) - labels.sum()),
        "ap": ap,
        "ap_lo": lo,
        "ap_hi": hi,
    }
    if folds is not None:
        out["ap_fold_mean"], out["n_folds_ap"] = fold_mean_ap(labels, scores, folds)
    for p in precisions:
        recall, threshold = recall_at_precision(labels, scores, p)
        if folds is not None:
            crossed = cross_fitted_recall(labels, scores, folds, p)
            out[f"recall@p{p}_oracle"] = recall
            recall, k, n_pos = crossed["recall"], crossed["k"], crossed["n_pos"]
        else:
            k = int(((scores >= threshold) & (labels == 1)).sum()) if np.isfinite(threshold) else 0
            n_pos = int(labels.sum())
        w_lo, w_hi = wilson_interval(k, n_pos)
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
