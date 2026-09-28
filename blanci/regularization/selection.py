"""Choix des réglages et entraînements légers : validation groupée (R26, R40, R75, R76), ce que le
choix n'a pas vu (R74), pseudo-étiquetage (R81), bagging (R79)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def sample_pool(
    con, store, filters: dict | None, by: str, exclude: set, size: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """R81 : (embeddings, groupe) d'au plus `size` fenêtres non annotées du stock, tirées au
    hasard partition par partition, hors fenêtres du benchmark (`exclude`) et fenêtres
    arrêtées par une porte."""
    from blanci.dataset import recordings_table
    from blanci.store import gated_mask

    group_of = recordings_table(con).set_index("recording_id")[by].astype(str)
    rng = np.random.default_rng(seed)
    fragments = list(store.fragments(filters))
    per_fragment = max(1, int(np.ceil(size / max(len(fragments), 1))))
    embs, groups = [], []
    for path in fragments:
        meta, emb = store.read(path)
        keep = np.flatnonzero(~gated_mask(meta) & ~meta["window_id"].isin(exclude).to_numpy())
        if not len(keep):
            continue
        keep = rng.choice(keep, size=min(per_fragment, len(keep)), replace=False)
        embs.append(np.asarray(emb[keep], dtype=np.float32))
        groups.append(group_of.loc[meta["recording_id"].to_numpy()[keep]].to_numpy())
    if not embs:
        raise ValueError(f"stock vide pour {store.encoder_id} : pas de réservoir (R81)")
    return np.vstack(embs)[:size], np.concatenate(groups)[:size]


# --- R26, R40, R50 : un réglage choisi par validation groupée -----------------------------------


def usable_folds(
    y: np.ndarray, groups: np.ndarray, n_splits: int | str, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plis groupés internes (micros entiers) dont l'entraînement et le test ont les deux
    classes ; [] s'il n'y a qu'un micro."""
    from blanci.evaluate import grouped_folds

    y, groups = np.asarray(y).astype(int), np.asarray(groups)
    if len(np.unique(groups)) < 2:
        return []
    return [
        (train, test)
        for train, test in grouped_folds(y, groups, n_splits, seed)
        if len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2
    ]


# R75 : règle de choix. "one_se" (défaut, `head.selection_rule`) : parmi les valeurs dont l'AP
# moyenne est à moins d'une erreur type de la meilleure, la plus régularisante ; "best" : la
# meilleure AP moyenne. `configure` la règle depuis la config au démarrage de la CLI.
SELECTION_RULES = ("one_se", "best")
_SELECTION = {"rule": "one_se"}


def configure(cfg: dict) -> None:
    """Réglages globaux de ce module tirés de la config (R75 : `head.selection_rule`)."""
    rule = (cfg.get("head", {}) or {}).get("selection_rule", "one_se")
    if rule not in SELECTION_RULES:
        raise ValueError(f"head.selection_rule : {rule!r} (connues : {SELECTION_RULES})")
    _SELECTION["rule"] = rule


def pick(
    means: dict,
    ses: dict,
    rule: str | None = None,
    more_regularized: str = "low",
) -> Any:
    """Valeur retenue parmi `means` (AP moyenne par valeur de la grille). R75, règle du
    « 1 écart-type » : toute valeur dont la moyenne est à moins d'une erreur type (`ses`) de la
    meilleure est indiscernable d'elle ; on prend alors la plus régularisante — la plus petite
    (`more_regularized="low"` : C) ou la plus grande ("high" : weight decay). None si aucune
    valeur n'a d'AP."""
    rule = rule or _SELECTION["rule"]
    valid = {k: v for k, v in means.items() if np.isfinite(v)}
    if not valid:
        return None
    best = max(valid, key=valid.get)
    if rule == "best":
        return best
    se = ses.get(best, 0.0)
    floor = valid[best] - (se if np.isfinite(se) else 0.0)
    close = [k for k, v in valid.items() if v >= floor]
    return min(close) if more_regularized == "low" else max(close)


def grouped_search(
    score,
    grid,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int | str = 5,
    seed: int = 0,
    rule: str | None = None,
    more_regularized: str = "low",
) -> tuple[Any, dict, dict]:
    """Force d'une régularisation choisie par validation groupée interne : C des têtes
    linéaires (R26, `head.select_C`), weight decay des têtes torch (R40), C de la fusion
    (R50). `score(valeur, train, test)` rend l'AP sur `test` d'un modèle appris sur `train`.

    Renvoie (valeur retenue par `pick` (R75), ou None ; {valeur: AP moyenne} ; {valeur :
    erreur type de cette moyenne, écart-type des AP des plis / √plis}). Les deux derniers
    forment le chemin de régularisation (R76)."""
    folds = usable_folds(y, groups, n_splits, seed)
    means, ses = {}, {}
    for value in grid:
        aps = [score(value, train, test) for train, test in folds]
        finite = np.array([ap for ap in aps if np.isfinite(ap)], dtype=float)
        means[value] = float(finite.mean()) if len(finite) else float("nan")
        ses[value] = float(finite.std(ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0
    return pick(means, ses, rule, more_regularized), means, ses


def fine_grid(grid, points: int = 13) -> list[float]:
    """R76 : grille plus fine, `points` valeurs régulièrement espacées en échelle log entre la
    plus petite et la plus grande de `grid` (13 entre 0,001 et 10 : un facteur ~2,2 entre deux
    voisines, au lieu de 10)."""
    low, high = float(min(grid)), float(max(grid))
    return [float(v) for v in np.logspace(np.log10(low), np.log10(high), int(points))]


# --- R74 : ce que le choix n'a pas vu -----------------------------------------------------------


def fold_ids(n: int, folds) -> np.ndarray:
    """Numéro du pli de test de chaque fenêtre, d'après les plis (train, test) d'un `OOFScores`."""
    out = np.full(n, -1)
    for f, (_, test) in enumerate(folds):
        out[test] = f
    return out


def cross_fitted_threshold(
    y: np.ndarray, scores: np.ndarray, folds: np.ndarray, min_precision: float
) -> dict[str, Any]:
    """R74 : seuil de précision plancher choisi pour chaque pli sur les scores hors-pli des
    *autres* plis, puis appliqué à ce pli. Précision et rappel obtenus : ce que le seuil fera
    sur des micros qu'il n'a pas vus (le seuil choisi et jugé sur les mêmes scores est
    optimiste). Le calcul vit dans `evaluate.cross_fitted_recall`, que toute évaluation avec
    plis utilise pour son rappel (DECISIONS n° 140)."""
    from blanci.evaluate import cross_fitted_recall

    return cross_fitted_recall(y, scores, folds, min_precision)


def selection_estimate(
    scores: dict[str, np.ndarray],
    y: np.ndarray,
    folds: np.ndarray,
    recordings: np.ndarray | None = None,
) -> dict[str, Any]:
    """R74 et R80 : ce que vaut la procédure « garder la variante à la meilleure AP ».

    Pour chaque pli, la variante est choisie sur les scores hors-pli des autres plis, puis
    jugée sur ce pli : le choix ne voit jamais les labels qu'on mesure. Comparée à l'AP, sur
    les mêmes plis, de la variante qui gagne le tableau (choisie en voyant tout), elle dit
    combien le gagnant doit à la chance. AP au niveau enregistrement si `recordings` est
    donné. Renvoie les variantes choisies, et les deux AP moyennes par pli."""
    from blanci.evaluate import average_precision, to_recordings

    y, folds = np.asarray(y).astype(int), np.asarray(folds)

    def ap(values: np.ndarray, mask: np.ndarray) -> float:
        if recordings is None:
            return average_precision(y[mask], values[mask])
        rec = to_recordings(values[mask], y[mask], np.asarray(recordings)[mask])
        return average_precision(rec["y"].to_numpy(), rec["score"].to_numpy())

    everything = folds >= 0
    overall = {name: ap(np.asarray(v, float), everything) for name, v in scores.items()}
    finite = {k: v for k, v in overall.items() if np.isfinite(v)}
    if not finite:
        return {}
    winner = max(finite, key=finite.get)
    chosen, selected, naive = {}, [], []
    for f in np.unique(folds[everything]):
        other, this = everything & (folds != f), folds == f
        perf = {name: ap(np.asarray(v, float), other) for name, v in scores.items()}
        perf = {k: v for k, v in perf.items() if np.isfinite(v)}
        if not perf:
            continue
        pick_f = max(perf, key=perf.get)
        chosen[int(f)] = pick_f
        selected.append(ap(np.asarray(scores[pick_f], float), this))
        naive.append(ap(np.asarray(scores[winner], float), this))
    return {
        "winner": winner,
        "winner_ap": finite[winner],
        "chosen": chosen,
        "fold_ap_winner": float(np.nanmean(naive)) if naive else float("nan"),
        "fold_ap_selection": float(np.nanmean(selected)) if selected else float("nan"),
    }


# --- R81 : pseudo-étiquetage ---------------------------------------------------------------------


def pseudo_positives(
    scores: np.ndarray, min_score: float = 3.0, max_fraction: float = 0.01
) -> np.ndarray:
    """R81 : indices des fenêtres non annotées retenues comme pseudo-positifs — score (logit de
    la tête) ≥ `min_score`, et au plus la fraction `max_fraction` du réservoir (les mieux
    notées) : un garde-fou si la tête note haut trop de fenêtres."""
    scores = np.asarray(scores, dtype=float)
    above = np.flatnonzero(scores >= min_score)
    cap = int(np.floor(max_fraction * len(scores)))
    if len(above) > cap:
        above = above[np.argsort(-scores[above])[:cap]]
    return np.sort(above)


def with_pseudo_labels(
    fit_rows,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    pool: np.ndarray,
    min_score: float = 3.0,
    max_fraction: float = 0.01,
    weight: float = 0.3,
    rounds: int = 1,
):
    """R81 (auto-apprentissage, Lee 2013) : une tête apprise sur les labels note le réservoir ;
    les fenêtres sûres (`pseudo_positives`) rejoignent l'entraînement comme positifs, au poids
    `weight` (un avis du modèle ne vaut pas un label d'expert), et la tête est réapprise,
    `rounds` fois. `fit_rows(X, y, poids)` apprend une tête. Pas de pseudo-négatifs : les
    fenêtres non annotées notées bas sont aussi incertaines que les négatifs présumés (n° 106).
    Risque : le biais de confirmation (la tête étiquette ce qu'elle reconnaît déjà, erreurs
    comprises)."""
    y = np.asarray(y).astype(int)
    base_w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
    model = fit_rows(X, y, sample_weight)
    chosen = np.array([], dtype=int)
    for _ in range(int(rounds)):
        chosen = pseudo_positives(model.decision(pool), min_score, max_fraction)
        if not len(chosen):
            break
        model = fit_rows(
            np.vstack([X, pool[chosen]]),
            np.r_[y, np.ones(len(chosen), dtype=int)],
            np.r_[base_w, np.full(len(chosen), float(weight))],
        )
    model.meta = getattr(model, "meta", {}) | {"pseudo_positives": int(len(chosen))}
    return model


# --- R79 : bagging -------------------------------------------------------------------------------


def bootstrap_weights(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """R79 : tirage bootstrap des **micros** (avec remise, autant que de micros) ; poids de
    chaque fenêtre = nombre de fois où son micro est tiré (0 : absent de ce tirage). Tirer
    des fenêtres ferait croire à des exemples indépendants."""
    names, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    counts = np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names))
    return counts[inverse].astype(float)


@dataclass
class Bagged:
    """Têtes apprises sur des tirages bootstrap ; score = moyenne de leurs scores."""

    models: list

    def decision(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.decision(X) for m in self.models], axis=0)


def bagged(
    fit,
    y: np.ndarray,
    groups: np.ndarray,
    bags: int,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
) -> Bagged:
    """R79 (Breiman 1996) : `bags` têtes, chacune apprise sur un tirage bootstrap des micros
    (`bootstrap_weights`), leurs scores moyennés. Chaque tête a ses lubies (les exemples
    qu'elle a vus) ; la moyenne les efface : la variance baisse, le biais ne bouge pas.
    `fit(lignes, poids)` apprend une tête sur ces lignes ; les tirages sans l'une des deux
    classes sont refaits."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    models, attempts = [], 0
    while len(models) < bags and attempts < 10 * bags:
        attempts += 1
        w = bootstrap_weights(groups, rng)
        rows = np.flatnonzero(w > 0)
        if len(np.unique(y[rows])) < 2:
            continue
        if sample_weight is not None:
            w = w * np.asarray(sample_weight, dtype=float)
        models.append(fit(rows, w[rows]))
    if not models:
        raise ValueError("R79 : aucun tirage bootstrap n'a les deux classes")
    return Bagged(models)
