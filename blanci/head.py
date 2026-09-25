"""Têtes sur embeddings gelés (§3) : recherche par l'exemple, prototypes simple et différentiel,
kNN cosinus, régression logistique, attentive probe et cascade (DECISIONS n° 92).

La tête retenue (logistique L2) est stockée sans pickle (JSON + npz, calcul en numpy) : elle se
recharge sur n'importe quelle machine, quelle que soit la version de scikit-learn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from blanci.evaluate import average_precision, grouped_folds
from blanci.index import l2_normalize
from blanci.sequential import GATED_SCORE


@dataclass(frozen=True)
class OOFScores:
    """Scores hors-pli : chaque score vient d'un modèle entraîné sans son micro (§3).

    Seul type accepté en entrée de la fusion (§13.7).
    """

    values: np.ndarray
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    method: str


# --- Prototype différentiel et kNN ----------------------------------------------------------


def differential_prototype(E_pos: np.ndarray, E_neg_paired: np.ndarray) -> tuple[np.ndarray, float]:
    """w = μ+ − μ− sur embeddings normalisés ; b place le seuil 0 au milieu des centroïdes.

    Avec des négatifs appariés (mêmes micros, heures), w retire le fond sonore partagé.
    """
    mu_pos = l2_normalize(E_pos).mean(axis=0)
    mu_neg = l2_normalize(E_neg_paired).mean(axis=0)
    w = mu_pos - mu_neg
    return w, float(-w @ (mu_pos + mu_neg) / 2)


def simple_prototype_scores(E_pos: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cosinus au centroïde des positifs, sans négatifs (baseline de similarité du §3).

    L'écart avec le prototype différentiel mesure le fond sonore capté par l'embedding : si
    retrancher le centroïde des négatifs appariés aide beaucoup, les positifs se ressemblent
    surtout par leur ambiance (même micro, même heure), pas par le chant.
    """
    return l2_normalize(X) @ l2_normalize(l2_normalize(E_pos).mean(axis=0, keepdims=True))[0]


def exemplar_scores(E_pos: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Recherche par l'exemple : cosinus au positif d'entraînement le plus proche.

    Aucun apprentissage, aucun négatif : ce que donne une requête « trouve-moi des fenêtres
    comme celles-ci » (§5, récolte par similarité)."""
    return (l2_normalize(X) @ l2_normalize(E_pos).T).max(axis=1)


def medoid_scores(E_pos: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Recherche par un seul exemple : w = l'embedding d'une fenêtre de référence, le positif le
    plus central (médoïde : cosinus moyen maximal aux autres positifs). Ce qu'on a en arrivant
    sur un nouveau site avec un seul chant validé."""
    P = l2_normalize(E_pos)
    reference = P[(P @ P.T).mean(axis=1).argmax()]
    return l2_normalize(X) @ reference


def cascade_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    T_train: np.ndarray,
    X_test: np.ndarray,
    T_test: np.ndarray,
    C: float = 1.0,
    fraction: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """Cascade : un linear probe trie toutes les fenêtres, une attentive probe reclasse les
    `fraction` meilleures (candidats), seules à avoir besoin de leurs jetons (mémoire, §3).

    Le seuil des candidats est pris sur l'entraînement ; l'attentive est apprise sur les
    candidats d'entraînement (les cas difficiles), ou sur tout l'entraînement s'ils n'ont
    qu'une classe. Un candidat reste toujours devant un non-candidat : l'étage 2 reclasse, il
    ne repêche pas.
    """
    from blanci.attentive import fit_attentive
    from blanci.pooling import flat_tokens

    stage1 = fit_logistic(X_train, y_train, C, seed)
    s_train, s_test = stage1.decision(X_train), stage1.decision(X_test)
    threshold = np.quantile(s_train, 1.0 - fraction)
    chosen = s_train >= threshold
    if len(np.unique(y_train[chosen])) < 2:
        chosen = np.ones(len(y_train), dtype=bool)
    attentive = fit_attentive(flat_tokens(T_train[chosen]), y_train[chosen], seed=seed)
    out = s_test.astype(float).copy()
    candidates = s_test >= threshold
    if candidates.any():
        a = attentive.decision(flat_tokens(T_test[candidates]))
        floor = s_test[~candidates].max() if (~candidates).any() else s_test.min()
        out[candidates] = floor + 1.0 + (a - a.min()) / (np.ptp(a) + 1e-9)
    return out


def prototype_scores(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return l2_normalize(X) @ w + b


def knn_scores(X_train: np.ndarray, y_train: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Marge top-1 : similarité au positif le plus proche − au négatif le plus proche."""
    y_train = np.asarray(y_train).astype(bool)
    if not y_train.any() or y_train.all():
        raise ValueError("la marge kNN exige des positifs et des négatifs dans l'entraînement")
    Xn, Tn = l2_normalize(X), l2_normalize(X_train)
    sims = Xn @ Tn.T
    return sims[:, y_train].max(axis=1) - sims[:, ~y_train].max(axis=1)


# --- Régression logistique --------------------------------------------------------------------


@dataclass
class Head:
    """Standardisation + régression logistique, en numpy pur à l'inférence."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    meta: dict[str, Any] = field(default_factory=dict)

    def decision(self, X: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(X, dtype=np.float32) - self.mean) / self.scale
        ) @ self.coef + self.intercept

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision(X)))

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "weights.npz", mean=self.mean, scale=self.scale, coef=self.coef)
        manifest = {"intercept": self.intercept, **self.meta}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> Head:
        directory = Path(directory)
        weights = np.load(directory / "weights.npz")
        meta = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        intercept = meta.pop("intercept")
        return cls(weights["mean"], weights["scale"], weights["coef"], intercept, meta)


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float, seed: int = 0) -> Head:
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed).fit(
        scaler.transform(X), y
    )
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    return Head(
        scaler.mean_.astype(np.float32),
        scale.astype(np.float32),
        model.coef_[0].astype(np.float32),
        float(model.intercept_[0]),
        {"C": C},
    )


def select_C(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C_grid: list[float],
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[float, dict[float, float]]:
    """C maximisant l'AP moyenne en validation groupée (plis internes)."""
    folds = grouped_folds(y, groups, n_splits, seed)
    results = {}
    for C in C_grid:
        aps = [
            average_precision(y[test], fit_logistic(X[train], y[train], C, seed).decision(X[test]))
            for train, test in folds
        ]
        results[C] = float(np.nanmean(aps))
    return max(results, key=results.get), results


def train_head(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C_grid: list[float],
    seed: int = 0,
    gated: np.ndarray | None = None,
) -> Head:
    """C choisi par validation groupée, puis réentraînement sur tous les labels (§4).

    `gated` : fenêtres arrêtées par une porte du seuillage en amont (embedding nul), écartées.
    """
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y).astype(int)
    groups = np.asarray(groups)
    if gated is not None:
        keep = ~np.asarray(gated, dtype=bool)
        X, y, groups = X[keep], y[keep], groups[keep]
    C, cv_ap = select_C(X, y, groups, C_grid, seed=seed)
    head = fit_logistic(X, y, C, seed)
    head.meta |= {
        "cv_ap": {str(k): v for k, v in cv_ap.items()},
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "seed": seed,
    }
    return head


METHODS = (
    "exemplar_medoid",
    "exemplar",
    "knn",
    "simple_prototype",
    "prototype",
    "logistic",
    "attentive",
    "cascade",
)


def _choose_C(X, y, groups, C_grid, n_splits, seed) -> float:
    """C par validation groupée interne, si la grille a plusieurs valeurs et l'entraînement
    plusieurs groupes ; sinon la première valeur."""
    if C_grid and len(C_grid) > 1 and len(np.unique(groups)) >= 2:
        return select_C(X, y, groups, C_grid, n_splits, seed)[0]
    return (C_grid or [1.0])[0]


def fit_and_score(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    C: float | None = None,
    C_grid: list[float] | None = None,
    n_splits: int = 5,
    seed: int = 0,
    tokens: np.ndarray | None = None,
    cascade_fraction: float = 0.2,
) -> np.ndarray:
    """Scores des fenêtres `test` par la tête `method` apprise sur les seules fenêtres `train`.

    `C` fixe le C des têtes logistiques ; sinon il est choisi sur `train` (validation groupée
    interne sur `C_grid`). Brique commune de la validation croisée (`oof_scores`) et de la
    courbe selon le nombre d'annotations (`head_benchmark.annotation_curve`).
    """
    Xtr, ytr = X[train], y[train]
    if method in ("logistic", "cascade") and C is None:
        C = _choose_C(Xtr, ytr, groups[train], C_grid, n_splits, seed)
    if method == "logistic":
        return fit_logistic(Xtr, ytr, C, seed).decision(X[test])
    if method == "prototype":
        w, b = differential_prototype(Xtr[ytr == 1], Xtr[ytr == 0])
        return prototype_scores(X[test], w, b)
    if method == "attentive":  # X = jetons (fenêtres, jetons, dim) ou grille 4-D
        from blanci.attentive import fit_attentive
        from blanci.pooling import flat_tokens

        return fit_attentive(flat_tokens(Xtr), ytr, seed=seed).decision(flat_tokens(X[test]))
    if method == "cascade":
        if tokens is None:
            raise ValueError("la cascade a besoin des jetons (tokens=…)")
        return cascade_scores(
            Xtr, ytr, tokens[train], X[test], tokens[test], C, cascade_fraction, seed
        )
    if method == "exemplar":
        return exemplar_scores(Xtr[ytr == 1], X[test])
    if method == "exemplar_medoid":
        return medoid_scores(Xtr[ytr == 1], X[test])
    if method == "simple_prototype":
        return simple_prototype_scores(Xtr[ytr == 1], X[test])
    if method == "knn":
        return knn_scores(Xtr, ytr, X[test])
    raise ValueError(f"méthode inconnue : {method} (connues : {METHODS})")


def oof_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    method: str = "logistic",
    C_grid: list[float] | None = None,
    seed: int = 0,
    gated: np.ndarray | None = None,
    assignment: dict[str, int] | None = None,
    tokens: np.ndarray | None = None,
    cascade_fraction: float = 0.2,
) -> OOFScores:
    """Scores hors-pli sur plis groupés (par micro).

    Méthodes (`METHODS`) : exemplar_medoid (un seul exemple), exemplar (plus proche positif),
    knn, simple_prototype, prototype (différentiel), logistic, attentive (X = jetons en
    (fenêtres, jetons, dim) ou (fenêtres, temps, fréquence, dim), `blanci/attentive.py`),
    cascade (X = embeddings, `tokens` = jetons des mêmes fenêtres).

    Pour `logistic`, le C est choisi dans chaque pli sur les seules données d'entraînement.
    `gated` (seuillage en amont) : fenêtres arrêtées, jamais apprises, score `GATED_SCORE`.
    `assignment` : plis communs à tous les modèles ({point: pli}, `dataset.folds_for`).
    """
    X, y, groups = np.asarray(X, dtype=np.float32), np.asarray(y).astype(int), np.asarray(groups)
    gated = np.zeros(len(y), dtype=bool) if gated is None else np.asarray(gated, dtype=bool)
    folds = grouped_folds(y, groups, n_splits, seed, assignment)
    out = np.full(len(y), np.nan, dtype=np.float64)
    for train, test in folds:
        train = train[~gated[train]]
        out[test] = fit_and_score(
            method,
            X,
            y,
            groups,
            train,
            test,
            C_grid=C_grid,
            n_splits=n_splits,
            seed=seed,
            tokens=tokens,
            cascade_fraction=cascade_fraction,
        )
    out[gated] = GATED_SCORE
    return OOFScores(out, tuple(folds), method)
