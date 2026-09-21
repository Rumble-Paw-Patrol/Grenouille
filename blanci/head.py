"""Têtes sur embeddings gelés (§3) : prototype différentiel, kNN cosinus, régression logistique.

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


def prototype_scores(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return l2_normalize(X) @ w + b


def knn_scores(X_train: np.ndarray, y_train: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Marge top-1 : similarité au positif le plus proche − au négatif le plus proche."""
    Xn, Tn = l2_normalize(X), l2_normalize(X_train)
    sims = Xn @ Tn.T
    y_train = np.asarray(y_train).astype(bool)
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
        return ((np.asarray(X, dtype=np.float32) - self.mean) / self.scale) @ self.coef + self.intercept

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
    model = LogisticRegression(
        C=C, class_weight="balanced", max_iter=2000, random_state=seed
    ).fit(scaler.transform(X), y)
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    return Head(
        scaler.mean_.astype(np.float32),
        scale.astype(np.float32),
        model.coef_[0].astype(np.float32),
        float(model.intercept_[0]),
        {"C": C},
    )


def select_C(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, C_grid: list[float], n_splits: int = 5, seed: int = 0
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
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, C_grid: list[float], seed: int = 0
) -> Head:
    """C choisi par validation groupée, puis réentraînement sur tous les labels (§4)."""
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y).astype(int)
    C, cv_ap = select_C(X, y, np.asarray(groups), C_grid, seed=seed)
    head = fit_logistic(X, y, C, seed)
    head.meta |= {"cv_ap": {str(k): v for k, v in cv_ap.items()}, "n_pos": int(y.sum()),
                  "n_neg": int(len(y) - y.sum()), "seed": seed}
    return head


def oof_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    method: str = "logistic",
    C_grid: list[float] | None = None,
    seed: int = 0,
) -> OOFScores:
    """Scores hors-pli sur plis groupés (par micro). Méthodes : logistic, prototype, knn.

    Pour `logistic`, le C est choisi dans chaque pli sur les seules données d'entraînement.
    """
    X, y, groups = np.asarray(X, dtype=np.float32), np.asarray(y).astype(int), np.asarray(groups)
    folds = grouped_folds(y, groups, n_splits, seed)
    out = np.full(len(y), np.nan, dtype=np.float64)
    for train, test in folds:
        if method == "logistic":
            if C_grid and len(np.unique(groups[train])) >= 2:
                C, _ = select_C(X[train], y[train], groups[train], C_grid, n_splits, seed)
            else:
                C = (C_grid or [1.0])[0]
            out[test] = fit_logistic(X[train], y[train], C, seed).decision(X[test])
        elif method == "prototype":
            w, b = differential_prototype(X[train][y[train] == 1], X[train][y[train] == 0])
            out[test] = prototype_scores(X[test], w, b)
        elif method == "knn":
            out[test] = knn_scores(X[train], y[train], X[test])
        else:
            raise ValueError(f"méthode inconnue : {method}")
    return OOFScores(out, tuple(folds), method)
