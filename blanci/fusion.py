"""Tête de fusion, niveau 2 du stacking (§3).

Régression logistique sur (logit du score de `head` hors-pli, 2–4 descripteurs séquentiels,
logits de congénères Perch en option). Le score de `head` doit être de type `OOFScores` :
un score en-pli ferait croire à la fusion que `head` est parfaite (§13.7).
Environ 10 positifs indépendants par coefficient.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from blanci.evaluate import grouped_folds
from blanci.head import OOFScores

POSITIVES_PER_COEF = 10


@dataclass
class Fusion:
    columns: list[str]
    scaler: StandardScaler
    model: LogisticRegression

    def decision(self, head_score: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        X = _design(head_score, features, self.columns)
        return self.model.decision_function(self.scaler.transform(X))


def _design(head_score: np.ndarray, features: pd.DataFrame, columns: list[str]) -> np.ndarray:
    X = np.column_stack([np.asarray(head_score, dtype=float), features[columns].to_numpy(float)])
    return np.nan_to_num(X, nan=0.0)


def _check(head_oof: OOFScores, y: np.ndarray, columns: list[str]) -> None:
    if not isinstance(head_oof, OOFScores):
        raise TypeError("la fusion n'accepte que des scores hors-pli (OOFScores, §13.7)")
    n_coef = 1 + len(columns)
    if int(np.sum(y)) < POSITIVES_PER_COEF * n_coef:
        warnings.warn(
            f"{int(np.sum(y))} positifs pour {n_coef} coefficients "
            f"(< {POSITIVES_PER_COEF} par coefficient) : fusion instable",
            stacklevel=3,
        )


def fit_fusion(
    head_oof: OOFScores, features: pd.DataFrame, y: np.ndarray, columns: list[str], C: float = 1.0
) -> Fusion:
    _check(head_oof, y, columns)
    X = _design(head_oof.values, features, columns)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(
        scaler.transform(X), y
    )
    return Fusion(columns, scaler, model)


def fusion_oof(
    head_oof: OOFScores,
    features: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    columns: list[str],
    n_splits: int = 5,
    seed: int = 0,
) -> OOFScores:
    """Scores hors-pli de la fusion, pour l'évaluer sur les mêmes groupes que `head`."""
    _check(head_oof, y, columns)
    y = np.asarray(y).astype(int)
    folds = grouped_folds(y, groups, n_splits, seed)
    out = np.full(len(y), np.nan)
    for train, test in folds:
        part = OOFScores(head_oof.values[train], (), head_oof.method)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fusion = fit_fusion(part, features.iloc[train], y[train], columns)
        out[test] = fusion.decision(head_oof.values[test], features.iloc[test])
    return OOFScores(out, tuple(folds), f"fusion({head_oof.method})")


@dataclass
class FusionWeights:
    """Fusion enregistrée sans pickle (JSON) et appliquée en numpy : se recharge partout (§7)."""

    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def from_fusion(cls, fusion: Fusion) -> FusionWeights:
        scale = np.where(fusion.scaler.scale_ > 0, fusion.scaler.scale_, 1.0)
        return cls(
            list(fusion.columns),
            fusion.scaler.mean_.astype(float),
            scale.astype(float),
            fusion.model.coef_[0].astype(float),
            float(fusion.model.intercept_[0]),
        )

    def to_dict(self) -> dict:
        return {
            "columns": self.columns,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FusionWeights:
        return cls(
            list(d["columns"]),
            np.asarray(d["mean"], dtype=float),
            np.asarray(d["scale"], dtype=float),
            np.asarray(d["coef"], dtype=float),
            float(d["intercept"]),
        )

    def decision(self, head_score: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        X = _design(head_score, features, self.columns)
        return ((X - self.mean) / self.scale) @ self.coef + self.intercept
