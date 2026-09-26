"""Tête de fusion, niveau 2 du stacking (§3, DECISIONS n° 94).

Entrées (colonnes de niveau 1), toutes hors-pli : le score de `head` (`OOFScores`, §13.7 : un
score en-pli ferait croire à la fusion que `head` est parfaite), les descripteurs du module
séquentiel, et autant d'autres sources qu'on veut : têtes d'autres encodeurs (ensemble de
modèles), logits des congénères Perch. Environ 10 positifs indépendants par coefficient.

Méthodes comparées (`FUSION_METHODS`), toutes sur entrées standardisées sur l'entraînement :

| méthode | combinaison | pondération |
|---|---|---|
| logistic | régression logistique (stacking) | apprise, C fixé (`fusion.C`) |
| logistic+R50 | idem | apprise, C choisi par validation groupée (`fusion.C_grid`) |
| weighted | somme pondérée | fixée à la main (`fusion.weights`) |
| weight_grid | somme pondérée | cherchée sur une grille (AP d'entraînement) |
| mean | moyenne des entrées | égale |
| rank_mean | moyenne des rangs (quantiles de l'entraînement) | égale, insensible aux échelles |
| max | maximum : une entrée forte suffit (règle OU) | — |
| min | minimum : toutes doivent être fortes (règle ET) | — |

Hors logistique, chaque entrée est d'abord orientée (signe de sa corrélation avec le label sur
l'entraînement) : un descripteur « plus c'est bas, plus c'est A. blanci » compte dans le bon
sens. La part de chaque entrée (`FusionModel.weights`) répond à « quel expert écouter en
priorité ? ».

`Fusion`, `fit_fusion`, `fusion_oof` et `FusionWeights` : la première version (logistique sur
le score de `head` et les descripteurs), gardée pour `blanci fusion` et `score --fusion`.
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
    assignment: dict[str, int] | None = None,
) -> OOFScores:
    """Scores hors-pli de la fusion, pour l'évaluer sur les mêmes groupes que `head`."""
    _check(head_oof, y, columns)
    y = np.asarray(y).astype(int)
    folds = grouped_folds(y, groups, n_splits, seed, assignment)
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


# --- Fusion à N entrées (DECISIONS n° 94) ---------------------------------------------------------

FUSION_METHODS = (
    "logistic",
    "logistic+R50",
    "weighted",
    "weight_grid",
    "mean",
    "rank_mean",
    "max",
    "min",
)
N_QUANTILES = 101
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)  # R50, faute de `fusion.C_grid`


@dataclass
class FusionModel:
    """Fusion apprise sur l'entraînement, appliquée en numpy, enregistrée en JSON (§7)."""

    method: str
    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    sign: np.ndarray
    coef: np.ndarray | None = None
    intercept: float = 0.0
    quantiles: np.ndarray | None = None  # (N_QUANTILES, entrées), pour rank_mean
    C: float | None = None  # logistique : le C retenu (R50 : choisi par validation groupée)

    def _z(self, X: np.ndarray) -> np.ndarray:
        z = (np.asarray(X, dtype=float) - self.mean) / self.scale
        return np.nan_to_num(z, nan=0.0)  # entrée manquante : la valeur moyenne

    def decision(self, X: np.ndarray) -> np.ndarray:
        z = self._z(X)
        if self.method in ("logistic", "weighted", "weight_grid", "mean"):
            return z @ self.coef + self.intercept
        oriented = z * self.sign
        if self.method == "max":
            return oriented.max(axis=1)
        if self.method == "min":
            return oriented.min(axis=1)
        if self.method == "rank_mean":
            ranks = np.column_stack(
                [
                    np.searchsorted(self.quantiles[:, j], oriented[:, j]) / N_QUANTILES
                    for j in range(oriented.shape[1])
                ]
            )
            return ranks.mean(axis=1)
        raise ValueError(f"méthode de fusion inconnue : {self.method}")

    def weights(self) -> dict[str, float]:
        """Part de chaque entrée dans la décision (|coefficient| normalisé, entrées
        standardisées) ; parts égales pour les règles sans poids."""
        if self.coef is None:
            return dict.fromkeys(self.columns, 1.0 / len(self.columns))
        magnitude = np.abs(self.coef)
        total = magnitude.sum() or 1.0
        return {c: float(m / total) for c, m in zip(self.columns, magnitude, strict=True)}

    def to_dict(self) -> dict:
        out = {
            "method": self.method,
            "columns": self.columns,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "sign": self.sign.tolist(),
            "intercept": self.intercept,
        }
        if self.coef is not None:
            out["coef"] = self.coef.tolist()
        if self.quantiles is not None:
            out["quantiles"] = self.quantiles.tolist()
        if self.C is not None:
            out["C"] = self.C
        return out

    @classmethod
    def from_dict(cls, d: dict) -> FusionModel:
        return cls(
            d["method"],
            list(d["columns"]),
            np.asarray(d["mean"], dtype=float),
            np.asarray(d["scale"], dtype=float),
            np.asarray(d["sign"], dtype=float),
            np.asarray(d["coef"], dtype=float) if "coef" in d else None,
            float(d.get("intercept", 0.0)),
            np.asarray(d["quantiles"], dtype=float) if "quantiles" in d else None,
            float(d["C"]) if "C" in d else None,
        )


def _simplex(n: int, step: float) -> np.ndarray:
    """Poids positifs de somme 1 sur une grille de pas `step` (n entrées)."""
    units = round(1 / step)

    def compositions(k: int, left: int):
        if k == 1:
            yield (left,)
            return
        for i in range(left + 1):
            for rest in compositions(k - 1, left - i):
                yield (i, *rest)

    return np.array(list(compositions(n, units)), dtype=float) / units


def _orientation(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """+1 si l'entrée monte avec le label sur l'entraînement, −1 sinon."""
    if len(np.unique(y)) < 2:
        return np.ones(z.shape[1])
    corr = [np.corrcoef(z[:, j], y)[0, 1] if z[:, j].std() > 0 else 0.0 for j in range(z.shape[1])]
    return np.where(np.nan_to_num(np.asarray(corr)) < 0, -1.0, 1.0)


def fit_fusion_model(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    C: float = 1.0,
    weights: dict[str, float] | None = None,
    grid_step: float = 0.1,
    max_grid: int = 5000,
    seed: int = 0,
    groups: np.ndarray | None = None,
    C_grid: list[float] | tuple[float, ...] | None = None,
    n_splits: int = 5,
) -> FusionModel:
    """Apprend une fusion `method` sur (X, y) : X = entrées de niveau 1, hors-pli.

    `logistic+R50` : C choisi sur `C_grid` (défaut `C_GRID`) par validation groupée interne
    sur `groups` (`regularization.choose_fusion_C`) ; sans groupes, le C fixé."""
    from blanci.evaluate import average_precision

    if method not in FUSION_METHODS:
        raise ValueError(f"méthode de fusion inconnue : {method!r} (connues : {FUSION_METHODS})")
    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
    if X.ndim != 2 or X.shape[1] != len(columns):
        raise ValueError("une colonne nommée par entrée")
    if method == "logistic+R50":
        if groups is not None:
            from blanci.regularization import choose_fusion_C

            chosen, _ = choose_fusion_C(X, y, groups, C_grid or C_GRID, n_splits, seed)
            C = chosen if chosen is not None else C
        method = "logistic"
    with warnings.catch_warnings():  # colonne entièrement manquante sur l'entraînement
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(X, axis=0)
        scale = np.nanstd(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    z = np.nan_to_num((X - mean) / scale, nan=0.0)
    sign = _orientation(z, y)
    model = FusionModel(method, list(columns), mean, scale, sign)
    if method == "logistic":
        fitted = LogisticRegression(
            C=C, class_weight="balanced", max_iter=2000, random_state=seed
        ).fit(z, y)
        model.coef, model.intercept = fitted.coef_[0].astype(float), float(fitted.intercept_[0])
        model.C = float(C)
    elif method == "weighted":
        if not weights:
            raise ValueError("fusion « weighted » : poids à fixer dans fusion.weights")
        unknown = set(weights) - set(columns)
        if unknown:
            raise ValueError(f"poids pour des entrées absentes : {sorted(unknown)}")
        model.coef = np.array([float(weights.get(c, 0.0)) for c in columns]) * sign
    elif method == "mean":
        model.coef = sign / len(columns)
    elif method == "weight_grid":
        grid = _simplex(len(columns), grid_step)
        if len(grid) > max_grid:  # trop d'entrées : tirage au hasard dans le simplexe
            grid = np.random.default_rng(seed).dirichlet(np.ones(len(columns)), max_grid)
        oriented = z * sign
        aps = np.array([average_precision(y, oriented @ w) for w in grid], dtype=float)
        model.coef = grid[int(np.nanargmax(aps))] * sign if np.isfinite(aps).any() else sign
    elif method == "rank_mean":
        model.quantiles = np.quantile(z * sign, np.linspace(0, 1, N_QUANTILES), axis=0)
    return model


def fusion_model_oof(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    columns: list[str],
    n_splits: int = 5,
    seed: int = 0,
    assignment: dict[str, int] | None = None,
    **options,
) -> OOFScores:
    """Scores hors-pli d'une fusion sur les plis communs : chaque pli apprend sa fusion sans
    le micro testé (les entrées sont déjà hors-pli, §3) ; R50 choisit son C dans chaque pli,
    sur les seuls micros d'entraînement."""
    y = np.asarray(y).astype(int)
    X = np.asarray(X, dtype=float)
    folds = grouped_folds(y, groups, n_splits, seed, assignment)
    out = np.full(len(y), np.nan)
    groups = np.asarray(groups)
    for train, test in folds:
        model = fit_fusion_model(
            method,
            X[train],
            y[train],
            columns,
            seed=seed,
            groups=groups[train],
            n_splits=n_splits,
            **options,
        )
        out[test] = model.decision(X[test])
    return OOFScores(out, tuple(folds), f"fusion:{method}")


def project_scores(target: pd.DataFrame, source: pd.DataFrame) -> np.ndarray:
    """Score d'une autre grille ramené sur les fenêtres cibles (ensemble de modèles).

    target : recording_id, offset_s, dur_s ; source : recording_id, offset_s, dur_s, score.
    Pour chaque fenêtre cible, le maximum des fenêtres sources qui la recouvrent sur au moins
    la moitié de la plus courte des deux ; à défaut, celle dont le centre est le plus proche
    (même enregistrement). NaN si l'enregistrement n'a aucune fenêtre source.
    """
    out = np.full(len(target), np.nan)
    rids = target["recording_id"].to_numpy()
    t_start = target["offset_s"].to_numpy(dtype=float)
    t_end = t_start + target["dur_s"].to_numpy(dtype=float)
    for rid, src in source.groupby("recording_id"):
        rows = np.flatnonzero(rids == rid)
        if not len(rows):
            continue
        s0 = src["offset_s"].to_numpy(dtype=float)
        s1 = s0 + src["dur_s"].to_numpy(dtype=float)
        scores = src["score"].to_numpy(dtype=float)
        for i in rows:
            overlap = np.minimum(t_end[i], s1) - np.maximum(t_start[i], s0)
            enough = overlap >= 0.5 * np.minimum(t_end[i] - t_start[i], s1 - s0) - 1e-9
            if enough.any():
                out[i] = np.nanmax(scores[enough])
            else:
                centers = (s0 + s1) / 2
                out[i] = scores[np.argmin(np.abs(centers - (t_start[i] + t_end[i]) / 2))]
    return out
