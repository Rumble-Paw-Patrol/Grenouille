"""Têtes sur embeddings gelés (§3) : recherche par l'exemple, prototypes simple et différentiel,
kNN cosinus, régression logistique, attentive probe et cascade (DECISIONS n° 92) ; logistique
tirée vers le prototype (R30) et LDA à covariance rétrécie (R31), DECISIONS n° 108 ; sonde à
portes (R85, n° 110).

La tête retenue (logistique L2) est stockée sans pickle (JSON + npz, calcul en numpy) : elle se
recharge sur n'importe quelle machine, quelle que soit la version de scikit-learn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from blanci.embedding.index import l2_normalize
from blanci.evaluation.evaluate import average_precision, grouped_folds
from blanci.heads.regularization import (
    bias_sigmas,
    group_bias_scale,
    grouped_search,
    nearest_similarity,
)
from blanci.heads.sequential import GATED_SCORE


@dataclass(frozen=True)
class OOFScores:
    """Scores hors-pli : chaque score vient d'un modèle entraîné sans son micro (§3).

    Seul type accepté en entrée de la fusion (§13.7).
    """

    values: np.ndarray
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    method: str
    raw: np.ndarray | None = None  # avant la recalibration par pli (`fold_calibration`)


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


def exemplar_scores(
    E_pos: np.ndarray, X: np.ndarray, k: int = 1, weighted: bool = False
) -> np.ndarray:
    """Recherche par l'exemple : cosinus au positif d'entraînement le plus proche (k = 1), ou
    moyenne des k plus proches (R39).

    Aucun apprentissage, aucun négatif : ce que donne une requête « trouve-moi des fenêtres
    comme celles-ci » (§5, récolte par similarité)."""
    return nearest_similarity(l2_normalize(X) @ l2_normalize(E_pos).T, k, weighted)


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
    **fit_kw,
) -> np.ndarray:
    """Cascade : un linear probe trie toutes les fenêtres, une attentive probe reclasse les
    `fraction` meilleures (candidats), seules à avoir besoin de leurs jetons (mémoire, §3).

    Le seuil des candidats est pris sur l'entraînement ; l'attentive est apprise sur les
    candidats d'entraînement (les cas difficiles), ou sur tout l'entraînement s'ils n'ont
    qu'une classe. Un candidat reste toujours devant un non-candidat : l'étage 2 reclasse, il
    ne repêche pas.
    """
    from blanci.heads.attentive import fit_attentive
    from blanci.heads.pooling import flat_tokens

    stage1 = fit_logistic(X_train, y_train, C, seed, **fit_kw)
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


def knn_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X: np.ndarray,
    k: int = 1,
    weighted: bool = False,
) -> np.ndarray:
    """Marge kNN : similarité aux positifs les plus proches − aux négatifs les plus proches.

    k = 1 (défaut) : le plus proche de chaque classe. k > 1 et `weighted` : R39
    (`nearest_similarity`). Les k voisins sont pris dans chaque classe séparément : un vote
    parmi les k plus proches, toutes classes confondues, serait écrasé par les ~20 négatifs
    par positif."""
    y_train = np.asarray(y_train).astype(bool)
    if not y_train.any() or y_train.all():
        raise ValueError("la marge kNN exige des positifs et des négatifs dans l'entraînement")
    Xn, Tn = l2_normalize(X), l2_normalize(X_train)
    sims = Xn @ Tn.T
    return nearest_similarity(sims[:, y_train], k, weighted) - nearest_similarity(
        sims[:, ~y_train], k, weighted
    )


_NEIGHBORS = re.compile(r"^(knn|exemplar):k=(\d+)(:w)?$")


def neighbor_options(method: str) -> tuple[str, int, bool] | None:
    """« knn:k=5:w » → ("knn", 5, True) ; None si `method` n'est pas une variante à k (R39)."""
    match = _NEIGHBORS.match(method)
    if not match:
        return None
    k = int(match.group(2))
    if k < 1:
        raise ValueError(f"{method} : k doit valoir au moins 1")
    return match.group(1), k, bool(match.group(3))


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
        from scipy.special import expit  # sans débordement sur GATED_SCORE (−1e6)

        return expit(self.decision(X))

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


_SKLEARN = tuple(int(v) for v in re.match(r"(\d+)\.(\d+)", sklearn.__version__).groups())


def _penalty(l1_ratio: float) -> dict[str, Any]:
    """Arguments de pénalité : l1_ratio 0 = L2 (R26), 1 = L1 (R27), entre les deux = Elastic
    Net (R28). `penalty` est déprécié depuis scikit-learn 1.8."""
    if not l1_ratio:
        return {"max_iter": 2000}
    kw: dict[str, Any] = {"l1_ratio": float(l1_ratio), "solver": "saga", "max_iter": 5000}
    if _SKLEARN < (1, 8):
        kw["penalty"] = "elasticnet"
    return kw


def standardize(
    X: np.ndarray, bias_columns: int = 0, bias_scale: float | np.ndarray = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X standardisé, moyenne, échelle) ; `(X − moyenne) / échelle` redonne la même chose.

    Les `bias_columns` dernières colonnes (R37 : indicatrices du micro) ne sont pas
    standardisées mais multipliées par `bias_scale` : sous la pénalité ½‖w‖², le biais du
    micro b = bias_scale · w_micro coûte ½ (b / bias_scale)². Les têtes passent
    bias_scale = σ / √C : rapportée au terme des données (C · Σ perte), la pénalité vaut
    b² / 2σ², l'a priori b ~ N(0, σ²) d'un modèle mixte, quel que soit C. σ petit : biais
    très pénalisés, proches de 0 ; σ grand : un biais libre par micro (effets fixes).
    `bias_scale` : une échelle, ou une par colonne (biais de micro et de site)."""
    X = np.asarray(X)
    n = int(bias_columns)
    d = X.shape[1] - n
    scaler = StandardScaler().fit(X[:, :d])
    mean = np.concatenate([scaler.mean_, np.zeros(n)])
    scale = np.concatenate(
        [
            np.where(scaler.scale_ > 0, scaler.scale_, 1.0),
            1.0 / np.broadcast_to(np.asarray(bias_scale, dtype=float), (n,)),
        ]
    )
    if not n:
        return scaler.transform(X), mean, scale
    return (X - mean) / scale, mean, scale


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
    l1_ratio: float = 0.0,
    bias_columns: int = 0,
    bias_scale: float | str | tuple = 1.0,
    bias_levels: np.ndarray | None = None,
    glmm_grid: tuple[float, ...] | None = None,
) -> Head:
    """Standardisation + logistique à classes équilibrées. `sample_weight` : R13, R15, R36 ;
    `l1_ratio` : R27, R28 ; `bias_columns`, `bias_scale` (σ, écart-type a priori des biais de
    micro, en logit ; un σ par niveau avec `bias_levels`, micro puis site ; « glmm » : σ estimé
    sur les données, `regularization.glmm_scales`) : R37 (`blanci/heads/regularization/`,
    `standardize`)."""
    scales = list(np.atleast_1d(np.asarray(bias_scale, dtype=object)))
    if bias_columns and "glmm" in [str(v) for v in scales]:
        from blanci.heads.regularization import GLMM_GRID, glmm_scales

        def fit_with(sigmas: list[float]) -> Head:
            return fit_logistic(
                X, y, C, seed, sample_weight, l1_ratio, bias_columns, tuple(sigmas), bias_levels
            )

        chosen, evidence = glmm_scales(
            fit_with,
            X,
            y,
            C,
            bias_columns,
            tuple(scales),
            bias_levels,
            sample_weight,
            glmm_grid or GLMM_GRID,
        )
        head = fit_with(chosen)
        head.meta["glmm_evidence"] = evidence
        return head
    sigmas = bias_sigmas(bias_scale, bias_columns, bias_levels)
    Z, mean, scale = standardize(X, bias_columns, group_bias_scale(sigmas, C))
    model = LogisticRegression(
        C=C, class_weight="balanced", random_state=seed, **_penalty(l1_ratio)
    ).fit(Z, y, sample_weight=sample_weight)
    meta: dict[str, Any] = {"C": C}
    if l1_ratio:
        meta["l1_ratio"] = float(l1_ratio)
    if bias_columns:
        shown = [float(v) for v in scales]
        meta |= {
            "group_biases": int(bias_columns),
            "bias_scale": shown[0] if len(shown) == 1 else shown,
        }
    return Head(
        mean.astype(np.float32),
        scale.astype(np.float32),
        model.coef_[0].astype(np.float32),
        float(model.intercept_[0]),
        meta,
    )


def fit_logistic_to_prototype(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
) -> Head:
    """R30 : logistique dont la pénalité tire w vers le prototype différentiel.

    Perte = ½‖w − w₀‖² + C · Σ perte logistique (la L2 ordinaire tire vers w = 0). w₀ = α·u,
    u = μ₊ − μ₋ normé, dans l'espace standardisé de la logistique ; α et le biais de départ
    viennent d'une logistique à une variable sur la projection x·u. C petit : la tête est le
    prototype différentiel (mis à l'échelle) ; C grand : la logistique libre ; entre les deux,
    elle ne s'écarte du prototype que là où les données le justifient. Mêmes C que la
    logistique, choisis de la même façon.
    """
    from scipy.optimize import minimize
    from scipy.special import expit

    y = np.asarray(y).astype(int)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X).astype(np.float64)
    u = Z[y == 1].mean(axis=0) - Z[y == 0].mean(axis=0)
    u /= np.linalg.norm(u) or 1.0
    projection = (Z @ u)[:, None]
    start = LogisticRegression(C=1.0, class_weight="balanced").fit(
        projection, y, sample_weight=sample_weight
    )
    alpha = float(start.coef_[0, 0])
    w0 = alpha * u
    counts = np.bincount(y, minlength=2)
    sw = (len(y) / (2.0 * np.maximum(counts, 1)))[y]  # class_weight="balanced"
    if sample_weight is not None:
        sw = sw * np.asarray(sample_weight, dtype=np.float64)
    t = 2.0 * y - 1.0

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = params[:-1], params[-1]
        margin = t * (Z @ w + b)
        g = -C * sw * t * expit(-margin)
        diff = w - w0
        loss = C * float(np.sum(sw * np.logaddexp(0.0, -margin))) + 0.5 * float(diff @ diff)
        return loss, np.append(Z.T @ g + diff, g.sum())

    result = minimize(
        objective,
        np.append(w0, start.intercept_[0]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000},
    )
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    return Head(
        scaler.mean_.astype(np.float32),
        scale.astype(np.float32),
        result.x[:-1].astype(np.float32),
        float(result.x[-1]),
        {"C": C, "shrink_to": "differential_prototype", "prototype_scale": alpha},
    )


def lda_shrunk_scores(X_train: np.ndarray, y_train: np.ndarray, X: np.ndarray) -> np.ndarray:
    """R31 : analyse discriminante linéaire à covariance rétrécie (Ledoit-Wolf).

    w = Σ⁻¹(μ₊ − μ₋) avec Σ = (1 − α)·Σ̂ + α·cible, α calculé sur les données (aucun
    hyperparamètre, solution fermée). α = 1 : prototype différentiel (à la variance de chaque
    dimension près : scikit-learn rétrécit la matrice de corrélation) ; α = 0 : LDA complète,
    proche de la logistique.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X_train, y_train)
    return lda.decision_function(X)


@dataclass
class MulticlassHead:
    """R67 : logistique multinomiale sur l'embedding standardisé ; score = logit de
    P(A. blanci)."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # (classes, d), ou (1, d) à deux classes
    intercept: np.ndarray
    classes: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def _logits(self, X: np.ndarray) -> np.ndarray:
        z = ((np.asarray(X, dtype=np.float32) - self.mean) / self.scale) @ self.coef.T
        return (z + self.intercept).astype(np.float64)

    def proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilité de chaque classe (n, classes), dans l'ordre de `classes`."""
        from scipy.special import expit, softmax

        z = self._logits(X)
        if len(self.classes) == 2:
            p1 = expit(z[:, 0])
            return np.column_stack([1.0 - p1, p1])
        return softmax(z, axis=1)

    def decision(self, X: np.ndarray) -> np.ndarray:
        """logit P(A. blanci) = z_blanci − log Σ exp(z_autres), en float64 et sans passer par
        la probabilité : arrondie à 1 en float32 dès un logit de ~17, elle donnait +inf, et
        l'AP refusait les scores."""
        from scipy.special import logsumexp

        z = self._logits(X)
        k = self.classes.index("blanci")
        if len(self.classes) == 2:  # scikit-learn : z est le logit de la seconde classe
            return z[:, 0] if k == 1 else -z[:, 0]
        return z[:, k] - logsumexp(np.delete(z, k, axis=1), axis=1)


def fit_multiclass(
    X: np.ndarray,
    classes: np.ndarray,
    C: float,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
    min_count: int = 10,
) -> MulticlassHead:
    """R67 : la tête apprend à distinguer toutes les classes de sons (A. blanci, amphibiens
    dont les congénères, orthoptères, oiseaux, pluie, fond…), pas seulement « A. blanci ou
    non » : la frontière doit se placer sur ce qui est propre à A. blanci. Classes rares
    fusionnées (`regularization.merge_rare_classes`), classes équilibrées, L2."""
    from blanci.heads.regularization import merge_rare_classes

    classes = merge_rare_classes(np.asarray(classes, dtype=object), min_count)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed).fit(
        scaler.transform(X), classes.astype(str), sample_weight=sample_weight
    )
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    names = [str(c) for c in model.classes_]
    return MulticlassHead(
        scaler.mean_.astype(np.float32),
        scale.astype(np.float32),
        model.coef_.astype(np.float32),
        model.intercept_.astype(np.float32),
        names,
        {"C": C, "classes": {n: int((classes == n).sum()) for n in names}},
    )


def select_C(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C_grid: list[float],
    n_splits: int = 5,
    seed: int = 0,
    fitter=None,
    sample_weight: np.ndarray | None = None,
    with_se: bool = False,
    **fit_kw,
) -> tuple:
    """C maximisant l'AP moyenne en validation groupée (plis internes,
    `regularization.grouped_search`).

    `fitter` : fit_logistic (défaut) ou fit_logistic_to_prototype (R30) ; `sample_weight` et
    `fit_kw` (l1_ratio) passent à chaque ajustement. Choix selon `head.selection_rule` (R75) ;
    `with_se` : rend aussi l'erreur type de chaque AP moyenne (chemin de régularisation, R76)."""
    fitter = fitter or fit_logistic

    def score(C, train, test):
        sw = None if sample_weight is None else sample_weight[train]
        head = fitter(X[train], y[train], C, seed, sample_weight=sw, **fit_kw)
        return average_precision(y[test], head.decision(X[test]))

    best, results, ses = grouped_search(score, C_grid, y, groups, n_splits, seed)
    chosen = C_grid[0] if best is None else best
    return (chosen, results, ses) if with_se else (chosen, results)


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
    C, cv_ap, cv_se = select_C(X, y, groups, C_grid, seed=seed, with_se=True)
    head = fit_logistic(X, y, C, seed)
    head.meta |= {
        "cv_ap": {str(k): v for k, v in cv_ap.items()},
        "cv_se": {str(k): v for k, v in cv_se.items()},  # chemin de régularisation (R76)
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
    "logistic_to_prototype",  # R30
    "lda_shrunk",  # R31
    "gated",  # R85 (torch)
    "dann",  # R66 (torch)
    "multiclass",  # R67
    "attentive",
    "proto_probe",  # n° 151 : prototypes sur les jetons (Bird-MAE, AudioProtoPNet)
    "cascade",
)
# Têtes à C, et leur ajustement. `loss:<nom>` : benchmark des pertes (`blanci/heads/losses.py`).
FITTERS = {
    "logistic": fit_logistic,
    "cascade": fit_logistic,
    "logistic_to_prototype": fit_logistic_to_prototype,
}


def _loss_fitter(name: str):
    from blanci.heads.losses import fit_loss

    def fitter(X, y, C, seed=0, sample_weight=None, **fit_kw):
        return fit_loss(X, y, C, seed, sample_weight, loss=name, **fit_kw)

    return fitter


def _fitter(method: str):
    """Ajustement d'une tête à C, ou None."""
    if method.startswith("loss:"):
        return _loss_fitter(method.split(":", 1)[1])
    return FITTERS.get(method)


def _choose_multiclass_C(X, y, classes, groups, C_grid, n_splits, seed, sw, min_count) -> float:
    """R67 : C de la tête multi-classes, jugé comme les autres sur l'AP d'A. blanci."""
    if not C_grid or len(C_grid) < 2 or len(np.unique(groups)) < 2:
        return (C_grid or [1.0])[0]

    def score(C, train, test):
        w = None if sw is None else sw[train]
        head = fit_multiclass(X[train], classes[train], C, seed, w, min_count)
        return average_precision(y[test], head.decision(X[test]))

    best, _, _ = grouped_search(score, C_grid, y, groups, n_splits, seed)
    return C_grid[0] if best is None else best


def _choose_C(X, y, groups, C_grid, n_splits, seed, fitter=None, **fit_kw) -> float:
    """C par validation groupée interne, si la grille a plusieurs valeurs et l'entraînement
    plusieurs groupes ; sinon la première valeur."""
    if C_grid and len(C_grid) > 1 and len(np.unique(groups)) >= 2:
        return select_C(X, y, groups, C_grid, n_splits, seed, fitter, **fit_kw)[0]
    return (C_grid or [1.0])[0]


def _prepared(X, y, train, regularizer, seed):
    """(X, poids des fenêtres `train`, options de la logistique) après les régularisations."""
    if regularizer is None:
        return X, None, {}
    return regularizer.prepare(X, y, train, seed)


def choose_C(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    rows: np.ndarray,
    C_grid: list[float] | None,
    n_splits: int = 5,
    seed: int = 0,
    regularizer=None,
) -> float | None:
    """C de la tête `method` choisi sur les fenêtres `rows`, régularisations comprises ; None
    pour une tête sans C."""
    fitter = _fitter(method)
    if fitter is None:
        return None
    if regularizer is not None:
        C_grid = regularizer.grid(C_grid)  # R76
    X, sw, fit_kw = _prepared(X, y, rows, regularizer, seed)
    return _choose_C(
        X[rows],
        y[rows],
        groups[rows],
        C_grid,
        n_splits,
        seed,
        fitter,
        sample_weight=sw,
        **fit_kw,
    )


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
    regularizer=None,
) -> np.ndarray:
    """Scores des fenêtres `test` par la tête `method` apprise sur les seules fenêtres `train`.

    `C` fixe le C des têtes logistiques ; sinon il est choisi sur `train` (validation groupée
    interne sur `C_grid`). Brique commune de la validation croisée (`oof_scores`) et de la
    courbe selon le nombre d'annotations (`head_benchmark.annotation_curve`).
    `regularizer` (`blanci/heads/regularization/`) : transformations de X ajustées sur `train`,
    poids des fenêtres et pénalité des têtes logistiques.
    """
    X, sw, fit_kw = _prepared(X, y, train, regularizer, seed)
    Xtr, ytr = X[train], y[train]
    fitter = _fitter(method)
    if regularizer is not None:
        C_grid = regularizer.grid(C_grid)  # R76
    if fitter is not None and C is None:
        C = _choose_C(
            Xtr,
            ytr,
            groups[train],
            C_grid,
            n_splits,
            seed,
            fitter,
            sample_weight=sw,
            **fit_kw,
        )
    if method == "logistic" or method == "logistic_to_prototype" or method.startswith("loss:"):
        kw = {} if method == "logistic_to_prototype" else fit_kw

        def fit_rows(rows, weights):
            return fitter(Xtr[rows], ytr[rows], C, seed, sample_weight=weights, **kw)

        bags = regularizer.bags if regularizer is not None else 0
        if regularizer is not None and 81 in regularizer.regs:  # R81
            from blanci.heads.regularization import with_pseudo_labels

            model = with_pseudo_labels(
                lambda Xa, ya, wa: fitter(Xa, ya, C, seed, sample_weight=wa, **kw),
                Xtr,
                ytr,
                sw,
                regularizer.pool_prepared,
                **regularizer.pseudo_options(),
            )
        elif bags:  # R79
            from blanci.heads.regularization import bagged

            model = bagged(fit_rows, ytr, groups[train], bags, seed, sw)
        else:
            model = fit_rows(np.arange(len(ytr)), sw)
        return model.decision(X[test])
    if method == "multiclass":  # R67
        ctx = None if regularizer is None else regularizer.context
        if ctx is None or ctx.classes is None:
            raise ValueError("multiclass : classes des fenêtres manquantes (Context.classes)")
        classes = np.asarray(ctx.classes, dtype=object)[train]
        min_count = int(regularizer.param(67, "min_count", 10))
        if C is None:
            C = _choose_multiclass_C(
                Xtr, ytr, classes, groups[train], C_grid, n_splits, seed, sw, min_count
            )
        return fit_multiclass(Xtr, classes, C, seed, sw, min_count).decision(X[test])
    if method == "lda_shrunk":
        return lda_shrunk_scores(Xtr, ytr, X[test])
    torch_options = {} if regularizer is None else regularizer.torch_options()
    if method == "gated":
        from blanci.heads.gated import fit_gated
        from blanci.heads.regularization import fit_with_options

        return fit_with_options(fit_gated, Xtr, ytr, groups[train], seed, **torch_options).decision(
            X[test]
        )
    if method == "dann":  # R66 : le micro (groupe du contexte) est le domaine à effacer
        from blanci.heads.dann import fit_dann
        from blanci.heads.regularization import fit_with_options

        domains = np.asarray(regularizer.context.groups)[train]
        options = regularizer.dann_options() | torch_options
        return fit_with_options(fit_dann, Xtr, ytr, domains, seed, **options).decision(X[test])
    if method == "prototype":
        w, b = differential_prototype(Xtr[ytr == 1], Xtr[ytr == 0])
        return prototype_scores(X[test], w, b)
    if method == "attentive":  # X = jetons (fenêtres, jetons, dim) ou grille 4-D
        from blanci.heads.attentive import fit_attentive
        from blanci.heads.pooling import flat_tokens
        from blanci.heads.regularization import fit_with_options

        T = flat_tokens(Xtr)
        if torch_options.get("shrink"):  # R47 : C de la logistique de départ, même grille
            torch_options["shrink_C"] = _choose_C(
                T.mean(axis=1), ytr, groups[train], C_grid, n_splits, seed, fit_logistic
            )
        return fit_with_options(
            fit_attentive, T, ytr, groups[train], seed, **torch_options
        ).decision(flat_tokens(X[test]))
    if method == "proto_probe":  # X = jetons, comme attentive (n° 151)
        from blanci.heads.pooling import flat_tokens
        from blanci.heads.proto_probe import fit_proto_probe

        return fit_proto_probe(flat_tokens(Xtr), ytr, seed=seed).decision(flat_tokens(X[test]))
    if method == "cascade":
        if tokens is None:
            raise ValueError("la cascade a besoin des jetons (tokens=…)")
        return cascade_scores(
            Xtr,
            ytr,
            tokens[train],
            X[test],
            tokens[test],
            C,
            cascade_fraction,
            seed,
            sample_weight=sw,
            **fit_kw,
        )
    if method == "exemplar":
        return exemplar_scores(Xtr[ytr == 1], X[test])
    if method == "exemplar_medoid":
        return medoid_scores(Xtr[ytr == 1], X[test])
    if method == "simple_prototype":
        return simple_prototype_scores(Xtr[ytr == 1], X[test])
    if method == "knn":
        return knn_scores(Xtr, ytr, X[test])
    neighbors = neighbor_options(method)
    if neighbors is not None:  # R39 : knn:k=5, exemplar:k=3, knn:k=5:w
        base, k, weighted = neighbors
        if base == "knn":
            return knn_scores(Xtr, ytr, X[test], k, weighted)
        return exemplar_scores(Xtr[ytr == 1], X[test], k, weighted)
    raise ValueError(
        f"méthode inconnue : {method} (connues : {METHODS} ; variantes knn:k=5, exemplar:k=3, "
        "knn:k=5:w)"
    )


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
    regularizer=None,
    calibration: str | None = None,
    calibration_splits: int = 3,
) -> OOFScores:
    """Scores hors-pli sur plis groupés (par micro).

    Méthodes (`METHODS`) : exemplar_medoid (un seul exemple), exemplar (plus proche positif),
    knn, leurs variantes à k voisins (R39 : knn:k=5, exemplar:k=3, knn:k=5:w),
    simple_prototype, prototype (différentiel), logistic, logistic_to_prototype (R30),
    lda_shrunk (R31), gated (R85, `blanci/heads/gated.py`), attentive (X = jetons en
    (fenêtres, jetons, dim) ou (fenêtres, temps, fréquence, dim), `blanci/heads/attentive.py`),
    cascade (X = embeddings, `tokens` = jetons des mêmes fenêtres).

    Pour `logistic`, le C est choisi dans chaque pli sur les seules données d'entraînement.
    `gated` (seuillage en amont) : fenêtres arrêtées, jamais apprises, score `GATED_SCORE`.
    `assignment` : plis communs à tous les modèles ({point: pli}, `dataset.folds_for`).
    `calibration` = "platt" : scores de chaque pli recalibrés avant d'être mis bout à bout
    (`calibrated_fold_scores`, DECISIONS n° 135) ; `raw` garde les scores d'avant.
    """
    X, y, groups = np.asarray(X, dtype=np.float32), np.asarray(y).astype(int), np.asarray(groups)
    gated = np.zeros(len(y), dtype=bool) if gated is None else np.asarray(gated, dtype=bool)
    folds = grouped_folds(y, groups, n_splits, seed, assignment)
    out = np.full(len(y), np.nan, dtype=np.float64)
    raw = None if calibration is None else np.full(len(y), np.nan, dtype=np.float64)
    options = {
        "C_grid": C_grid,
        "n_splits": n_splits,
        "seed": seed,
        "tokens": tokens,
        "cascade_fraction": cascade_fraction,
        "regularizer": regularizer,
    }
    for train, test in folds:
        train = train[~gated[train]]
        if calibration is None:
            out[test] = fit_and_score(method, X, y, groups, train, test, **options)
        else:
            out[test], raw[test] = calibrated_fold_scores(
                method, X, y, groups, train, test, calibration, calibration_splits, **options
            )
    out[gated] = GATED_SCORE
    if raw is not None:
        raw[gated] = GATED_SCORE
    return OOFScores(out, tuple(folds), method, raw)


FOLD_CALIBRATIONS = ("platt",)
# Têtes entraînées sans C mais dont l'échelle dépend de l'entraînement du pli (recalibrées).
TRAINED_METHODS = ("gated", "dann", "multiclass", "attentive")


def calibration_options(cfg: dict) -> dict[str, Any]:
    """Options de `oof_scores` et `fusion_model_oof` pour la recalibration par pli, d'après
    `benchmark.fold_calibration` (none | platt) et `benchmark.calibration_splits` ; {} sans."""
    bench = cfg.get("benchmark") or {}
    method = bench.get("fold_calibration")
    if method in (None, "none", False):
        return {}
    splits = int(bench.get("calibration_splits", 3))
    return {"calibration": str(method), "calibration_splits": splits}


def calibrated_fold_scores(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    calibration: str = "platt",
    splits: int = 3,
    **options,
) -> tuple[np.ndarray, np.ndarray]:
    """(scores recalibrés, scores bruts) des fenêtres `test` : remède (b) du n° 133.

    Chaque pli choisit ses réglages (C, R26) sur son entraînement ; un C petit resserre les
    scores. Mis bout à bout, des scores d'échelles différentes faussent le classement commun.
    Ici, le C du pli est choisi une fois ; des plis internes (par micro, `splits`) donnent,
    **avec ce même C**, des scores hors-pli de l'entraînement : même échelle que le modèle du
    pli, sans qu'il les ait vus. Une calibration de Platt à classes équilibrées apprise sur
    eux (`regularization.fold_platt`) ramène les scores du pli testé sur une échelle commune :
    la cote « A. blanci contre fond », indépendante de la part de positifs du pli. Dans un pli,
    le classement ne change pas (a > 0) ; seule la mise bout à bout change.

    Les réglages sans C (époques R42, weight decay R40, σ de R37=glmm) sont re-choisis dans
    chaque pli interne : approximation. Les têtes par similarité (prototypes, exemplar, knn,
    LDA) ne choisissent rien pli par pli et gardent la même échelle d'un pli à l'autre : non
    recalibrées (la recalibration n'y ajouterait que le bruit de son estimation, mesuré au
    n° 135). Sans plis internes utilisables (moins de deux micros à positifs dans
    l'entraînement), les scores restent tels quels."""
    from blanci.heads.regularization import fold_platt, usable_folds

    if calibration not in FOLD_CALIBRATIONS:
        raise ValueError(f"recalibration par pli inconnue : {calibration!r} ({FOLD_CALIBRATIONS})")
    if _fitter(method) is None and method not in TRAINED_METHODS:
        raw = fit_and_score(method, X, y, groups, train, test, **options)
        return raw, raw
    C = choose_C(
        method,
        X,
        y,
        groups,
        train,
        options.get("C_grid"),
        options.get("n_splits", 5),
        options.get("seed", 0),
        options.get("regularizer"),
    )
    fixed = options | ({"C": C} if C is not None else {})
    raw = fit_and_score(method, X, y, groups, train, test, **fixed)
    inner_folds = usable_folds(y[train], groups[train], splits, options.get("seed", 0))
    if not inner_folds:
        return raw, raw
    inner = np.full(len(train), np.nan)
    for inner_train, inner_test in inner_folds:
        inner[inner_test] = fit_and_score(
            method, X, y, groups, train[inner_train], train[inner_test], **fixed
        )
    seen = np.isfinite(inner)
    a, b = fold_platt(inner[seen], y[train][seen])
    return a * raw + b, raw
