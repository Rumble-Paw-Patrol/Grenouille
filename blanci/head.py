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

from blanci.evaluate import average_precision, grouped_folds
from blanci.index import l2_normalize
from blanci.regularization import group_bias_scale, grouped_search, nearest_similarity
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
    from blanci.attentive import fit_attentive
    from blanci.pooling import flat_tokens

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
    X: np.ndarray, bias_columns: int = 0, bias_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X standardisé, moyenne, échelle) ; `(X − moyenne) / échelle` redonne la même chose.

    Les `bias_columns` dernières colonnes (R37 : indicatrices du micro) ne sont pas
    standardisées mais multipliées par `bias_scale` : sous la pénalité ½‖w‖², le biais du
    micro b = bias_scale · w_micro coûte ½ (b / bias_scale)². Les têtes passent
    bias_scale = σ / √C : rapportée au terme des données (C · Σ perte), la pénalité vaut
    b² / 2σ², l'a priori b ~ N(0, σ²) d'un modèle mixte, quel que soit C. σ petit : biais
    très pénalisés, proches de 0 ; σ grand : un biais libre par micro (effets fixes)."""
    X = np.asarray(X)
    n = int(bias_columns)
    d = X.shape[1] - n
    scaler = StandardScaler().fit(X[:, :d])
    mean = np.concatenate([scaler.mean_, np.zeros(n)])
    scale = np.concatenate(
        [np.where(scaler.scale_ > 0, scaler.scale_, 1.0), np.full(n, 1.0 / float(bias_scale))]
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
    bias_scale: float = 1.0,
) -> Head:
    """Standardisation + logistique à classes équilibrées. `sample_weight` : R13, R15, R36 ;
    `l1_ratio` : R27, R28 ; `bias_columns`, `bias_scale` (σ, écart-type a priori des biais de
    micro, en logit) : R37 (`blanci/regularization.py`, `standardize`)."""
    Z, mean, scale = standardize(X, bias_columns, group_bias_scale(bias_scale, C))
    model = LogisticRegression(
        C=C, class_weight="balanced", random_state=seed, **_penalty(l1_ratio)
    ).fit(Z, y, sample_weight=sample_weight)
    meta: dict[str, Any] = {"C": C}
    if l1_ratio:
        meta["l1_ratio"] = float(l1_ratio)
    if bias_columns:
        meta |= {"group_biases": int(bias_columns), "bias_scale": float(bias_scale)}
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


def select_C(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C_grid: list[float],
    n_splits: int = 5,
    seed: int = 0,
    fitter=None,
    sample_weight: np.ndarray | None = None,
    **fit_kw,
) -> tuple[float, dict[float, float]]:
    """C maximisant l'AP moyenne en validation groupée (plis internes,
    `regularization.grouped_search`).

    `fitter` : fit_logistic (défaut) ou fit_logistic_to_prototype (R30) ; `sample_weight` et
    `fit_kw` (l1_ratio) passent à chaque ajustement."""
    fitter = fitter or fit_logistic

    def score(C, train, test):
        sw = None if sample_weight is None else sample_weight[train]
        head = fitter(X[train], y[train], C, seed, sample_weight=sw, **fit_kw)
        return average_precision(y[test], head.decision(X[test]))

    best, results = grouped_search(score, C_grid, y, groups, n_splits, seed)
    return (C_grid[0] if best is None else best), results


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
    "logistic_to_prototype",  # R30
    "lda_shrunk",  # R31
    "gated",  # R85 (torch)
    "attentive",
    "cascade",
)
# Têtes à C, et leur ajustement. `loss:<nom>` : benchmark des pertes (`blanci/losses.py`).
FITTERS = {
    "logistic": fit_logistic,
    "cascade": fit_logistic,
    "logistic_to_prototype": fit_logistic_to_prototype,
}


def _loss_fitter(name: str):
    from blanci.losses import fit_loss

    def fitter(X, y, C, seed=0, sample_weight=None, **fit_kw):
        return fit_loss(X, y, C, seed, sample_weight, loss=name, **fit_kw)

    return fitter


def _fitter(method: str):
    """Ajustement d'une tête à C, ou None."""
    if method.startswith("loss:"):
        return _loss_fitter(method.split(":", 1)[1])
    return FITTERS.get(method)


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
    `regularizer` (`blanci/regularization.py`) : transformations de X ajustées sur `train`,
    poids des fenêtres et pénalité des têtes logistiques.
    """
    X, sw, fit_kw = _prepared(X, y, train, regularizer, seed)
    Xtr, ytr = X[train], y[train]
    fitter = _fitter(method)
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
    if method == "logistic":
        return fit_logistic(Xtr, ytr, C, seed, sample_weight=sw, **fit_kw).decision(X[test])
    if method == "logistic_to_prototype":
        return fit_logistic_to_prototype(Xtr, ytr, C, seed, sample_weight=sw).decision(X[test])
    if method.startswith("loss:"):
        return fitter(Xtr, ytr, C, seed, sample_weight=sw, **fit_kw).decision(X[test])
    if method == "lda_shrunk":
        return lda_shrunk_scores(Xtr, ytr, X[test])
    torch_options = {} if regularizer is None else regularizer.torch_options()
    if method == "gated":
        from blanci.gated import fit_gated
        from blanci.regularization import fit_with_options

        return fit_with_options(fit_gated, Xtr, ytr, groups[train], seed, **torch_options).decision(
            X[test]
        )
    if method == "prototype":
        w, b = differential_prototype(Xtr[ytr == 1], Xtr[ytr == 0])
        return prototype_scores(X[test], w, b)
    if method == "attentive":  # X = jetons (fenêtres, jetons, dim) ou grille 4-D
        from blanci.attentive import fit_attentive
        from blanci.pooling import flat_tokens
        from blanci.regularization import fit_with_options

        T = flat_tokens(Xtr)
        if torch_options.get("shrink"):  # R47 : C de la logistique de départ, même grille
            torch_options["shrink_C"] = _choose_C(
                T.mean(axis=1), ytr, groups[train], C_grid, n_splits, seed, fit_logistic
            )
        return fit_with_options(
            fit_attentive, T, ytr, groups[train], seed, **torch_options
        ).decision(flat_tokens(X[test]))
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
) -> OOFScores:
    """Scores hors-pli sur plis groupés (par micro).

    Méthodes (`METHODS`) : exemplar_medoid (un seul exemple), exemplar (plus proche positif),
    knn, leurs variantes à k voisins (R39 : knn:k=5, exemplar:k=3, knn:k=5:w),
    simple_prototype, prototype (différentiel), logistic, logistic_to_prototype (R30),
    lda_shrunk (R31), gated (R85, `blanci/gated.py`), attentive (X = jetons en
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
            regularizer=regularizer,
        )
    out[gated] = GATED_SCORE
    return OOFScores(out, tuple(folds), method)
