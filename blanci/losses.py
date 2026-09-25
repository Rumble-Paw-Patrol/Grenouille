"""Benchmark des pertes (R34, R35, DECISIONS n° 112) : une même tête linéaire sur l'embedding,
entraînée avec des fonctions de perte différentes.

Toutes : standardisation, classes équilibrées (comme la logistique), pénalité L2 et même
convention que scikit-learn, C · Σ poids · perte + ½‖w‖², C choisi par validation groupée.
Têtes `loss:<nom>` du benchmark (`blanci heads --methods losses` les prend toutes, avec
`logistic` en référence). m = marge = t · (w·x + b), t = ±1 ; p = σ(w·x + b) côté vrai.

| nom | perte | ce qu'elle change |
|---|---|---|
| logistic (référence) | log(1 + e^−m) | — |
| hinge (R34) | max(0, 1 − m) | SVM : ignore les exemples bien classés au-delà de la marge |
| squared_hinge (R34) | max(0, 1 − m)² | SVM lisse (défaut de LinearSVC) |
| least_squares | (1 − m)² | moindres carrés sur ±1 (proche de la LDA, R31) |
| focal (R35) | −(1 − p)^γ log p, γ = 2 | écrase les exemples faciles, insiste sur les durs |
| gce (R35) | (1 − p^q)/q, q = 0,7 | robuste au bruit d'étiquette ; q → 0 : logistic, 1 : sigmoid |
| sce (R35) | α·(−log p) + β·4·(1 − p), α = 0,1, β = 1 | entropie croisée symétrique (robuste) |
| sigmoid (R35) | 1 − p | bornée : un label faux ne coûte jamais plus que 1 |

hinge, squared_hinge : `LinearSVC` ; least_squares : `RidgeClassifier` (α = 1 / 2C) ; les
quatre autres : L-BFGS (scipy), départ depuis la logistique de même C (gce, sce et sigmoid ne
sont pas convexes : le départ compte).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit

LOSSES = ("hinge", "squared_hinge", "least_squares", "focal", "gce", "sce", "sigmoid")
PARAMS: dict[str, dict[str, float]] = {
    "focal": {"gamma": 2.0},
    "gce": {"q": 0.7},
    "sce": {"alpha": 0.1, "beta": 1.0, "A": -4.0},
}


def loss_and_grad(name: str, m: np.ndarray, **params: float) -> tuple[np.ndarray, np.ndarray]:
    """(perte, dérivée par rapport à la marge m), élément par élément."""
    log_p = -np.logaddexp(0.0, -m)  # log σ(m)
    p, q = expit(m), expit(-m)
    if name == "logistic":
        return -log_p, -q
    if name == "focal":
        gamma = params.get("gamma", PARAMS["focal"]["gamma"])
        qg = q**gamma
        return -qg * log_p, gamma * p * qg * log_p - qg * q
    if name == "gce":
        k = params.get("q", PARAMS["gce"]["q"])
        pk = np.exp(k * log_p)
        return (1.0 - pk) / k, -pk * q
    if name == "sce":
        a = params.get("alpha", PARAMS["sce"]["alpha"])
        b = params.get("beta", PARAMS["sce"]["beta"]) * abs(params.get("A", PARAMS["sce"]["A"]))
        return -a * log_p + b * q, -a * q - b * p * q
    if name == "sigmoid":
        return q, -p * q
    raise ValueError(f"perte inconnue : {name!r} (connues : {LOSSES})")


def _balanced(y: np.ndarray, sample_weight: np.ndarray | None) -> np.ndarray:
    counts = np.bincount(y, minlength=2)
    sw = (len(y) / (2.0 * np.maximum(counts, 1)))[y]
    return sw if sample_weight is None else sw * np.asarray(sample_weight, dtype=np.float64)


def fit_loss(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
    loss: str = "hinge",
    **params: float,
):
    """Tête linéaire (`head.Head`) entraînée avec la perte `loss`."""
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    from blanci.head import Head

    y = np.asarray(y).astype(int)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X).astype(np.float64)
    meta: dict[str, Any] = {"C": C, "loss": loss}
    if loss in ("hinge", "squared_hinge"):
        model = LinearSVC(
            C=C, loss=loss, class_weight="balanced", max_iter=20000, random_state=seed
        ).fit(Z, y, sample_weight=sample_weight)
        w, b = model.coef_[0], float(model.intercept_[0])
    elif loss == "least_squares":
        model = RidgeClassifier(alpha=1.0 / (2.0 * C), class_weight="balanced").fit(
            Z, y, sample_weight=sample_weight
        )
        # coef_ : (1, d) ou (d,) selon la version de scikit-learn.
        w, b = np.ravel(model.coef_), float(np.ravel(model.intercept_)[0])
    elif loss in LOSSES:
        from scipy.optimize import minimize

        start = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(
            Z, y, sample_weight=sample_weight
        )
        sw = _balanced(y, sample_weight)
        t = 2.0 * y - 1.0
        options = PARAMS.get(loss, {}) | params

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            w, b = theta[:-1], theta[-1]
            value, grad = loss_and_grad(loss, t * (Z @ w + b), **options)
            g = C * sw * grad * t
            return C * float(sw @ value) + 0.5 * float(w @ w), np.append(Z.T @ g + w, g.sum())

        result = minimize(
            objective,
            np.append(start.coef_[0], start.intercept_[0]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 2000},
        )
        w, b = result.x[:-1], float(result.x[-1])
        meta |= options
    else:
        raise ValueError(f"perte inconnue : {loss!r} (connues : {LOSSES})")
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    return Head(
        scaler.mean_.astype(np.float32),
        scale.astype(np.float32),
        np.asarray(w, dtype=np.float32),
        b,
        meta,
    )
