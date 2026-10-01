"""R37 : échelle des biais de micro (et de site), σ fixé ou estimé sur les données (GLMM)."""

from __future__ import annotations

import numpy as np

# --- R37 : échelle des colonnes de biais ---------------------------------------------------------


def group_bias_scale(sigma, C: float):
    """Échelle des indicatrices de micro (`head.standardize`) : σ / √C. Sous la pénalité ½‖w‖²
    et le terme C · Σ perte, le biais b d'un micro coûte alors b² / 2σ² rapporté aux données,
    l'a priori N(0, σ²), quel que soit C. Un σ par colonne possible (niveaux micro et site)."""
    out = np.asarray(sigma, dtype=float) / float(np.sqrt(C))
    return float(out) if out.ndim == 0 else out


def bias_sigmas(bias_scale, bias_columns: int, bias_levels: np.ndarray | None = None) -> np.ndarray:
    """σ de chaque colonne de biais : `bias_scale` est un σ, ou un σ par niveau (micro, site),
    `bias_levels` le niveau de chaque colonne (tout au niveau 0 par défaut)."""
    scales = np.atleast_1d(np.asarray(bias_scale, dtype=object))
    if any(str(v) == "glmm" for v in scales):
        raise ValueError("R37 : σ « glmm » à estimer d'abord (`glmm_scales`)")
    levels = np.zeros(int(bias_columns), dtype=int) if bias_levels is None else bias_levels
    return scales.astype(float)[np.asarray(levels, dtype=int)]


# --- R37 en GLMM : σ estimé sur les données ------------------------------------------------------

GLMM_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)  # σ essayés (logit), faute de `R37.glmm_grid`


def glmm_evidence(
    decision: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    indicators: np.ndarray,
    biases: np.ndarray,
    sigmas: np.ndarray,
    weight_penalty: float,
) -> float:
    """Log-vraisemblance marginale approchée (Laplace) d'une logistique à biais aléatoires
    b ~ N(0, σ²) : on intègre les biais autour de leur valeur ajustée b̂, les poids w restent
    à leur valeur ajustée (effets fixes, déjà pénalisés par C).

        log p(y | σ) ≈ − Σ poids · perte(w, b̂) − ½‖w‖² / C − Σ b̂² / 2σ² − ½ log det(I + Σ H)

    avec H = Zᵀ diag(poids · p(1 − p)) Z la courbure de la perte le long des biais (Z : les
    indicatrices) et Σ = diag(σ²). Les deux derniers termes font l'arbitrage : σ grand laisse
    chaque micro coller à ses données mais se paie en log det (un paramètre libre par micro) ;
    σ petit coûte en ajustement si les micros diffèrent vraiment."""
    y = np.asarray(y).astype(int)
    p = 1.0 / (1.0 + np.exp(-decision))
    t = 2.0 * y - 1.0
    loss = float(sample_weight @ np.logaddexp(0.0, -t * decision))
    curvature = (indicators * (sample_weight * p * (1 - p))[:, None]).T @ indicators
    root = np.asarray(sigmas, dtype=float)  # Σ^½ = diag(σ)
    _, logdet = np.linalg.slogdet(np.eye(len(root)) + root[:, None] * curvature * root[None, :])
    prior = float((np.asarray(biases) ** 2 / (2.0 * np.asarray(sigmas) ** 2)).sum())
    return -loss - weight_penalty - prior - 0.5 * float(logdet)


def glmm_scales(
    fit,
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    bias_columns: int,
    bias_scale,
    bias_levels: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    grid=GLMM_GRID,
) -> tuple[list[float], dict[str, float]]:
    """R37 en GLMM : σ de chaque niveau marqué « glmm » dans `bias_scale`, choisi sur `grid` en
    maximisant `glmm_evidence` (Bayes empirique : l'a priori des biais est appris sur les
    données, comme la variance d'un effet aléatoire d'un GLMM). `fit(σ par niveau)` renvoie la
    tête logistique (`head.Head`) ajustée avec ces σ. Deux niveaux : une coordonnée après
    l'autre, deux passes. Renvoie (σ par niveau, log-vraisemblance approchée de chaque essai)."""
    y = np.asarray(y).astype(int)
    X = np.asarray(X, dtype=np.float32)
    counts = np.bincount(y, minlength=2)
    weights = (len(y) / (2.0 * np.maximum(counts, 1)))[y]
    if sample_weight is not None:
        weights = weights * np.asarray(sample_weight, dtype=float)
    n = int(bias_columns)
    indicators = X[:, -n:].astype(np.float64)
    scales = list(np.atleast_1d(np.asarray(bias_scale, dtype=object)))
    free = [k for k, v in enumerate(scales) if str(v) == "glmm"]
    current = [1.0 if str(v) == "glmm" else float(v) for v in scales]
    tried: dict[tuple[float, ...], float] = {}

    def evidence(sigmas: list[float]) -> float:
        key = tuple(sigmas)
        if key not in tried:
            head = fit(list(sigmas))
            sig = bias_sigmas(sigmas, n, bias_levels)
            biases = head.coef[-n:] / head.scale[-n:]  # b = coefficient · échelle de la colonne
            penalty = 0.5 * float(head.coef[:-n].astype(float) @ head.coef[:-n]) / C
            tried[key] = glmm_evidence(
                head.decision(X).astype(float), y, weights, indicators, biases, sig, penalty
            )
        return tried[key]

    for _ in range(2 if len(free) > 1 else 1):
        for k in free:
            options = [[*current[:k], float(v), *current[k + 1 :]] for v in grid]
            current = max(options, key=evidence)
    table = {"/".join(f"{v:g}" for v in key): value for key, value in tried.items()}
    return current, table
