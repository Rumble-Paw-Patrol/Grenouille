"""Assemblage : ce que les régularisations savent des fenêtres (`Context`), leur application dans
l'ordre (`Regularizer`), et `regularizer_for` (nom de tête → régularisation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from blanci.heads.regularization.glmm import GLMM_GRID
from blanci.heads.regularization.names import (
    CONTEXT_HEADS,
    MAIN_PARAMETER,
    head_name,
    parse_head,
    validate,
)
from blanci.heads.regularization.selection import fine_grid
from blanci.heads.regularization.windows import (
    DomainStats,
    bias_indicators,
    mean_directions,
    nuisance_directions,
    sample_weights,
)

# --- Assemblage -------------------------------------------------------------------------------


@dataclass
class Context:
    """Ce que certaines régularisations savent de chaque fenêtre, hors embedding."""

    groups: np.ndarray  # (n,) groupe `regularization.by` (R13, R21)
    hard: np.ndarray | None = None  # (n,) négatif annoté (R15)
    domain: DomainStats | None = None  # R19, R20
    domain_rows: np.ndarray | None = None  # (n,) indice dans `domain`
    classes: np.ndarray | None = None  # (n,) classe du son (R67, `window_classes`)
    # R81 : fenêtres non annotées du stock (embeddings, groupe, indice dans `domain`)
    pool: np.ndarray | None = None
    pool_groups: np.ndarray | None = None
    pool_domain_rows: np.ndarray | None = None

    def subset(self, rows: np.ndarray) -> Context:
        return Context(
            self.groups[rows],
            None if self.hard is None else self.hard[rows],
            self.domain,
            None if self.domain_rows is None else self.domain_rows[rows],
            None if self.classes is None else self.classes[rows],
            self.pool,
            self.pool_groups,
            self.pool_domain_rows,
        )


@dataclass
class Regularizer:
    """Les régularisations d'une tête, avec leurs réglages et le contexte des fenêtres."""

    regs: dict[int, float | str | None]
    params: dict[str, Any]
    context: Context

    def param(self, number: int, key: str, default: Any) -> Any:
        value = self.regs.get(number)
        if value is not None and MAIN_PARAMETER.get(number) == key:
            return value
        return (self.params.get(f"R{number}") or {}).get(key, default)

    def window_transform(self, X: np.ndarray, domain_rows: np.ndarray | None = None) -> np.ndarray:
        """R19 / R20 puis R17 : fenêtre par fenêtre, sans apprentissage. `domain_rows` : pour
        d'autres fenêtres que celles du contexte (le réservoir de R81)."""
        X = np.asarray(X, dtype=np.float32)
        if {19, 20} & set(self.regs):
            ctx = self.context
            rows = ctx.domain_rows if domain_rows is None else domain_rows
            if ctx.domain is None or rows is None:
                raise ValueError("R19/R20 demandent les statistiques du stock (Context.domain)")
            X = X - ctx.domain.mean[rows]
            if 20 in self.regs:
                eps = float(self.param(20, "eps", 1e-6))
                X = X / (ctx.domain.std[rows] + eps)
        if 17 in self.regs:
            from blanci.embedding.index import l2_normalize

            X = l2_normalize(X)
        return X.astype(np.float32)

    def fit_projection(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0):
        """R18 puis R21, ajustées sur les fenêtres d'entraînement : renvoie x ↦ projection."""
        steps = []
        Z = np.asarray(X, dtype=np.float64)
        if 18 in self.regs:
            from sklearn.decomposition import PCA

            k = int(self.param(18, "components", 32))
            pca = PCA(n_components=max(1, min(k, *Z.shape)), random_state=seed).fit(Z)
            steps.append(pca.transform)
            Z = pca.transform(Z)
        if 21 in self.regs:
            negatives = np.asarray(y) == 0
            how = self.param(21, "method", "means")
            if how == "means":
                Q = mean_directions(Z[negatives], np.asarray(groups)[negatives])
            elif how == "inlp":
                Q = nuisance_directions(
                    Z[negatives],
                    np.asarray(groups)[negatives],
                    iterations=int(self.param(21, "iterations", 3)),
                    max_directions=int(self.param(21, "max_directions", 128)),
                    C=float(self.param(21, "C", 1.0)),
                    seed=seed,
                    tolerance=float(self.param(21, "tolerance", 0.05)),
                )
            else:
                raise ValueError(f"R21 : méthode inconnue {how!r} (means ou inlp)")
            steps.append(lambda A, Q=Q: A - (A @ Q) @ Q.T)
        if not steps:
            return None

        def project(A: np.ndarray) -> np.ndarray:
            A = np.asarray(A, dtype=np.float64)
            for step in steps:
                A = step(A)
            return A.astype(np.float32)

        return project

    pool_prepared: np.ndarray | None = None  # R81 : réservoir transformé par `prepare`
    pool_prepared_groups: np.ndarray | None = None

    def pool_windows(self, project, train: np.ndarray) -> np.ndarray:
        """R81 : le réservoir de fenêtres non annotées, passé par les mêmes transformations que
        les fenêtres annotées. `pool_from: train` (défaut) : seulement les micros
        d'entraînement ; `all` : aussi ceux du pli jugé (adaptation sans labels à un micro
        nouveau)."""
        ctx = self.context
        if ctx.pool is None or ctx.pool_groups is None:
            raise ValueError("R81 demande un réservoir de fenêtres non annotées (Context.pool)")
        keep = np.ones(len(ctx.pool), dtype=bool)
        if self.param(81, "pool_from", "train") == "train":
            keep = np.isin(
                np.asarray(ctx.pool_groups).astype(str),
                np.unique(np.asarray(ctx.groups)[train].astype(str)),
            )
        rows = None if ctx.pool_domain_rows is None else ctx.pool_domain_rows[keep]
        pool = self.window_transform(ctx.pool[keep], rows)
        if project is not None:
            pool = project(pool)
        self.pool_prepared_groups = np.asarray(ctx.pool_groups)[keep]
        return np.asarray(pool, dtype=np.float32)

    def grid(self, C_grid):
        """Grille de C de la tête : R76 la remplace par une grille plus fine."""
        if 76 in self.regs and C_grid:
            return fine_grid(C_grid, int(self.param(76, "points", 13)))
        return C_grid

    def pseudo_options(self) -> dict[str, Any]:
        """Réglages de R81 (`with_pseudo_labels`), section `regularization.R81`."""
        return {
            "min_score": float(self.param(81, "min_score", 3.0)),
            "max_fraction": float(self.param(81, "max_fraction", 0.01)),
            "weight": float(self.param(81, "weight", 0.3)),
            "rounds": int(self.param(81, "rounds", 1)),
        }

    def dann_options(self) -> dict[str, Any]:
        """Réglages de la tête `dann` (R66), section `regularization.R66` de la config."""
        defaults = {
            "hidden": 32,
            "strength": 1.0,
            "domain_on": "negatives",
            "weight_decay": 1e-2,
            "epochs": 300,
            "lr": 1e-2,
        }
        return {k: type(v)(self.param(66, k, v)) for k, v in defaults.items()}

    @property
    def bags(self) -> int:
        """R79 : nombre de tirages bootstrap (0 : pas de bagging)."""
        return int(self.param(79, "bags", 20)) if 79 in self.regs else 0

    def torch_options(self) -> dict[str, Any]:
        """Options des têtes torch (`fit_with_options`) : R40–R47, R59, R64."""
        options: dict[str, Any] = {}
        if 40 in self.regs:
            options["weight_decays"] = [
                float(v) for v in self.param(40, "grid", [1e-4, 1e-3, 1e-2, 1e-1])
            ]
            options["n_splits"] = int(self.param(40, "n_splits", 3))
        if 41 in self.regs:
            options["optimizer"] = "adamw"
            options["weight_decay"] = float(self.param(41, "weight_decay", 1e-2))
        if 42 in self.regs:
            options["early_stopping"] = True
            options["monitor"] = str(self.param(42, "monitor", "ap"))  # provisoire (n° 119)
            options["epochs"] = int(self.param(42, "max_epochs", 300))
            options["n_splits"] = int(self.param(42, "n_splits", options.get("n_splits", 3)))
        if 43 in self.regs:
            options["entropy"] = float(self.param(43, "strength", 0.01))
        if 45 in self.regs:
            options["token_dropout"] = float(self.param(45, "p", 0.2))
        if 46 in self.regs:
            options["dim_dropout"] = float(self.param(46, "p", 0.2))
        if 47 in self.regs:
            options["shrink"] = float(self.param(47, "strength", 1e-2))
        if 59 in self.regs:
            options["warmup"] = int(self.param(59, "warmup", 20))
            options["clip_norm"] = float(self.param(59, "clip_norm", 1.0))
        if 64 in self.regs:
            options["average"] = str(self.param(64, "method", "ema"))
            options["ema_decay"] = float(self.param(64, "ema_decay", 0.99))
            options["swa_start"] = float(self.param(64, "swa_start", 0.75))
        return options

    def bias_scales(self) -> tuple[float | str, ...]:
        """R37 : σ de chaque niveau de biais, (micro,) ou (micro, site) ; « glmm » : σ estimé
        sur les données (`glmm_scales`)."""

        def read(value: Any) -> float | str:
            return "glmm" if str(value) == "glmm" else float(value)

        scales = (read(self.param(37, "scale", 3.0)),)
        site = self.param(37, "site_scale", None)
        return scales if site is None else (*scales, read(site))

    def fit_options(
        self, bias_columns: int = 0, bias_levels: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Options de l'ajustement : pénalité (R27, R28 : l1_ratio, 0 = L2, R26) et biais par
        micro (R37 : nombre de colonnes d'indicatrices, σ par niveau, niveau de chaque colonne,
        grille du GLMM)."""
        options: dict[str, Any] = {}
        if 27 in self.regs:
            options["l1_ratio"] = 1.0
        elif 28 in self.regs:
            options["l1_ratio"] = float(self.param(28, "l1_ratio", 0.5))
        if bias_columns:
            scales = self.bias_scales()
            options |= {
                "bias_columns": bias_columns,
                "bias_scale": scales[0] if len(scales) == 1 else scales,
            }
            if bias_levels is not None and len(scales) > 1:
                options["bias_levels"] = np.asarray(bias_levels)
            if "glmm" in scales:
                options["glmm_grid"] = tuple(
                    float(v) for v in self.param(37, "glmm_grid", GLMM_GRID)
                )
        return options

    def prepare(
        self, X: np.ndarray, y: np.ndarray, train: np.ndarray, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, float]]:
        """(X transformé, toutes fenêtres ; poids des fenêtres `train` ou None ; options de la
        logistique)."""
        X = self.window_transform(X)
        project = self.fit_projection(X[train], y[train], self.context.groups[train], seed)
        if project is not None:
            X = project(X)
        pool = self.pool_windows(project, train) if 81 in self.regs else None
        bias_columns, bias_levels = 0, None
        if 37 in self.regs:
            groups = np.asarray(self.context.groups).astype(str)
            sites = len(self.bias_scales()) > 1 and all("/" in g for g in groups[train])
            indicators, bias_levels = bias_indicators(groups, train, sites)
            bias_columns = indicators.shape[1]
            X = np.hstack([np.asarray(X, dtype=np.float32), indicators])
            if pool is not None:
                pool_groups = np.asarray(self.pool_prepared_groups).astype(str)
                both = np.concatenate([groups[train], pool_groups])
                own = np.arange(len(groups[train]))
                pool = np.hstack([pool, bias_indicators(both, own, sites)[0][len(own) :]])
        self.pool_prepared = pool
        weights = sample_weights(
            y[train],
            self.context.groups[train],
            None if self.context.hard is None else self.context.hard[train],
            self.regs,
            float(self.param(15, "hard_weight", 3.0)),
            float(self.param(36, "power", 0.0)),
        )
        return X, weights, self.fit_options(bias_columns, bias_levels)


def regularizer_for(
    spec: str, cfg: dict, context: Context | None
) -> tuple[str, str, Regularizer | None]:
    """(nom canonique, tête de base, régularisation ou None) d'une tête du benchmark."""
    base, regs = parse_head(spec)
    validate(base, regs)
    name = head_name(base, regs)
    if not regs and base not in CONTEXT_HEADS:
        return name, base, None
    if context is None:
        raise ValueError(f"{name} : contexte des fenêtres manquant")
    return name, base, Regularizer(regs, cfg.get("regularization", {}) or {}, context)
