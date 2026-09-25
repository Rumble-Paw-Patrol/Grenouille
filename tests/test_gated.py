"""R85 : sonde à portes, le poids de chaque dimension dépend de la fenêtre (DECISIONS n° 110)."""

import numpy as np
import pytest

pytest.importorskip("torch")

from blanci.evaluate import average_precision  # noqa: E402
from blanci.gated import fit_gated  # noqa: E402
from blanci.head import fit_logistic, oof_scores  # noqa: E402

DIM = 16


def interaction(n=900, seed=0):
    """Le chant (dimension 0) ne compte que si le contexte (dimension 1) est positif : un poids
    fixe par dimension ne sait pas l'exprimer."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, DIM)).astype(np.float32)
    return X, ((X[:, 0] > 0.5) & (X[:, 1] > 0)).astype(int)


def test_gates_start_half_open_like_a_logistic():
    X, y = interaction()
    head = fit_gated(X, y, epochs=0)
    assert np.allclose(head.gates(X), 0.5)


def test_gated_probe_learns_a_context_the_logistic_cannot():
    X, y = interaction(seed=0)
    Xt, yt = interaction(seed=1)
    gated = average_precision(yt, fit_gated(X, y).decision(Xt))
    linear = average_precision(yt, fit_logistic(X, y, C=1.0).decision(Xt))
    assert gated > linear + 0.03


def test_gated_probe_runs_in_the_head_benchmark_folds():
    X, y = interaction(n=300)
    groups = np.array([f"m{i % 6}" for i in range(len(y))])
    out = oof_scores(X, y, groups, n_splits=3, method="gated").values
    assert np.isfinite(out).all()
