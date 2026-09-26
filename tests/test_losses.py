"""Benchmark des pertes (R34, R35, DECISIONS n° 112)."""

import numpy as np
import pytest

from blanci.evaluate import average_precision
from blanci.head import oof_scores
from blanci.head_benchmark import expand_methods
from blanci.losses import LOSSES, fit_loss, loss_and_grad
from blanci.regularization import Context, Regularizer

SMOOTH = ("logistic", "focal", "gce", "sce", "sigmoid")


@pytest.mark.parametrize("name", SMOOTH)
def test_gradients_match_finite_differences(name):
    m = np.linspace(-6, 6, 41)
    h = 1e-6
    _, grad = loss_and_grad(name, m)
    numeric = (loss_and_grad(name, m + h)[0] - loss_and_grad(name, m - h)[0]) / (2 * h)
    assert np.allclose(grad, numeric, atol=1e-5)


def test_limits_of_the_robust_losses():
    m = np.linspace(-5, 5, 11)
    logistic = loss_and_grad("logistic", m)[0]
    assert np.allclose(loss_and_grad("focal", m, gamma=0.0)[0], logistic)
    assert np.allclose(loss_and_grad("gce", m, q=1.0)[0], loss_and_grad("sigmoid", m)[0])
    assert np.allclose(loss_and_grad("gce", m, q=1e-6)[0], logistic, atol=1e-4)
    assert loss_and_grad("sigmoid", np.array([-1e3]))[0][0] == pytest.approx(1.0)  # bornée


def corpus(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 12)).astype(np.float32)
    y = (np.arange(n) < n // 3).astype(int)
    X[y == 1, 0] += 2.0
    return X, y, np.array([f"m{i % 6}" for i in range(n)])


@pytest.mark.parametrize("name", LOSSES)
def test_every_loss_head_ranks_positives_first(name):
    X, y, groups = corpus()
    out = oof_scores(X, y, groups, n_splits=3, method=f"loss:{name}", C_grid=[0.1, 1.0]).values
    assert np.isfinite(out).all() and average_precision(y, out) > 0.8


def test_loss_heads_take_the_R15_weights():
    X, y, groups = corpus()
    reg = Regularizer({15: 4.0}, {}, Context(groups, y == 0))
    out = oof_scores(X, y, groups, n_splits=3, method="loss:focal", regularizer=reg).values
    assert np.isfinite(out).all()


def test_unknown_loss_is_refused():
    X, y, _ = corpus(60)
    with pytest.raises(ValueError, match="perte inconnue"):
        fit_loss(X, y, 1.0, loss="magie")


def test_losses_shorthand_expands_to_every_loss_and_the_reference():
    out = expand_methods(["prototype", "losses"])
    assert out[:2] == ["prototype", "logistic"]
    assert out[2:] == [f"loss:{name}" for name in LOSSES]


def test_neighbors_shorthand_expands_to_the_similarity_heads():
    out = expand_methods(["neighbors"])
    assert out[:2] == ["logistic", "prototype"]
    assert {"exemplar_medoid", "knn", "knn:k=3", "knn:k=5:w", "exemplar:k=3"} <= set(out)
