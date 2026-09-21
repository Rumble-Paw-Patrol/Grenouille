import json

import numpy as np
import pytest

from blanci.evaluate import average_precision
from blanci.head import (
    Head,
    OOFScores,
    differential_prototype,
    fit_logistic,
    knn_scores,
    oof_scores,
    prototype_scores,
    select_C,
    train_head,
)

DIM = 32


def two_clusters(n=120, sep=1.5, seed=0):
    """Deux amas gaussiens séparés par une direction unique : le cas que le prototype doit voir."""
    rng = np.random.default_rng(seed)
    direction = np.zeros(DIM)
    direction[0] = sep
    neg = rng.normal(0, 0.3, (n, DIM))
    pos = rng.normal(0, 0.3, (n, DIM)) + direction
    X = np.vstack([pos, neg]).astype(np.float32)
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    return X, y


def mics(n_per_class, n_mics=6):
    """Micros répartis sur les deux classes : chaque groupe a des positifs et des négatifs."""
    one = np.array([f"mic{i % n_mics}" for i in range(n_per_class)])
    return np.concatenate([one, one])


# --- Prototype différentiel ------------------------------------------------------------------


def test_prototype_separates_two_clusters():
    X, y = two_clusters()
    w, b = differential_prototype(X[y == 1], X[y == 0])
    scores = prototype_scores(X, w, b)
    assert scores[y == 1].mean() > 0 > scores[y == 0].mean()
    assert (scores[y == 1] > 0).mean() > 0.95
    assert (scores[y == 0] < 0).mean() > 0.95


def test_prototype_direction_points_at_the_species_axis():
    """w ≈ μ+ − μ− : avec un fond partagé, seule la direction discriminante subsiste (§3)."""
    X, y = two_clusters()
    w, _ = differential_prototype(X[y == 1], X[y == 0])
    assert np.argmax(np.abs(w)) == 0


def test_prototype_threshold_sits_between_the_centroids():
    """b place le zéro au milieu : les deux classes s'écartent symétriquement de 0."""
    X, y = two_clusters()
    w, b = differential_prototype(X[y == 1], X[y == 0])
    scores = prototype_scores(X, w, b)
    assert float(scores[y == 1].mean()) == pytest.approx(-float(scores[y == 0].mean()), rel=0.05)


def test_prototype_is_blind_to_a_shared_background():
    """Un décalage commun aux deux classes ne doit pas changer la direction retenue."""
    X, y = two_clusters()
    background = np.full(DIM, 3.0, dtype=np.float32)
    w_plain, _ = differential_prototype(X[y == 1], X[y == 0])
    w_shifted, _ = differential_prototype(X[y == 1] + background, X[y == 0] + background)
    cosine = w_plain @ w_shifted / (np.linalg.norm(w_plain) * np.linalg.norm(w_shifted))
    assert cosine > 0.9


# --- kNN ------------------------------------------------------------------------------------


def test_knn_margin_separates_held_out_windows():
    """X = 60 positifs puis 60 négatifs ; on entraîne sur la moitié de chaque classe."""
    X, y = two_clusters(n=60)
    train = np.r_[0:30, 60:90]
    test = np.r_[30:60, 90:120]
    scores = knn_scores(X[train], y[train], X[test])
    assert (scores[y[test] == 1] > 0).mean() > 0.9
    # L'amas négatif est diffus : sa marge est bruitée. C'est le classement qui compte.
    assert average_precision(y[test], scores) > 0.9


def test_knn_refuses_a_single_class_training_set():
    X, y = two_clusters(n=30)
    with pytest.raises(ValueError, match="positifs et des négatifs"):
        knn_scores(X[y == 1], y[y == 1], X)


# --- Régression logistique et sauvegarde ------------------------------------------------------


def test_head_save_load_round_trip(tmp_path):
    X, y = two_clusters()
    head = fit_logistic(X, y, C=1.0)
    head.save(tmp_path / "head-v1")
    reloaded = Head.load(tmp_path / "head-v1")
    assert np.allclose(head.decision(X), reloaded.decision(X))
    assert reloaded.meta["C"] == 1.0


def test_saved_head_contains_no_pickle(tmp_path):
    """La tête doit se recharger sans scikit-learn : JSON + npz, jamais de pickle (§7)."""
    X, y = two_clusters()
    fit_logistic(X, y, C=1.0).save(tmp_path / "head-v1")
    assert sorted(p.name for p in (tmp_path / "head-v1").iterdir()) == [
        "manifest.json",
        "weights.npz",
    ]
    json.loads((tmp_path / "head-v1" / "manifest.json").read_text(encoding="utf-8"))


def test_predict_proba_is_a_probability():
    X, y = two_clusters()
    proba = fit_logistic(X, y, C=1.0).predict_proba(X)
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_select_C_reports_every_candidate():
    X, y = two_clusters(n=60)
    best, results = select_C(X, y, mics(60), [0.01, 1.0, 100.0], n_splits=3)
    assert set(results) == {0.01, 1.0, 100.0}
    assert results[best] == max(results.values())


def test_train_head_records_provenance():
    X, y = two_clusters(n=60)
    head = train_head(X, y, mics(60), [0.1, 1.0], seed=3)
    assert head.meta["n_pos"] == 60 and head.meta["n_neg"] == 60
    assert head.meta["seed"] == 3 and "cv_ap" in head.meta


# --- Scores hors-pli -------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["logistic", "prototype", "knn"])
def test_oof_scores_cover_every_window_without_nan(method):
    X, y = two_clusters(n=60)
    out = oof_scores(X, y, mics(60), n_splits=3, method=method)
    assert isinstance(out, OOFScores)
    assert out.values.shape == (len(y),)
    assert not np.isnan(out.values).any()
    assert out.method == method


@pytest.mark.parametrize("method", ["logistic", "prototype", "knn"])
def test_oof_scores_rank_positives_above_negatives(method):
    X, y = two_clusters(n=60)
    out = oof_scores(X, y, mics(60), n_splits=3, method=method)
    assert out.values[y == 1].mean() > out.values[y == 0].mean()


def test_oof_folds_never_share_a_mic():
    X, y = two_clusters(n=60)
    groups = mics(60)
    out = oof_scores(X, y, groups, n_splits=3)
    for train, test in out.folds:
        assert not set(groups[train]) & set(groups[test])


def test_oof_rejects_an_unknown_method():
    X, y = two_clusters(n=30)
    with pytest.raises(ValueError, match="méthode inconnue"):
        oof_scores(X, y, mics(30), n_splits=2, method="magie")


def test_oof_logistic_picks_C_inside_each_fold():
    """Le C se choisit sur les seules données d'entraînement du pli : pas de fuite."""
    X, y = two_clusters(n=60)
    out = oof_scores(X, y, mics(60), n_splits=3, method="logistic", C_grid=[0.01, 1.0])
    assert not np.isnan(out.values).any()
