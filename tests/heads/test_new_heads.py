"""R66 (DANN), R67 (tête multi-classes), R81 (pseudo-étiquetage) : DECISIONS n° 124."""

import numpy as np
import pytest

from blanci.evaluation.evaluate import average_precision
from blanci.heads.head import fit_multiclass, oof_scores
from blanci.heads.regularization import (
    Context,
    Regularizer,
    merge_rare_classes,
    pseudo_positives,
    regularizer_for,
    window_classes,
)

DIM = 16


def sounds(n=60, seed=0):
    """A. blanci (axe 0), un faux ami qui partage une partie de sa signature (axes 0 et 1),
    la pluie (axe 2) et du fond, sur 6 micros."""
    rng = np.random.default_rng(seed)
    blocks, classes = [], []
    for name, center in (
        ("blanci", [1.5, 0.0, 0.0]),
        ("orthoptère", [1.0, 1.5, 0.0]),
        ("pluie", [0.0, 0.0, 1.5]),
        ("fond", [0.0, 0.0, 0.0]),
    ):
        mean = np.zeros(DIM)
        mean[:3] = center
        count = n * (3 if name == "fond" else 1)
        blocks.append(mean + rng.normal(0, 0.6, (count, DIM)))
        classes += [name] * count
    X = np.vstack(blocks).astype(np.float32)
    classes = np.array(classes, dtype=object)
    y = (classes == "blanci").astype(int)
    groups = np.array([f"m{i % 6}" for i in range(len(y))])
    return X, y, classes, groups


# --- R67 ------------------------------------------------------------------------------------


def test_window_classes_follow_the_annotation_labels():
    labels = np.array(["blanci_solo", "orthoptera", None, "rain", "amphibian_contact_call", "x"])
    y = np.array([1, 0, 0, 0, 0, 0])
    presumed = np.array([False, False, True, False, False, False])
    out = window_classes(labels, y, presumed)
    assert out.tolist() == ["blanci", "orthoptère", "fond", "pluie", "amphibien", "autre"]


def test_rare_classes_are_merged():
    classes = np.array(["blanci"] * 12 + ["fond"] * 30 + ["pluie"] * 3 + ["oiseau"] * 2)
    merged = merge_rare_classes(classes, min_count=10)
    assert set(merged) == {"blanci", "fond"}  # « autre » (5) trop petit : dans le fond
    classes = np.array(["blanci"] * 12 + ["fond"] * 30 + ["pluie"] * 8 + ["oiseau"] * 4)
    assert set(merge_rare_classes(classes, min_count=10)) == {"blanci", "fond", "autre"}


def test_multiclass_head_scores_blanci_and_knows_the_false_friend():
    X, y, classes, groups = sounds()
    head = fit_multiclass(X, classes, C=1.0, min_count=5)
    assert set(head.classes) == {"blanci", "orthoptère", "pluie", "fond"}
    assert average_precision(y, head.decision(X)) > 0.9
    proba = head.proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
    friends = classes == "orthoptère"
    assert (proba[friends].argmax(axis=1) == head.classes.index("orthoptère")).mean() > 0.8


def test_multiclass_score_stays_finite_and_ordered_on_very_sure_windows():
    """Un logit de 20 arrondissait P(A. blanci) à 1 en float32 : score +inf, AP refusée."""
    from blanci.heads.head import MulticlassHead

    classes = ["blanci", "fond", "pluie"]
    coef = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    head = MulticlassHead(np.zeros(2), np.ones(2), coef, np.zeros(3), classes)
    X = np.array([[20.0, 0.0], [40.0, 0.0], [0.0, 0.0], [-40.0, 0.0]], dtype=np.float32)
    scores = head.decision(X)
    assert np.isfinite(scores).all() and (np.diff(scores[[3, 2, 0, 1]]) > 0).all()
    p = head.proba(X)[:, 0]
    assert np.allclose(scores[2], np.log(p[2]) - np.log1p(-p[2]))  # loin des bornes : le logit
    binary = MulticlassHead(np.zeros(2), np.ones(2), coef[:1], np.zeros(1), ["fond", "blanci"])
    assert np.allclose(binary.decision(X), X[:, 0])  # deux classes : z, logit de la seconde
    assert average_precision(np.array([1, 1, 0, 0]), scores) == 1.0


def test_multiclass_runs_in_the_benchmark_and_needs_classes():
    X, y, classes, groups = sounds()
    name, base, reg = regularizer_for("multiclass", {}, Context(groups, classes=classes))
    assert reg is not None  # la tête lit les classes même sans suffixe
    out = oof_scores(
        X, y, groups, n_splits=3, method="multiclass", C_grid=[0.1, 1.0], regularizer=reg
    ).values
    binary = oof_scores(X, y, groups, n_splits=3, method="logistic", C_grid=[0.1, 1.0]).values
    # Ici, apprendre le faux ami comme une classe aide : 0,83 contre 0,74 (données simulées).
    assert np.isfinite(out).all() and average_precision(y, out) > average_precision(y, binary)
    with pytest.raises(ValueError, match="classes"):
        oof_scores(
            X,
            y,
            groups,
            n_splits=3,
            method="multiclass",
            regularizer=regularizer_for("multiclass", {}, Context(groups))[2],
        )
    with pytest.raises(ValueError, match="multiclass"):
        regularizer_for("multiclass+R13", {}, Context(groups))


# --- R81 ------------------------------------------------------------------------------------


def test_pseudo_positives_take_the_sure_windows_up_to_a_cap():
    scores = np.array([5.0, 4.0, 3.5, 1.0, -2.0, 6.0])
    assert pseudo_positives(scores, min_score=3.0, max_fraction=1.0).tolist() == [0, 1, 2, 5]
    assert pseudo_positives(scores, min_score=3.0, max_fraction=0.34).tolist() == [0, 5]


def test_R81_adds_pseudo_positives_from_the_pool_of_training_mics():
    X, y, classes, groups = sounds()
    rng = np.random.default_rng(3)
    pool = np.vstack([X[y == 1][:20] + rng.normal(0, 0.1, (20, DIM)), X[y == 0][:200]])
    pool_groups = np.array([f"m{i % 6}" for i in range(len(pool))])
    context = Context(groups, pool=pool.astype(np.float32), pool_groups=pool_groups)
    reg = Regularizer({81: None}, {"R81": {"min_score": 1.0, "max_fraction": 0.2}}, context)
    train = np.flatnonzero(np.isin(groups, ["m0", "m1", "m2", "m3"]))
    reg.prepare(X, y, train)
    assert set(reg.pool_prepared_groups) <= {"m0", "m1", "m2", "m3"}  # pool_from: train
    out = oof_scores(X, y, groups, n_splits=3, method="logistic", regularizer=reg).values
    plain = oof_scores(X, y, groups, n_splits=3, method="logistic").values
    assert np.isfinite(out).all()
    assert average_precision(y, out) > average_precision(y, plain) - 0.1
    with pytest.raises(ValueError, match="réservoir"):
        Regularizer({81: None}, {}, Context(groups)).prepare(X, y, train)
    with pytest.raises(ValueError, match="R81"):
        regularizer_for("knn+R81", {}, Context(groups))


# --- R66 ------------------------------------------------------------------------------------


def test_R66_adversary_hides_the_mic_in_the_representation():
    pytest.importorskip("torch")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    from blanci.heads.dann import fit_dann
    from tests.heads.test_regularization import shortcut_corpus

    X, y, groups, held = shortcut_corpus(seed=0)
    train = np.flatnonzero(~held)
    accuracy = {}
    for strength in (0.0, 1.0):
        head = fit_dann(
            X[train], y[train], groups=groups[train], strength=strength, hidden=8, epochs=600
        )
        rep, neg = head.representation(X[train]), y[train] == 0
        accuracy[strength] = cross_val_score(
            LogisticRegression(max_iter=1000), rep[neg], groups[train][neg], cv=3
        ).mean()
        assert head.meta["adversary"] == (strength > 0)
    assert accuracy[1.0] < accuracy[0.0] - 0.03


def test_R66_head_runs_in_the_benchmark_with_torch_regularizations():
    pytest.importorskip("torch")
    X, y, classes, groups = sounds(n=40)
    cfg = {"regularization": {"R66": {"epochs": 60}, "R42": {"max_epochs": 40}}}
    name, base, reg = regularizer_for("dann+R42", cfg, Context(groups))
    assert name == "dann+R42" and reg.dann_options()["epochs"] == 60
    out = oof_scores(X, y, groups, n_splits=3, method="dann", regularizer=reg).values
    assert np.isfinite(out).all() and average_precision(y, out) > 0.4  # hasard : 0,17
    with pytest.raises(ValueError, match="R41"):
        regularizer_for("dann+R41", {}, Context(groups))
