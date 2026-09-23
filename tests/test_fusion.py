import warnings

import numpy as np
import pandas as pd
import pytest

from blanci.evaluate import average_precision
from blanci.fusion import Fusion, fit_fusion, fusion_oof
from blanci.head import OOFScores

COLUMNS = ["frac_windows", "frac_ioi_blanci"]


def make_case(n=200, seed=0):
    """Enregistrements où `head` est moyenne et où la persistance ajoute de l'information."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    head = y + rng.normal(0, 1.2, n)  # tête bruitée
    features = pd.DataFrame(
        {
            "frac_windows": np.clip(y * 0.6 + rng.normal(0, 0.15, n), 0, 1),
            "frac_ioi_blanci": np.clip(y * 0.5 + rng.normal(0, 0.2, n), 0, 1),
        }
    )
    groups = np.array([f"mic{i % 8}" for i in range(n)])
    return head, features, y, groups


def as_oof(values, folds=()):
    return OOFScores(np.asarray(values, dtype=float), folds, "logistic")


# --- La règle du §13.7 : seuls des scores hors-pli entrent dans la fusion ---------------------


def test_fusion_refuses_a_raw_array():
    head, features, y, _ = make_case()
    with pytest.raises(TypeError, match="hors-pli"):
        fit_fusion(head, features, y, COLUMNS)


def test_fusion_refuses_a_head_score_in_disguise():
    """Un score en-pli ferait croire à la fusion que `head` est parfaite."""
    head, features, y, _ = make_case()
    for impostor in (list(head), pd.Series(head), float(head[0])):
        with pytest.raises(TypeError, match="hors-pli"):
            fit_fusion(impostor, features, y, COLUMNS)


def test_fusion_oof_refuses_a_raw_array():
    head, features, y, groups = make_case()
    with pytest.raises(TypeError, match="hors-pli"):
        fusion_oof(head, features, y, groups, COLUMNS)


def test_fusion_accepts_oof_scores():
    head, features, y, _ = make_case()
    assert isinstance(fit_fusion(as_oof(head), features, y, COLUMNS), Fusion)


# --- Effectifs : environ 10 positifs par coefficient ------------------------------------------


def test_fusion_warns_when_positives_are_too_few():
    head, features, y, _ = make_case(n=20)
    with pytest.warns(UserWarning, match="par coefficient"):
        fit_fusion(as_oof(head), features, y, COLUMNS)


def test_fusion_is_quiet_with_enough_positives():
    head, features, y, _ = make_case(n=400)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fit_fusion(as_oof(head), features, y, COLUMNS)


# --- Comportement de la fusion ----------------------------------------------------------------


def test_fusion_improves_on_a_noisy_head():
    """Le score séquentiel module la décision : il doit aider, jamais mettre un veto (§3)."""
    head, features, y, groups = make_case()
    fused = fusion_oof(as_oof(head), features, y, groups, COLUMNS)
    assert average_precision(y, fused.values) > average_precision(y, head)


def test_fusion_oof_returns_oof_scores_over_the_same_folds():
    head, features, y, groups = make_case()
    fused = fusion_oof(as_oof(head), features, y, groups, COLUMNS, n_splits=4)
    assert isinstance(fused, OOFScores)
    assert fused.values.shape == y.shape
    assert not np.isnan(fused.values).any()
    assert fused.method == "fusion(logistic)"
    for train, test in fused.folds:
        assert not set(groups[train]) & set(groups[test])


def test_fusion_handles_missing_sequential_features():
    """Un enregistrement sans onset donne des IOI NaN : la fusion ne doit pas s'effondrer."""
    head, features, y, _ = make_case()
    features = features.copy()
    features.loc[:20, "frac_ioi_blanci"] = np.nan
    fusion = fit_fusion(as_oof(head), features, y, COLUMNS)
    assert not np.isnan(fusion.decision(head, features)).any()


def test_fusion_uses_only_the_declared_columns():
    head, features, y, _ = make_case()
    features = features.assign(piege=np.arange(len(y), dtype=float))
    fusion = fit_fusion(as_oof(head), features, y, COLUMNS)
    assert fusion.columns == COLUMNS
    without = features.drop(columns="piege")
    assert np.allclose(fusion.decision(head, features), fusion.decision(head, without))


def test_saved_fusion_weights_reproduce_the_sklearn_decision():
    """Fusion enregistrée en JSON, rechargée et appliquée en numpy : même score (§7)."""
    import json

    from blanci.fusion import FusionWeights, fit_fusion
    from blanci.head import OOFScores

    rng = np.random.default_rng(0)
    y = np.r_[np.ones(60), np.zeros(140)].astype(int)
    head = OOFScores(rng.normal(y * 2, 1.0), (), "logistic")
    features = pd.DataFrame({"a": rng.normal(y, 1.0), "b": rng.normal(0, 1.0, len(y))})
    features.loc[3, "a"] = np.nan  # descripteur manquant : 0 comme à l'entraînement
    fusion = fit_fusion(head, features, y, ["a", "b"])
    saved = FusionWeights.from_dict(
        json.loads(json.dumps(FusionWeights.from_fusion(fusion).to_dict()))
    )
    assert np.allclose(
        saved.decision(head.values, features), fusion.decision(head.values, features)
    )
