import numpy as np
import pytest

from blanci.evaluation.evaluate import (
    average_precision,
    bootstrap_ci,
    evaluate,
    grouped_folds,
    paired_bootstrap,
    recall_at_precision,
    to_recordings,
    wilson_interval,
)

# --- Wilson : les deux intervalles cités au §6 de la feuille de route ------------------------


def test_wilson_matches_roadmap_n30():
    """n = 30, rappel 0,9 → [0,74 ; 0,97] (§6)."""
    lo, hi = wilson_interval(27, 30)
    assert (round(lo, 2), round(hi, 2)) == (0.74, 0.97)


def test_wilson_matches_roadmap_n345():
    """n = 345, rappel ≈ 0,9 → [0,86 ; 0,93] (§6) : l'intervalle se resserre avec l'effectif."""
    lo, hi = wilson_interval(310, 345)
    assert (round(lo, 2), round(hi, 2)) == (0.86, 0.93)


def test_wilson_is_narrower_with_more_recordings():
    small = wilson_interval(27, 30)
    large = wilson_interval(310, 345)
    assert large[1] - large[0] < small[1] - small[0]


def test_wilson_stays_inside_unit_interval_at_the_extremes():
    lo, hi = wilson_interval(30, 30)
    assert 0.0 < lo < 1.0 and hi <= 1.0 + 1e-12
    lo, hi = wilson_interval(0, 30)
    assert lo >= -1e-12 and 0.0 < hi < 1.0


def test_wilson_undefined_without_positives():
    assert all(np.isnan(v) for v in wilson_interval(0, 0))


# --- Plis groupés : la règle du §13.7 -------------------------------------------------------


def test_no_group_is_split_across_folds():
    """Un micro ne doit jamais se retrouver des deux côtés d'un pli (§6, plis par micro)."""
    rng = np.random.default_rng(0)
    groups = np.repeat([f"mic{i}" for i in range(10)], 20)
    y = rng.integers(0, 2, len(groups))
    for train, test in grouped_folds(y, groups, n_splits=5, seed=0):
        assert not set(groups[train]) & set(groups[test])


def test_every_example_is_tested_exactly_once():
    groups = np.repeat([f"mic{i}" for i in range(10)], 20)
    y = np.tile([0, 1], len(groups) // 2)
    tested = np.concatenate([test for _, test in grouped_folds(y, groups, 5, 0)])
    assert sorted(tested) == list(range(len(y)))


def test_folds_require_explicit_groups():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        grouped_folds(y, None)
    with pytest.raises(ValueError):
        grouped_folds(y, np.array(["a", "a"]))  # longueur différente


def test_folds_require_at_least_two_groups():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="deux groupes"):
        grouped_folds(y, np.array(["mic1"] * 4))


def test_n_splits_capped_by_group_count():
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    assert len(grouped_folds(y, groups, n_splits=5)) == 3


def test_fold_assignment_puts_each_group_in_exactly_one_fold():
    """Plis communs (DECISIONS n° 91) : calculés sur les enregistrements, un pli par micro."""
    from blanci.evaluation.evaluate import fold_assignment

    groups = np.repeat([f"mic{i}" for i in range(8)], 5)
    has_positive = np.tile([1, 0, 0, 0, 0], 8)
    assignment = fold_assignment(groups, has_positive, n_splits=4, seed=0)
    assert set(assignment) == {f"mic{i}" for i in range(8)}
    assert set(assignment.values()) == {0, 1, 2, 3}


def test_folds_from_an_assignment_do_not_depend_on_the_examples():
    """Deux jeux de fenêtres différents (deux grilles, deux tirages de négatifs), mêmes micros :
    exactement les mêmes micros tenus à l'écart dans chaque pli."""
    from blanci.evaluation.evaluate import fold_assignment

    assignment = fold_assignment(np.array(["a", "b", "c", "d"]), np.array([1, 1, 0, 1]), 2, 0)
    rng = np.random.default_rng(0)
    for n in (40, 97):
        groups = rng.choice(list("abcd"), n)
        y = rng.integers(0, 2, n)
        folds = grouped_folds(y, groups, n_splits=2, assignment=assignment)
        held_out = [sorted(set(groups[test])) for _, test in folds]
        expected = [sorted(g for g, f in assignment.items() if f == k) for k in (0, 1)]
        assert held_out == expected


def test_assignment_must_cover_every_group():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="sans pli"):
        grouped_folds(y, np.array(["a", "a", "z", "z"]), assignment={"a": 0, "b": 1})


# --- Rappel à précision fixée ---------------------------------------------------------------


def test_recall_at_precision_on_separable_scores():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    recall, threshold = recall_at_precision(y, scores, 0.5)
    assert recall == 1.0 and threshold == 5.0


def test_recall_at_precision_trades_precision_for_recall():
    """Un négatif en tête : à P ≥ 0,5 on prend tout, à P ≥ 0,9 aucun seuil ne convient."""
    y = np.array([0, 1, 1])
    scores = np.array([3.0, 2.0, 1.0])
    assert recall_at_precision(y, scores, 0.5) == (1.0, 1.0)
    assert recall_at_precision(y, scores, 0.9) == (0.0, float("inf"))


def test_recall_at_precision_undefined_without_positives():
    y = np.zeros(5, dtype=int)
    assert all(np.isnan(v) for v in recall_at_precision(y, np.arange(5.0), 0.1))


def test_recall_at_precision_is_monotone_in_the_floor():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    scores = y + rng.normal(0, 1.0, 200)
    high, _ = recall_at_precision(y, scores, 0.8)
    low, _ = recall_at_precision(y, scores, 0.1)
    assert low >= high


# --- AP et bootstrap ------------------------------------------------------------------------


def test_average_precision_is_one_when_separable():
    y = np.array([1, 1, 0, 0])
    assert average_precision(y, np.array([4.0, 3.0, 2.0, 1.0])) == 1.0


def test_average_precision_undefined_on_a_single_class():
    assert np.isnan(average_precision(np.zeros(4, dtype=int), np.arange(4.0)))
    assert np.isnan(average_precision(np.ones(4, dtype=int), np.arange(4.0)))


def test_bootstrap_resamples_recordings_not_windows():
    """L'intervalle encadre l'AP observée et reste borné par [0, 1]."""
    rng = np.random.default_rng(0)
    units = np.repeat([f"rec{i}" for i in range(40)], 5)
    y = np.repeat(rng.integers(0, 2, 40), 5)
    scores = y + rng.normal(0, 0.8, len(y))
    lo, hi = bootstrap_ci(y, scores, units, n_boot=200, seed=0)
    assert 0.0 <= lo <= average_precision(y, scores) <= hi <= 1.0


def test_bootstrap_is_reproducible_with_a_fixed_seed():
    units = np.repeat([f"rec{i}" for i in range(20)], 3)
    y = np.tile([1, 0, 0], 20)
    scores = np.linspace(0, 1, len(y))
    assert bootstrap_ci(y, scores, units, n_boot=100, seed=7) == bootstrap_ci(
        y, scores, units, n_boot=100, seed=7
    )


def test_paired_bootstrap_detects_a_real_difference():
    """A meilleur que B seulement si l'intervalle apparié exclut zéro (§6)."""
    rng = np.random.default_rng(0)
    units = np.repeat([f"rec{i}" for i in range(60)], 4)
    y = np.repeat(rng.integers(0, 2, 60), 4)
    good = y + rng.normal(0, 0.4, len(y))
    noise = rng.normal(0, 1.0, len(y))
    better = paired_bootstrap(y, good, noise, units, n_boot=200, seed=0)
    assert better["diff"] > 0 and better["significant"]


def test_paired_bootstrap_calls_a_tie_a_tie():
    rng = np.random.default_rng(0)
    units = np.repeat([f"rec{i}" for i in range(60)], 4)
    y = np.repeat(rng.integers(0, 2, 60), 4)
    scores = y + rng.normal(0, 0.6, len(y))
    tie = paired_bootstrap(y, scores, scores.copy(), units, n_boot=200, seed=0)
    assert tie["diff"] == 0.0 and not tie["significant"]


# --- Fenêtre → enregistrement ----------------------------------------------------------------


def test_to_recordings_takes_the_max_window():
    scores = np.array([0.1, 0.9, 0.2, 0.3])
    labels = np.array([0, 1, 0, 0])
    rec = to_recordings(scores, labels, np.array(["r1", "r1", "r2", "r2"])).set_index(
        "recording_id"
    )
    assert rec.loc["r1", "score"] == 0.9 and rec.loc["r1", "y"] == 1
    assert rec.loc["r2", "score"] == 0.3 and rec.loc["r2", "y"] == 0


def test_to_recordings_top3_averages_the_three_best():
    scores = np.array([1.0, 0.8, 0.6, 0.0])
    rec = to_recordings(scores, np.ones(4, dtype=int), np.array(["r"] * 4), how="top3")
    assert rec.loc[0, "score"] == pytest.approx(0.8)


def test_to_recordings_rejects_an_unknown_aggregation():
    with pytest.raises(ValueError, match="agrégation inconnue"):
        to_recordings(np.zeros(2), np.zeros(2, dtype=int), np.array(["r", "r"]), how="median")


def test_evaluate_reports_both_levels():
    rng = np.random.default_rng(0)
    recordings = np.repeat([f"rec{i}" for i in range(30)], 8)
    y = np.repeat(rng.integers(0, 2, 30), 8)
    scores = y + rng.normal(0, 0.5, len(y))
    window = evaluate(scores, y, recordings, level="window", n_boot=100)
    recording = evaluate(scores, y, recordings, level="recording", n_boot=100)
    assert window["n_pos"] + window["n_neg"] == len(y)
    assert recording["n_pos"] + recording["n_neg"] == 30
    assert set(window) == set(recording)
    for key in ("recall@p0.1", "recall@p0.5", "threshold@p0.1", "ap"):
        assert key in window


# --- Rappel par strate et fausses alarmes (§6) ------------------------------------------------


def test_recall_by_quality_counts_positives_only():
    from blanci.evaluation.evaluate import recall_by_group

    scores = np.array([0.9, 0.8, 0.2, 0.9, 0.1, 0.7, 0.95])
    labels = np.array([1, 1, 1, 1, 1, 0, 0])
    quality = np.array(["A", "A", "B", "B", None, "A", "C"])
    table = recall_by_group(scores, labels, quality, threshold=0.5).set_index("stratum")
    assert table.loc["A", "n_pos"] == 2 and table.loc["A", "recall"] == 1.0
    assert table.loc["B", "recall"] == 0.5
    assert table.loc["?", "recall"] == 0.0  # qualité non renseignée
    assert "C" not in table.index  # que des négatifs
    assert (table["recall_lo"] <= table["recall"]).all()


def test_false_alarms_per_hour():
    from blanci.evaluation.evaluate import false_alarms_per_hour

    scores = np.array([0.9, 0.8, 0.2, 0.7])
    labels = np.array([1, 0, 0, 0])
    assert false_alarms_per_hour(scores, labels, 0.5, audio_hours=2.0) == 1.0
    assert np.isnan(false_alarms_per_hour(scores, labels, 0.5, audio_hours=0))


def test_R77_leave_one_micro_out_puts_each_positive_mic_alone():
    from blanci.evaluation.evaluate import fold_assignment, grouped_folds, lomo_assignment

    groups = np.array(["a", "a", "b", "b", "c", "c", "n1", "n2", "n3"])
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0])
    out = lomo_assignment(groups, y)
    assert [out[g] for g in "abc"] == [0, 1, 2]
    assert {out[g] for g in ("n1", "n2", "n3")} <= {0, 1, 2}
    folds = grouped_folds(y, groups, "lomo")
    assert len(folds) == 3
    for _, test in folds:
        assert len(set(groups[test]) & {"a", "b", "c"}) == 1
    assert fold_assignment(groups, y, "lomo") == out
    with pytest.raises(ValueError, match="deux micros"):
        lomo_assignment(np.array(["a", "b"]), np.array([1, 0]))


# --- AP moyenne par pli (DECISIONS n° 133, 135) ---------------------------------------------------


def two_folds_on_different_scales():
    """Deux plis qui classent parfaitement, l'un avec des scores resserrés et décalés : mis bout
    à bout, les négatifs du premier passent devant les positifs du second."""
    rng = np.random.default_rng(0)
    y = np.tile(np.r_[np.ones(5), np.zeros(45)], 2).astype(int)
    scores = y * 2.0 + rng.uniform(0, 0.5, len(y))
    folds = np.repeat([0, 1], 50)
    scores[folds == 1] = scores[folds == 1] * 0.1 - 1.0
    return y, scores, folds


def test_fold_mean_ap_sees_only_the_ranking_within_each_fold():
    from blanci.evaluation.evaluate import fold_mean_ap

    y, scores, folds = two_folds_on_different_scales()
    assert average_precision(y, scores) < 0.9
    assert fold_mean_ap(y, scores, folds) == (pytest.approx(1.0), 2)
    one_class = np.where(folds == 1, 0, y)  # un pli sans positif ne compte pas
    assert fold_mean_ap(one_class, scores, folds)[1] == 1


def test_evaluate_adds_the_fold_mean_at_both_levels():
    y, scores, folds = two_folds_on_different_scales()
    recordings = np.array([f"r{i // 2}" for i in range(len(y))])  # deux fenêtres par enregistrement
    for level in ("window", "recording"):
        out = evaluate(scores, y, recordings, level, n_boot=10, folds=folds)
        assert out["ap_fold_mean"] == pytest.approx(1.0) and out["n_folds_ap"] == 2
        assert out["ap"] < 0.95
    assert "ap_fold_mean" not in evaluate(scores, y, recordings, "window", n_boot=10)


def test_bootstrap_p_is_never_zero_and_floors_at_two_over_n_plus_one():
    from blanci.evaluation.evaluate import bootstrap_p

    d = np.full(199, 0.3)  # aucun tirage ne traverse zéro
    assert bootstrap_p(d) == pytest.approx(2 / 200)
    assert bootstrap_p(np.r_[d, np.nan]) == pytest.approx(2 / 200)  # les NaN ne comptent pas
    assert bootstrap_p(np.array([-1.0, 1.0] * 50)) == 1.0


def test_paired_bootstrap_p_is_positive_when_every_difference_has_one_sign():
    rng = np.random.default_rng(0)
    y = np.tile([1, 0], 60)
    good = y + rng.normal(0, 0.05, len(y))
    bad = rng.normal(0, 1, len(y))
    out = paired_bootstrap(y, good, bad, np.arange(len(y)), n_boot=100, seed=0)
    assert out["p"] == pytest.approx(2 / 101)
    from blanci.evaluation.evaluate import with_holm

    pd = pytest.importorskip("pandas")
    p = with_holm(pd.DataFrame({"p": [out["p"]] * 115}))["p_holm"]
    assert (p >= 0.11).all()  # Holm ne garde plus un p nul
