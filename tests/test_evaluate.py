import numpy as np
import pytest

from blanci.evaluate import (
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
    from blanci.evaluate import recall_by_group

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
    from blanci.evaluate import false_alarms_per_hour

    scores = np.array([0.9, 0.8, 0.2, 0.7])
    labels = np.array([1, 0, 0, 0])
    assert false_alarms_per_hour(scores, labels, 0.5, audio_hours=2.0) == 1.0
    assert np.isnan(false_alarms_per_hour(scores, labels, 0.5, audio_hours=0))


def test_snr_bins():
    from blanci.evaluate import snr_bins

    bins = snr_bins(np.array([3.0, 6.0, 11.9, 12.0, np.nan]))
    assert bins.tolist() == ["<6 dB", "6–12 dB", "6–12 dB", "≥12 dB", "?"]
