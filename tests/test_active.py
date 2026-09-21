import numpy as np
import pandas as pd
import pytest

from blanci.active import build_queue

MICS = ["M1", "M2", "M3", "M4"]
HOURS = [7, 8, 15, 16]  # heures de pic (§5)


def pool(n=400, seed=0):
    """Scores par enregistrement, répartis sur quatre micros et quatre heures."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "recording_id": [f"rec{i:04d}" for i in range(n)],
            "score": rng.normal(0, 2.0, n),
            "mic_id": [MICS[i % len(MICS)] for i in range(n)],
            "hour": [HOURS[(i // len(MICS)) % len(HOURS)] for i in range(n)],
        }
    )


def labelled(recording_ids):
    return pd.DataFrame({"recording_id": list(recording_ids)})


# --- Composition de la file -------------------------------------------------------------------


def test_queue_respects_the_60_20_20_mix():
    queue = build_queue(pool(), labelled([]), n=50)
    counts = queue["reason"].value_counts()
    assert counts["uncertain"] == 30 and counts["top"] == 10 and counts["random"] == 10


def test_queue_has_the_requested_size():
    assert len(build_queue(pool(), labelled([]), n=50)) == 50


def test_queue_mix_must_sum_to_one():
    with pytest.raises(ValueError, match="sommer à 1"):
        build_queue(pool(), labelled([]), n=50, mix=(0.5, 0.2, 0.2))


def test_custom_mix_is_honoured():
    queue = build_queue(pool(), labelled([]), n=100, mix=(0.5, 0.25, 0.25))
    counts = queue["reason"].value_counts()
    assert counts["uncertain"] == 50 and counts["top"] == 25 and counts["random"] == 25


# --- Exclusion des enregistrements déjà vus ---------------------------------------------------


def test_already_labelled_recordings_are_excluded():
    scores = pool()
    seen = scores["recording_id"].iloc[:100]
    queue = build_queue(scores, labelled(seen), n=50)
    assert not set(queue["recording_id"]) & set(seen)


def test_queue_never_repeats_a_recording():
    queue = build_queue(pool(), labelled([]), n=120)
    assert queue["recording_id"].is_unique


def test_queue_shrinks_to_the_remaining_pool():
    scores = pool(n=30)
    assert len(build_queue(scores, labelled(scores["recording_id"].iloc[:20]), n=50)) == 10


def test_empty_pool_gives_an_empty_queue():
    scores = pool(n=10)
    assert build_queue(scores, labelled(scores["recording_id"]), n=50).empty


# --- Sélection par catégorie ------------------------------------------------------------------


def test_uncertain_are_the_closest_to_the_threshold():
    scores = pool()
    queue = build_queue(scores, labelled([]), n=50, threshold=0.0)
    uncertain = queue[queue["reason"] == "uncertain"]["score"].abs().max()
    top = queue[queue["reason"] == "top"]["score"].min()
    assert uncertain < top


def test_threshold_moves_the_uncertain_band():
    """Le seuil de décision courant déplace les incertains : ce sont les plus proches de lui."""
    scores = pool()
    around_zero = build_queue(scores, labelled([]), n=50, threshold=0.0)
    around_three = build_queue(scores, labelled([]), n=50, threshold=3.0)
    centre_zero = around_zero[around_zero["reason"] == "uncertain"]["score"].mean()
    centre_three = around_three[around_three["reason"] == "uncertain"]["score"].mean()
    assert abs(centre_zero) < 0.5 and centre_three == pytest.approx(3.0, abs=0.5)


def test_top_holds_the_highest_scores_left():
    scores = pool()
    queue = build_queue(scores, labelled([]), n=50)
    top = queue[queue["reason"] == "top"]
    others = queue[queue["reason"] == "random"]
    assert top["score"].min() >= others["score"].max()


def test_random_stratum_spreads_across_mics_and_hours():
    """La strate aléatoire mesure les faux négatifs confiants : elle doit couvrir les strates."""
    queue = build_queue(pool(), labelled([]), n=100)
    random = queue[queue["reason"] == "random"]
    assert random["mic_id"].nunique() == len(MICS)
    assert random["hour"].nunique() == len(HOURS)


def test_random_stratum_is_not_just_the_best_scores():
    queue = build_queue(pool(), labelled([]), n=100)
    random = queue[queue["reason"] == "random"]
    assert random["score"].min() < 0


# --- Reproductibilité -------------------------------------------------------------------------


def test_queue_is_reproducible_with_a_fixed_seed():
    scores = pool()
    first = build_queue(scores, labelled([]), n=50, seed=42)
    second = build_queue(scores, labelled([]), n=50, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_seed_changes_the_random_stratum():
    scores = pool()
    first = build_queue(scores, labelled([]), n=50, seed=1)
    second = build_queue(scores, labelled([]), n=50, seed=2)
    a = set(first[first["reason"] == "random"]["recording_id"])
    b = set(second[second["reason"] == "random"]["recording_id"])
    assert a != b


def test_queue_order_is_shuffled():
    """Ordre mélangé : l'annotateur ne doit pas deviner la catégorie d'un enregistrement."""
    queue = build_queue(pool(), labelled([]), n=100)
    reasons = list(queue["reason"])
    assert reasons != sorted(reasons, key=["uncertain", "top", "random"].index)
