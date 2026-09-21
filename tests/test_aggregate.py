import numpy as np
import pandas as pd

from blanci.aggregate import (
    POINT_CONFIRMED,
    POINT_NOT_DETECTED,
    POINT_TO_CHECK,
    RECORDING_STATUSES,
    aggregate_recording,
    rank_points,
)

WINDOWS = 40
OFFSETS = np.arange(WINDOWS) * 3.0  # grille de 3 s sur 2 min


def scores_with_positives(indices, value=0.9):
    scores = np.zeros(WINDOWS)
    scores[list(indices)] = value
    return scores


# --- Enregistrement ---------------------------------------------------------------------------


def test_continuous_song_is_positive():
    """Chant continu (H20) : toutes les fenêtres au-dessus du seuil."""
    fraction, status = aggregate_recording(np.full(WINDOWS, 0.9), 0.5, OFFSETS)
    assert fraction == 1.0 and status == "positive"


def test_silence_is_negative():
    fraction, status = aggregate_recording(np.zeros(WINDOWS), 0.5, OFFSETS)
    assert fraction == 0.0 and status == "negative"


def test_single_detection_is_suspect_not_positive():
    """Une détection isolée ne confirme rien : elle part en file « suspect » (§1)."""
    _, status = aggregate_recording(scores_with_positives([10]), 0.5, OFFSETS)
    assert status == "suspect"


def test_a_few_clustered_detections_are_suspect():
    """Trois fenêtres contiguës couvrent 6 s : trop resserré pour un chant continu."""
    _, status = aggregate_recording(scores_with_positives([10, 11, 12]), 0.5, OFFSETS)
    assert status == "suspect"


def test_detections_spread_over_time_are_positive():
    _, status = aggregate_recording(scores_with_positives([0, 15, 39]), 0.5, OFFSETS)
    assert status == "positive"


def test_span_is_ignored_when_offsets_are_absent():
    """Sans les offsets, seul le nombre de fenêtres décide."""
    _, status = aggregate_recording(scores_with_positives([10, 11, 12]), 0.5)
    assert status == "positive"


def test_fraction_counts_every_window():
    fraction, _ = aggregate_recording(scores_with_positives(range(10)), 0.5, OFFSETS)
    assert fraction == 0.25


def test_threshold_is_inclusive():
    _, status = aggregate_recording(np.full(WINDOWS, 0.5), 0.5, OFFSETS)
    assert status == "positive"


def test_empty_recording_is_negative():
    fraction, status = aggregate_recording(np.array([]), 0.5)
    assert fraction == 0.0 and status == "negative"


def test_statuses_are_declared():
    for scores in (np.zeros(WINDOWS), scores_with_positives([1]), np.full(WINDOWS, 0.9)):
        assert aggregate_recording(scores, 0.5, OFFSETS)[1] in RECORDING_STATUSES


# --- Point ------------------------------------------------------------------------------------


def decisions(rows):
    """rows : (site, mic_id, jour, fraction, statut)."""
    return pd.DataFrame(
        [
            {
                "dataset": "2026",
                "site": site,
                "mic_id": mic,
                "recording_id": f"{site}{mic}{day}{i}",
                "start_utc": f"2026-01-{day:02d}T10:00:00Z",
                "fraction": fraction,
                "status": status,
            }
            for i, (site, mic, day, fraction, status) in enumerate(rows)
        ]
    )


def test_a_verified_point_is_confirmed():
    points = rank_points(decisions([("mataroni", "M1", 5, 0.8, "verified_positive")]))
    assert points.loc[0, "status"] == POINT_CONFIRMED
    assert points.loc[0, "n_verified"] == 1


def test_an_unverified_detection_is_only_to_check():
    """Jamais « présence confirmée » sans validation humaine (§1)."""
    points = rank_points(decisions([("mataroni", "M1", 5, 0.8, "positive")]))
    assert points.loc[0, "status"] == POINT_TO_CHECK


def test_a_suspect_alone_still_asks_for_a_check():
    points = rank_points(decisions([("mataroni", "M1", 5, 0.1, "suspect")]))
    assert points.loc[0, "status"] == POINT_TO_CHECK


def test_a_silent_point_is_not_detected_never_absent():
    points = rank_points(decisions([("kaw", "K1", 5, 0.0, "negative")]))
    assert points.loc[0, "status"] == POINT_NOT_DETECTED


def test_confirmed_points_rank_before_the_rest():
    points = rank_points(
        decisions(
            [
                ("kaw", "K1", 5, 0.0, "negative"),
                ("tresor", "T1", 5, 0.9, "positive"),
                ("mataroni", "M1", 5, 0.2, "verified_positive"),
            ]
        )
    )
    assert list(points["status"]) == [POINT_CONFIRMED, POINT_TO_CHECK, POINT_NOT_DETECTED]


def test_points_are_ranked_by_strength():
    """À statut égal : d'abord le nombre d'enregistrements positifs, puis les jours."""
    points = rank_points(
        decisions(
            [
                ("tresor", "T1", 5, 0.9, "positive"),
                ("tresor", "T2", 5, 0.5, "positive"),
                ("tresor", "T2", 6, 0.5, "positive"),
                ("tresor", "T2", 7, 0.5, "positive"),
            ]
        )
    )
    assert list(points["mic_id"]) == ["T2", "T1"]
    assert list(points["n_positive"]) == [3, 1]


def test_days_break_a_tie_on_recordings():
    points = rank_points(
        decisions(
            [
                ("kaw", "K1", 5, 0.9, "positive"),
                ("kaw", "K1", 5, 0.9, "positive"),
                ("kaw", "K2", 5, 0.9, "positive"),
                ("kaw", "K2", 6, 0.9, "positive"),
            ]
        )
    )
    assert list(points["mic_id"]) == ["K2", "K1"]
    assert list(points["n_days_positive"]) == [2, 1]


def test_max_fraction_breaks_the_last_tie():
    points = rank_points(
        decisions(
            [
                ("kaw", "K1", 5, 0.2, "positive"),
                ("kaw", "K2", 5, 0.7, "positive"),
            ]
        )
    )
    assert list(points["mic_id"]) == ["K2", "K1"]


def test_suspects_are_counted_separately_from_positives():
    points = rank_points(
        decisions(
            [
                ("mataroni", "M1", 5, 0.9, "positive"),
                ("mataroni", "M1", 6, 0.1, "suspect"),
                ("mataroni", "M1", 7, 0.0, "negative"),
            ]
        )
    )
    assert points.loc[0, "n_recordings"] == 3
    assert points.loc[0, "n_positive"] == 1
    assert points.loc[0, "n_suspect"] == 1
