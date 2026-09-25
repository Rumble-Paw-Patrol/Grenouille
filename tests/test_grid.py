import numpy as np
import pytest

from blanci.grid import (
    containing_windows,
    hop_for_overlap,
    max_hop_without_cut,
    overlap_from_cfg,
    overlap_of,
    window_grid,
)

NOTE_S = 0.103


def test_grid_3s_on_2min_recording():
    windows = window_grid(120.0, 3.0, 1.5)
    assert len(windows) == 79
    assert windows[0] == (0.0, 3.0)
    assert windows[-1] == (117.0, 3.0)


def test_grid_5s_on_2min_recording():
    windows = window_grid(120.0, 5.0, 2.5)
    assert len(windows) == 47
    assert windows[-1] == (115.0, 5.0)


def test_uncovered_tail_gets_end_aligned_window():
    windows = window_grid(121.0, 3.0, 1.5)
    assert windows[-2:] == [(117.0, 3.0), (118.0, 3.0)]


def test_small_overrun_is_tolerated():
    assert window_grid(119.99, 3.0, 1.5)[-1] == (117.0, 3.0)


def test_short_recording_gives_one_nominal_window():
    assert window_grid(2.0, 3.0, 1.5) == [(0.0, 3.0)]


@pytest.mark.parametrize("window_s, hop_s", [(3.0, 3.5), (3.0, 0.0), (3.0, 1.005)])
def test_invalid_hop_rejected(window_s, hop_s):
    with pytest.raises(ValueError):
        window_grid(120.0, window_s, hop_s)


@pytest.mark.parametrize("window_s, hop_s", [(3.0, 1.5), (5.0, 2.5)])
def test_no_note_is_cut(window_s, hop_s):
    assert hop_s <= max_hop_without_cut(window_s, NOTE_S)
    windows = window_grid(120.0, window_s, hop_s)
    starts = np.random.default_rng(0).uniform(0, 120.0 - NOTE_S, 5000)
    assert all(containing_windows(s, s + NOTE_S, windows) for s in starts)


def test_non_overlapping_grid_cuts_notes():
    windows = window_grid(120.0, 3.0, 3.0)
    assert containing_windows(2.95, 2.95 + NOTE_S, windows) == []


@pytest.mark.parametrize("window_s, hop_s", [(3.0, 1.5), (5.0, 2.5)])
def test_biophonia_3s_annotations_fit_in_one_window(window_s, hop_s):
    windows = window_grid(120.0, window_s, hop_s)
    for k in range(40):
        assert containing_windows(3.0 * k, 3.0 * k + 3.0, windows), k


# --- Chevauchement ajustable (DECISIONS n° 89) ---------------------------------------------------


def test_overlap_zero_gives_a_standard_grid():
    assert hop_for_overlap(3.0, 0.0) == 3.0
    offsets = [o for o, _ in window_grid(120.0, 3.0, hop_for_overlap(3.0, 0.0))]
    assert offsets[:3] == [0.0, 3.0, 6.0] and len(offsets) == 40


def test_overlap_half_is_the_half_window_and_max_is_99_percent():
    assert hop_for_overlap(5.0, 0.5) == 2.5
    assert hop_for_overlap(3.0, 0.99) == 0.03
    assert overlap_of(3.0, 0.03) == pytest.approx(0.99)
    # 3 901 fenêtres pleines, plus une qui déborde de 0,03 s (< tolérance de 0,05 s)
    assert len(window_grid(120.0, 3.0, 0.03)) == 3902


def test_overlap_is_rounded_to_the_hundredth_and_never_below():
    assert hop_for_overlap(0.96, 0.99) == 0.01  # 0,0096 s arrondi, jamais 0
    assert hop_for_overlap(5.0, 0.25) == 3.75
    assert hop_for_overlap(3.0, 0.9) == 0.3


@pytest.mark.parametrize("bad", [-0.1, 0.995, 1.0])
def test_overlap_outside_the_range_is_refused(bad):
    with pytest.raises(ValueError, match="chevauchement"):
        hop_for_overlap(3.0, bad)


def test_overlap_is_read_from_the_config_or_the_old_hop_ratio():
    assert overlap_from_cfg({"encoders": {"overlap": 0.75}}) == 0.75
    assert overlap_from_cfg({"encoders": {"grid_hop_ratio": 0.25}}) == 0.75
    assert overlap_from_cfg({}) == 0.5
