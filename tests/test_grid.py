import numpy as np
import pytest

from blanci.grid import containing_windows, max_hop_without_cut, window_grid

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
