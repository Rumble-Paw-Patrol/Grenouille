import numpy as np
import pytest

from blanci.attention_map import (
    TokenGrid,
    attention_grid,
    band_share,
    hz_to_mel,
    mel_to_hz,
    peak_ratio,
    uniform_band_share,
)

GRID = TokenGrid(time=16, freq=4, fmin_hz=0.0, fmax_hz=16_000.0, scale="linear")


def test_mel_round_trip():
    f = np.array([60.0, 1000.0, 4750.0, 16000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(f)), f)


def test_band_edges_span_the_range_bottom_up():
    edges = TokenGrid(16, 4, 60.0, 16_000.0).band_edges_hz()
    assert edges[0] == pytest.approx(60.0) and edges[-1] == pytest.approx(16_000.0)
    assert np.all(np.diff(edges) > 0)
    assert np.allclose(GRID.band_edges_hz(), [0, 4000, 8000, 12000, 16000])


def test_attention_grid_follows_time_major_order():
    w = np.zeros((1, 64))
    w[0, 2 * 4 + 1] = 1.0  # temps 2, fréquence 1
    g = attention_grid(w, GRID)
    assert g.shape == (1, 16, 4) and g[0, 2, 1] == 1.0
    flipped = attention_grid(w, TokenGrid(16, 4, 0.0, 16_000.0, "linear", low_first=False))
    assert flipped[0, 2, 2] == 1.0


def test_band_share_counts_overlap_pro_rata():
    g = np.zeros((1, 16, 4))
    g[0, 5, 1] = 1.0  # tout le poids sur la bande 4–8 kHz
    # 4,4–5,5 kHz couvre 1,1 kHz des 4 kHz de la bande
    assert band_share(g, GRID, (4400, 5500))[0] == pytest.approx(1.1 / 4)
    g2 = np.zeros((1, 16, 4))
    g2[0, 5, 3] = 1.0
    assert band_share(g2, GRID, (4400, 5500))[0] == 0.0


def test_uniform_share_is_the_band_fraction_of_the_axis():
    assert uniform_band_share(GRID, (4400, 5500)) == pytest.approx(1.1 / 16)


def test_peak_ratio():
    assert peak_ratio(np.full((1, 64), 1 / 64))[0] == pytest.approx(1.0)
    one = np.zeros((1, 64))
    one[0, 7] = 1.0
    assert peak_ratio(one)[0] == pytest.approx(64.0)


def test_plot_attention_draws_one_patch_per_token():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from blanci.attention_map import plot_attention

    fig, ax = plt.subplots()
    plot_attention(ax, np.full((16, 4), 1 / 64), GRID, window_s=5.0)
    assert len(ax.patches) == 64
    plt.close(fig)
