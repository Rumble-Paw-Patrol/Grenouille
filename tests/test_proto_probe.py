"""Sonde à prototypes (n° 151) : une note brève dans un seul jeton suffit à la trouver."""

import numpy as np
import pytest

pytest.importorskip("torch")

from blanci.head import fit_and_score  # noqa: E402
from blanci.proto_probe import fit_proto_probe  # noqa: E402


def _windows(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, t, d = 160, 12, 16
    tokens = rng.normal(size=(n, t, d)).astype(np.float32)
    y = (np.arange(n) % 4 == 0).astype(int)
    note = rng.normal(size=d).astype(np.float32) * 3
    where = rng.integers(0, t, n)
    tokens[y == 1, where[y == 1]] += note  # un seul jeton porte la note
    return tokens, y


def test_proto_probe_trouve_la_note():
    tokens, y = _windows()
    head = fit_proto_probe(tokens[:120], y[:120], seed=0, epochs=150)
    scores = head.decision(tokens[120:])
    assert scores[y[120:] == 1].mean() > scores[y[120:] == 0].mean()
    assert head.activations(tokens[:3]).shape == (3, 8)


def test_proto_probe_par_fit_and_score_et_grille():
    tokens, y = _windows(1)
    grid = tokens.reshape(len(tokens), 6, 2, -1)  # (fenêtres, temps, fréquence, dim)
    groups = np.repeat(["a", "b"], len(y) // 2)
    train, test = np.arange(120), np.arange(120, 160)
    scores = fit_and_score("proto_probe", grid, y, groups, train, test, seed=0)
    assert scores.shape == (40,) and np.isfinite(scores).all()
