"""Attentive probing (§3) : l'attention retrouve une note que la moyenne des jetons dilue."""

import numpy as np
import pytest

pytest.importorskip("torch")

from blanci.attentive import AttentiveHead, TokenStore, fit_attentive  # noqa: E402
from blanci.evaluate import average_precision  # noqa: E402
from blanci.head import oof_scores  # noqa: E402

N_TOKENS, DIM = 16, 8


def windows(n_per_class=120, strength=3.0, seed=0):
    """Fond propre à chaque fenêtre ; chez les positifs, un seul jeton porte la « note »."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    tokens = rng.normal(0, 1.0, (n, N_TOKENS, DIM)) + rng.normal(0, 1.0, (n, 1, DIM))
    y = np.r_[np.ones(n_per_class), np.zeros(n_per_class)].astype(int)
    where = rng.integers(N_TOKENS, size=n_per_class)
    tokens[np.arange(n_per_class), where, 0] += strength
    groups = np.array([f"m{i % 6}" for i in range(n)])
    return tokens.astype(np.float32), y, groups, where


def test_attention_beats_the_mean_of_tokens_on_a_brief_note():
    tokens, y, groups, _ = windows()
    attentive = oof_scores(tokens, y, groups, n_splits=3, method="attentive")
    pooled = oof_scores(tokens.mean(axis=1), y, groups, n_splits=3, method="logistic")
    ap_attentive = average_precision(y, attentive.values)
    ap_pooled = average_precision(y, pooled.values)  # ~0,5 : la note est diluée dans 16 jetons
    assert ap_attentive > ap_pooled + 0.1


def test_attention_points_at_the_note():
    tokens, y, _, where = windows()
    head = fit_attentive(tokens, y)
    weights = head.attention(tokens[y == 1])
    assert (weights.argmax(axis=1) == where).mean() > 0.7


def test_saved_head_gives_the_same_scores(tmp_path):
    tokens, y, _, _ = windows(n_per_class=40)
    head = fit_attentive(tokens, y, epochs=50)
    head.save(tmp_path / "att")
    again = AttentiveHead.load(tmp_path / "att")
    assert np.allclose(head.decision(tokens), again.decision(tokens), atol=1e-5)
    assert sorted(p.name for p in (tmp_path / "att").iterdir()) == ["manifest.json", "weights.npz"]


def test_token_store_replaces_and_refuses_partial_loads(tmp_path):
    store = TokenStore(tmp_path, "perch_v2-1")
    assert store.load(["a"]) is None
    store.write(["a", "b"], np.zeros((2, 3, 4)))
    store.write(["b", "c"], np.ones((2, 3, 4)))
    got = store.load(["c", "a", "b"])
    assert got.shape == (3, 3, 4) and got[0].max() == 1 and got[1].max() == 0 and got[2].max() == 1
    assert store.load(["a", "z"]) is None  # une fenêtre manque : pas de sonde attentive


def test_probe_table_adds_the_attentive_probe_when_tokens_exist():
    from blanci.benchmark import probe_table

    tokens, y, groups, _ = windows(n_per_class=60)
    recordings = np.array([f"r{i // 3}" for i in range(len(y))])
    table, scores = probe_table(
        tokens.mean(axis=1), y, groups, recordings, n_splits=3, n_boot=10, tokens=tokens
    )
    assert "attentive" in set(table["probe"]) and "attentive" in scores
