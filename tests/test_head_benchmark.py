"""Têtes à comparer et leur benchmark (DECISIONS n° 92–93)."""

import json

import numpy as np
import pandas as pd
import pytest

from blanci.attentive import TokenStore
from blanci.config import load_config
from blanci.db import connect, recording_id_for, utc_now, window_id_for
from blanci.head import (
    cascade_scores,
    exemplar_scores,
    fit_and_score,
    medoid_scores,
    oof_scores,
)
from blanci.head_benchmark import (
    annotation_curve,
    head_methods,
    plot_curve,
    run_head_benchmark,
)
from blanci.pooling import as_grid, available_poolings, flat_tokens, pool
from blanci.store import EmbeddingStore

DIM = 8
WINDOW_S = 3.0
N_WINDOWS = 6


# --- Poolings ----------------------------------------------------------------------------------


def grid_tokens():
    """2 fenêtres, 3 temps × 2 fréquences, dim 1 : valeurs 0..5 et 10..15."""
    return np.arange(12, dtype=np.float32).reshape(2, 3, 2, 1) + np.array([0, 4]).reshape(
        2, 1, 1, 1
    )


def test_poolings_on_a_time_frequency_grid():
    t = grid_tokens()[:1]  # temps × fréquence : [[0, 1], [2, 3], [4, 5]]
    assert pool(t, "mean")[0, 0] == 2.5
    assert pool(t, "max")[0, 0] == 5
    assert pool(t, "meanmax").shape == (1, 2)
    assert pool(t, "meanstd").shape == (1, 2)
    assert pool(t, "topk")[0, 0] == 4.5  # moyenne des deux plus fortes
    assert pool(t, "mean_f_max_t")[0, 0] == 4.5  # moyennes en fréquence 0,5 / 2,5 / 4,5
    assert pool(t, "max_f_mean_t")[0, 0] == 3.0  # maxima en fréquence 1 / 3 / 5


def test_grid_poolings_need_a_grid():
    flat = flat_tokens(grid_tokens())
    assert flat.shape == (2, 6, 1)
    assert "mean_f_max_t" not in available_poolings(flat)
    assert "mean_f_max_t" in available_poolings(grid_tokens())
    assert available_poolings(None) == []
    with pytest.raises(ValueError, match="grille"):
        pool(flat, "mean_f_max_t")


def test_as_grid_restores_both_token_orders():
    t = grid_tokens()
    time_major = t.reshape(2, 6, 1)
    assert np.array_equal(as_grid(time_major, 3, 2, "time_major"), t)
    freq_major = t.transpose(0, 2, 1, 3).reshape(2, 6, 1)
    assert np.array_equal(as_grid(freq_major, 3, 2, "freq_major"), t)
    with pytest.raises(ValueError, match="jetons"):
        as_grid(time_major, 4, 2)


# --- Recherche par l'exemple et cascade -----------------------------------------------------------


def test_exemplar_scores_the_nearest_positive():
    positives = np.array([[1.0, 0.0], [0.0, 1.0]])
    X = np.array([[1.0, 0.1], [-1.0, -1.0]])
    scores = exemplar_scores(positives, X)
    assert scores[0] > 0.99 and scores[1] < 0


def test_medoid_uses_a_single_central_reference():
    positives = np.array([[1.0, 0.0], [0.9, 0.1], [1.0, 0.1], [0.0, 1.0]])  # un écart
    X = np.array([[1.0, 0.05], [0.0, 1.0]])
    scores = medoid_scores(positives, X)
    assert scores[0] > 0.99 and scores[1] < 0.2  # la référence n'est pas l'exemple isolé


def test_cascade_keeps_candidates_ahead_and_reranks_them():
    rng = np.random.default_rng(0)
    n = 80
    y = np.array([1] * 20 + [0] * 60)
    X = rng.normal(0, 1, (n, 4))
    X[y == 1, 0] += 2.0
    T = rng.normal(0, 1, (n, 5, 4)).astype(np.float32)
    T[y == 1, 2, 1] += 4.0  # la note : un jeton
    out = cascade_scores(X[:60], y[:60], T[:60], X[60:], T[60:], fraction=0.3)
    assert out.shape == (20,) and np.isfinite(out).all()
    groups = np.tile(np.arange(4), 20)
    oof = oof_scores(X, y, groups, n_splits=4, method="cascade", tokens=T).values
    assert np.isfinite(oof).all()


def test_fit_and_score_refuses_an_unknown_head():
    X, y = np.ones((4, 2)), np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="inconnue"):
        fit_and_score("magic", X, y, np.array(list("aabb")), np.array([0, 1]), np.array([2, 3]))


def test_head_methods_depend_on_tokens():
    assert "attentive" not in head_methods(None) and "logistic" in head_methods(None)
    names = head_methods(np.zeros((2, 3, 2, 4)))
    assert {"attentive", "cascade", "logistic:max", "logistic:mean_f_max_t"} <= set(names)


# --- Benchmark des têtes de bout en bout ----------------------------------------------------------


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = {key: str(tmp_path / key) for key in cfg["paths"]}
    cfg["benchmark"] |= {"n_boot": 30, "negatives_per_positive": 6}
    cfg["head"] |= {"n_splits": 3, "C_grid": [1.0]}
    cfg["head"]["curve"] = {
        "by": "point",
        "k": [0, 1, 2],
        "repeats": 2,
        "methods": ["prototype", "logistic"],
    }
    return cfg


@pytest.fixture
def corpus(tmp_path, cfg):
    """5 micros × 8 jours à 10 h. Jours 0 à 3 : A. blanci ; les autres : fond (négatifs
    appariés). Chaque micro a son fond propre (c_site) ; la note ajoute une direction commune
    et un jeton fort."""
    con = connect(tmp_path / "db.sqlite")
    rng = np.random.default_rng(0)
    eid = "toy-1"
    params = {
        "sample_rate": 32000,
        "window_s": WINDOW_S,
        "hop_s": 1.5,
        "dim": DIM,
        "has_tokens": True,
        "last_run": {},
    }
    con.execute(
        "INSERT INTO models (model_id, kind, name, version, params_json, created_at) "
        "VALUES (?, 'encoder', 'toy', '1', ?, ?)",
        (eid, json.dumps(params), utc_now()),
    )
    metas, embs, token_rows, token_ids = [], [], [], []
    species = np.zeros(DIM)
    species[0] = 1.5
    for m in range(5):
        site = rng.normal(0, 1.0, DIM)
        for day in range(8):
            rel = f"2026/mataroni/M{m}/M{m}_d{day}.wav"  # le nom seul identifie
            rid = recording_id_for(rel)
            con.execute(
                "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
                "duration_s, sample_rate, channels, qc_flags) VALUES (?, ?, '2026', 'mataroni', "
                "?, ?, 120.0, 32000, 1, '{}')",
                (rid, rel, f"M{m}", f"2026-02-{10 + day:02d}T13:00:00Z"),
            )
            offsets = [round(1.5 * i, 2) for i in range(N_WINDOWS)]
            ids = [window_id_for(rid, o) for o in offsets]
            positive = day < 4
            emb = site + rng.normal(0, 0.5, (N_WINDOWS, DIM))
            tokens = rng.normal(0, 0.5, (N_WINDOWS, 4, 2, DIM)) + site
            if positive:
                emb += species
                tokens[:, 1, 0, 0] += 4.0
                con.executemany(
                    "INSERT INTO windows (window_id, recording_id, offset_s, dur_s) "
                    "VALUES (?, ?, ?, ?)",
                    [(w, rid, o, WINDOW_S) for w, o in zip(ids, offsets, strict=True)],
                )
                con.executemany(
                    "INSERT INTO labels (window_id, label, source, created_at) "
                    "VALUES (?, 'blanci_solo', 'import', ?)",
                    [(w, utc_now()) for w in ids],
                )
            metas.append(pd.DataFrame({"window_id": ids, "recording_id": rid, "offset_s": offsets}))
            embs.append(emb)
            token_rows.append(tokens)
            token_ids += ids
    con.commit()
    EmbeddingStore(cfg["paths"]["embeddings"], eid).write(
        pd.concat(metas, ignore_index=True), np.concatenate(embs), "2026", "mataroni", "202602"
    )
    TokenStore(cfg["paths"]["tokens"], eid).write(token_ids, np.concatenate(token_rows))
    return con, eid


def test_head_benchmark_compares_every_head_on_the_same_folds(corpus, cfg):
    con, eid = corpus
    methods = [
        "prototype",
        "simple_prototype",
        "logistic",
        "exemplar_medoid",
        "logistic:max",
        "attentive",
    ]
    out = run_head_benchmark(con, cfg, eid, methods)
    table = out["table"]
    assert set(table["head"]) == set(methods)
    assert set(table["level"]) == {"window", "recording"}
    assert set(out["comparisons"]["head"]) == set(methods) - {"logistic"}
    assert out["background"]["verdict"]
    from blanci.oof import load_oof

    oof = load_oof(cfg)
    assert set(oof["source"]) == {f"{eid}/{m}" for m in methods}
    assert (oof.groupby("point")["fold"].nunique() == 1).all()


def test_differential_prototype_sees_the_site_background(corpus, cfg):
    """Fond de micro fort : retrancher les négatifs appariés doit aider (diagnostic du §3)."""
    con, eid = corpus
    out = run_head_benchmark(con, cfg, eid, ["prototype", "simple_prototype"])
    recording = out["table"].set_index(["head", "level"])["ap"]
    assert recording[("prototype", "recording")] >= recording[("simple_prototype", "recording")]


def test_annotation_curve_runs_nested_k_on_a_fixed_test_half(corpus, cfg):
    con, eid = corpus
    out = annotation_curve(con, cfg, eid)
    runs = out["runs"]
    assert set(runs["head"]) == {"prototype", "logistic"}
    assert set(runs["k"]) == {0, 1, 2}
    assert runs["target"].nunique() == 5
    assert (runs.groupby(["target", "repeat"])["n_test_pos_recordings"].nunique() == 1).all()
    assert set(out["summary"].columns) >= {"head", "k", "ap_mean", "n_targets"}
    gaps = out["gaps"]
    assert set(gaps["head"]) == {"prototype"} and set(gaps["k"]) == {0, 1, 2}
    assert (gaps["lo"] <= gaps["gap"]).all() and (gaps["gap"] <= gaps["hi"]).all()


def test_annotation_curve_is_reproducible(corpus, cfg):
    con, eid = corpus
    first = annotation_curve(con, cfg, eid)["runs"]
    second = annotation_curve(con, cfg, eid)["runs"]
    pd.testing.assert_frame_equal(first, second)


def test_curve_plot_is_written_when_matplotlib_is_there(corpus, cfg, tmp_path):
    pytest.importorskip("matplotlib")
    con, eid = corpus
    summary = annotation_curve(con, cfg, eid)["summary"]
    assert plot_curve(summary, tmp_path / "courbe.png") and (tmp_path / "courbe.png").exists()


# --- Régularisations (DECISIONS n° 108) -----------------------------------------------------------


def test_regularized_heads_are_named_by_what_they_used(corpus, cfg):
    con, eid = corpus
    methods = [
        "logistic",
        "logistic+R18=4+R13",
        "logistic+R19",
        "prototype+R19",
        "logistic+R21+R15",
        "logistic_to_prototype",
        "lda_shrunk",
        "logistic:gem",
    ]
    out = run_head_benchmark(con, cfg, eid, methods)
    names = {
        "logistic",
        "logistic+R13+R18=4",
        "logistic+R19",
        "prototype+R19",
        "logistic+R15+R21",
        "logistic_to_prototype",
        "lda_shrunk",
        "logistic:gem",
    }
    assert set(out["table"]["head"]) == names
    assert not out["table"]["ap"].isna().any()
    from blanci.oof import load_oof

    assert set(load_oof(cfg)["source"]) == {f"{eid}/{m}" for m in names}


def test_config_variants_join_the_default_heads(corpus, cfg):
    con, eid = corpus
    cfg["regularization"]["variants"] = ["logistic+R17"]
    assert "logistic+R17" in head_methods(None, cfg["regularization"]["variants"])
    out = run_head_benchmark(con, cfg, eid, None)
    assert "logistic+R17" in set(out["table"]["head"])


def test_reference_follows_its_regularizations(corpus, cfg):
    con, eid = corpus
    cfg["head"]["reference"] = "logistic+R19"
    out = run_head_benchmark(con, cfg, eid, ["logistic+R19", "prototype"])
    assert set(out["comparisons"]["reference"]) == {"logistic+R19"}


def test_annotation_curve_accepts_regularized_heads(corpus, cfg):
    con, eid = corpus
    out = annotation_curve(con, cfg, eid, ["logistic", "logistic+R19", "lda_shrunk"])
    assert set(out["runs"]["head"]) == {"logistic", "logistic+R19", "lda_shrunk"}
    assert set(out["gaps"]["head"]) == {"logistic+R19", "lda_shrunk"}
