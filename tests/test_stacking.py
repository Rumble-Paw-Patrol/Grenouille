"""Fusion à N entrées et emplacement du module séquentiel (DECISIONS n° 94–95)."""

import json

import numpy as np
import pandas as pd
import pytest

from blanci.config import load_config
from blanci.db import connect, recording_id_for, utc_now, window_id_for
from blanci.evaluate import average_precision
from blanci.fusion import (
    FUSION_METHODS,
    FusionModel,
    fit_fusion_model,
    fusion_model_oof,
    project_scores,
)
from blanci.sequential import store_onsets
from blanci.stacking import (
    build_level1,
    design_matrix,
    fusion_benchmark,
    fusion_options,
    onset_counts,
    position_tag,
    positions_from,
    upstream_pass,
)
from blanci.store import EmbeddingStore

# --- Modèles de fusion ---------------------------------------------------------------------------


def two_experts(n=300, seed=0):
    """Deux experts bruités et un descripteur inversé (plus bas = plus A. blanci)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = np.column_stack(
        [
            y * 2.0 + rng.normal(0, 1, n),
            y * 1.0 + rng.normal(0, 1, n),
            -y * 1.5 + rng.normal(0, 1, n),
        ]
    )
    return X, y, ["head", "seq", "inverse"]


@pytest.mark.parametrize("method", [m for m in FUSION_METHODS if m != "weighted"])
def test_every_fusion_method_ranks_the_positives_first(method):
    X, y, columns = two_experts()
    model = fit_fusion_model(method, X, y, columns)
    assert average_precision(y, model.decision(X)) > 0.75


def test_inverse_inputs_are_oriented_before_the_rules():
    X, y, columns = two_experts()
    model = fit_fusion_model("mean", X, y, columns)
    assert model.sign.tolist() == [1.0, 1.0, -1.0]


def test_weighted_fusion_uses_the_given_weights():
    X, y, columns = two_experts()
    model = fit_fusion_model("weighted", X, y, columns, weights={"head": 1.0})
    assert model.weights() == {"head": 1.0, "seq": 0.0, "inverse": 0.0}
    with pytest.raises(ValueError, match="poids"):
        fit_fusion_model("weighted", X, y, columns)
    with pytest.raises(ValueError, match="absentes"):
        fit_fusion_model("weighted", X, y, columns, weights={"magic": 1.0})


def test_weight_grid_listens_most_to_the_best_expert():
    X, y, columns = two_experts()
    weights = fit_fusion_model("weight_grid", X, y, columns).weights()
    assert weights["head"] == max(weights.values())


def test_logistic_weights_tell_which_expert_matters():
    X, y, columns = two_experts()
    weights = fit_fusion_model("logistic", X, y, columns).weights()
    assert abs(sum(weights.values()) - 1) < 1e-9 and weights["head"] > weights["seq"]


def test_fusion_model_round_trips_through_json():
    X, y, columns = two_experts()
    for method in ("logistic", "rank_mean", "max"):
        model = fit_fusion_model(method, X, y, columns)
        again = FusionModel.from_dict(json.loads(json.dumps(model.to_dict())))
        assert np.allclose(again.decision(X), model.decision(X))


def test_missing_inputs_take_the_mean():
    X, y, columns = two_experts()
    model = fit_fusion_model("logistic", X, y, columns)
    holed = X.copy()
    holed[:, 1] = np.nan
    assert np.isfinite(model.decision(holed)).all()


def test_fusion_oof_uses_the_common_folds():
    X, y, columns = two_experts()
    groups = np.repeat(list("abcdef"), 50)
    assignment = {g: i % 3 for i, g in enumerate("abcdef")}
    oof = fusion_model_oof("mean", X, y, groups, columns, assignment=assignment)
    held = [sorted(set(groups[test])) for _, test in oof.folds]
    assert held == [["a", "d"], ["b", "e"], ["c", "f"]]
    assert np.isfinite(oof.values).all()


def test_three_or_more_inputs_are_accepted():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    X = np.column_stack([y + rng.normal(0, 1, 200) for _ in range(5)])
    model = fit_fusion_model("weight_grid", X, y, [f"e{i}" for i in range(5)], grid_step=0.25)
    assert len(model.weights()) == 5


def test_project_scores_takes_the_best_overlapping_window():
    target = pd.DataFrame(
        {"recording_id": ["r", "r", "q"], "offset_s": [0.0, 5.0, 0.0], "dur_s": [5.0, 5.0, 5.0]}
    )
    source = pd.DataFrame(
        {
            "recording_id": ["r"] * 4,
            "offset_s": [0.0, 1.5, 3.0, 6.0],
            "dur_s": [3.0] * 4,
            "score": [0.1, 0.9, 0.4, 0.3],
        }
    )
    out = project_scores(target, source)
    assert out[0] == 0.9 and out[1] == 0.3 and np.isnan(out[2])


# --- Emplacement du module séquentiel -----------------------------------------------------------


def test_positions_are_parsed_and_ordered():
    assert positions_from("downstream,upstream") == ["upstream", "downstream"]
    assert positions_from("none") == [] and positions_from(None) == []
    assert position_tag([]) == "aucun" and position_tag(["parallel"]) == "parallel"
    with pytest.raises(ValueError, match="inconnus"):
        positions_from(["sideways"])


def test_onset_counts_and_upstream_gate():
    windows = pd.DataFrame(
        {"recording_id": ["r", "r", "x"], "offset_s": [0.0, 3.0, 0.0], "dur_s": [3.0, 3.0, 3.0]}
    )
    onsets = {"r": np.array([0.2, 1.6, 2.2])}  # intervalles 1,4 (A. blanci) et 0,6
    counts = onset_counts(windows, onsets, (1.2, 1.906))
    assert counts.loc[0].tolist() == [3.0, 1.0] and counts.loc[1].tolist() == [0.0, 0.0]
    assert np.isnan(counts.loc[2, "notes"])
    cfg = load_config()
    cfg["sequential"]["upstream"]["gates"]["notes"]["enabled"] = True
    cfg["sequential"]["upstream"]["gates"]["rhythm"]["enabled"] = True
    assert upstream_pass(counts, cfg).tolist() == [True, False, True]


def test_unset_weights_share_what_is_left():
    cfg = load_config()
    cfg["fusion"]["weights"] = {"head": 0.7}
    options = fusion_options(cfg, ["head", "a", "b"])
    assert options["weights"] == pytest.approx({"head": 0.7, "a": 0.15, "b": 0.15})


# --- De bout en bout ------------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = {key: str(tmp_path / key) for key in cfg["paths"]}
    cfg["benchmark"] |= {"n_boot": 30, "negatives_per_positive": 6}
    cfg["head"] |= {"n_splits": 3, "C_grid": [1.0]}
    return cfg


def write_stock(con, cfg, eid, window_s, hop, rows, seed):
    """Stock d'un encodeur : fond propre au micro, direction commune pour A. blanci."""
    rng = np.random.default_rng(seed)
    params = {
        "sample_rate": 32000,
        "window_s": window_s,
        "hop_s": hop,
        "dim": 6,
        "has_tokens": False,
        "last_run": {},
    }
    con.execute(
        "INSERT INTO models (model_id, kind, name, version, params_json, created_at) "
        "VALUES (?, 'encoder', ?, '1', ?, ?)",
        (eid, eid.split("-")[0], json.dumps(params), utc_now()),
    )
    metas, embs = [], []
    for rid, mic, positive in rows:
        offsets = [round(hop * i, 2) for i in range(int((20 - window_s) / hop) + 1)]
        ids = [window_id_for(rid, o, window_s) for o in offsets]
        emb = rng.normal(0, 0.6, (len(ids), 6)) + (mic * 0.3)
        if positive:
            emb[:, 0] += 1.5
        metas.append(pd.DataFrame({"window_id": ids, "recording_id": rid, "offset_s": offsets}))
        embs.append(emb)
        con.executemany(  # comme `embed` : les fenêtres de la grille existent en base
            "INSERT OR IGNORE INTO windows (window_id, recording_id, offset_s, dur_s) "
            "VALUES (?, ?, ?, ?)",
            [(w, rid, o, window_s) for w, o in zip(ids, offsets, strict=True)],
        )
    EmbeddingStore(cfg["paths"]["embeddings"], eid).write(
        pd.concat(metas, ignore_index=True), np.concatenate(embs), "2026", "mataroni", "202602"
    )


@pytest.fixture
def corpus(tmp_path, cfg):
    con = connect(tmp_path / "db.sqlite")
    rows = []
    for m in range(4):
        for day in range(6):
            rel = f"2026/mataroni/M{m}/M{m}_d{day}.wav"
            rid = recording_id_for(rel)
            positive = day < 3
            con.execute(
                "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
                "duration_s, sample_rate, channels, qc_flags) VALUES (?, ?, '2026', 'mataroni', "
                "?, ?, 20.0, 32000, 1, '{}')",
                (rid, rel, f"M{m}", f"2026-02-{10 + day:02d}T13:00:00Z"),
            )
            if positive:
                for offset in (0.0, 6.0, 12.0):
                    wid = window_id_for(rid, offset)
                    con.execute(
                        "INSERT INTO windows (window_id, recording_id, offset_s, dur_s) "
                        "VALUES (?, ?, ?, 3.0)",
                        (wid, rid, offset),
                    )
                    con.execute(
                        "INSERT INTO labels (window_id, label, source, created_at) "
                        "VALUES (?, 'blanci_solo', 'import', ?)",
                        (wid, utc_now()),
                    )
                notes = np.arange(0.4, 20, 1.414)
            else:
                notes = np.array([5.0])
            store_onsets(con, rid, notes, 0)
            rows.append((rid, m, positive))
    con.commit()
    write_stock(con, cfg, "main-1", 3.0, 1.5, rows, seed=0)
    write_stock(con, cfg, "other-1", 5.0, 2.5, rows, seed=1)
    # logits de congénères sur la grille de 5 s de « other »
    meta, _ = EmbeddingStore(cfg["paths"]["embeddings"], "other-1").load()
    positive_rids = {rid for rid, _, p in rows if p}
    con.executemany(
        "INSERT OR IGNORE INTO windows (window_id, recording_id, offset_s, dur_s) "
        "VALUES (?, ?, ?, 5.0)",
        meta[["window_id", "recording_id", "offset_s"]].itertuples(index=False),
    )
    con.executemany(
        "INSERT INTO scores (window_id, model_id, score) VALUES (?, 'other-1:logit:A. baeo', ?)",
        [
            (w, 2.0 if r in positive_rids else -2.0)
            for w, r in zip(meta["window_id"], meta["recording_id"], strict=True)
        ],
    )
    con.commit()
    return con


def test_level1_holds_every_input_out_of_fold(corpus, cfg):
    level1 = build_level1(corpus, cfg, "main-1", ["head:other-1", "congeners:other-1"])
    assert np.isfinite(level1.head.values).all()
    assert level1.sources["head:other-1"].notna().all()
    assert level1.sources["congeners:other-1"].notna().all()
    X, columns = design_matrix(
        level1, ["parallel", "downstream"], cfg, ["head:other-1", "congeners:other-1"]
    )
    assert columns == [
        "head",
        "frac_ioi_blanci",
        "onset_rate_hz",
        "frac_windows",
        "longest_run",
        "head:other-1",
        "congeners:other-1",
    ]
    assert X.shape == (len(level1.y), 7)
    assert design_matrix(level1, [], cfg)[1] == ["head"]


def test_fusion_benchmark_compares_positions_and_methods(corpus, cfg):
    out = fusion_benchmark(
        corpus,
        cfg,
        "main-1",
        methods=["logistic", "mean", "max"],
        positions=[[], ["upstream"], ["parallel", "downstream"]],
        sources=["head:other-1"],
    )
    table = out["table"]
    assert set(table["position"]) == {"aucun", "upstream", "parallel+downstream"}
    assert len(table[table["level"] == "recording"]) == 3 * 3
    assert set(out["weights"]["method"]) == {"logistic", "mean", "max"}
    assert out["comparisons"]["diff"].notna().all()


def test_upstream_gate_stops_windows_without_notes(corpus, cfg):
    from blanci.sequential import GATED_SCORE
    from blanci.stacking import fused_oof

    level1 = build_level1(corpus, cfg, "main-1")
    gates = cfg["sequential"]["upstream"]["gates"]
    gates["notes"] = {"enabled": True, "min_count": 2}
    gates["rhythm"] = {"enabled": True, "min_count": 1}
    values, _ = fused_oof(level1, "logistic", ["upstream"], cfg)
    negatives_only = level1.counts["notes"] < 2
    assert (values[negatives_only.to_numpy()] == GATED_SCORE).all()


@pytest.mark.filterwarnings("ignore")
def test_production_fusion_with_a_congener_source_decides(corpus, cfg):
    """`blanci fusion` puis `score --fusion` avec une autre entrée et le module en aval seul."""
    from blanci.service import score_and_decide, train_and_register, train_fusion

    cfg["fusion"] |= {"method": "mean", "sources": ["congeners:other-1"]}
    cfg["sequential"]["position"] = ["downstream"]
    train_and_register(corpus, "main-1", cfg)
    out = train_fusion(corpus, "main-1", cfg)
    assert out["columns"] == ["head", "frac_windows", "longest_run", "congeners:other-1"]
    assert abs(sum(out["weights"].values()) - 1) < 1e-9
    result = score_and_decide(corpus, "main-1", cfg, fusion=True)
    assert result.threshold_id.startswith("main-1:fusion:") and len(result.decisions) == 24


def test_production_fusion_needs_something_to_fuse(corpus, cfg):
    from blanci.service import train_and_register, train_fusion

    cfg["sequential"]["position"] = []
    train_and_register(corpus, "main-1", cfg)
    with pytest.raises(ValueError, match="rien à fusionner"):
        train_fusion(corpus, "main-1", cfg)


# --- Ensemble de modèles (DECISIONS n° 96) ----------------------------------------------------


@pytest.mark.filterwarnings("ignore")
def test_ensemble_combines_sources_by_recording(corpus, cfg):
    from blanci.benchmark import benchmark_encoder
    from blanci.ensemble import recording_table, run_ensemble

    for eid in ("main-1", "other-1"):
        benchmark_encoder(corpus, eid, cfg["paths"]["embeddings"], cfg)
    sources = ["main-1/logistic", "other-1/logistic"]
    table = recording_table(cfg, sources)
    assert set(sources) <= set(table.columns) and table["fold"].ge(0).all()
    out = run_ensemble(corpus, cfg, sources, ["mean", "max", "logistic"])
    assert set(out["table"]["kind"]) == {"source", "ensemble"}
    assert len(out["comparisons"]) == 3
    from blanci.oof import list_sources

    assert any(s.startswith("ensemble:mean(") for s in list_sources(cfg)["source"])


def test_ensemble_refuses_sources_from_other_labels(corpus, cfg, tmp_path):
    from blanci.benchmark import benchmark_encoder
    from blanci.ensemble import run_ensemble

    benchmark_encoder(corpus, "main-1", cfg["paths"]["embeddings"], cfg)
    rid = corpus.execute("SELECT recording_id FROM recordings LIMIT 1").fetchone()[0]
    corpus.execute(
        "INSERT INTO windows (window_id, recording_id, offset_s, dur_s) VALUES (?, ?, 18.0, 3.0)",
        (window_id_for(rid, 18.0), rid),
    )
    corpus.execute(
        "INSERT INTO labels (window_id, label, source, created_at) VALUES (?, 'bird', 'active', ?)",
        (window_id_for(rid, 18.0), utc_now()),
    )
    corpus.commit()
    benchmark_encoder(corpus, "other-1", cfg["paths"]["embeddings"], cfg)
    with pytest.raises(ValueError, match="labels différents"):
        run_ensemble(corpus, cfg, ["main-1/logistic", "other-1/logistic"])


def test_concatenation_needs_the_same_grid(corpus, cfg):
    from blanci.ensemble import concat_training_set

    with pytest.raises(ValueError, match="grilles différentes"):
        concat_training_set(corpus, cfg, ["main-1", "other-1"])


def test_concatenation_of_two_same_grid_stocks(corpus, cfg):
    from blanci.ensemble import concat_benchmark, concat_training_set

    meta, emb = EmbeddingStore(cfg["paths"]["embeddings"], "main-1").load()
    rng = np.random.default_rng(3)
    params = {
        "sample_rate": 32000,
        "window_s": 3.0,
        "hop_s": 1.5,
        "dim": 4,
        "has_tokens": False,
        "last_run": {},
    }
    corpus.execute(
        "INSERT INTO models (model_id, kind, name, version, params_json, created_at) "
        "VALUES ('twin-1', 'encoder', 'twin', '1', ?, ?)",
        (json.dumps(params), utc_now()),
    )
    EmbeddingStore(cfg["paths"]["embeddings"], "twin-1").write(
        meta, rng.normal(0, 1, (len(meta), 4)), "2026", "mataroni", "202602"
    )
    data, X = concat_training_set(corpus, cfg, ["main-1", "twin-1"])
    assert X.shape == (len(data), 6 + 4)
    table = concat_benchmark(corpus, cfg, ["main-1", "twin-1"], ["logistic"])
    assert set(table["encoders"]) == {"main-1+twin-1"}


# --- Benchmark complet (DECISIONS n° 98) -----------------------------------------------------


def test_full_benchmark_ranks_every_source_on_common_recordings(corpus, cfg):
    from blanci.benchmark import benchmark_encoder
    from blanci.full_benchmark import external_source, run_full_benchmark

    for eid in ("main-1", "other-1"):
        benchmark_encoder(corpus, eid, cfg["paths"]["embeddings"], cfg)
    # Blancinet : une détection forte sur chaque enregistrement positif
    positives = [
        r[0]
        for r in corpus.execute(
            "SELECT DISTINCT w.recording_id FROM labels l JOIN windows w USING (window_id)"
        )
    ]
    for rid in positives:
        wid = window_id_for(rid, 6.0)
        corpus.execute(
            "INSERT INTO scores (window_id, model_id, score) VALUES (?, 'blancinet', 0.9)", (wid,)
        )
    corpus.commit()
    assert external_source(corpus, cfg, "blancinet") == "external/blancinet"
    assert external_source(corpus, cfg, "other-1:logit") == "external/other-1/congeners"
    out = run_full_benchmark(corpus, cfg)
    table = out["table"]
    assert {"main-1/logistic", "other-1/prototype", "external/blancinet"} <= set(table["source"])
    assert table["n_recordings"].nunique() == 1  # enregistrements communs
    assert table["up_to_date"].all()
    assert set(out["comparisons"]["reference"]) == {out["reference"]}
    assert table.loc[table["source"] == "main-1/logistic", "dim"].iloc[0] == 6
    assert len(out["by_site"])
