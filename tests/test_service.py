import json

import numpy as np
import pandas as pd
import pytest

from blanci.config import load_config
from blanci.db import connect, next_version, recording_id_for, register_model, utc_now
from blanci.head import Head
from blanci.service import (
    append_label,
    evaluate_holdout,
    head_id,
    load_head,
    make_queue,
    ranked_points,
    score_and_decide,
    similarity_search,
    train_and_register,
)
from blanci.store import EmbeddingStore
from tests.test_benchmark import build_recordings, write_embeddings

DIM = 16


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = {key: str(tmp_path / key) for key in cfg["paths"]}
    cfg["paths"]["db"] = str(tmp_path / "db" / "blanci.sqlite")
    cfg["benchmark"]["n_boot"] = 30
    cfg["benchmark"]["negatives_per_positive"] = 4
    cfg["head"]["n_splits"] = 3
    cfg["head"]["C_grid"] = [1.0]
    cfg["active"]["batch_recordings"] = 6
    return cfg


@pytest.fixture
def con(cfg):
    return connect(cfg["paths"]["db"])


@pytest.fixture
def trained(con, tmp_path, cfg):
    """Corpus séparable + tête entraînée : le point de départ de score, queue et search."""
    layout = build_recordings(con)
    write_embeddings(con, tmp_path, "good-1", layout, separation=4.0, seed=0)
    return layout, train_and_register(con, "good-1", cfg)


# --- Registre des modèles ---------------------------------------------------------------------


def test_next_version_counts_up(con):
    assert next_version(con, "head", "good-1") == "v1"
    register_model(con, head_id("good-1", "v1"), "head", "good-1", "v1", {})
    assert next_version(con, "head", "good-1") == "v2"


def test_next_version_is_per_encoder(con):
    register_model(con, head_id("good-1", "v1"), "head", "good-1", "v1", {})
    assert next_version(con, "head", "autre-1") == "v1"


# --- Entraînement -----------------------------------------------------------------------------


def test_train_registers_head_and_threshold(con, tmp_path, cfg, trained):
    _, result = trained
    assert result.version == "v1"
    assert (result.directory / "manifest.json").exists()
    assert (result.directory / "weights.npz").exists()
    kinds = dict(con.execute("SELECT kind, COUNT(*) FROM models GROUP BY kind").fetchall())
    assert kinds["head"] == 1 and kinds["threshold"] == 1 and kinds["encoder"] == 1


def test_train_threshold_comes_from_out_of_fold_scores(con, cfg, trained):
    """Un seuil calibré sur des scores en-pli serait optimiste (§6)."""
    _, result = trained
    assert np.isfinite(result.threshold)
    assert result.min_precision == cfg["benchmark"]["precisions"][0]
    assert 0.0 <= result.recall_at_threshold <= 1.0


def test_train_records_metrics_and_provenance(con, trained):
    _, result = trained
    assert result.metrics["level"] == "recording"
    assert result.metrics["n_pos"] > 0 and result.metrics["n_neg"] > 0
    params = json.loads(
        con.execute(
            "SELECT params_json FROM models WHERE model_id = ?", (result.model_id,)
        ).fetchone()["params_json"]
    )
    assert params["encoder_id"] == "good-1" and params["threshold_id"] == result.threshold_id
    assert "trained_at" in params


def test_train_separable_corpus_scores_well(con, trained):
    _, result = trained
    assert result.metrics["ap"] > 0.9


def test_second_training_creates_a_new_version(con, tmp_path, cfg, trained):
    second = train_and_register(con, "good-1", cfg)
    assert second.version == "v2"
    assert second.directory != trained[1].directory


def test_train_rejects_an_unknown_encoder(con, cfg):
    with pytest.raises(ValueError, match="encodeur inconnu"):
        train_and_register(con, "jamais-1", cfg)


def test_train_rejects_a_single_class_corpus(con, tmp_path, cfg):
    cfg["benchmark"]["negatives_per_positive"] = 0
    write_embeddings(con, tmp_path, "good-1", build_recordings(con), separation=3.0)
    with pytest.raises(ValueError, match="une seule classe"):
        train_and_register(con, "good-1", cfg)


# --- Chargement de la tête --------------------------------------------------------------------


def test_load_head_by_version(con, cfg, trained):
    head, params = load_head(con, "good-1", "v1")
    assert isinstance(head, Head)
    assert params["version"] == "v1"


def test_load_head_latest_takes_the_newest(con, cfg, trained):
    train_and_register(con, "good-1", cfg)
    _, params = load_head(con, "good-1", "latest")
    assert params["version"] == "v2"


def test_load_head_without_training_is_explicit(con):
    with pytest.raises(ValueError, match="aucune tête entraînée"):
        load_head(con, "good-1", "latest")


# --- Score et décisions -----------------------------------------------------------------------


def test_score_fills_scores_and_decisions(con, cfg, trained):
    result = score_and_decide(con, "good-1", cfg)
    n_windows = con.execute("SELECT COUNT(*) FROM windows").fetchone()[0]
    assert result.windows == n_windows
    assert con.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == n_windows
    assert con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == len(result.decisions)


def test_decisions_are_stamped_with_the_model(con, cfg, trained):
    result = score_and_decide(con, "good-1", cfg)
    row = con.execute("SELECT * FROM decisions LIMIT 1").fetchone()
    assert row["encoder_id"] == "good-1"
    assert row["head_version"] == result.version
    assert row["threshold_id"] == trained[1].threshold_id


def test_scoring_twice_replaces_decisions_not_duplicates(con, cfg, trained):
    """Les décisions sont reproductibles : elles se remplacent, contrairement aux labels."""
    first = score_and_decide(con, "good-1", cfg)
    score_and_decide(con, "good-1", cfg)
    assert con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == len(first.decisions)


def test_score_finds_the_positive_recordings(con, cfg, trained):
    layout, _ = trained
    result = score_and_decide(con, "good-1", cfg)
    expected = {rid for rid, _, _, positive in layout if positive}
    flagged = set(
        result.decisions.loc[
            result.decisions["status"].isin(["positive", "suspect"]), "recording_id"
        ]
    )
    assert expected <= flagged


def test_score_without_a_head_is_explicit(con, cfg, tmp_path):
    write_embeddings(con, tmp_path, "good-1", build_recordings(con), separation=3.0)
    with pytest.raises(ValueError, match="aucune tête entraînée"):
        score_and_decide(con, "good-1", cfg)


# --- Classement des points --------------------------------------------------------------------


def test_ranked_points_reads_the_stored_decisions(con, cfg, trained):
    score_and_decide(con, "good-1", cfg)
    points = ranked_points(con, "good-1")
    assert len(points) == 4  # quatre micros
    assert {"site", "mic_id", "status", "n_positive"} <= set(points.columns)


def test_ranked_points_without_decisions_is_explicit(con, cfg, trained):
    with pytest.raises(ValueError, match="aucune décision"):
        ranked_points(con, "good-1")


# --- File de vérification ---------------------------------------------------------------------


def test_make_queue_excludes_already_labelled_recordings(con, cfg, trained):
    score_and_decide(con, "good-1", cfg)
    rows = make_queue(con, "good-1", cfg, n=6)
    labelled = {
        r["recording_id"]
        for r in con.execute(
            "SELECT DISTINCT w.recording_id FROM labels l JOIN windows w USING (window_id)"
        )
    }
    assert not set(rows["recording_id"]) & labelled


def test_make_queue_reports_reasons_and_context(con, cfg, trained):
    score_and_decide(con, "good-1", cfg)
    rows = make_queue(con, "good-1", cfg, n=6)
    assert set(rows["reason"]) <= {"uncertain", "top", "random"}
    assert rows["path"].notna().all() and rows["site"].notna().all()
    assert rows["hour"].notna().all()


def test_make_queue_defaults_to_the_configured_batch(con, cfg, trained):
    score_and_decide(con, "good-1", cfg)
    assert len(make_queue(con, "good-1", cfg)) == cfg["active"]["batch_recordings"]


def test_make_queue_without_scores_is_explicit(con, cfg, trained):
    with pytest.raises(ValueError, match="aucun score"):
        make_queue(con, "good-1", cfg, n=6)


# --- Annotation -------------------------------------------------------------------------------


def a_window(con):
    return con.execute("SELECT window_id FROM windows LIMIT 1").fetchone()["window_id"]


def test_append_label_writes_a_row(con, trained):
    window_id = a_window(con)
    before = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    label_id = append_label(con, window_id, "bird", "active", quality="B", species="Manakin")
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == before + 1
    row = con.execute("SELECT * FROM labels WHERE label_id = ?", (label_id,)).fetchone()
    assert row["label"] == "bird" and row["quality"] == "B" and row["species"] == "Manakin"


def test_append_label_stores_conditions_as_json(con, trained):
    label_id = append_label(
        con, a_window(con), "blanci_solo", "audit", conditions={"pluie": True, "tags": ["lointain"]}
    )
    row = con.execute("SELECT conditions FROM labels WHERE label_id = ?", (label_id,)).fetchone()
    assert json.loads(row["conditions"])["tags"] == ["lointain"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"label": "grenouille_verte", "source": "active"}, "label inconnu"),
        ({"label": "bird", "source": "devinette"}, "source inconnue"),
        ({"label": "bird", "source": "active", "quality": "Z"}, "qualité inconnue"),
    ],
)
def test_append_label_validates_its_vocabulary(con, trained, kwargs, message):
    with pytest.raises(ValueError, match=message):
        append_label(con, a_window(con), **kwargs)


def test_append_label_rejects_an_unknown_window(con, trained):
    with pytest.raises(ValueError, match="fenêtre inconnue"):
        append_label(con, "inexistante:0.00", "bird", "active")


def test_a_correction_is_a_new_row(con, trained):
    """Les labels ne se modifient jamais : la correction est une ligne plus récente (§13.7)."""
    window_id = a_window(con)
    append_label(con, window_id, "bird", "active")
    append_label(con, window_id, "blanci_solo", "audit")
    rows = con.execute(
        "SELECT label FROM labels WHERE window_id = ? ORDER BY label_id", (window_id,)
    ).fetchall()
    assert [r["label"] for r in rows][-2:] == ["bird", "blanci_solo"]


# --- Recherche par similarité -----------------------------------------------------------------


def test_similarity_search_ranks_positives_first(con, cfg, trained):
    found = similarity_search(con, "good-1", cfg, k=20)
    assert len(found) == 20
    assert found["score"].is_monotonic_decreasing
    assert found["path"].notna().all()


def test_similarity_search_can_drop_paired_negatives(con, cfg, trained):
    with_negatives = similarity_search(con, "good-1", cfg, k=10)
    without = similarity_search(con, "good-1", cfg, k=10, use_paired_negatives=False)
    assert (without["sim_neg"] == 0).all()
    assert (with_negatives["sim_neg"] != 0).any()


def test_similarity_search_can_target_a_site(con, cfg, trained):
    found = similarity_search(con, "good-1", cfg, k=10, filters={"site": "mataroni"})
    assert (found["site"] == "mataroni").all()


# --- Évaluation sur sites tenus à l'écart -----------------------------------------------------


def multi_site_corpus(con, tmp_path, encoder_id="good-1", separation=4.0, seed=0):
    """Mataroni (entraînement) + Trésor (tenu à l'écart), avec des positifs des deux côtés."""
    rng = np.random.default_rng(seed)
    from tests.test_benchmark import register_encoder

    register_encoder(con, encoder_id)
    direction = np.zeros(DIM, dtype=np.float32)
    direction[0] = separation
    metas, embs = [], []
    for site in ("mataroni", "tresor"):
        for mic in range(3):
            for day in range(4):
                rel = f"2026/{site}/{site[0].upper()}{mic}/{site}_{mic}_d{day}.wav"
                rid = recording_id_for(rel)
                con.execute(
                    "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, "
                    "start_utc, duration_s, sample_rate, channels, qc_flags) "
                    "VALUES (?, ?, '2026', ?, ?, ?, 120.0, 32000, 1, '{}')",
                    (
                        rid,
                        rel,
                        site,
                        f"{site[0].upper()}{mic}",
                        f"2026-02-{10 + day:02d}T13:00:00Z",
                    ),
                )
                offsets = [round(i * 1.5, 2) for i in range(8)]
                ids = [f"{rid}:{o:.2f}" for o in offsets]
                con.executemany(
                    "INSERT INTO windows (window_id, recording_id, offset_s, dur_s) "
                    "VALUES (?, ?, ?, 3.0)",
                    [(w, rid, o) for w, o in zip(ids, offsets, strict=True)],
                )
                emb = rng.normal(0, 0.3, (8, DIM)).astype(np.float32)
                if day == 0:
                    emb += direction
                    con.executemany(
                        "INSERT INTO labels (window_id, label, source, created_at) "
                        "VALUES (?, 'blanci_solo', 'import', ?)",
                        [(w, utc_now()) for w in ids],
                    )
                metas.append(
                    pd.DataFrame({"window_id": ids, "recording_id": rid, "offset_s": offsets})
                )
                embs.append(emb)
    con.commit()
    EmbeddingStore(tmp_path / "embeddings", encoder_id).write(
        pd.concat(metas, ignore_index=True), np.concatenate(embs), "2026", "tous", "202602"
    )


def test_evaluate_without_holdout_uses_grouped_folds(con, cfg, trained):
    metrics = evaluate_holdout(con, "good-1", cfg)
    assert metrics["protocol"] == "plis groupés par micro"
    assert metrics["holdout"] == []
    assert metrics["ap"] > 0.9


def test_evaluate_holdout_trains_elsewhere(con, cfg, tmp_path):
    """Niveau 2 du §6 : entraîner sur Mataroni, mesurer sur Trésor."""
    multi_site_corpus(con, tmp_path)
    metrics = evaluate_holdout(con, "good-1", cfg, ["tresor"])
    assert metrics["holdout"] == ["tresor"]
    assert "mataroni" not in metrics["protocol"]
    assert metrics["ap"] > 0.8


def test_evaluate_holdout_is_case_insensitive(con, cfg, tmp_path):
    multi_site_corpus(con, tmp_path)
    assert evaluate_holdout(con, "good-1", cfg, ["TRESOR"])["holdout"] == ["tresor"]


def test_evaluate_holdout_window_level(con, cfg, tmp_path):
    multi_site_corpus(con, tmp_path)
    window = evaluate_holdout(con, "good-1", cfg, ["tresor"], level="window")
    recording = evaluate_holdout(con, "good-1", cfg, ["tresor"], level="recording")
    assert window["level"] == "window" and recording["level"] == "recording"
    assert window["n_pos"] > recording["n_pos"]


def test_evaluate_holdout_rejects_an_unknown_site(con, cfg, tmp_path):
    multi_site_corpus(con, tmp_path)
    with pytest.raises(ValueError, match="aucune fenêtre"):
        evaluate_holdout(con, "good-1", cfg, ["molokoi"])
