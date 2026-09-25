"""Outil de sélection des candidats (DECISIONS n° 99)."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from blanci.dataset import current_labels
from blanci.selection import (
    SELECTION_METHODS,
    cluster_status,
    embedding_map,
    export_clips,
    import_clip_labels,
    label_cluster,
    map_selection,
    select_candidates,
    write_queue,
)
from blanci.service import append_label, score_and_decide, train_and_register
from blanci.workbench import CANDIDATE_COLUMNS, ensure_window
from tests.test_stacking import cfg, corpus  # noqa: F401  (fixtures partagées)

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def scored(corpus, cfg):  # noqa: F811
    train_and_register(corpus, "main-1", cfg)
    score_and_decide(corpus, "main-1", cfg)
    return corpus


def test_every_method_is_known():
    assert set(SELECTION_METHODS) >= {
        "active",
        "similarity",
        "coverage",
        "cluster",
        "audit",
        "negative_mining",
        "phenology",
        "suspects",
    }


def test_active_queue_listens_to_the_best_window_with_adjustable_mix(scored, cfg):  # noqa: F811
    queue = select_candidates(scored, cfg, "active", "main-1", n=10, mix=[0.2, 0.3, 0.5])
    assert list(queue.columns[: len(CANDIDATE_COLUMNS)]) == CANDIDATE_COLUMNS
    assert (queue["source"] == "active").all() and queue["reason"].str.startswith("active_").all()
    assert queue["reason"].value_counts().get("active_random", 0) == 5
    assert queue["path"].notna().all() and (queue["dur_s"] == 3.0).all()


def test_similarity_and_coverage_propose_unlabelled_windows(scored, cfg):  # noqa: F811
    labelled = set(current_labels(scored)["window_id"])
    sim = select_candidates(scored, cfg, "similarity", "main-1", n=15)
    assert len(sim) and (sim["source"] == "similarity").all()
    cover = select_candidates(scored, cfg, "coverage", "main-1", n=8, pool=500)
    assert len(cover) == 8 and (cover["source"] == "coverage").all()
    from blanci.db import window_id_for

    ids = {
        window_id_for(r, o, d)
        for r, o, d in cover[["recording_id", "offset_s", "dur_s"]].itertuples(index=False)
    }
    assert not ids & labelled


def test_coverage_goes_where_nothing_was_heard(scored, cfg):  # noqa: F811
    """Tous les positifs annotés sont dans le même coin de l'espace : la couverture choisit
    surtout des enregistrements sans positif."""
    cover = select_candidates(scored, cfg, "coverage", "main-1", n=10, pool=500)
    positives = set(current_labels(scored)["recording_id"])
    assert (~cover["recording_id"].isin(positives)).mean() >= 0.5


def test_negative_mining_both_modes(scored, cfg):  # noqa: F811
    unlikely = select_candidates(scored, cfg, "negative_mining", "main-1", n=5)
    assert len(unlikely) == 5 and (unlikely["reason"] == "mining_unlikely").all()
    with pytest.raises(ValueError, match="faux ami"):
        select_candidates(scored, cfg, "negative_mining", "main-1", n=5, mode="false_friends")
    queue = select_candidates(scored, cfg, "random", n=2, peak_hours=False)
    wid = ensure_window(scored, queue.loc[0, "recording_id"], 0.0, 3.0)
    append_label(scored, wid, "bird", "active", species="Fourmilier tacheté")
    friends = select_candidates(scored, cfg, "negative_mining", "main-1", n=5, mode="false_friends")
    assert len(friends) and (friends["source"] == "mining").all()


def test_phenology_template_fills_its_strata(scored, cfg):  # noqa: F811
    queue = select_candidates(scored, cfg, "phenology", n=10)
    assert (queue["source"] == "phenology").all()
    assert set(queue["reason"]) <= {
        "phenology_priority",
        "phenology_secondary",
        "phenology_outside",
    }
    whole = select_candidates(scored, cfg, "phenology", n=4, whole=True)
    assert (whole["offset_s"] == 0.0).all() and (whole["dur_s"] == 20.0).all()


def test_suspects_come_from_the_decisions(scored, cfg):  # noqa: F811
    scored.execute(
        "UPDATE decisions SET status = 'suspect' WHERE rowid IN "
        "(SELECT rowid FROM decisions LIMIT 3)"
    )
    scored.commit()
    expected = scored.execute(
        "SELECT COUNT(DISTINCT recording_id) FROM decisions WHERE status = 'suspect'"
    ).fetchone()[0]
    queue = select_candidates(scored, cfg, "suspects", "main-1")
    assert 3 <= len(queue) <= expected and (queue["source"] == "suspect").all()


def test_methods_needing_an_encoder_say_so(scored, cfg):  # noqa: F811
    with pytest.raises(ValueError, match="encodeur"):
        select_candidates(scored, cfg, "coverage")
    with pytest.raises(ValueError, match="inconnue"):
        select_candidates(scored, cfg, "magic")


def test_cluster_then_bulk_labelling_of_a_homogeneous_group(scored, cfg):  # noqa: F811
    from blanci.workbench import save_answer

    cfg["cluster"] |= {
        "min_cluster_size": 5,
        "min_samples": 2,
        "pca_components": 3,
        "c0_sample": 400,
    }
    queue = select_candidates(scored, cfg, "cluster", "main-1", n=3)
    assert "cluster" in queue.columns and (queue["source"] == "cluster").all()
    before = cluster_status(scored, cfg, "main-1", min_checked=3).set_index("cluster")
    sizes = queue["cluster"].value_counts()
    fresh = [g for g in sizes.index if sizes[g] >= 3 and before.loc[g, "n_listened"] == 0]
    assert fresh, "un groupe jamais écouté, avec trois candidats"
    group = int(fresh[0])
    with pytest.raises(ValueError, match="non homogène"):
        label_cluster(scored, cfg, "main-1", group, "background", min_checked=3)
    for c in queue[queue["cluster"] == group].to_dict("records"):
        save_answer(scored, c, "background", "expert")
    status = cluster_status(scored, cfg, "main-1", min_checked=3).set_index("cluster")
    assert status.loc[group, "homogeneous"] and status.loc[group, "label"] == "background"
    with pytest.raises(ValueError, match="entendu"):
        label_cluster(scored, cfg, "main-1", group, "bird", min_checked=3)
    written = label_cluster(scored, cfg, "main-1", group, min_checked=3, annotator="expert")
    assert written == status.loc[group, "n_windows"] - status.loc[group, "n_listened"]
    sources = pd.read_sql_query("SELECT source, COUNT(*) n FROM labels GROUP BY source", scored)
    assert sources.set_index("source").loc["bulk", "n"] == written


def test_embedding_map_and_selection_on_the_map(scored, cfg):  # noqa: F811
    pytest.importorskip("sklearn.manifold")
    points = embedding_map(scored, cfg, "main-1", n=300, method="pca")
    assert {"x", "y", "label", "window_id"} <= set(points.columns)
    assert "non écouté" in set(points["label"])
    chosen = points[points["label"] == "non écouté"].head(4)
    queue = map_selection(scored, chosen)
    assert len(queue) == 4 and (queue["reason"] == "map").all()


def test_yapat_bridge_exports_clips_and_reads_answers_back(scored, cfg, tmp_path):  # noqa: F811
    queue = select_candidates(scored, cfg, "random", n=3, peak_hours=False)
    raw = tmp_path / "raw"
    cfg["paths"]["raw"] = str(raw)
    for path in queue["path"]:
        (raw / path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(raw / path, np.zeros(32000 * 20, dtype=np.float32), 32000)
    manifest = export_clips(cfg, queue, "essai", context_s=1.0)
    clips = pd.read_csv(manifest)
    assert len(clips) == len(queue) >= 2 and (manifest.parent / clips.loc[0, "clip"]).exists()
    answers = tmp_path / "reponses.csv"
    pd.DataFrame(
        {
            "filename": clips["clip"],
            "label": (["Oiseau", "rien"] * 3)[: len(clips)],
            "commentaire": (["fourmilier", ""] * 3)[: len(clips)],
        }
    ).to_csv(answers, index=False)
    with pytest.raises(ValueError, match="correspondance"):
        import_clip_labels(scored, manifest, answers)
    before = len(current_labels(scored))
    written = import_clip_labels(
        scored, manifest, answers, {"Oiseau": "bird", "rien": "background"}
    )
    assert written == len(clips) and len(current_labels(scored)) == before + len(clips)


def test_write_queue_is_read_by_the_workbench(scored, cfg):  # noqa: F811
    from blanci.workbench import load_candidates

    queue = select_candidates(scored, cfg, "audit", n=2)
    path = write_queue(cfg, queue, "audit_essai")
    assert path.name == "candidats_audit_essai.csv"
    again = load_candidates(path, scored)
    assert len(again) == 2 and (again["source"] == "audit").all()


def test_gaps_in_the_model_scores_are_proposed(scored, cfg):  # noqa: F811
    """Faux négatifs suspects du modèle : une fenêtre sous le seuil entre deux au-dessus."""
    from blanci.service import load_head

    _, params = load_head(scored, "main-1", "default")
    rid = scored.execute("SELECT recording_id FROM windows LIMIT 1").fetchone()[0]
    rows = scored.execute(
        "SELECT window_id, offset_s FROM windows WHERE recording_id = ? AND dur_s = 3.0 "
        "ORDER BY offset_s",
        (rid,),
    ).fetchall()
    high, low = params["threshold"] + 5.0, params["threshold"] - 5.0
    pattern = [high, low, high] + [low] * (len(rows) - 3)
    scored.executemany(
        "UPDATE scores SET score = ? WHERE window_id = ? AND model_id = 'main-1:head:v1'",
        [(v, r[0]) for v, r in zip(pattern, rows, strict=True)],
    )
    scored.commit()
    queue = select_candidates(scored, cfg, "gaps", "main-1", n=50)
    mine = queue[queue["recording_id"] == rid]
    assert len(mine) == 1 and mine["offset_s"].iloc[0] == rows[1][1]
    assert (queue["source"] == "gap").all() and (queue["reason"] == "gap_scores").all()


def test_annotated_negatives_between_positives_are_listened_again(scored, cfg):  # noqa: F811
    rid = scored.execute(
        "SELECT w.recording_id FROM labels l JOIN windows w USING (window_id) LIMIT 1"
    ).fetchone()[0]  # positifs annotés à 0, 6 et 12 s
    wid = ensure_window(scored, rid, 3.0, 3.0)
    append_label(scored, wid, "bird", "active")
    queue = select_candidates(scored, cfg, "gaps", mode="labels")
    assert (queue["recording_id"] == rid).any() and (queue["reason"] == "gap_labels").all()
    assert 3.0 in set(queue.loc[queue["recording_id"] == rid, "offset_s"])
