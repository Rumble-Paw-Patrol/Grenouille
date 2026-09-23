"""Négatifs suspects (DECISIONS n° 80) et commentaires accolés aux fenêtres annotées."""

import json

import numpy as np
import pandas as pd
import pytest

from blanci.config import load_config
from blanci.dataset import (
    current_labels,
    detected_blanci,
    suspect_count,
    suspect_negatives,
    training_set,
    usable_labels,
)
from blanci.db import connect, recording_id_for, window_id_for
from blanci.labels import comment_fields, import_detections
from blanci.service import append_label
from blanci.workbench import save_answer, suspect_candidates

CFG = load_config()


@pytest.fixture
def con(tmp_path):
    return connect(tmp_path / "blanci.sqlite")


def add_recording(con, name, mic="M1", start="2026-02-10T13:00:00Z"):
    rel = f"2026/mataroni/{mic}/{name}.wav"
    rid = recording_id_for(rel)
    con.execute(
        "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
        "duration_s, sample_rate, channels) VALUES (?, ?, '2026', 'mataroni', ?, ?, 120, 48000, 2)",
        (rid, rel, mic, start),
    )
    con.commit()
    return rid


def window(con, rid, offset, dur=3.0):
    wid = window_id_for(rid, offset, dur)
    con.execute(
        "INSERT OR IGNORE INTO windows (window_id, recording_id, offset_s, dur_s) "
        "VALUES (?, ?, ?, ?)",
        (wid, rid, offset, dur),
    )
    return wid


def label(con, rid, offset, value, comment=None, dur=3.0):
    conditions = {"comment": comment} if comment else None
    return append_label(con, window(con, rid, offset, dur), value, "import", conditions=conditions)


def detection(con, rid, offset, score):
    con.execute(
        "INSERT INTO scores (window_id, model_id, score) VALUES (?, 'blancinet', ?)",
        (window(con, rid, offset), score),
    )
    con.commit()


# --- Règle ----------------------------------------------------------------------------------


def test_negative_next_to_an_unheard_detection_is_suspect(con):
    rid = add_recording(con, "a")
    label(con, rid, 30.0, "bird")
    detection(con, rid, 33.0, 0.9)  # fenêtre voisine, jamais écoutée
    assert usable_labels(con)["suspect"].tolist() == [True]
    assert suspect_count(con) == {"negatives": 1, "suspect": 1}


@pytest.mark.parametrize(
    "offset, score, suspect",
    [
        (33.0, 0.3, False),  # score trop bas : Blancinet n'y croit pas
        (36.0, 0.9, False),  # deux fenêtres plus loin : hors du voisinage
        (27.0, 0.9, True),  # voisine avant
        (30.0, 0.9, False),  # la fenêtre elle-même : écoutée, c'est le négatif
    ],
)
def test_neighbourhood_and_score(con, offset, score, suspect):
    rid = add_recording(con, "a")
    label(con, rid, 30.0, "bird")
    detection(con, rid, offset, score)
    assert bool(usable_labels(con)["suspect"].iloc[0]) is suspect


def test_listened_neighbour_lifts_the_suspicion(con):
    rid = add_recording(con, "a")
    label(con, rid, 30.0, "bird")
    detection(con, rid, 33.0, 0.9)
    label(con, rid, 33.0, "background")  # voisin écouté : pas A. blanci
    labels = usable_labels(con)
    assert not labels["suspect"].any()


def test_blanci_heard_next_door_keeps_it_suspect(con):
    rid = add_recording(con, "a")
    label(con, rid, 30.0, "bird")
    detection(con, rid, 33.0, 0.9)
    label(con, rid, 33.0, "blanci")  # voisin écouté : c'est A. blanci
    labels = usable_labels(con).set_index("offset_s")
    assert labels.at[30.0, "suspect"] and not labels.at[33.0, "suspect"]


def test_whole_recording_listened_covers_its_detections(con):
    """Jeu gelé, audit : l'enregistrement entier a été écouté, ses détections aussi."""
    rid = add_recording(con, "a")
    label(con, rid, 0.0, "background", dur=120.0)
    detection(con, rid, 33.0, 0.9)
    labels = current_labels(con)
    assert detected_blanci(con, labels).empty
    assert not suspect_negatives(labels, detected_blanci(con, labels)).any()


def test_suspect_negative_leaves_training_and_paired_pool(con):
    pos = add_recording(con, "pos", start="2026-02-10T13:00:00Z")
    neg = add_recording(con, "neg", start="2026-02-11T13:00:00Z")
    label(con, pos, 30.0, "blanci")
    label(con, neg, 30.0, "bird")
    detection(con, neg, 33.0, 0.9)
    offsets = np.arange(0, 115, 2.5)
    grid = pd.DataFrame(
        [
            {"window_id": window_id_for(r, o, 5.0), "recording_id": r, "offset_s": o, "dur_s": 5.0}
            for r in (pos, neg)
            for o in offsets
        ]
    )
    data = training_set(con, grid, per_positive=100)
    kept = data[~data["presumed"]]
    assert set(kept["label"]) == {"blanci"}  # le négatif suspect n'est plus là
    presumed = data[data["presumed"]]
    # Aucun négatif présumé à moins de 3 s de la détection non écoutée (33–36 s).
    near = (presumed["offset_s"] < 39.0) & (presumed["offset_s"] + 5.0 > 30.0)
    assert len(presumed) > 0 and not near.any()


def test_suspect_queue_lists_the_unheard_neighbours(con):
    rid = add_recording(con, "a")
    label(con, rid, 30.0, "bird")
    detection(con, rid, 33.0, 0.9)
    detection(con, rid, 60.0, 0.9)  # loin de tout négatif : pas dans la file
    queue = suspect_candidates(con)
    assert queue["offset_s"].tolist() == [33.0]
    assert queue["reason"].iloc[0] == "voisin_negatif_suspect"
    assert queue["score"].iloc[0] == pytest.approx(0.9)


# --- Import des détections ------------------------------------------------------------------


def test_detections_are_scores_not_labels(con, tmp_path):
    rid = add_recording(con, "2LA03550_20260108_143000")
    table = tmp_path / "detections.csv"
    pd.DataFrame(
        {
            "file_s3_key": ["2353462-2la03550_20260108_143000.flac"] * 3 + ["inconnu.flac"],
            "start_time": [87, 90, 30, 0],
            "score": [0.94, 0.2, 0.8, 0.5],
            "vérification": [None, None, "True", None],
        }
    ).to_csv(table, index=False)
    report = import_detections(con, table, CFG)
    assert report.stored == 3 and report.not_found == 1 and report.unverified == 2
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 0
    scores = dict(con.execute("SELECT window_id, score FROM scores").fetchall())
    assert scores[window_id_for(rid, 87.0)] == pytest.approx(0.94)
    # Réimporter remplace, ne duplique pas.
    assert import_detections(con, table, CFG).stored == 3
    assert con.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 3


# --- Commentaires accolés -------------------------------------------------------------------


def test_comment_follows_the_window(con):
    rid = add_recording(con, "a")
    label(con, rid, 30.0, "bird", comment="Fourmilier tacheté, pluie légère")
    assert current_labels(con)["comment"].tolist() == ["Fourmilier tacheté, pluie légère"]
    grid = pd.DataFrame(
        {"window_id": ["g"], "recording_id": [rid], "offset_s": [29.0], "dur_s": [5.0]}
    )
    data = training_set(con, grid)
    assert data["comment"].tolist() == ["Fourmilier tacheté, pluie légère"]


def test_station_comment_is_read_like_an_import(con):
    rid = add_recording(con, "a")
    candidate = {"recording_id": rid, "offset_s": 12.0, "dur_s": 3.0, "reason": "lot1"}
    save_answer(con, candidate, "bird", "leonard", comment="Fourmilier tacheté sous la pluie")
    conditions = json.loads(con.execute("SELECT conditions FROM labels").fetchone()[0])
    assert conditions["comment"] == "Fourmilier tacheté sous la pluie"
    assert "rain" in conditions["tags"]
    assert conditions["co_occurring"] == ["Fourmilier tacheté"]
    # « pluie » écrit au poste pose le drapeau pluie de l'enregistrement.
    flags = json.loads(con.execute("SELECT qc_flags FROM recordings").fetchone()[0])
    assert flags["annotated"] == ["rain"]


def test_empty_comment_gives_no_field():
    assert comment_fields("  ") == {} and comment_fields(None) == {}
