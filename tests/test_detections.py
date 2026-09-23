"""Détections Blancinet (scores, pas labels), négatifs présumés tirés loin d'elles
(DECISIONS n° 80, 85), commentaires accolés aux fenêtres annotées (n° 82)."""

import json

import numpy as np
import pandas as pd
import pytest

from blanci.config import load_config
from blanci.dataset import current_labels, detected_blanci, training_set
from blanci.db import connect, recording_id_for, window_id_for
from blanci.labels import comment_fields, import_detections
from blanci.service import append_label
from blanci.workbench import save_answer

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


def grid_of(*recordings, dur=5.0):
    return pd.DataFrame(
        [
            {"window_id": window_id_for(r, o, dur), "recording_id": r, "offset_s": o, "dur_s": dur}
            for r in recordings
            for o in np.arange(0, 115, 2.5)
        ]
    )


# --- Négatifs annotés : négatifs, quel que soit le voisinage (n° 85) ------------------------


def test_annotated_negative_stays_negative_next_to_a_detection(con):
    rid = add_recording(con, "a")
    label(con, rid, 30.0, "bird")
    detection(con, rid, 33.0, 0.9)  # fenêtre voisine, jamais écoutée
    data = training_set(con, grid_of(rid))
    assert data["label"].tolist() == ["bird"] and data["y"].tolist() == [0]


# --- Où A. blanci est détecté ---------------------------------------------------------------


@pytest.mark.parametrize("score, counted", [(0.9, True), (0.3, False)])
def test_only_confident_detections_count(con, score, counted):
    rid = add_recording(con, "a")
    detection(con, rid, 33.0, score)
    assert (len(detected_blanci(con, current_labels(con))) == 1) is counted


def test_a_listened_detection_no_longer_counts(con):
    rid = add_recording(con, "a")
    detection(con, rid, 33.0, 0.9)
    label(con, rid, 33.0, "background")  # écoutée : pas A. blanci
    assert detected_blanci(con, current_labels(con)).empty


def test_whole_recording_listened_covers_its_detections(con):
    """Jeu gelé, audit : l'enregistrement entier a été écouté, ses détections aussi."""
    rid = add_recording(con, "a")
    label(con, rid, 0.0, "background", dur=120.0)
    detection(con, rid, 33.0, 0.9)
    assert detected_blanci(con, current_labels(con)).empty


# --- Négatifs présumés ----------------------------------------------------------------------


def test_presumed_negatives_are_drawn_away_from_unheard_detections(con):
    pos = add_recording(con, "pos", start="2026-02-10T13:00:00Z")
    other = add_recording(con, "autre", start="2026-02-11T13:00:00Z")
    label(con, pos, 30.0, "blanci")
    detection(con, other, 33.0, 0.9)
    data = training_set(con, grid_of(pos, other), per_positive=100)
    presumed = data[data["presumed"]]
    # Aucun négatif présumé à moins de 3 s de la détection non écoutée (33–36 s).
    near = (presumed["offset_s"] < 39.0) & (presumed["offset_s"] + 5.0 > 30.0)
    assert len(presumed) > 0 and not near.any()


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
