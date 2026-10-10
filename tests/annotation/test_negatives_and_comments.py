"""Négatifs annotés négatifs quel que soit le voisinage d'un score (DECISIONS n° 85, 87),
commentaires accolés aux fenêtres annotées (n° 82)."""

import json

import numpy as np
import pandas as pd
import pytest

from blanci.annotation.workbench import save_answer
from blanci.core.config import load_config
from blanci.core.db import connect, recording_id_for, window_id_for
from blanci.inputs.dataset import current_labels, training_set
from blanci.inputs.labels import comment_fields
from blanci.service import append_label

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
    return append_label(con, window(con, rid, offset, dur), value, "random", conditions=conditions)


def detection(con, rid, offset, score):
    con.execute(
        "INSERT INTO scores (window_id, model_id, score) VALUES (?, 'externe', ?)",
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


def test_station_comment_is_parsed(con):
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
