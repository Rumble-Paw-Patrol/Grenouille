"""Drapeaux posés à l'écoute, contrôle audio pendant `embed`, règle d'exclusion (DECISIONS 79)."""

import json

import numpy as np
import pytest
import soundfile as sf

from blanci.core.config import load_config
from blanci.core.db import connect, recording_id_for, window_id_for
from blanci.embedding.embed import embed_recordings, select_recordings
from blanci.inputs.qc import (
    annotation_flags,
    apply_annotation_flags,
    apply_audio_flags,
    flag_raised,
    is_excluded,
)
from blanci.service import append_label
from tests.conftest import write_wav
from tests.embedding.test_embed import SR, FakeEncoder

QC = load_config()["qc"]


@pytest.fixture
def workspace(tmp_path):
    return connect(tmp_path / "blanci.sqlite"), tmp_path / "raw", tmp_path / "embeddings"


def add(con, raw, name, silent=False, qc=None):
    rel = f"2026/mataroni/M1/{name}.wav"
    if silent:
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(raw / rel, np.zeros(SR * 6, dtype=np.float32), SR, subtype="FLOAT")
    else:
        write_wav(raw / rel, sr=SR, duration_s=6.0)
    rid = recording_id_for(rel)
    con.execute(
        "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
        "duration_s, sample_rate, channels, qc_flags) "
        "VALUES (?, ?, '2026', 'mataroni', 'M1', '2026-02-10T13:00:00Z', 6.0, ?, 1, ?)",
        (rid, rel, SR, json.dumps(qc) if qc is not None else None),
    )
    con.commit()
    return rid


def label(con, rid, value, offset=0.0, conditions=None):
    wid = window_id_for(rid, offset)
    con.execute(
        "INSERT OR IGNORE INTO windows (window_id, recording_id, offset_s, dur_s) "
        "VALUES (?, ?, ?, 3.0)",
        (wid, rid, offset),
    )
    append_label(con, wid, value, "similarity", conditions=conditions)
    return wid


def flags(con, rid):
    return json.loads(
        con.execute("SELECT qc_flags FROM recordings WHERE recording_id = ?", (rid,)).fetchone()[0]
        or "{}"
    )


# --- Règle d'exclusion ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "qc, excluded",
    [
        ({"silent": True}, True),
        ({"in_bag": True}, True),
        ({"duration_off": True}, True),
        ({"off_campaign": True}, True),
        ({"annotated": ["in_bag"]}, True),
        ({"rain": True, "saturation": True}, False),  # remarques, pas exclusions
        ({"annotated": ["rain"]}, False),
        ({"annotated": []}, False),
        (None, False),
    ],
)
def test_only_silent_in_bag_and_inventory_flags_exclude(qc, excluded):
    assert is_excluded(json.dumps(qc) if qc is not None else None) is excluded


def test_flag_raised_reads_both_computed_and_heard():
    assert flag_raised({"rain": True}, "rain")
    assert flag_raised({"rain": False, "annotated": ["rain"]}, "rain")
    assert not flag_raised({"annotated": ["in_bag"]}, "rain")


# --- Drapeaux posés à l'écoute --------------------------------------------------------------


def test_every_annotated_recording_is_marked(workspace):
    con, raw, _ = workspace
    bird, bag, rain, untouched = (add(con, raw, n) for n in ("bird", "bag", "rain", "none"))
    label(con, bird, "bird")
    label(con, bag, "artefact_in_bag")
    label(con, rain, "blanci", conditions={"tags": ["rain"], "comment": "pluie"})
    assert annotation_flags(con) == {bird: [], bag: ["in_bag"], rain: ["rain"]}
    assert flags(con, bird) == {"annotated": []}
    assert flags(con, bag)["annotated"] == ["in_bag"]
    assert "annotated" not in flags(con, untouched)


def test_in_bag_heard_excludes_rain_heard_does_not(workspace):
    con, raw, _ = workspace
    bag, rain = add(con, raw, "bag"), add(con, raw, "rain")
    label(con, bag, "artefact_in_bag")
    label(con, rain, "rain")
    selected = set(select_recordings(con)["recording_id"])
    assert rain in selected and bag not in selected


def test_corrected_label_withdraws_the_flag(workspace):
    con, raw, _ = workspace
    rid = add(con, raw, "bag")
    wid = label(con, rid, "artefact_in_bag")
    append_label(con, wid, "background", "audit")
    assert flags(con, rid)["annotated"] == []
    assert rid in set(select_recordings(con)["recording_id"])


def test_recording_with_blanci_heard_is_never_excluded(workspace):
    con, raw, _ = workspace
    rid = add(con, raw, "chant", qc={"in_bag": True})
    label(con, rid, "blanci")
    assert rid in set(select_recordings(con)["recording_id"])


def test_recompute_keeps_other_flags(workspace):
    con, raw, _ = workspace
    rid = add(con, raw, "bag", qc={"duration_off": False, "rain": True})
    label(con, rid, "artefact_in_bag")
    counts = apply_annotation_flags(con)
    assert counts == {"annotated": 1, "in_bag": 1, "rain": 0}
    assert flags(con, rid) == {"duration_off": False, "rain": True, "annotated": ["in_bag"]}


# --- Contrôle audio -------------------------------------------------------------------------


def test_embed_checks_audio_and_skips_silent_recordings(workspace):
    con, raw, store = workspace
    loud, silent = add(con, raw, "loud"), add(con, raw, "silent", silent=True)
    report = embed_recordings(
        con, FakeEncoder(), select_recordings(con), raw, store, qc_thresholds=QC
    )
    assert report.qc_checked == 2 and report.qc_excluded == 1 and report.recordings == 1
    assert flags(con, silent)["silent"] is True
    assert flags(con, loud)["silent"] is False and "indices" in flags(con, loud)
    # Le passage suivant (autre encodeur) ne le sélectionne plus, et ne recontrôle rien.
    assert silent not in set(select_recordings(con)["recording_id"])
    again = embed_recordings(
        con, FakeEncoder(), select_recordings(con), raw, store, qc_thresholds=QC
    )
    assert again.qc_checked == 0 and again.skipped == 1


def test_embed_encodes_a_silent_recording_where_blanci_was_heard(workspace):
    con, raw, store = workspace
    rid = add(con, raw, "silent", silent=True)
    label(con, rid, "blanci")
    report = embed_recordings(
        con, FakeEncoder(), select_recordings(con), raw, store, qc_thresholds=QC
    )
    assert report.qc_excluded == 0 and report.recordings == 1
    assert flags(con, rid)["silent"] is True  # la remarque reste


def test_embed_without_qc_touches_no_flag(workspace):
    con, raw, store = workspace
    rid = add(con, raw, "silent", silent=True)
    embed_recordings(con, FakeEncoder(), select_recordings(con), raw, store)
    assert flags(con, rid) == {}


def test_new_threshold_is_applied_without_reading_audio(workspace):
    con, raw, store = workspace
    rid = add(con, raw, "loud")
    embed_recordings(con, FakeEncoder(), select_recordings(con), raw, store, qc_thresholds=QC)
    (raw / "2026/mataroni/M1/loud.wav").unlink()  # plus d'audio : seuls les indices servent
    assert flags(con, rid)["in_bag"] is False
    counts = apply_audio_flags(con, QC | {"in_bag_hf_ratio": 2.0, "in_bag_min_run": 1})
    assert counts["in_bag"] == 1 and flags(con, rid)["in_bag"] is True


def _series(con, mic, minutes, ratios, dataset="2026"):
    """Une série d'enregistrements d'un micro, aux minutes données, de ratio hf donné."""
    ids = []
    for i, (minute, hf) in enumerate(zip(minutes, ratios, strict=True)):
        rid = f"{dataset}{mic}{i}"
        qc = {"indices": {"rms_dbfs": -30.0, "peak": 0.5, "clip_fraction": 0.0,
                          "hf_ratio": hf, "flatness_1_10k": 0.3}}
        con.execute(
            "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
            "duration_s, sample_rate, channels, qc_flags) "
            "VALUES (?, ?, ?, 'S', ?, ?, 120.0, 48000, 1, ?)",
            (rid, f"{rid}.wav", dataset, mic,
             f"2026-02-10T{minute // 60:02d}:{minute % 60:02d}:00Z", json.dumps(qc)),
        )
        ids.append(rid)
    con.commit()
    return ids


def test_in_bag_needs_a_run_of_low_ratio_recordings_of_the_same_mic(workspace):
    con = workspace[0]
    low, high = 0.005, 0.3
    steps = [0, 30, 60, 90, 120, 150, 180, 210]
    run = _series(con, "A", steps, [high, low, low, low, low, high, low, high])
    isolated = _series(con, "B", steps, [high, high, low, high, high, high, high, high])
    counts = apply_audio_flags(con, QC)
    assert counts["in_bag"] == 4
    assert [flags(con, r)["in_bag"] for r in run] == [False, True, True, True, True] + [False] * 3
    assert not any(flags(con, r)["in_bag"] for r in isolated)  # 0,005 isolé : pas un sac


def test_in_bag_run_is_broken_by_a_gap_and_stops_at_the_threshold(workspace):
    con = workspace[0]
    gap = _series(con, "A", [0, 30, 60, 300, 330, 360], [0.005] * 6)  # 3 + 3, trou de 4 h
    near = _series(con, "B", [0, 30, 60, 90], [0.06] * 4)  # sous 0,2 mais pas sous 0,05
    assert apply_audio_flags(con, QC)["in_bag"] == 0
    assert not any(flags(con, r)["in_bag"] for r in gap + near)
    # Le seuil de durée se règle sans relire l'audio.
    assert apply_audio_flags(con, QC | {"in_bag_min_run": 3})["in_bag"] == 6


def test_in_bag_runs_do_not_mix_micros(workspace):
    con = workspace[0]
    _series(con, "A", [0, 30], [0.005] * 2)
    _series(con, "B", [60, 90], [0.005] * 2)  # 4 candidats au total, mais 2 par micro
    assert apply_audio_flags(con, QC)["in_bag"] == 0
