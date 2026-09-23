"""Poste d'annotation (§5) : files de candidats, écoute, réponses en ajout seul."""

import io
import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from blanci.db import connect, window_id_for
from blanci.ingest import ingest
from blanci.workbench import (
    ANSWERS,
    CANDIDATE_COLUMNS,
    blancinet_candidates,
    clip_spectrogram,
    ensure_window,
    load_candidates,
    local_time,
    progress,
    random_candidates,
    read_clip,
    save_answer,
    wav_bytes,
)

SR = 16_000
DURATION_S = 12.0
SITES = {"CDR": ["C1", "C2", "C3"], "Patawa": ["P1", "P2"]}


@pytest.fixture
def corpus(tmp_path, cfg):
    """2 sites, 5 micros, 3 jours à 7 h (heure de pic) et 1 à 12 h ; stéréo, canal 1 = ×4."""
    raw = tmp_path / "raw"
    cfg["paths"]["raw"] = str(raw)
    cfg["qc"]["expected_duration_s"] = DURATION_S
    rng = np.random.default_rng(0)
    for site, mics in SITES.items():
        for mic in mics:
            for day, hour in ((10, 7), (11, 7), (12, 7), (13, 12)):
                x = rng.normal(0, 0.02, int(SR * DURATION_S)).astype(np.float32)
                path = raw / "2026" / site / mic / f"{mic}_202602{day}_{hour:02d}0000.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.stack([x, 4 * x], axis=1), SR, subtype="PCM_16")
    con = connect(tmp_path / "db.sqlite")
    ingest(con, raw, "2026", cfg, run_qc=False, hash_file=False)
    return con, cfg, raw


def blancinet_export(tmp_path, rows):
    """Export au format Blancinet : clé S3, station, début, score, vérification."""
    df = pd.DataFrame(
        rows, columns=["file_s3_key", "station", "start_time", "score", "vérification"]
    )
    df.insert(2, "label_name", "Anomaloglossus blanci")
    path = tmp_path / "export.xlsx"
    df.to_excel(path, index=False)
    return path


def all_detections(verified=()):
    """Une détection par fenêtre de 3 s de chaque enregistrement, scores étalés sur [0,1]."""
    rows, k = [], 0
    for site, mics in SITES.items():
        for mic in mics:
            for day, hour in ((10, 7), (11, 7), (12, 7), (13, 12)):
                for offset in (0, 3, 6, 9):
                    name = f"{1000 + k}-{mic.lower()}_202602{day}_{hour:02d}0000.flac"
                    verdict = True if (mic, day, offset) in verified else None
                    rows.append((name, site, offset, round((k % 10) / 10 + 0.05, 2), verdict))
                    k += 1
    return rows


# --- Files de candidats ---------------------------------------------------------------------


def test_blancinet_candidates_are_spread_over_sites_scores_and_mics(corpus, tmp_path):
    con, cfg, _ = corpus
    export = blancinet_export(tmp_path, all_detections())
    queue = blancinet_candidates(con, export, cfg, per_site=6)
    assert list(queue.columns) == CANDIDATE_COLUMNS
    assert queue.groupby("site").size().to_dict() == {"CDR": 6, "Patawa": 6}
    # Une seule fenêtre par enregistrement ; les trois tranches de score représentées.
    assert not queue.duplicated(["recording_id"]).any()
    for _, part in queue.groupby("site"):
        assert part["reason"].nunique() == 3
    assert queue.groupby("site")["mic_id"].nunique().to_dict() == {"CDR": 3, "Patawa": 2}
    assert (queue["source"] == "active").all() and (queue["dur_s"] == 3.0).all()


def test_blancinet_candidates_skip_verified_rows_and_labelled_windows(corpus, tmp_path):
    con, cfg, _ = corpus
    # La vérification remplie pour M1 : ce n'est plus un candidat.
    verified = {("C1", day, off) for day in (10, 11, 12, 13) for off in (0, 3, 6, 9)}
    export = blancinet_export(tmp_path, all_detections(verified))
    queue = blancinet_candidates(con, export, cfg, per_site=50)
    assert "C1" not in set(queue["mic_id"])

    # Une fenêtre déjà étiquetée (par le poste) ne revient pas.
    first = queue.iloc[0].to_dict()
    save_answer(con, first, "background", "léonard")
    again = blancinet_candidates(con, export, cfg, per_site=50)
    pairs = zip(again["recording_id"], again["offset_s"], strict=True)
    ids = {window_id_for(r, o) for r, o in pairs}
    assert window_id_for(first["recording_id"], first["offset_s"]) not in ids


def test_blancinet_candidates_skip_flagged_recordings_and_filter_sites(corpus, tmp_path):
    con, cfg, _ = corpus
    con.execute(
        "UPDATE recordings SET qc_flags = ? WHERE path LIKE '%P1_20260210%'",
        (json.dumps({"off_campaign": True}),),
    )
    export = blancinet_export(tmp_path, all_detections())
    queue = blancinet_candidates(con, export, cfg, per_site=50, sites=["patawa"])
    assert set(queue["site"]) == {"Patawa"}
    assert not queue["path"].str.contains("P1_20260210").any()


def test_random_candidates_draw_peak_hours_evenly_across_sites(corpus):
    con, cfg, _ = corpus
    queue = random_candidates(con, cfg, n=6, seed=1)
    assert len(queue) == 6
    assert queue.groupby("site").size().to_dict() == {"CDR": 3, "Patawa": 3}
    assert not queue["path"].str.contains("_120000").any()  # 12 h : hors heures de pic
    assert (queue["reason"] == "random").all() and (queue["source"] == "random").all()
    assert queue["offset_s"].between(0, DURATION_S - 3).all()


def test_load_candidates_completes_a_recording_level_queue(corpus, tmp_path):
    """Une file `blanci queue` n'a que recording_id, score, reason : départ à 0 s, 3 s."""
    con, cfg, _ = corpus
    rid = con.execute("SELECT recording_id FROM recordings LIMIT 1").fetchone()[0]
    path = tmp_path / "queue_toy-1.csv"
    pd.DataFrame({"recording_id": [rid], "score": [0.4], "reason": ["uncertain"]}).to_csv(
        path, index=False
    )
    queue = load_candidates(path, con)
    row = queue.iloc[0]
    assert row["offset_s"] == 0.0 and row["dur_s"] == 3.0 and row["source"] == "active"
    assert row["site"] in SITES and row["path"].endswith(".wav")


# --- Réponses --------------------------------------------------------------------------------


def test_save_answer_creates_the_window_and_appends_labels(corpus):
    con, cfg, _ = corpus
    rid = con.execute("SELECT recording_id FROM recordings LIMIT 1").fetchone()[0]
    candidate = {
        "recording_id": rid,
        "offset_s": 4.5,
        "dur_s": 3.0,
        "score": 0.82,
        "reason": "blancinet_0.7-1.0",
        "source": "active",
    }
    save_answer(con, candidate, "blanci", "léonard", quality="C", comment="lointain", channel=1)
    save_answer(con, candidate, "blanci_uncertain", "tuteur")  # seconde écoute : nouvelle ligne
    rows = con.execute(
        "SELECT label, quality, annotator, source, conditions FROM labels ORDER BY label_id"
    ).fetchall()
    assert [r["label"] for r in rows] == ["blanci", "blanci_uncertain"]
    first = json.loads(rows[0]["conditions"])
    assert first == {
        "candidate_reason": "blancinet_0.7-1.0",
        "comment": "lointain",
        "channel_listened": 1,
        "previous_model_score": 0.82,
    }
    assert rows[0]["quality"] == "C" and rows[0]["source"] == "active"
    queue = pd.DataFrame([candidate])
    assert progress(con, queue).tolist() == ["blanci_uncertain"]


def test_ensure_window_is_idempotent(corpus):
    con, _, _ = corpus
    rid = con.execute("SELECT recording_id FROM recordings LIMIT 1").fetchone()[0]
    assert ensure_window(con, rid, 3.0, 3.0) == ensure_window(con, rid, 3.0, 3.0)
    assert con.execute("SELECT COUNT(*) FROM windows").fetchone()[0] == 1


def test_every_answer_is_a_known_label(corpus):
    from blanci.labels import LABELS

    assert {label for label, _ in ANSWERS} <= set(LABELS)


# --- Écoute ----------------------------------------------------------------------------------


def test_read_clip_adds_context_within_the_file_and_picks_the_channel(corpus):
    con, _, raw = corpus
    path = con.execute("SELECT path FROM recordings LIMIT 1").fetchone()[0]
    before = (raw / path).stat().st_mtime_ns
    wav0, sr, start = read_clip(raw, path, 1.0, 3.0, context_s=2.0, channel=0)
    wav1, _, _ = read_clip(raw, path, 1.0, 3.0, context_s=2.0, channel=1)
    assert sr == SR and start == 0.0  # le contexte s'arrête au début du fichier
    assert len(wav0) == int(6.0 * SR)
    assert np.std(wav1) == pytest.approx(4 * np.std(wav0), rel=0.01)
    _, _, start = read_clip(raw, path, 8.0, 3.0, context_s=2.0)
    assert start == 6.0
    assert (raw / path).stat().st_mtime_ns == before


def test_spectrogram_and_player_bytes():
    x = np.sin(2 * np.pi * 5000 * np.arange(SR) / SR).astype(np.float32) * 0.1
    freqs, times, db = clip_spectrogram(x, SR, fmax_hz=8000)
    assert freqs.max() <= 8000 and db.shape == (len(freqs), len(times))
    assert abs(freqs[db.mean(axis=1).argmax()] - 5000) < 50
    data, rate = sf.read(io.BytesIO(wav_bytes(x, SR, gain_db=6)))
    assert rate == SR and np.abs(data).max() == pytest.approx(0.2, rel=0.01)


def test_local_time_is_shown_in_guiana_time():
    assert local_time("2026-01-10T10:00:00Z", -3) == "2026-01-10 07:00"
    assert local_time(None, -3) == "?"
