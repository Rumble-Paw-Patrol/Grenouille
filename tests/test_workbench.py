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
    agreement,
    blancinet_candidates,
    clip_spectrogram,
    congener_candidates,
    ensure_window,
    load_candidates,
    local_time,
    progress,
    random_candidates,
    read_clip,
    recording_candidates,
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
    ids = {window_id_for(r, o) for r, o in pairs}  # fenêtres de 3 s
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
        "tags": ["distant"],  # lu comme à l'import (DECISIONS n° 82)
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


# --- Enregistrements entiers, identifiants, accord ---------------------------------------------


def test_window_ids_keep_the_duration_apart_from_the_legacy_3_s():
    assert window_id_for("r", 0.0) == window_id_for("r", 0.0, 3.0) == "r:0.00"
    assert window_id_for("r", 0.0, 120.0) == "r:0.00/120.00"
    assert window_id_for("r", 5.0, 5.0) != window_id_for("r", 5.0, 3.0)


def test_a_whole_recording_label_does_not_land_on_the_first_3_s_window(corpus):
    con, _, _ = corpus
    rid = con.execute("SELECT recording_id FROM recordings LIMIT 1").fetchone()[0]
    short = ensure_window(con, rid, 0.0, 3.0)
    whole = ensure_window(con, rid, 0.0, DURATION_S)
    assert short != whole
    durations = dict(con.execute("SELECT window_id, dur_s FROM windows").fetchall())
    assert durations == {short: 3.0, whole: DURATION_S}


def test_recording_candidates_cover_every_mic_and_the_whole_file(corpus):
    con, cfg, _ = corpus
    queue = recording_candidates(con, cfg, n=10, sites=["cdr"], reason="jeu_gele", seed=2)
    # 3 micros × 4 enregistrements disponibles : 10 répartis en 4 + 3 + 3.
    assert len(queue) == 10
    assert sorted(queue.groupby("mic_id").size()) == [3, 3, 4]
    assert (queue["offset_s"] == 0).all() and (queue["dur_s"] == DURATION_S).all()
    assert (queue["source"] == "audit").all() and (queue["reason"] == "jeu_gele").all()
    # Chaque micro : ses heures les moins servies d'abord, donc 7 h et 12 h représentées.
    assert queue.groupby("mic_id")["start_utc"].apply(lambda s: s.str[11:13].nunique()).min() == 2


def test_recording_candidates_skip_recordings_already_heard_in_full(corpus):
    con, cfg, _ = corpus
    first = recording_candidates(con, cfg, n=2, sites=["patawa"]).iloc[0].to_dict()
    save_answer(con, first, "background", "léonard")
    again = recording_candidates(con, cfg, n=50, sites=["patawa"])
    assert first["recording_id"] not in set(again["recording_id"])


def test_calibration_hides_only_my_own_answers(corpus):
    con, _, _ = corpus
    rid = con.execute("SELECT recording_id FROM recordings LIMIT 1").fetchone()[0]
    candidate = {"recording_id": rid, "offset_s": 3.0, "dur_s": 3.0, "source": "audit"}
    save_answer(con, candidate, "blanci", "tuteur")
    queue = pd.DataFrame([candidate])
    assert progress(con, queue).tolist() == ["blanci"]
    assert progress(con, queue, annotator="léonard").tolist() == [None]
    assert progress(con, queue, annotator="tuteur").tolist() == ["blanci"]


def test_agreement_between_two_annotators(corpus):
    con, _, _ = corpus
    rid = con.execute("SELECT recording_id FROM recordings LIMIT 1").fetchone()[0]
    answers = [  # (léonard, tuteur)
        ("blanci", "blanci"),
        ("blanci", "blanci_chorus"),
        ("blanci", "bird"),
        ("bird", "bird"),
        ("background", "background"),
        ("blanci_uncertain", "background"),
    ]
    for k, (first, second) in enumerate(answers):
        candidate = {"recording_id": rid, "offset_s": 1.5 * k, "dur_s": 3.0, "source": "audit"}
        save_answer(con, candidate, first, "léonard")
        save_answer(con, candidate, second, "tuteur")
    summary, table = agreement(con, "léonard", "tuteur")
    assert summary["n_windows"] == 6
    assert summary["label_agreement"] == pytest.approx(3 / 6)
    assert summary["blanci_agreement"] == pytest.approx(5 / 6)  # l'incertain compte comme non
    assert summary["positive_agreement"] == pytest.approx(2 * 2 / (2 * 2 + 1))
    assert table.loc["blanci", "bird"] == 1


# --- Congénères de Perch 2.0 ---------------------------------------------------------------------


class LogitToy:
    """Encodeur factice qui, comme perch_v2, garde des logits de congénères pendant `embed`."""

    name, version, sample_rate, window_s, dim, has_tokens = "perchtoy", "1", SR, 3.0, 2, False
    logit_names = ["Anomaloglossus stepheni", "Anomaloglossus surinamensis"]

    def __init__(self):
        self._pending = []

    def embed(self, wav, sr):
        wav = np.atleast_2d(wav)
        # « Logits » : niveau du canal lu, et son opposé ; le micro P2 sera le plus « congénère ».
        level = wav.std(axis=1)
        self._pending.append(np.stack([level, -level], axis=1))
        return np.stack([wav.mean(axis=1), level], axis=1).astype(np.float32)

    def embed_tokens(self, wav, sr):
        return None

    def pop_logits(self):
        out = np.concatenate(self._pending) if self._pending else np.zeros((0, 2))
        self._pending = []
        return out


def test_embed_stores_congener_logits_and_they_rank_candidates(corpus, tmp_path):
    from blanci.embed import embed_recordings, select_recordings

    con, cfg, raw = corpus
    report = embed_recordings(
        con, LogitToy(), select_recordings(con), raw, tmp_path / "emb", channel=1
    )
    n_scores = con.execute(
        "SELECT COUNT(*) FROM scores WHERE model_id LIKE 'perchtoy-1:logit:%'"
    ).fetchone()[0]
    assert n_scores == 2 * report.windows

    queue = congener_candidates(con, "perchtoy-1", per_site=3)
    assert queue.groupby("site").size().to_dict() == {"CDR": 3, "Patawa": 3}
    assert not queue.duplicated("recording_id").any()
    assert (queue["reason"] == "congeneres_perch").all()
    # Le meilleur de chaque micro d'abord : trois micros différents à CDR.
    assert queue[queue["site"] == "CDR"]["mic_id"].nunique() == 3


def test_congener_candidates_need_logits(corpus):
    con, _, _ = corpus
    with pytest.raises(ValueError, match="aucun logit"):
        congener_candidates(con, "perch_v2-bacpipe1.3.5")
