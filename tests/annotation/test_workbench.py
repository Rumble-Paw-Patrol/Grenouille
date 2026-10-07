"""Poste d'annotation (§5) : files de candidats, écoute, réponses en ajout seul."""

import io
import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from blanci.annotation.workbench import (
    ANSWERS,
    CANDIDATE_COLUMNS,
    agreement,
    clip_spectrogram,
    ensure_window,
    latest_labels,
    load_candidates,
    local_time,
    next_position,
    ordered_classes,
    progress,
    random_candidates,
    read_clip,
    recording_candidates,
    save_answer,
    split_windows,
    wav_bytes,
)
from blanci.core.db import connect, window_id_for
from blanci.inputs.ingest import ingest

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
        "reason": "random",
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
        "candidate_reason": "random",
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
    from blanci.inputs.labels import LABELS

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


def test_player_bytes_keep_only_the_band():
    """« N'écouter que la bande » : un son hors bande disparaît, un son dans la bande reste."""
    t = np.arange(2 * SR) / SR
    inside = np.sin(2 * np.pi * 5000 * t).astype(np.float32) * 0.1
    outside = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.1
    band = (4400, 5500)
    kept, _ = sf.read(io.BytesIO(wav_bytes(inside, SR, band_hz=band)))
    removed, _ = sf.read(io.BytesIO(wav_bytes(outside, SR, band_hz=band)))
    middle = slice(SR // 2, 3 * SR // 2)  # hors des bords du filtre
    assert np.abs(kept[middle]).max() == pytest.approx(0.1, rel=0.05)
    assert np.abs(removed[middle]).max() < 0.005


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


# --- Logits de classes gardés à l'encodage (`logit_classes`) --------------------------------------


class LogitToy:
    """Encodeur factice qui, comme birdnet_v3, garde des logits de classes pendant `embed`."""

    name, version, sample_rate, window_s, dim, has_tokens = "logittoy", "1", SR, 3.0, 2, False
    logit_names = ["Dendropsophus minutus", "Leptodactylus latrans"]

    def __init__(self):
        self._pending = []

    def embed(self, wav, sr):
        wav = np.atleast_2d(wav)
        # « Logits » : niveau du canal lu, et son opposé.
        level = wav.std(axis=1)
        self._pending.append(np.stack([level, -level], axis=1))
        return np.stack([wav.mean(axis=1), level], axis=1).astype(np.float32)

    def embed_tokens(self, wav, sr):
        return None

    def pop_logits(self):
        out = np.concatenate(self._pending) if self._pending else np.zeros((0, 2))
        self._pending = []
        return out


def test_embed_stores_class_logits(corpus, tmp_path):
    from blanci.embedding.embed import embed_recordings, select_recordings

    con, cfg, raw = corpus
    report = embed_recordings(
        con, LogitToy(), select_recordings(con), raw, tmp_path / "emb", channel=1
    )
    n_scores = con.execute(
        "SELECT COUNT(*) FROM scores WHERE model_id LIKE 'logittoy-1:logit:%'"
    ).fetchone()[0]
    assert n_scores == 2 * report.windows


def test_flagged_recordings_go_whole_to_the_listening_queue(corpus):
    """Écartés par un drapeau (horloge douteuse), jamais encodés : on les écoute en entier."""
    from blanci.annotation.workbench import flagged_candidates
    from blanci.service import append_label

    con, _, _ = corpus
    con.execute(
        "UPDATE recordings SET qc_flags = ? WHERE path LIKE '%P1_2026021%'",
        (json.dumps({"clock_off": True}),),
    )
    con.execute(  # test de quelques secondes : déjà écarté pour sa durée, pas à écouter
        "UPDATE recordings SET qc_flags = ? WHERE path LIKE '%P2_20260210%'",
        (json.dumps({"clock_off": True, "duration_off": True}),),
    )
    queue = flagged_candidates(con, "clock_off")
    assert list(queue.columns) == CANDIDATE_COLUMNS and len(queue) == 4
    assert queue["path"].str.contains("P1_2026021").all()
    assert (queue["offset_s"] == 0).all() and (queue["dur_s"] == DURATION_S).all()
    assert set(queue["source"]) == {"flag"} and set(queue["reason"]) == {"clock_off"}
    heard = queue.iloc[0]
    wid = ensure_window(con, heard["recording_id"], 0.0, DURATION_S)
    append_label(con, wid, "background", "flag")
    assert len(flagged_candidates(con, "clock_off")) == 3  # déjà écouté : sauté
    assert flagged_candidates(con, "clock_off", sites=["CDR"]).empty


# --- Navigation, découpage, plusieurs classes ---------------------------------------------------


def test_next_position_counts_in_the_whole_queue():
    done = pd.Series([None, "bird", None, None], dtype=object)
    assert next_position(done, 0, unheard_only=False) == 1
    assert next_position(done, 0, unheard_only=True) == 2  # saute le candidat déjà écouté
    assert next_position(done, 3, unheard_only=True) == 0  # repart du début
    assert next_position(done, -1, unheard_only=True) == 0  # ouverture de la file
    assert next_position(pd.Series(["bird"] * 2, dtype=object), 1, unheard_only=True) == 1
    assert next_position(done, 3, unheard_only=False) == 3


def test_next_position_backwards_and_without_wrapping():
    done = pd.Series([None, "bird", None, None], dtype=object)
    assert next_position(done, 2, unheard_only=False, step=-1) == 1
    assert next_position(done, 0, unheard_only=False, step=-1) == 0
    assert next_position(done, 2, unheard_only=True, step=-1) == 0  # saute le 2e
    assert next_position(done, 0, unheard_only=True, step=-1) == 3  # fait le tour
    assert next_position(done, 0, unheard_only=True, step=-1, wrap=False) == 0
    assert next_position(done, 3, unheard_only=True, wrap=False) == 3  # bout de la file


def test_split_windows_are_anchored_on_the_candidate():
    assert split_windows(0.0, 12.0, 4.5, 3.0) == [1.5, 4.5, 7.5]
    assert split_windows(0.0, 12.0, 3.0, 3.0) == [0.0, 3.0, 6.0, 9.0]
    assert split_windows(1.5, 10.5, 4.5, 3.0) == [1.5, 4.5, 7.5]
    assert split_windows(0.0, 2.0, 0.0, 3.0) == []
    with pytest.raises(ValueError):
        split_windows(0.0, 2.0, 0.0, 0.0)


def test_ordered_classes_puts_blanci_first():
    assert ordered_classes([]) == ("background", [])
    assert ordered_classes(["rain", "blanci"]) == ("blanci", ["rain"])
    assert ordered_classes(["background", "bird"]) == ("bird", [])
    assert ordered_classes(["orthoptera", "bird", "bird"]) == ("bird", ["orthoptera"])


def test_save_answer_keeps_the_other_classes(corpus):
    con, _, _ = corpus
    rid = con.execute("SELECT recording_id FROM recordings LIMIT 1").fetchone()[0]
    candidate = {"recording_id": rid, "offset_s": 0.0, "dur_s": 3.0, "source": "random"}
    save_answer(con, candidate, "blanci", "léonard", extra_labels=["rain", "bird"])
    conditions = json.loads(con.execute("SELECT conditions FROM labels").fetchone()[0])
    assert conditions["extra_labels"] == ["rain", "bird"]
    with pytest.raises(ValueError, match="classe inconnue"):
        save_answer(con, candidate, "blanci", "léonard", extra_labels=["dragon"])
    wid = window_id_for(rid, 0.0, 3.0)
    assert latest_labels(con, [wid, window_id_for(rid, 3.0, 3.0)]) == {wid: "blanci"}
    assert latest_labels(con, [wid], annotator="tuteur") == {}


def test_viewer_serves_its_files_next_to_the_page(tmp_path, monkeypatch):
    """L'écoute est rangée à côté de la page du composant, nommée par son contenu : les
    arguments ne transportent que son chemin ; la page embarque sa palette."""
    from blanci.annotation import viewer

    monkeypatch.setattr(viewer, "VIEWER_DIR", tmp_path / "viewer")
    src = viewer.asset(b"wav", ".wav")
    assert (tmp_path / "viewer" / src).read_bytes() == b"wav"
    assert viewer.asset(b"wav", ".wav") == src
    args = viewer.viewer_args(
        1.0,
        7.0,
        [("micro 1", src, 1.0)],
        [{"t0": 3.0, "t1": 6.0, "label": None, "current": True, "candidate": True}],
        (4400.0, 5500.0),
        key="a.wav:1.00:7.00",
        labels=[("blanci", "A. blanci")],
        intervals=[(2.0, 2.5, "blanci")],
        interval_mode=True,
        channel=1,
    )
    assert args["extract"] == "a.wav:1.00:7.00" and args["channel"] == 1
    assert args["intervals"] == [[2.0, 2.5, "blanci"]]
    assert args["audios"] == [{"name": "micro 1", "src": src, "start": 1.0}]
    page = (viewer._prepare() / "index.html").read_text(encoding="utf-8")
    assert "__MAGMA__" not in page and "[0, 0, 4]" in page


def test_save_span_derives_the_other_label(corpus):
    """Les fenêtres hors intervalles prennent la première autre classe cochée ; A. blanci
    sans intervalle est refusé."""
    from blanci.annotation.workbench import save_span, span_intervals

    con, _, _ = corpus
    rid = con.execute("SELECT recording_id FROM recordings LIMIT 1").fetchone()[0]
    candidate = {"recording_id": rid, "offset_s": 3.0, "dur_s": 3.0, "source": "random"}
    with pytest.raises(ValueError, match="sans intervalle"):
        save_span(con, candidate, 0.0, 12.0, [], ["blanci"], "léonard")
    assert span_intervals(con, rid, 0.0, 12.0) is None
    save_span(con, candidate, 0.0, 12.0, [(4.0, 5.0, "blanci")], ["rain", "bird"], "léonard")
    other, classes = con.execute("SELECT other_label, classes FROM spans").fetchone()
    assert other == "bird" and json.loads(classes) == ["rain", "bird", "blanci"]
    assert span_intervals(con, rid, 0.0, 12.0) == [(4.0, 5.0, "blanci")]
    queue = pd.DataFrame({"recording_id": [rid] * 2, "offset_s": [3.0, 6.0], "dur_s": 3.0})
    assert progress(con, queue).tolist() == ["blanci", "bird"]
