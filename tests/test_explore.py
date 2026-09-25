"""Exploration en notebook (`blanci/explore.py`) : lecture seule, fenêtres d'un enregistrement,
module séquentiel, négatifs appariés, embeddings et prototypes."""

import sqlite3

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from blanci import explore as ex
from blanci.db import connect, register_model, window_id_for
from blanci.embed import month_of
from blanci.ingest import ingest
from blanci.sequential import GATES, upstream_from_cfg
from blanci.service import append_label
from blanci.store import EmbeddingStore

SR = 24_000
DURATION_S = 12.0


def soundscape(seed: int, song_until_s: float = 0.0) -> np.ndarray:
    """Fond large bande ; notes de 0,1 s à 4,9 kHz toutes les 1,4 s jusqu'à `song_until_s`."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.02, int(SR * DURATION_S))
    note = np.hanning(int(0.1 * SR)) * np.sin(2 * np.pi * 4900 * np.arange(int(0.1 * SR)) / SR)
    for start in np.arange(0.4, song_until_s - 0.2, 1.4):
        i = int(start * SR)
        x[i : i + len(note)] += 0.3 * note
    return x.astype(np.float32)


@pytest.fixture
def corpus(tmp_path, cfg):
    """2 micros × 3 jours à 10 h, stéréo. Premier jour : chant de 0 à 9 s, annoté à 0, 3 et 6 s
    sur M1, à 0 et 6 s sur M2 (la fenêtre à 3 s est un trou). BlanciNet : une détection à 0 s
    et une à 9 s (hors annotation) sur M1."""
    raw = tmp_path / "raw"
    cfg["paths"]["raw"] = str(raw)
    cfg["qc"]["expected_duration_s"] = DURATION_S
    cfg["benchmark"] |= {"negatives_per_positive": 4, "pairing": "nearest"}
    seed = 0
    for mic in ("M1", "M2"):
        for day in (10, 11, 12):
            wav = soundscape(seed, song_until_s=9.0 if day == 10 else 0.0)
            path = raw / "2026" / "mataroni" / mic / f"{mic}_202602{day}_100000.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, np.stack([wav, wav], axis=1), SR, subtype="PCM_16")
            seed += 1
    con = connect(tmp_path / "db.sqlite")
    ingest(con, raw, "2026", cfg, run_qc=False, hash_file=False)
    rids = {
        path.split("/")[-1][:2]: rid
        for rid, path in con.execute(
            "SELECT recording_id, path FROM recordings WHERE path LIKE '%20260210%'"
        ).fetchall()
    }
    for mic, offsets in (("M1", (0.0, 3.0, 6.0)), ("M2", (0.0, 6.0))):
        for offset in offsets:
            wid = window_id_for(rids[mic], offset)
            con.execute(
                "INSERT INTO windows (window_id, recording_id, offset_s, dur_s) VALUES (?,?,?,3.0)",
                (wid, rids[mic], offset),
            )
            append_label(con, wid, "blanci", "import")
    register_model(con, "blancinet", "detector", "blancinet", "v0", {"source": "test"})
    for offset, score in ((0.0, 0.9), (9.0, 0.7)):
        wid = window_id_for(rids["M1"], offset)
        con.execute(
            "INSERT OR IGNORE INTO windows (window_id, recording_id, offset_s, dur_s) "
            "VALUES (?,?,?,3.0)",
            (wid, rids["M1"], offset),
        )
        con.execute("INSERT INTO scores VALUES (?, 'blancinet', ?)", (wid, score))
    con.commit()
    readonly = ex.open_readonly(tmp_path / "db.sqlite")
    yield readonly, cfg, rids, raw
    readonly.close()
    con.close()


def test_the_database_is_opened_read_only(corpus):
    con, *_ = corpus
    with pytest.raises(sqlite3.OperationalError):
        con.execute("CREATE TABLE x (a INTEGER)")


def test_overview_lists_positives_first_with_detections(corpus):
    con, cfg, rids, _ = corpus
    overview = ex.recordings_overview(con, cfg)
    assert len(overview) == 6
    assert overview["n_positive"].head(2).tolist() == [3, 2]
    m1 = overview.set_index("recording_id").loc[rids["M1"]]
    assert m1["n_blancinet"] == 2 and m1["max_blancinet"] == pytest.approx(0.9)
    assert str(m1["local"]) == "2026-02-10 10:00:00"


def test_recording_windows_know_labels_gaps_and_detections(corpus):
    con, cfg, rids, _ = corpus
    windows = ex.recording_windows(con, cfg, rids["M1"])
    assert windows["offset_s"].tolist() == [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0]
    assert windows.loc[windows["y"] == 1, "offset_s"].tolist() == [0.0, 3.0, 6.0]
    assert windows["overlaps_positive"].tolist() == [True] * 6 + [False]
    assert windows["distance_to_positive_s"].iloc[-1] == 0.0
    # Une détection couvre une fenêtre dont elle occupe au moins la moitié.
    expected = [0.9, 0.9, np.nan, np.nan, np.nan, 0.7, 0.7]
    np.testing.assert_allclose(windows["blancinet"], expected)

    gap = ex.recording_windows(con, cfg, rids["M2"]).set_index("offset_s")
    assert gap.at[3.0, "suspect_fn"] and not gap.at[3.0, "overlaps_positive"]
    assert not gap["suspect_fn"].drop(3.0).any()


def test_window_indices_count_notes_from_the_recording_onsets(corpus):
    con, cfg, rids, _ = corpus
    windows = ex.recording_windows(con, cfg, rids["M1"])
    wav = np.zeros(int(SR * DURATION_S), dtype=np.float32)
    values = ex.window_indices(wav, SR, windows, cfg, onsets=np.array([0.5, 1.9, 3.3]))
    assert list(values.columns) == list(GATES) and values.index.equals(windows.index)
    assert values.loc[0, "notes"] == 2 and values.loc[0, "rhythm"] == 1  # 0,5 et 1,9 s
    assert values.loc[1, "notes"] == 2 and values.loc[6, "notes"] == 0


def test_read_segment_matches_the_recording_and_pads_after_the_end(corpus):
    con, cfg, rids, raw = corpus
    before = {p: p.stat().st_mtime_ns for p in raw.rglob("*.wav")}
    path = con.execute("SELECT path FROM recordings WHERE recording_id = ?", (rids["M1"],))
    path = path.fetchone()[0]
    wav, sr = ex.read_recording(cfg, path)
    segment, sr2 = ex.read_segment(cfg, path, 3.0, 3.0)
    assert sr == sr2 == SR
    np.testing.assert_allclose(segment, ex.cut(wav, sr, 3.0, 3.0))
    tail, _ = ex.read_segment(cfg, path, 11.0, 3.0)
    assert len(tail) == 3 * SR and not tail[SR + 10 :].any()
    assert {p: p.stat().st_mtime_ns for p in raw.rglob("*.wav")} == before  # jamais écrit


def test_upstream_stages_and_gate_report(corpus):
    _, cfg, _, _ = corpus
    upstream = upstream_from_cfg(cfg, only=["bandpass", "denoise", "notes"])
    stages = ex.upstream_stages(soundscape(0, 3.0)[: 3 * SR], SR, upstream)
    assert [name for name, _ in stages] == ["brut", "bp3-7k", "dn1"]
    assert {len(wav) for _, wav in stages} == {3 * SR}
    values = pd.Series({"band_energy": 9.0, "band_contrast": 1.0, "notes": 0.0, "rhythm": np.nan})
    report = ex.gate_report(values, cfg, upstream).set_index("gate")
    assert report["enabled"].to_dict() == {
        "band_energy": False,
        "band_contrast": False,
        "notes": True,
        "rhythm": False,
    }
    assert report["passes"].to_dict() == {
        "band_energy": True,
        "band_contrast": False,
        "notes": False,
        "rhythm": True,  # valeur manquante : laisse passer
    }


def test_nearest_negatives_of_a_recording(corpus):
    """Annotations à 0, 3 et 6 s sur 12 s : seule la fenêtre à 9 s est libre dans
    l'enregistrement ; le reste vient d'un autre jour (même micro, même créneau)."""
    con, cfg, rids, _ = corpus
    negatives = ex.paired_for_recording(con, cfg, rids["M1"])
    assert len(negatives) == 4
    same = negatives[negatives["pairing"] == "same_recording"]
    assert same["offset_s"].tolist() == [9.0] and same["distance_to_positive_s"].tolist() == [0.0]
    others = negatives.drop(same.index)
    assert (others["pairing"] == "other_day").all()
    assert set(others["minutes_from_positive"]) <= {1440.0, 2880.0}
    assert not others["recording_id"].isin(rids.values()).any()
    ordered = ex.nearest_first(negatives, 6.0)
    assert ordered["pairing"].iloc[0] == "same_recording"
    assert ordered["minutes_from_positive"].abs().is_monotonic_increasing


def test_a_window_without_annotation_gets_hypothetical_negatives(corpus):
    con, cfg, _, _ = corpus
    rid = con.execute("SELECT recording_id FROM recordings WHERE path LIKE '%M1_20260211%'")
    rid = rid.fetchone()[0]
    with pytest.raises(ValueError, match="pas d'annotation positive"):
        ex.paired_for_recording(con, cfg, rid)
    chosen = pd.DataFrame([{"recording_id": rid, "offset_s": 4.5, "dur_s": 3.0}])
    negatives = ex.paired_for_recording(con, cfg, rid, positive_windows=chosen)
    # 3 à 6 s chevauchent la fenêtre choisie (4,5–7,5 s) ; les voisines d'abord.
    assert (negatives["pairing"] == "same_recording").all()
    assert set(negatives["offset_s"]) == {0.0, 1.5, 7.5, 9.0}


def test_annotation_coverage_counts_detections_outside_annotations(corpus):
    con, _, rids, _ = corpus
    coverage = ex.annotation_coverage(con).set_index("recording_id")
    assert coverage.at[rids["M1"], "covered_s"] == 9.0
    assert coverage.at[rids["M2"], "covered_s"] == 6.0
    assert coverage.at[rids["M1"], "detections_outside"] == 1
    assert coverage.at[rids["M1"], "median_score_outside"] == pytest.approx(0.7)
    assert coverage.at[rids["M2"], "detections_outside"] == 0


def test_onset_sweep_finds_notes_when_the_threshold_is_low_enough(corpus):
    _, cfg, _, _ = corpus
    song, background = soundscape(1, 3.0)[: 3 * SR], soundscape(2)[: 3 * SR]
    sweep = ex.onset_sweep([song, background], SR, cfg, (2.0, 4.0), (0.01,))
    assert len(sweep) == 2 * 2 * 1
    found = sweep.set_index(["segment", "k_mad"])["notes"]
    assert found[(0, 2.0)] >= 1 and found[(0, 2.0)] >= found[(1, 2.0)]


def test_subtracting_the_background_of_the_paired_negative():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, 3 * SR).astype(np.float32)
    np.testing.assert_allclose(
        ex.subtract_background(noise, np.zeros_like(noise), SR), noise, atol=1e-4
    )
    cleaned = ex.subtract_background(noise, rng.normal(0, 0.1, 3 * SR), SR)
    assert len(cleaned) == len(noise)
    assert np.mean(cleaned**2) < 0.5 * np.mean(noise**2)


def test_embeddings_of_a_recording_and_prototypes(corpus):
    """Stock synthétique : la dimension 0 porte le chant, les autres le fond."""
    con, cfg, rids, _ = corpus
    rng = np.random.default_rng(0)
    overview = ex.recordings_overview(con, cfg)
    store = EmbeddingStore(cfg["paths"]["embeddings"], "toy")
    for rec in overview.itertuples():
        windows = ex.recording_windows(con, cfg, rec.recording_id)
        emb = rng.normal(0, 0.1, (len(windows), 8)) + np.array([0, 1, 1, 1, 1, 1, 1, 1])
        emb[windows["overlaps_positive"].to_numpy(), 0] += 3.0
        meta = windows[["window_id", "recording_id", "offset_s"]]
        store.write(meta, emb, rec.dataset, rec.site, month_of(rec.start_utc))
    assert ex.embedding_stocks(cfg) == ["toy"]

    rec = overview.loc[overview["recording_id"] == rids["M1"]].iloc[0]
    meta, emb = ex.recording_embeddings(cfg, "toy", rec)
    assert (meta["recording_id"] == rids["M1"]).all() and len(meta) == 7
    assert meta["offset_s"].is_monotonic_increasing and emb.dtype == np.float32
    row, vector = ex.nearest_embedding(meta, emb, 4.4, ex.stock_window_s(con, "toy"))
    assert row["offset_s"] == 3.0 and vector.shape == (8,)

    pair = ex.compare_embeddings(emb[0], emb[-1])
    assert -1 <= pair["cosine"] < 0.9 and pair["difference"].shape == (8,)
    prototypes = ex.corpus_prototypes(con, cfg, "toy")
    assert prototypes["n_pos"] == 5 and prototypes["w"].shape == (8,)
    assert np.argmax(np.abs(prototypes["w"])) == 0  # w garde le chant, retire le fond
    scores = ex.prototype_scores(emb, prototypes)
    assert list(scores.columns) == ["differential", "simple"]
    assert scores["differential"].iloc[0] > 0 > scores["differential"].iloc[-1]
