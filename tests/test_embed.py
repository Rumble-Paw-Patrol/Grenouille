import json

import numpy as np
import pandas as pd
import pytest

from blanci.db import connect, recording_id_for
from blanci.embed import embed_recordings, month_of, select_recordings
from blanci.encoders.base import BaseEncoder
from blanci.store import EmbeddingStore
from tests.conftest import write_wav

SR = 16000
WINDOW_S = 3.0
DURATION_S = 12.0


class FakeEncoder(BaseEncoder):
    """Encodeur déterministe et instantané : on teste la mécanique d'extraction, pas un modèle."""

    name = "fake"
    version = "1"
    sample_rate = SR
    window_s = WINDOW_S
    dim = 4
    has_tokens = False

    def __init__(self):
        self.calls = 0

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        self.calls += 1
        base = np.stack([batch.mean(1), batch.std(1), batch.max(1), batch.min(1)], axis=1)
        return base.astype(np.float32)


@pytest.fixture
def workspace(tmp_path):
    raw = tmp_path / "raw"
    store_root = tmp_path / "embeddings"
    con = connect(tmp_path / "blanci.sqlite")
    return con, raw, store_root


def add_recording(con, raw, rel_path, site="mataroni", mic="M1", start="2026-02-10T13:00:00Z"):
    write_wav(raw / rel_path, sr=SR, duration_s=DURATION_S)
    rid = recording_id_for(rel_path)
    con.execute(
        "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
        "duration_s, sample_rate, channels, qc_flags) "
        "VALUES (?, ?, '2026', ?, ?, ?, ?, ?, 1, '{}')",
        (rid, rel_path, site, mic, start, DURATION_S, SR),
    )
    con.commit()
    return rid


def recordings_of(con):
    return pd.read_sql_query("SELECT * FROM recordings ORDER BY path", con)


# --- Sélection des enregistrements ------------------------------------------------------------


def test_select_filters_by_dataset_and_site(workspace):
    con, raw, _ = workspace
    add_recording(con, raw, "a.wav", site="mataroni")
    add_recording(con, raw, "b.wav", site="tresor")
    assert len(select_recordings(con, site="tresor")) == 1
    assert len(select_recordings(con, site="TRESOR")) == 1  # insensible à la casse
    assert len(select_recordings(con, dataset="2023")) == 0


def test_select_keeps_only_peak_hours(workspace):
    """Heures de pic locales 7–9 h et 15–17 h (§5) ; le fichier est à 13 h UTC = 10 h locales."""
    con, raw, _ = workspace
    add_recording(con, raw, "midmorning.wav", start="2026-02-10T13:00:00Z")  # 10 h locales
    add_recording(con, raw, "peak.wav", start="2026-02-10T11:00:00Z")  # 8 h locales
    kept = select_recordings(con, peak_hours=[[7, 9], [15, 17]], utc_offset_h=-3)
    assert list(kept["path"]) == ["peak.wav"]


def test_select_drops_flagged_recordings(workspace):
    """« micro dans sac » et silence : exclus de l'entraînement (§5, negative mining)."""
    con, raw, _ = workspace
    add_recording(con, raw, "good.wav")
    bad = add_recording(con, raw, "bagged.wav")
    con.execute(
        "UPDATE recordings SET qc_flags = ? WHERE recording_id = ?",
        (json.dumps({"in_bag": True}), bad),
    )
    con.commit()
    assert list(select_recordings(con)["path"]) == ["good.wav"]


def test_month_of_handles_a_missing_timestamp():
    assert month_of("2026-02-10T13:00:00Z") == "202602"
    assert month_of(None) == "unknown"


# --- Extraction -------------------------------------------------------------------------------


def test_embed_writes_one_partition_per_site_and_month(workspace):
    con, raw, store_root = workspace
    add_recording(con, raw, "m.wav", site="mataroni", start="2026-02-10T13:00:00Z")
    add_recording(con, raw, "t.wav", site="tresor", start="2026-03-10T13:00:00Z")
    report = embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root)
    store = EmbeddingStore(store_root, "fake-1")
    assert report.recordings == 2
    assert sorted(p.relative_to(store.directory).as_posix() for p in store.fragments()) == [
        "2026/mataroni/202602.parquet",
        "2026/tresor/202603.parquet",
    ]


def test_embed_fills_the_windows_table(workspace):
    con, raw, store_root = workspace
    rid = add_recording(con, raw, "a.wav")
    embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root)
    rows = con.execute(
        "SELECT offset_s, dur_s FROM windows WHERE recording_id = ? ORDER BY offset_s", (rid,)
    ).fetchall()
    assert [r["dur_s"] for r in rows] == [WINDOW_S] * len(rows)
    assert rows[0]["offset_s"] == 0.0
    assert rows[1]["offset_s"] == pytest.approx(WINDOW_S / 2)  # chevauchement 0,5


def test_other_overlap_writes_a_separate_stock(workspace):
    """Chevauchement 75 % : grille au quart de fenêtre, stock `fake-1@o75`, rien dans `fake-1`
    (DECISIONS n° 89)."""
    con, raw, store_root = workspace
    add_recording(con, raw, "2026/mataroni/M1/a.wav")
    report = embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root, overlap=0.75)
    assert report.encoder_id == "fake-1@o75"
    meta, _ = EmbeddingStore(store_root, "fake-1@o75").load()
    first = meta[meta["recording_id"] == meta["recording_id"].iloc[0]]["offset_s"].sort_values()
    assert first.iloc[1] == pytest.approx(round(WINDOW_S / 4, 2))
    assert not EmbeddingStore(store_root, "fake-1").directory.exists()
    params = json.loads(
        con.execute("SELECT params_json FROM models WHERE model_id = 'fake-1@o75'").fetchone()[0]
    )
    assert params["overlap"] == pytest.approx(0.75, abs=0.01)


def test_embeddings_match_the_windows_table(workspace):
    con, raw, store_root = workspace
    add_recording(con, raw, "a.wav")
    report = embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root)
    store = EmbeddingStore(store_root, "fake-1")
    meta, emb = store.load()
    assert len(meta) == len(emb) == report.windows
    stored = {r["window_id"] for r in con.execute("SELECT window_id FROM windows")}
    assert set(meta["window_id"]) == stored
    assert emb.shape[1] == FakeEncoder.dim
    assert emb.dtype == np.float16  # stock en float16 (§4)


# --- Reprise ----------------------------------------------------------------------------------


def test_second_run_skips_what_is_already_done(workspace):
    """Reprenable : relancer après une coupure ne recalcule rien."""
    con, raw, store_root = workspace
    add_recording(con, raw, "a.wav")
    first = embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root)
    second = embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root)
    assert first.recordings == 1 and first.skipped == 0
    assert second.recordings == 0 and second.skipped == 1


def test_resume_adds_only_the_new_recordings(workspace):
    con, raw, store_root = workspace
    add_recording(con, raw, "a.wav", start="2026-02-10T13:00:00Z")
    embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root)
    add_recording(con, raw, "b.wav", start="2026-02-11T13:00:00Z")
    second = embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root)
    assert second.recordings == 1 and second.skipped == 1
    meta, _ = EmbeddingStore(store_root, "fake-1").load()
    assert meta["recording_id"].nunique() == 2


def test_flush_does_not_lose_earlier_batches(workspace):
    """Le tampon est vidé plusieurs fois par partition : rien ne doit être écrasé."""
    con, raw, store_root = workspace
    for i in range(5):
        add_recording(con, raw, f"r{i}.wav", start="2026-02-10T13:00:00Z")
    report = embed_recordings(
        con, FakeEncoder(), recordings_of(con), raw, store_root, flush_every=2
    )
    meta, emb = EmbeddingStore(store_root, "fake-1").load()
    assert meta["recording_id"].nunique() == 5
    assert len(meta) == len(emb) == report.windows


def test_unreadable_file_is_counted_not_fatal(workspace):
    con, raw, store_root = workspace
    add_recording(con, raw, "good.wav")
    (raw / "broken.wav").write_bytes(b"pas un WAV")
    con.execute(
        "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
        "duration_s, sample_rate, channels, qc_flags) VALUES "
        "('broken', 'broken.wav', '2026', 'mataroni', 'M1', '2026-02-10T13:00:00Z', 120, 16000, "
        "1, '{}')"
    )
    con.commit()
    report = embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root)
    assert report.errors == 1 and report.recordings == 1


# --- Débit enregistré -------------------------------------------------------------------------


def test_encoder_and_throughput_are_registered(workspace):
    """La vitesse est la colonne « vitesse » du benchmark (§2) : elle doit être en base."""
    con, raw, store_root = workspace
    add_recording(con, raw, "a.wav")
    report = embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root)
    row = con.execute("SELECT * FROM models WHERE model_id = 'fake-1'").fetchone()
    params = json.loads(row["params_json"])
    assert row["kind"] == "encoder" and row["name"] == "fake"
    assert params["sample_rate"] == SR and params["dim"] == FakeEncoder.dim
    assert params["hop_s"] == pytest.approx(WINDOW_S / 2)
    assert params["last_run"]["windows_per_s"] > 0
    assert report.realtime_factor > 0


def test_rerunning_updates_the_throughput_without_duplicating_the_model(workspace):
    con, raw, store_root = workspace
    add_recording(con, raw, "a.wav")
    embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root)
    embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root)
    assert con.execute("SELECT COUNT(*) FROM models").fetchone()[0] == 1


def test_report_counts_windows_and_audio(workspace):
    con, raw, store_root = workspace
    add_recording(con, raw, "a.wav")
    report = embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root)
    assert report.audio_s == pytest.approx(DURATION_S)
    assert report.windows == len(EmbeddingStore(store_root, "fake-1").load()[0])
