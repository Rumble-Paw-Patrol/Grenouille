import json
import sqlite3
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from blanci.cli import app
from blanci.core.db import connect, recording_id_for, window_id_for
from blanci.inputs.ingest import ingest, parse_songmeter_name, read_guano
from blanci.service import append_label
from tests.conftest import write_wav


@pytest.fixture
def raw(tmp_path):
    """Arborescence type : raw/2026/<site>/<micro>/<préfixe>_<date>_<heure>.wav."""
    root = tmp_path / "raw"
    base = root / "2026" / "Mataroni"
    write_wav(
        base / "M01" / "SMM01_20260212_070000.wav",
        duration_s=120.0,
        guano={"Timestamp": "2026-02-12T07:00:00-03:00", "Serial": "SMM01"},
    )
    write_wav(base / "M01" / "SMM01_20260212_073000.wav", duration_s=120.0)
    write_wav(base / "M02" / "SMM02_20260212_070000.WAV", duration_s=10.0)
    write_wav(base / "SMM03_20260213_150000.wav", duration_s=5.0, guano={"Serial": "SMM03"})
    (base / "M01" / "._SMM01_20260212_070000.wav").write_bytes(b"\x00\x05\x16\x07")  # AppleDouble
    (base / "M02" / "SMM02_20260212_080000.wav").write_bytes(b"RIFF\x00\x00")  # fichier tronqué
    return root


def rows_by_name(con):
    return {Path(r["path"]).name: r for r in con.execute("SELECT * FROM recordings")}


def test_songmeter_name_and_guano(raw):
    prefix, stamp = parse_songmeter_name("SMM01_20260212_073000")
    assert prefix == "SMM01" and stamp.hour == 7 and stamp.minute == 30
    assert parse_songmeter_name("notes") is None
    guano = read_guano(raw / "2026/Mataroni/M01/SMM01_20260212_070000.wav")
    assert guano["Serial"] == "SMM01"


def test_ingest_inventories_and_flags(raw, cfg):
    con = connect(Path(cfg["paths"]["db"]))
    report = ingest(con, raw, "2026", cfg)
    assert report.added == 4 and report.skipped == 0
    assert [Path(p).name for p, _ in report.errors] == ["SMM02_20260212_080000.wav"]

    rows = rows_by_name(con)
    first = rows["SMM01_20260212_070000.wav"]
    assert first["recording_id"] == recording_id_for("2026/Mataroni/M01/SMM01_20260212_070000.wav")
    # Micro = numéro de série (GUANO), pas le nom du dossier : sur le terrain, le même
    # enregistreur s'appelle « SM4 A », « SM A_SMA13417 » ou « SMA13417 » selon le relevé.
    assert (first["dataset"], first["site"], first["mic_id"]) == ("2026", "Mataroni", "SMM01")
    assert first["start_utc"] == "2026-02-12T10:00:00Z"  # GUANO, UTC−3
    assert first["duration_s"] == 120.0 and first["sample_rate"] == 16000
    assert len(first["sha256"]) == 64
    assert set(json.loads(first["qc_flags"])) >= {"in_bag", "silent", "indices"}
    # Sans GUANO : horodatage du nom de fichier + décalage configuré (UTC−3).
    assert rows["SMM01_20260212_073000.wav"]["start_utc"] == "2026-02-12T10:30:00Z"
    # Extension en majuscules acceptée ; sans GUANO, le micro vient du préfixe du nom.
    assert rows["SMM02_20260212_070000.WAV"]["mic_id"] == "SMM02"
    assert rows["SMM03_20260213_150000.wav"]["mic_id"] == "SMM03"

    again = ingest(con, raw, "2026", cfg)
    assert again.added == 0 and again.skipped == 4


def add_label(con, name: str, offset: float, dur: float = 3.0) -> None:
    """Un label positif sur l'enregistrement inventorié `name`."""
    (rid,) = con.execute("SELECT recording_id FROM recordings WHERE path LIKE ?", (f"%/{name}",))
    wid = window_id_for(rid[0], offset, dur)
    con.execute(
        "INSERT OR IGNORE INTO windows (window_id, recording_id, offset_s, dur_s) "
        "VALUES (?, ?, ?, ?)",
        (wid, rid[0], offset, dur),
    )
    append_label(con, wid, "blanci", "random")


def test_labels_are_append_only(raw, cfg):
    con = connect(Path(cfg["paths"]["db"]))
    ingest(con, raw, "2026", cfg, run_qc=False)
    add_label(con, "SMM01_20260212_070000.wav", 36.0)
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 1
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE labels SET label = 'bird'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM labels")


def test_cli_end_to_end(raw, cfg, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"paths": {**cfg["paths"], "raw": str(raw)}}), encoding="utf-8"
    )
    runner = CliRunner()
    result = runner.invoke(app, ["--config", str(config), "ingest", "--dataset", "2026"])
    assert result.exit_code == 0, result.output
    assert "4 ajoutés" in result.output and "1 erreurs" in result.output

    con = connect(Path(cfg["paths"]["db"]))
    add_label(con, "SMM01_20260212_070000.wav", 117.0)
    con.close()

    result = runner.invoke(app, ["-c", str(config), "check-grid"])
    assert result.exit_code == 0, result.output
    assert "hors de toute fenêtre : 0/1" in result.output

    result = runner.invoke(app, ["-c", str(config), "status"])
    assert result.exit_code == 0, result.output
    assert "2026 / Mataroni : 4, 3 micros" in result.output
