import json
import sqlite3
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from blanci.cli import app
from blanci.db import connect, recording_id_for
from blanci.ingest import ingest, parse_songmeter_name, read_guano
from blanci.labels import import_label_file
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
    assert (first["dataset"], first["site"], first["mic_id"]) == ("2026", "Mataroni", "M01")
    assert first["start_utc"] == "2026-02-12T10:00:00Z"  # GUANO, UTC−3
    assert first["duration_s"] == 120.0 and first["sample_rate"] == 16000
    assert len(first["sha256"]) == 64
    assert set(json.loads(first["qc_flags"])) >= {"rain", "saturation", "in_bag", "silent"}
    # Sans GUANO : horodatage du nom de fichier + décalage configuré (UTC−3).
    assert rows["SMM01_20260212_073000.wav"]["start_utc"] == "2026-02-12T10:30:00Z"
    # Extension en majuscules acceptée ; micro pris dans GUANO si le dossier micro manque.
    assert rows["SMM02_20260212_070000.WAV"]["mic_id"] == "M02"
    assert rows["SMM03_20260213_150000.wav"]["mic_id"] == "SMM03"

    again = ingest(con, raw, "2026", cfg)
    assert again.added == 0 and again.skipped == 4


def write_csv(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_import_labels_is_atomic_idempotent_and_append_only(raw, cfg, tmp_path):
    con = connect(Path(cfg["paths"]["db"]))
    ingest(con, raw, "2026", cfg, run_qc=False)
    positives = write_csv(
        tmp_path / "positifs.csv",
        [
            "Fichier;Début (s);Commentaire",
            "SMM01_20260212_070000.wav;36;chants audibles en second plan",
            "SMM01_20260212_073000;01:30;",
        ],
    )
    report = import_label_file(con, positives, cfg, kind="positive")
    assert report.inserted == 2 and not report.unresolved
    labels = con.execute(
        "SELECT l.label, l.quality, w.offset_s, w.dur_s FROM labels l "
        "JOIN windows w USING (window_id) ORDER BY w.offset_s"
    ).fetchall()
    assert [tuple(r) for r in labels] == [("blanci", "C", 36.0, 3.0), ("blanci", "A", 90.0, 3.0)]

    assert import_label_file(con, positives, cfg, kind="positive").already_imported
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 2

    broken = write_csv(
        tmp_path / "faux_amis.csv",
        [
            "fichier,debut,commentaire",
            "SMM01_20260212_070000.wav,3,oiseau Fourmilier tacheté",
            "INCONNU_20260101_000000.wav,0,micro dans sac",
        ],
    )
    report = import_label_file(con, broken, cfg, kind="negative")
    assert report.inserted == 0 and len(report.unresolved) == 1
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 2

    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE labels SET label = 'bird'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM labels")


def test_cli_end_to_end(raw, cfg, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"paths": {**cfg["paths"], "raw": str(raw)}}), encoding="utf-8"
    )
    positives = write_csv(
        tmp_path / "positifs.csv",
        ["fichier;debut;commentaire", "SMM01_20260212_070000.wav;117;chant lointain"],
    )
    runner = CliRunner()
    result = runner.invoke(app, ["--config", str(config), "ingest", "--dataset", "2026"])
    assert result.exit_code == 0, result.output
    assert "4 ajoutés" in result.output and "1 erreurs" in result.output

    result = runner.invoke(
        app, ["-c", str(config), "import-labels", str(positives), "--kind", "positive"]
    )
    assert result.exit_code == 0, result.output
    assert "1 labels ajoutés" in result.output

    result = runner.invoke(app, ["-c", str(config), "check-grid"])
    assert result.exit_code == 0, result.output
    assert "hors de toute fenêtre : 0/1" in result.output

    result = runner.invoke(app, ["-c", str(config), "status"])
    assert result.exit_code == 0, result.output
    assert "2026 / Mataroni : 4, 3 micros" in result.output
