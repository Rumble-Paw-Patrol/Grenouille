"""Le format réel des données ONF : enregistrements de 2 min, arborescence des disques.

Nom de fichier : `2la04530_20260106_103000` — micro 2LA04530, 6 janvier 2026 à 10 h 30.
Un même nom en `.wav` et en `.flac` est un seul enregistrement.
"""

import datetime as dt
import json

import numpy as np
import pytest
import soundfile as sf

from blanci.core.db import connect, window_id_for
from blanci.inputs.ingest import ingest as run_ingest
from blanci.inputs.ingest import iter_audio_files, parse_songmeter_name, start_utc
from blanci.service import append_label

SR = 32000
DURATION_S = 120.0
WINDOW_S = 3.0


def write_recording(path, seed=0, tone_hz=None):
    """Enregistrement de 2 min, large bande (un sinus pur serait classé « micro dans sac »).

    Le format est déduit de l'extension : soundfile écrit du WAV ou du FLAC indifféremment.
    """
    rng = np.random.default_rng(seed)
    n = int(SR * DURATION_S)
    x = rng.normal(0, 0.05, n)
    if tone_hz is not None:
        x += 0.2 * np.sin(2 * np.pi * tone_hz * np.arange(n) / SR)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x.astype(np.float32), SR, subtype="PCM_16")
    return path


STEMS = [
    "2la04530_20260106_103000",
    "2la04530_20260106_110000",
    "2la04531_20260106_103000",
]


@pytest.fixture
def onf_corpus(tmp_path, cfg):
    """Deux micros, deux créneaux : l'arborescence réelle du disque externe."""
    raw = tmp_path / "data" / "raw"
    for seed, stem in enumerate(STEMS):
        write_recording(raw / "2026" / "mataroni" / f"{stem}.wav", seed=seed, tone_hz=4750.0)
    cfg["paths"]["raw"] = str(raw)
    return raw, [f"{stem}.flac" for stem in STEMS]


# --- Nommage et inventaire ---------------------------------------------------------------


def test_songmeter_name_parses_the_real_pattern():
    parsed = parse_songmeter_name("2la04530_20260106_103000")
    assert parsed is not None
    prefix, stamp = parsed
    assert prefix == "2la04530"
    assert stamp == dt.datetime(2026, 1, 6, 10, 30, 0)


def test_both_formats_are_scanned(tmp_path):
    write_recording(tmp_path / "a.flac")
    write_recording(tmp_path / "b.wav")
    (tmp_path / "c.txt").write_text("non", encoding="utf-8")
    (tmp_path / "._a.flac").write_bytes(b"\x00")  # AppleDouble du disque externe
    assert [p.name for p in iter_audio_files(tmp_path)] == ["a.flac", "b.wav"]


def test_ingest_reads_the_corpus(tmp_path, cfg, onf_corpus):
    raw, _ = onf_corpus
    con = connect(cfg["paths"]["db"])
    report = run_ingest(con, raw, "2026", cfg, hash_file=False)
    assert report.added == 3 and not report.errors

    rows = {r["path"].rsplit("/", 1)[-1]: r for r in con.execute("SELECT * FROM recordings")}
    first = rows["2la04530_20260106_103000.wav"]
    assert first["sample_rate"] == SR
    assert first["duration_s"] == pytest.approx(DURATION_S)
    assert first["mic_id"] == "2la04530"  # préfixe du nom, faute de dossier micro
    assert first["site"] == "mataroni"


def test_ingest_mixes_wav_and_flac(tmp_path, cfg):
    """La base peut contenir les deux formats côte à côte."""
    raw = tmp_path / "raw"
    write_recording(raw / "2026" / "mataroni" / "2la04530_20260106_103000.wav")
    write_recording(raw / "2026" / "mataroni" / "2la04531_20260106_103000.flac")
    con = connect(cfg["paths"]["db"])
    report = run_ingest(con, raw, "2026", cfg, hash_file=False)
    assert report.added == 2 and not report.errors
    suffixes = {r["path"][-5:] for r in con.execute("SELECT path FROM recordings")}
    assert suffixes == {"0.wav", ".flac"}


@pytest.mark.parametrize(
    "guano, stem, expected",
    [
        # Chaînes relevées sur le disque : décalage sans zéro devant l'heure.
        ("2025-12-27 15:30:00-3:00", "2LA04186_20251227_153000", "2025-12-27T18:30:00Z"),
        # Enregistreur réglé en UTC : le nom de fichier est aussi en UTC. Avant correction,
        # le repli sur « nom + UTC−3 » donnait 15:10Z, soit 3 h d'erreur sans avertissement.
        ("2024-10-22 12:10:16+0:00", "2LA02707_20241022_121016", "2024-10-22T12:10:16Z"),
        ("2024-02-20 14:30:00-3:00", "SMA14163_20240220_143000", "2024-02-20T17:30:00Z"),
        ("2026-02-12T07:00:00-03:00", "SMM01_20260212_070000", "2026-02-12T10:00:00Z"),
        ("2026-01-06 10:30:00+10:30", "X_20260106_103000", "2026-01-06T00:00:00Z"),
    ],
)
def test_guano_timestamp_keeps_the_recorder_timezone(guano, stem, expected):
    assert start_utc({"Timestamp": guano}, stem, -3) == expected


def test_unreadable_guano_falls_back_to_the_filename():
    assert start_utc({"Timestamp": "pas une date"}, "X_20260106_103000", -3) == (
        "2026-01-06T13:30:00Z"
    )


def test_ingest_dates_from_the_filename(tmp_path, cfg, onf_corpus):
    """Horodatage tiré du nom (le FLAC n'a pas de bloc GUANO), en heure locale UTC−3."""
    raw, _ = onf_corpus
    con = connect(cfg["paths"]["db"])
    run_ingest(con, raw, "2026", cfg, hash_file=False)
    row = con.execute(
        "SELECT start_utc FROM recordings WHERE path LIKE '%2la04530_20260106_103000%'"
    ).fetchone()
    assert row["start_utc"] == "2026-01-06T13:30:00Z"  # 10 h 30 locales


def test_ingest_separates_the_two_mics(tmp_path, cfg, onf_corpus):
    raw, _ = onf_corpus
    con = connect(cfg["paths"]["db"])
    run_ingest(con, raw, "2026", cfg, hash_file=False)
    mics = {r["mic_id"] for r in con.execute("SELECT DISTINCT mic_id FROM recordings")}
    assert mics == {"2la04530", "2la04531"}


# --- Un même nom en deux formats -----------------------------------------------------------


def test_same_recording_in_two_formats_is_inventoried_once(tmp_path, cfg):
    """Un même nom en .wav et en .flac est un seul enregistrement : l'autre est un doublon."""
    raw = tmp_path / "raw"
    write_recording(raw / "2026" / "mataroni" / f"{STEMS[0]}.wav", seed=0)
    write_recording(raw / "2026" / "mataroni" / f"{STEMS[0]}.flac", seed=0)
    cfg["paths"]["raw"] = str(raw)
    con = connect(cfg["paths"]["db"])
    report = run_ingest(con, raw, "2026", cfg, hash_file=False)
    assert report.added == 1 and len(report.duplicates) == 1


# --- Doublons de relevés -------------------------------------------------------------------


def test_sd_card_leftovers_are_inventoried_once(tmp_path, cfg):
    """Carte SD rapportée à Mataroni avec les fichiers de CDR : copies identiques écartées,
    le relevé inventorié en premier (le plus ancien) garde le fichier et son site."""
    raw = tmp_path / "raw"
    releve_1 = raw / "Projet" / "RELEVE 1 CDR" / "2LA03021" / "Data"
    releve_3 = raw / "Projet" / "RELEVE 3 Mataroni" / "2LA03021_GI18" / "Data"
    write_recording(releve_1 / "2LA03021_20251221_093000.wav", seed=0)
    write_recording(releve_3 / "2LA03021_20251221_093000.wav", seed=0)  # reste de carte SD
    write_recording(releve_3 / "2LA03021_20260108_073000.wav", seed=1)
    con = connect(cfg["paths"]["db"])

    first = run_ingest(con, raw, "2026", cfg, hash_file=False, scan=releve_1.parents[1], site="CDR")
    second = run_ingest(
        con, raw, "2026", cfg, hash_file=False, scan=releve_3.parents[1], site="Mataroni"
    )
    assert first.added == 1 and second.added == 1 and len(second.duplicates) == 1
    sites = dict(con.execute("SELECT substr(path, -28, 24), site FROM recordings").fetchall())
    assert sites == {
        "2LA03021_20251221_093000": "CDR",
        "2LA03021_20260108_073000": "Mataroni",
    }
    mics = {r[0] for r in con.execute("SELECT DISTINCT mic_id FROM recordings")}
    assert mics == {"2LA03021"}  # numéro de série, pas « 2LA03021_GI18 » ni « Data »


def test_scan_folder_must_be_under_the_raw_root(tmp_path, cfg):
    elsewhere = tmp_path / "ailleurs"
    write_recording(elsewhere / "2LA03021_20251221_093000.wav")
    con = connect(cfg["paths"]["db"])
    with pytest.raises(ValueError, match="n'est pas sous la racine"):
        run_ingest(con, tmp_path / "raw", "2026", cfg, scan=elsewhere)


def test_each_survey_keeps_its_own_ingest_report(tmp_path):
    """Deux relevés du même jeu : le rapport du second n'écrase pas celui du premier."""
    import yaml
    from typer.testing import CliRunner

    from blanci.cli import app

    raw = tmp_path / "raw"
    cdr = raw / "RELEVE 1 CDR" / "2LA03021" / "Data"
    mataroni = raw / "RELEVE 3 Mataroni" / "2LA03021_GI18" / "Data"
    write_recording(cdr / "2LA03021_20251221_093000.wav")
    (cdr / "2LA03021_20251221_100000.wav").write_bytes(b"")  # fichier vide, comme SMA14826
    write_recording(mataroni / "2LA03021_20251221_093000.wav")  # reste de carte SD
    config = tmp_path / "local.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "raw": str(raw),
                    "db": str(tmp_path / "db.sqlite"),
                    "reports": str(tmp_path / "reports"),
                }
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    for site, folder in (("CDR", cdr.parents[1]), ("Mataroni", mataroni.parents[1])):
        result = runner.invoke(
            app,
            [
                "--config",
                str(config),
                "ingest",
                "--dataset",
                "2026",
                "--site",
                site,
                "--no-qc",
                "--no-hash",
                str(folder),
            ],
        )
        assert result.exit_code == 0, result.output
    reports = sorted(p.name for p in (tmp_path / "reports").iterdir())
    assert reports == ["ingest_duplicates_2026_Mataroni.csv", "ingest_errors_2026_CDR.csv"]


# --- Copie sur un autre disque ------------------------------------------------------------


def test_recording_id_ignores_folder_and_extension():
    """L'identité d'un enregistrement est son nom : copie, réorganisation, WAV ↔ FLAC."""
    from blanci.core.db import recording_id_for

    base = recording_id_for(
        "Projet blanci 2025/RELEVE 3 Mataroni/2LA04530_MGM06/Data/2LA04530_20260106_103000.wav"
    )
    assert recording_id_for("2026/Mataroni/2LA04530/2LA04530_20260106_103000.flac") == base
    assert recording_id_for("2la04530_20260106_103000") == base


def test_moving_to_a_new_disk_keeps_labels(tmp_path, cfg):
    """Copie sur un nouveau disque, dossiers réorganisés : l'inventaire suit les fichiers,
    les labels restent attachés, rien n'est compté en doublon."""
    old_disk, new_disk = tmp_path / "ancien", tmp_path / "nouveau"
    old = old_disk / "RELEVE 3 Mataroni - 06-13 janv 2026" / "2LA04530_MGM06" / "Data"
    write_recording(old / f"{STEMS[0].upper()}.wav", seed=0)
    con = connect(cfg["paths"]["db"])
    run_ingest(con, old_disk, "2026", cfg, hash_file=False, scan=old_disk, site="Mataroni")
    (rid,) = con.execute("SELECT recording_id FROM recordings").fetchone()
    wid = window_id_for(rid, 36.0, WINDOW_S)
    con.execute(
        "INSERT INTO windows (window_id, recording_id, offset_s, dur_s) VALUES (?, ?, 36, ?)",
        (wid, rid, WINDOW_S),
    )
    append_label(con, wid, "blanci", "random")
    before = con.execute("SELECT recording_id, qc_flags FROM recordings").fetchone()

    # Nouveau disque, arborescence <jeu>/<site>/<micro>/ ; l'ancien n'est plus branché.
    new = new_disk / "2026" / "Mataroni" / "2LA04530"
    write_recording(new / f"{STEMS[0].upper()}.wav", seed=0)
    report = run_ingest(con, new_disk, "2026", cfg, run_qc=False, hash_file=False)
    assert report.relocated == 1 and report.added == 0 and not report.duplicates

    row = con.execute("SELECT * FROM recordings").fetchone()
    assert row["recording_id"] == before["recording_id"]
    assert row["path"] == f"2026/Mataroni/2LA04530/{STEMS[0].upper()}.wav"
    assert json.loads(row["qc_flags"]) == json.loads(before["qc_flags"])  # QC gardé
    labelled = con.execute(
        "SELECT COUNT(*) FROM labels l JOIN windows w USING (window_id) "
        "JOIN recordings r USING (recording_id)"
    ).fetchone()[0]
    assert labelled == 1


def test_a_leftover_copy_on_another_disk_is_not_moved_to_another_site(tmp_path, cfg):
    """Relevé 1 inventorié depuis le disque A ; le disque B ne porte que le relevé 2, dont les
    cartes SD contiennent encore les fichiers du relevé 1. Ces copies ne sont pas un
    déplacement : le fichier garde son site, la copie est signalée en doublon (n° 144)."""
    disk_a, disk_b = tmp_path / "A", tmp_path / "B"
    write_recording(disk_a / "RELEVE 1" / "2LA04530" / f"{STEMS[0].upper()}.wav", seed=0)
    con = connect(cfg["paths"]["db"])
    run_ingest(con, disk_a, "2026", cfg, hash_file=False, scan=disk_a, site="Mataroni")
    write_recording(disk_b / "RELEVE 2" / "2LA04530" / f"{STEMS[0].upper()}.wav", seed=0)
    report = run_ingest(
        con, disk_b, "2026", cfg, run_qc=False, hash_file=False, scan=disk_b, site="CDR"
    )
    assert report.relocated == 0 and report.added == 0 and len(report.duplicates) == 1
    row = con.execute("SELECT site, path FROM recordings").fetchone()
    assert row["site"] == "Mataroni" and row["path"].startswith("RELEVE 1/")


def test_surveys_are_scanned_in_natural_order():
    from pathlib import Path

    from blanci.inputs.ingest import natural_key

    names = ["RELEVE 10 X/a.wav", "RELEVE 2 Y/a.wav", "RELEVE 1 Z/b.wav"]
    ordered = sorted((Path(n) for n in names), key=natural_key)
    assert [p.parts[0] for p in ordered] == ["RELEVE 1 Z", "RELEVE 2 Y", "RELEVE 10 X"]
