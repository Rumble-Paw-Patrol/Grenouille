"""Le format réel des données ONF : enregistrements de 2 min + Excel d'une ligne par fenêtre.

Nom de fichier : `2la04530_20260106_103000` — micro 2LA04530, 6 janvier 2026 à 10 h 30.
Colonnes reçues : nom de l'enregistrement, timecode, score de l'ancien prestataire,
vérification manuelle.

Le disque contient des `.wav`, mais l'Excel les cite en `.flac` : l'appariement se fait sur
le nom sans extension, et les deux formats peuvent coexister dans la base.
"""

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from blanci.db import connect
from blanci.ingest import ingest as run_ingest
from blanci.ingest import iter_audio_files, parse_songmeter_name
from blanci.labels import import_label_file, parse_offset, parse_verdict

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


# Sur le disque : des .wav. Dans l'Excel : les mêmes, cités en .flac.
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
    return raw, [f"{stem}.flac" for stem in STEMS]  # les noms tels que l'Excel les cite


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


# --- Verdict de la colonne « vérif manuelle » -----------------------------------------------


@pytest.mark.parametrize(
    "value", ["oui", "OUI", "Oui", "o", "yes", "x", "ok", "vrai", "confirmé", "valide", 1, True]
)
def test_verdict_yes(value):
    assert parse_verdict(value) is True


@pytest.mark.parametrize(
    "value", ["non", "NON", "n", "no", "faux", "ko", "rejeté", "invalide", 0, False]
)
def test_verdict_no(value):
    assert parse_verdict(value) is False


@pytest.mark.parametrize(
    "value, expected",
    [
        ("blanci", True),
        ("blanci lointain", True),
        ("blanci malgré la pluie", True),
        ("pas blanci", False),
        ("non blanci", False),
        ("aucun blanci", False),
    ],
)
def test_verdict_reads_a_written_answer(value, expected):
    assert parse_verdict(value) is expected


@pytest.mark.parametrize("value", [None, float("nan"), "", "à revoir", "?", 0.7])
def test_verdict_refuses_to_guess(value):
    """Mieux vaut signaler la ligne que de la ranger en négatif sans le dire."""
    assert parse_verdict(value) is None


def test_verdict_alone_does_not_read_a_species():
    """Nommer un faux ami n'est pas un oui/non : c'est l'import qui en tire un négatif."""
    assert parse_verdict("fourmilier tacheté") is None


# --- Timecode ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (36, 36.0),
        (36.5, 36.5),
        ("36", 36.0),
        ("36,5", 36.5),
        ("1:30", 90.0),
        ("00:01:30", 90.0),
        (dt.time(0, 1, 30), 90.0),
        (dt.timedelta(seconds=90), 90.0),
        (pd.Timedelta(seconds=90), 90.0),
        (dt.datetime(1900, 1, 1, 0, 1, 30), 90.0),
    ],
)
def test_parse_offset_handles_excel_shapes(value, expected):
    assert parse_offset(value, "seconds", WINDOW_S) == pytest.approx(expected)


def test_parse_offset_window_index():
    assert parse_offset(12, "window_index", WINDOW_S) == 36.0


# --- Import de la feuille réelle ------------------------------------------------------------


def write_sheet(path, rows):
    """Feuille au format reçu : nom, timecode, score du prestataire, vérif manuelle."""
    pd.DataFrame(
        rows, columns=["Nom de l'enregistrement", "Timecode", "Score", "Vérif manuelle"]
    ).to_excel(path, index=False)
    return path


@pytest.fixture
def ingested(tmp_path, cfg, onf_corpus):
    raw, names = onf_corpus
    con = connect(cfg["paths"]["db"])
    run_ingest(con, raw, "2026", cfg, hash_file=False)
    return con, names


def test_import_real_sheet(tmp_path, cfg, ingested):
    con, names = ingested
    sheet = write_sheet(
        tmp_path / "annotations.xlsx",
        [
            (names[0], 36, 0.91, "oui"),
            (names[0], 78, 0.44, "oui"),
            (names[1], 12, 0.83, "non"),
            (names[2], 105, 0.62, "fourmilier tacheté"),
        ],
    )
    report = import_label_file(con, sheet, cfg, kind=None)
    assert report.columns["file"].startswith("Nom")
    assert report.columns["offset_s"] == "Timecode"
    assert report.columns["verdict"] == "Vérif manuelle"
    assert report.columns["score"] == "Score"
    assert report.inserted == 4

    rows = con.execute(
        "SELECT l.label, l.species, l.quality, l.conditions, w.offset_s, w.dur_s "
        "FROM labels l JOIN windows w USING (window_id) ORDER BY w.offset_s"
    ).fetchall()
    assert [r["offset_s"] for r in rows] == [12.0, 36.0, 78.0, 105.0]
    assert all(r["dur_s"] == WINDOW_S for r in rows)


def test_import_keeps_the_previous_model_score(tmp_path, cfg, ingested):
    """Le score de l'ancien prestataire est le repère chiffré du §6 : on le garde."""
    con, names = ingested
    sheet = write_sheet(tmp_path / "a.xlsx", [(names[0], 36, 0.91, "oui")])
    import_label_file(con, sheet, cfg, kind=None)
    conditions = con.execute("SELECT conditions FROM labels").fetchone()["conditions"]
    assert json.loads(conditions)["previous_model_score"] == pytest.approx(0.91)


def test_import_labels_positives_and_negatives(tmp_path, cfg, ingested):
    con, names = ingested
    sheet = write_sheet(
        tmp_path / "a.xlsx",
        [(names[0], 36, 0.91, "oui"), (names[1], 12, 0.83, "non")],
    )
    import_label_file(con, sheet, cfg, kind=None)
    labels = [r["label"] for r in con.execute("SELECT label FROM labels ORDER BY label_id")]
    assert labels[0] in ("blanci", "blanci_solo", "blanci_chorus")
    assert labels[1] not in ("blanci", "blanci_solo", "blanci_chorus")


def test_verification_text_names_the_false_friend(tmp_path, cfg, ingested):
    """« fourmilier tacheté » dans la vérif : l'espèce doit ressortir en base."""
    con, names = ingested
    sheet = write_sheet(tmp_path / "a.xlsx", [(names[2], 105, 0.62, "fourmilier tacheté")])
    import_label_file(con, sheet, cfg, kind=None)
    row = con.execute("SELECT label, species FROM labels").fetchone()
    assert row["label"] == "bird"
    assert row["species"] and "ourmilier" in row["species"]


def test_unreadable_verdict_blocks_the_import(tmp_path, cfg, ingested):
    """Un « à revoir » ne doit pas devenir un négatif en silence."""
    con, names = ingested
    sheet = write_sheet(
        tmp_path / "a.xlsx",
        [(names[0], 36, 0.91, "oui"), (names[1], 12, 0.83, "à revoir")],
    )
    report = import_label_file(con, sheet, cfg, kind=None)
    assert report.inserted == 0
    assert len(report.unresolved) == 1
    assert "vérification illisible" in report.unresolved[0][1]
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 0


def test_kind_option_bypasses_the_verdict_column(tmp_path, cfg, ingested):
    """Fichier sans vérif exploitable : --kind positive tranche pour tout le fichier."""
    con, names = ingested
    sheet = write_sheet(tmp_path / "a.xlsx", [(names[0], 36, 0.91, "à revoir")])
    report = import_label_file(con, sheet, cfg, kind="positive")
    assert report.inserted == 1
    assert con.execute("SELECT label FROM labels").fetchone()["label"].startswith("blanci")


def test_timecode_beyond_the_recording_is_reported(tmp_path, cfg, ingested):
    con, names = ingested
    sheet = write_sheet(tmp_path / "a.xlsx", [(names[0], 119.5, 0.9, "oui")])
    report = import_label_file(con, sheet, cfg, kind=None)
    assert report.inserted == 0
    assert "hors de l'enregistrement" in report.unresolved[0][1]


def test_dry_run_reports_the_score_range(tmp_path, cfg, ingested):
    con, names = ingested
    sheet = write_sheet(
        tmp_path / "a.xlsx",
        [(names[0], 36, 0.91, "oui"), (names[0], 78, 0.44, "oui"), (names[1], 12, 0.6, "oui")],
    )
    report = import_label_file(con, sheet, cfg, kind=None, dry_run=True)
    summary = report.summary()
    assert "score de l'ancien modèle" in summary
    assert "0.440" in summary and "0.910" in summary
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 0


def test_unknown_recording_is_reported(tmp_path, cfg, ingested):
    con, _ = ingested
    sheet = write_sheet(tmp_path / "a.xlsx", [("2la99999_20260106_103000.flac", 36, 0.9, "oui")])
    report = import_label_file(con, sheet, cfg, kind=None)
    assert report.inserted == 0
    assert "introuvable" in report.unresolved[0][1]


# --- Extension citée ≠ extension sur le disque -------------------------------------------


def test_flac_in_the_sheet_matches_a_wav_on_disk(tmp_path, cfg, ingested):
    """Le cas réel : le disque contient des .wav, l'Excel les cite en .flac."""
    con, names = ingested
    assert names[0].endswith(".flac")
    stored = con.execute("SELECT path FROM recordings LIMIT 1").fetchone()["path"]
    assert stored.endswith(".wav")

    report = import_label_file(
        con, write_sheet(tmp_path / "a.xlsx", [(names[0], 36, 0.9, "oui")]), cfg, kind=None
    )
    assert report.inserted == 1 and not report.unresolved


def test_a_bare_name_without_extension_also_matches(tmp_path, cfg, ingested):
    con, _ = ingested
    sheet = write_sheet(tmp_path / "a.xlsx", [(STEMS[0], 36, 0.9, "oui")])
    report = import_label_file(con, sheet, cfg, kind=None)
    assert report.inserted == 1 and not report.unresolved


def test_a_full_path_in_the_sheet_also_matches(tmp_path, cfg, ingested):
    """Certains tableurs collent le chemin complet du disque externe."""
    con, _ = ingested
    sheet = write_sheet(
        tmp_path / "a.xlsx", [(f"E:\\audio\\2026\\{STEMS[0]}.flac", 36, 0.9, "oui")]
    )
    report = import_label_file(con, sheet, cfg, kind=None)
    assert report.inserted == 1 and not report.unresolved


def test_same_stem_in_both_formats_prefers_the_cited_extension(tmp_path, cfg):
    """Si les deux formats coexistent, l'extension citée départage au lieu d'être ambiguë."""
    raw = tmp_path / "raw"
    write_recording(raw / "2026" / "mataroni" / f"{STEMS[0]}.wav", seed=0)
    write_recording(raw / "2026" / "mataroni" / f"{STEMS[0]}.flac", seed=1)
    cfg["paths"]["raw"] = str(raw)
    con = connect(cfg["paths"]["db"])
    run_ingest(con, raw, "2026", cfg, hash_file=False)

    sheet = write_sheet(tmp_path / "a.xlsx", [(f"{STEMS[0]}.flac", 36, 0.9, "oui")])
    report = import_label_file(con, sheet, cfg, kind=None)
    assert report.inserted == 1 and not report.unresolved
    path = con.execute(
        "SELECT r.path FROM labels l JOIN windows w USING (window_id) "
        "JOIN recordings r USING (recording_id)"
    ).fetchone()["path"]
    assert path.endswith(".flac")


def test_same_stem_both_formats_without_extension_is_ambiguous(tmp_path, cfg):
    """Sans extension citée, deux candidats : la ligne est signalée, pas devinée."""
    raw = tmp_path / "raw"
    write_recording(raw / "2026" / "mataroni" / f"{STEMS[0]}.wav", seed=0)
    write_recording(raw / "2026" / "mataroni" / f"{STEMS[0]}.flac", seed=1)
    cfg["paths"]["raw"] = str(raw)
    con = connect(cfg["paths"]["db"])
    run_ingest(con, raw, "2026", cfg, hash_file=False)

    sheet = write_sheet(tmp_path / "a.xlsx", [(STEMS[0], 36, 0.9, "oui")])
    report = import_label_file(con, sheet, cfg, kind=None)
    assert report.inserted == 0
    assert "ambigu" in report.unresolved[0][1]
