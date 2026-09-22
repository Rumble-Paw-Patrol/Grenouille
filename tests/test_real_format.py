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
from blanci.ingest import iter_audio_files, parse_songmeter_name, start_utc
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
    """Un « peut-être » ne doit pas devenir un négatif en silence."""
    con, names = ingested
    sheet = write_sheet(
        tmp_path / "a.xlsx",
        [(names[0], 36, 0.91, "oui"), (names[1], 12, 0.83, "peut-être")],
    )
    report = import_label_file(con, sheet, cfg, kind=None)
    assert report.inserted == 0
    assert len(report.unresolved) == 1
    assert "vérification illisible" in report.unresolved[0][1]
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 0


@pytest.mark.parametrize("pending", ["à vérif", "à conf", "à revoir", "À vérifier écouteurs"])
def test_pending_verdict_is_set_aside_without_blocking(tmp_path, cfg, ingested, pending):
    """« à vérif » : l'expert n'a pas tranché. Pas de label inventé, pas de blocage."""
    con, names = ingested
    sheet = write_sheet(
        tmp_path / "a.xlsx",
        [(names[0], 36, 0.91, "oui"), (names[1], 12, 0.83, pending)],
    )
    report = import_label_file(con, sheet, cfg, kind=None)
    assert report.inserted == 1 and not report.unresolved
    assert len(report.pending) == 1 and report.pending[0][0] == 3
    assert "attente" in report.summary()


def test_unverified_detections_are_not_labels(tmp_path, cfg, ingested):
    """Le fichier de l'ancien prestataire liste toutes ses détections ; seules les vérifiées
    sont des labels. Une cellule de vérification vide n'est ni positive ni négative."""
    con, names = ingested
    sheet = write_sheet(
        tmp_path / "a.xlsx",
        [
            (names[0], 36, 0.91, True),
            (names[0], 39, 0.35, None),
            (names[1], 12, 0.12, None),
            (names[2], 3, 0.83, False),
        ],
    )
    report = import_label_file(con, sheet, cfg, kind=None)
    assert report.inserted == 2 and report.unverified == 2 and not report.unresolved
    assert "jamais écoutées" in report.summary()


def test_boolean_verdicts_from_excel(tmp_path, cfg, ingested):
    """Le vrai fichier range True / False en booléens Excel."""
    con, names = ingested
    sheet = write_sheet(
        tmp_path / "a.xlsx", [(names[0], 36, 0.91, True), (names[1], 12, 0.83, False)]
    )
    import_label_file(con, sheet, cfg, kind=None)
    labels = [r["label"] for r in con.execute("SELECT label FROM labels ORDER BY label_id")]
    assert labels[0].startswith("blanci") and not labels[1].startswith("blanci")


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


def test_same_recording_in_two_formats_is_inventoried_once(tmp_path, cfg):
    """Un même nom en .wav et en .flac est un seul enregistrement : l'autre est un doublon."""
    raw = tmp_path / "raw"
    write_recording(raw / "2026" / "mataroni" / f"{STEMS[0]}.wav", seed=0)
    write_recording(raw / "2026" / "mataroni" / f"{STEMS[0]}.flac", seed=0)
    cfg["paths"]["raw"] = str(raw)
    con = connect(cfg["paths"]["db"])
    report = run_ingest(con, raw, "2026", cfg, hash_file=False)
    assert report.added == 1 and len(report.duplicates) == 1

    # L'extension citée n'a alors plus d'importance : il n'y a qu'un candidat.
    for cited in (f"{STEMS[0]}.wav", f"{STEMS[0]}.flac", STEMS[0]):
        sheet = write_sheet(tmp_path / f"{cited}.xlsx", [(cited, 36, 0.9, "oui")])
        imported = import_label_file(con, sheet, cfg, kind=None, dry_run=True)
        assert len(imported.rows) == 1 and not imported.unresolved, cited


def test_cited_extension_breaks_a_tie_in_an_older_database(tmp_path, cfg, ingested):
    """Base constituée avant la détection des doublons : deux formats pour un même nom.
    L'extension citée départage ; sans elle, la ligne est ambiguë et signalée."""
    con, _ = ingested
    original = con.execute(
        "SELECT * FROM recordings WHERE path LIKE ?", (f"%{STEMS[0]}.wav",)
    ).fetchone()
    flac_path = original["path"][: -len(".wav")] + ".flac"
    con.execute(
        "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
        "duration_s, sample_rate, channels) VALUES ('copie', ?, '2026', ?, ?, ?, ?, ?, 1)",
        (
            flac_path,
            original["site"],
            original["mic_id"],
            original["start_utc"],
            original["duration_s"],
            original["sample_rate"],
        ),
    )
    con.commit()

    cited = write_sheet(tmp_path / "a.xlsx", [(f"{STEMS[0]}.flac", 36, 0.9, "oui")])
    report = import_label_file(con, cited, cfg, kind=None, dry_run=True)
    assert len(report.rows) == 1 and report.rows[0]["recording_id"] == "copie"

    bare = write_sheet(tmp_path / "b.xlsx", [(STEMS[0], 36, 0.9, "oui")])
    report = import_label_file(con, bare, cfg, kind=None, dry_run=True)
    assert "ambigu" in report.unresolved[0][1]


# --- Fichier de l'ancien prestataire : clés S3, commentaires, doublons de relevés --------


def test_s3_key_matches_the_file_on_disk(tmp_path, cfg, ingested):
    """« 2353462-2la04530_20260106_103000.flac » désigne 2la04530_20260106_103000.wav."""
    con, names = ingested
    sheet = write_sheet(tmp_path / "a.xlsx", [(f"2353462-{names[0]}", 36, 0.9, True)])
    report = import_label_file(con, sheet, cfg, kind=None)
    assert report.inserted == 1 and not report.unresolved


def test_file_key_strips_only_an_s3_prefix():
    from blanci.labels import file_key

    assert file_key("2353462-2la03550_20260108_143000.flac") == "2la03550_20260108_143000"
    assert file_key("D:/x/2LA03550_20260108_143000.wav") == "2la03550_20260108_143000"
    assert file_key("12-notes.wav") == "12-notes"  # pas un nom Song Meter : intact


def provider_sheet(path, rows):
    """Colonnes du fichier Blancinet réel, commentaires dans des colonnes sans nom."""
    pd.DataFrame(
        rows,
        columns=[
            "file_s3_key",
            "station",
            "label_name",
            "start_time",
            "end_time",
            "score",
            "vérification",
            "Colonne1",
            "Unnamed: 12",
        ],
    ).to_excel(path, index=False)
    return path


def test_provider_sheet_columns_are_detected(tmp_path, cfg, ingested):
    con, names = ingested
    sheet = provider_sheet(
        tmp_path / "blancinet.xlsx",
        [(f"1-{names[0]}", "Mataroni_crique2_RB04", "ANOBLA", 36, 39, 0.9, True, None, None)],
    )
    report = import_label_file(con, sheet, cfg, kind=None, dry_run=True)
    assert report.columns["file"] == "file_s3_key"
    assert report.columns["offset_s"] == "start_time"
    assert report.columns["verdict"] == "vérification"
    assert report.columns["score"] == "score"
    assert len(report.rows) == 1 and report.rows[0]["offset_s"] == 36.0


def test_anonymous_columns_feed_the_comment(tmp_path, cfg, ingested):
    """Les commentaires du vrai fichier sont dans « Colonne1 » et « Unnamed: 12 »."""
    con, names = ingested
    sheet = provider_sheet(
        tmp_path / "blancinet.xlsx",
        [
            (f"1-{names[2]}", "st", "ANOBLA", 3, 6, 0.8, False, "oiseau Fourmilier tacheté", None),
            (f"2-{names[1]}", "st", "ANOBLA", 9, 12, 0.7, False, None, "A. andreae"),
            (
                f"3-{names[0]}",
                "st",
                "ANOBLA",
                36,
                39,
                0.9,
                True,
                "chants audibles en second plan",
                "Colonne1",
            ),
        ],
    )
    report = import_label_file(con, sheet, cfg, kind=None, dry_run=True)
    by_offset = {r["offset_s"]: r for r in report.rows}
    assert by_offset[3.0]["label"] == "bird" and "ourmilier" in by_offset[3.0]["species"]
    assert by_offset[9.0]["label"] == "amphibian" and by_offset[9.0]["species"] == (
        "Adenomera andreae"
    )
    positive = by_offset[36.0]
    assert positive["quality"] == "C"  # « second plan »
    assert "Colonne1" not in positive["comment"]  # en-tête recopié dans une cellule, écarté


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
    from blanci.db import recording_id_for

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
    sheet = write_sheet(tmp_path / "a.xlsx", [(f"{STEMS[0]}.flac", 36, 0.9, True)])
    assert import_label_file(con, sheet, cfg, kind=None).inserted == 1
    before = con.execute("SELECT recording_id, qc_flags FROM recordings").fetchone()

    # Nouveau disque, arborescence <jeu>/<site>/<micro>/ ; l'ancien n'est plus branché.
    new = new_disk / "2026" / "Mataroni" / "2LA04530"
    write_recording(new / f"{STEMS[0].upper()}.wav", seed=0)
    report = run_ingest(con, new_disk, "2026", cfg, run_qc=False, hash_file=False)
    assert report.relocated == 1 and report.added == 0 and not report.duplicates

    row = con.execute("SELECT * FROM recordings").fetchone()
    assert row["recording_id"] == before["recording_id"]
    assert row["path"] == f"2026/Mataroni/2LA04530/{STEMS[0].upper()}.wav"
    assert row["qc_flags"] == before["qc_flags"]  # QC déjà calculé : pas effacé
    labelled = con.execute(
        "SELECT COUNT(*) FROM labels l JOIN windows w USING (window_id) "
        "JOIN recordings r USING (recording_id)"
    ).fetchone()[0]
    assert labelled == 1
