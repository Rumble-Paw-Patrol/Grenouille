"""Commandes d'annotation en ligne de commande, sur une base minimale (sans audio)."""

import pandas as pd
import yaml
from typer.testing import CliRunner

from blanci.annotation.workbench import progress
from blanci.cli import app
from blanci.core.db import connect, recording_id_for, window_id_for
from blanci.service import append_label, append_span


def test_export_labels_says_who_annotated(tmp_path):
    """Fenêtres et intervalles exportés gardent l'annotateur."""
    db = tmp_path / "blanci.sqlite"
    con = connect(db)
    rel = "2026/CDR/C1/C1_20260210_070000.wav"
    rid = recording_id_for(rel)
    con.execute(
        "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
        "duration_s, sample_rate, channels, qc_flags) "
        "VALUES (?, ?, '2026', 'CDR', 'C1', '2026-02-10T10:00:00Z', 120.0, 32000, 1, '{}')",
        (rid, rel),
    )
    wid = window_id_for(rid, 0.0)
    con.execute(
        "INSERT INTO windows (window_id, recording_id, offset_s, dur_s) VALUES (?, ?, 0, 3)",
        (wid, rid),
    )
    append_label(con, wid, "bird", "active", annotator="léonard")
    append_span(con, rid, 30.0, 60.0, [(40.0, 42.0, "blanci")], "background", "audit",
                annotator="tuteur")  # fmt: skip
    config = tmp_path / "blanci.yaml"
    paths = {"db": str(db), "reports": str(tmp_path / "reports")}
    config.write_text(yaml.safe_dump({"paths": paths}), encoding="utf-8")
    result = CliRunner().invoke(app, ["--config", str(config), "export-labels"])
    assert result.exit_code == 0, result.output
    reports = tmp_path / "reports"
    windows = pd.read_csv(reports / "fenetres_annotees.csv")
    intervals = pd.read_csv(reports / "intervalles_annotes.csv")
    assert windows["annotator"].tolist() == ["léonard"]
    assert intervals["annotator"].tolist() == ["tuteur"]


def _base(path, rel):
    con = connect(path)
    rid = recording_id_for(rel)
    con.execute(
        "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
        "duration_s, sample_rate, channels, qc_flags) "
        "VALUES (?, ?, '2026', 'CDR', 'C1', '2026-02-10T10:00:00Z', 120.0, 32000, 1, '{}')",
        (rid, rel),
    )
    con.commit()
    return con, rid


def test_import_labels_resumes_on_another_machine(tmp_path):
    """export-labels sur une machine, import-labels sur une autre : mêmes extraits, mêmes
    intervalles, même progression ; relancer n'ajoute rien."""
    rel = "2026/CDR/C1/C1_20260210_070000.wav"
    con, rid = _base(tmp_path / "a.sqlite", rel)
    wid = window_id_for(rid, 0.0, 120.0)
    con.execute(
        "INSERT INTO windows (window_id, recording_id, offset_s, dur_s) VALUES (?, ?, 0, 120)",
        (wid, rid),
    )
    append_label(con, wid, "amphibian", "flag", quality="B", annotator="Léonard")
    append_span(con, rid, 30.0, 60.0, [(40.0, 42.0, "blanci", "A")], "background", "plan",
                annotator="Léonard", conditions={"multiclass": False})  # fmt: skip
    config = tmp_path / "a.yaml"
    paths = {"db": str(tmp_path / "a.sqlite"), "reports": str(tmp_path / "export")}
    config.write_text(yaml.safe_dump({"paths": paths}), encoding="utf-8")
    assert CliRunner().invoke(app, ["--config", str(config), "export-labels"]).exit_code == 0

    other, _ = _base(tmp_path / "b.sqlite", "autre/disque/C1_20260210_070000.wav")
    other.close()
    config_b = tmp_path / "b.yaml"
    config_b.write_text(yaml.safe_dump({"paths": {"db": str(tmp_path / "b.sqlite")}}),
                        encoding="utf-8")  # fmt: skip
    args = ["--config", str(config_b), "import-labels", "--dossier", str(tmp_path / "export")]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "1 extraits (1 intervalles) et 1 fenêtres importés" in first.output
    again = CliRunner().invoke(app, args)
    assert "0 extraits (0 intervalles) et 0 fenêtres importés ; 2 déjà" in again.output

    b = connect(tmp_path / "b.sqlite")
    spans = b.execute("SELECT start_s, end_s, annotator, conditions FROM spans").fetchall()
    assert [tuple(s)[:3] for s in spans] == [(30.0, 60.0, "Léonard")]
    assert '"multiclass": false' in spans[0][3]
    assert [
        tuple(r) for r in b.execute("SELECT start_s, end_s, label, quality FROM intervals")
    ] == [(40.0, 42.0, "blanci", "A")]
    queue = pd.DataFrame({"recording_id": [rid, rid], "offset_s": [40.0, 90.0],
                          "dur_s": [3.0, 3.0]})  # fmt: skip
    assert progress(b, queue, "Léonard").tolist() == progress(con, queue, "Léonard").tolist()
