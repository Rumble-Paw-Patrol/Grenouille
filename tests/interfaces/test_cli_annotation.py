"""Commandes d'annotation en ligne de commande, sur une base minimale (sans audio)."""

import pandas as pd
import yaml
from typer.testing import CliRunner

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
