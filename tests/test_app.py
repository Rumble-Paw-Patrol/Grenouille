"""Poste d'annotation Streamlit : l'écran s'affiche et un clic ajoute un label."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from blanci.db import connect  # noqa: E402
from blanci.ingest import ingest  # noqa: E402

SR = 16_000


@pytest.fixture
def app_config(tmp_path, cfg, monkeypatch):
    raw = tmp_path / "raw"
    path = raw / "2026" / "CDR" / "C1" / "C1_20260210_070000.wav"
    path.parent.mkdir(parents=True)
    x = np.random.default_rng(0).normal(0, 0.02, SR * 12).astype(np.float32)
    sf.write(path, np.stack([x, 4 * x], axis=1), SR, subtype="PCM_16")
    cfg["paths"]["raw"] = str(raw)
    cfg["qc"]["expected_duration_s"] = 12.0
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    con = connect(cfg["paths"]["db"])
    ingest(con, raw, "2026", cfg, run_qc=False, hash_file=False)
    rid = con.execute("SELECT recording_id FROM recordings").fetchone()[0]
    reports = tmp_path / "data" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "recording_id": [rid],
            "offset_s": [3.0],
            "dur_s": [3.0],
            "reason": ["random"],
            "source": ["random"],
        }
    ).to_csv(reports / "candidats_test.csv", index=False)
    monkeypatch.setenv("BLANCI_CONFIG", str(config))
    return cfg


def test_app_shows_a_candidate_and_saves_an_answer(app_config):
    at = AppTest.from_file("blanci/app.py", default_timeout=60).run()
    assert not at.exception
    assert "CDR" in at.subheader[0].value
    at.sidebar.text_input[0].input("léonard").run()
    next(b for b in at.button if b.label == "rien").click().run()
    assert not at.exception
    con = connect(app_config["paths"]["db"])
    rows = con.execute("SELECT label, annotator, source FROM labels").fetchall()
    assert [tuple(r) for r in rows] == [("background", "léonard", "random")]
    assert any("File terminée" in s.value for s in at.success)
