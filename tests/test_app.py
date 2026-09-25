"""Poste d'annotation Streamlit : l'écran s'affiche et un clic ajoute un label."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from blanci.db import connect  # noqa: E402
from blanci.ingest import ingest  # noqa: E402

# Chemin absolu : Streamlit ≥ 1.5x résout un chemin relatif depuis le fichier de test.
APP = str(Path(__file__).resolve().parents[1] / "blanci" / "app.py")

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


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_app_shows_a_candidate_and_saves_an_answer(app_config):
    """Classe, espèce, commentaire, puis « Envoyer » : un label, et la fenêtre suivante."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "CDR" in at.subheader[0].value
    at.sidebar.text_input[0].input("léonard").run()
    next(r for r in at.radio if r.label == "Classe").set_value("bird")
    next(t for t in at.text_input if t.label.startswith("Espèce")).input("Fourmilier tacheté")
    next(t for t in at.text_input if t.label.startswith("Commentaire")).input("chant lointain")
    _button(at, "Envoyer ▶").click().run()
    assert not at.exception
    con = connect(app_config["paths"]["db"])
    rows = con.execute("SELECT label, annotator, source, species FROM labels").fetchall()
    assert [tuple(r) for r in rows] == [("bird", "léonard", "random", "Fourmilier tacheté")]
    assert any("File terminée" in s.value for s in at.success)


def test_sending_without_an_annotator_is_refused(app_config):
    at = AppTest.from_file(APP, default_timeout=60).run()
    _button(at, "Envoyer ▶").click().run()
    assert any("annotateur" in e.value for e in at.error)
    con = connect(app_config["paths"]["db"])
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 0


def test_a_queue_is_drawn_in_the_app_and_opened(app_config):
    """Mode de sélection « fenêtres au hasard » : la file est écrite puis ouverte sur place."""
    from pathlib import Path

    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox(key="mode").set_value("random").run()
    at.sidebar.number_input(key="n::random").set_value(1).run()
    _button(at, "Générer la file").click().run()
    assert not at.exception
    queues = list(Path(app_config["paths"]["reports"]).glob("candidats_random_*.csv"))
    assert len(queues) == 1
    assert at.sidebar.selectbox(key="mode").value == "existing"
    assert "CDR" in at.subheader[0].value


def test_modes_needing_an_encoder_say_so(app_config):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox(key="mode").set_value("map").run()
    assert not at.exception
    assert any("encodeur" in i.value.lower() for i in at.info)
