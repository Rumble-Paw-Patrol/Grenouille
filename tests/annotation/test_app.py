"""Poste d'annotation Streamlit : l'écran s'affiche et un clic ajoute un label."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from blanci.core.db import connect  # noqa: E402
from blanci.inputs.ingest import ingest  # noqa: E402

pytestmark = pytest.mark.slow  # pytest -m 'not slow' : suite rapide

# Chemin absolu : Streamlit ≥ 1.5x résout un chemin relatif depuis le fichier de test.
APP = str(Path(__file__).resolve().parents[2] / "blanci" / "annotation" / "app.py")

SR = 16_000


@pytest.fixture
def n_candidates():
    return 1


@pytest.fixture
def app_config(tmp_path, cfg, monkeypatch, n_candidates):
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
            "recording_id": [rid] * n_candidates,
            "offset_s": [3.0 * (k + 1) for k in range(n_candidates)],
            "dur_s": [3.0] * n_candidates,
            "reason": ["random"] * n_candidates,
            "source": ["random"] * n_candidates,
        }
    ).to_csv(reports / "candidats_test.csv", index=False)
    monkeypatch.setenv("BLANCI_CONFIG", str(config))
    return cfg


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _tick(at, name):
    next(c for c in at.checkbox if c.label == name).check()


def _labels(cfg):
    con = connect(cfg["paths"]["db"])
    return con.execute(
        "SELECT w.offset_s, l.label, l.conditions FROM labels l JOIN windows w USING (window_id) "
        "ORDER BY l.label_id"
    ).fetchall()


def _caption(at):
    return next(c.value for c in at.caption if c.value.startswith("Candidat "))


def test_app_shows_a_candidate_and_saves_an_answer(app_config):
    """Classe, espèce, commentaire, puis « Envoyer » : un label, et la fenêtre suivante."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "CDR" in at.subheader[0].value
    at.sidebar.text_input[0].input("léonard").run()
    _tick(at, "oiseau")
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


@pytest.mark.parametrize("n_candidates", [3])
def test_numbering_follows_the_whole_queue_and_going_back_works(app_config):
    """Après « 1 / 3 » vient « 2 / 3 » (et non « 1 / 2 ») ; on revient sur un candidat
    déjà écouté, par ◀ ou par la liste des candidats, et on le corrige."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].input("léonard").run()
    assert _caption(at).startswith("Candidat 1 / 3")
    _button(at, "Envoyer ▶").click().run()
    assert not at.exception
    assert _caption(at).startswith("Candidat 2 / 3")
    _button(at, "◀ Candidat précédent").click().run()
    assert _caption(at).startswith("Candidat 1 / 3")
    assert "déjà écouté : rien" in _caption(at)
    _tick(at, "oiseau")
    _button(at, "Envoyer ▶").click().run()
    assert _caption(at).startswith("Candidat 2 / 3")  # le prochain jamais écouté
    strip = at.sidebar.selectbox(key=next(k for k in at.session_state if k.startswith("strip::")))
    strip.set_value(2).run()
    assert _caption(at).startswith("Candidat 3 / 3")
    assert [(o, label) for o, label, _ in _labels(app_config)] == [
        (3.0, "background"),
        (3.0, "bird"),
    ]


def test_several_classes_are_saved_together(app_config):
    """A. blanci et pluie cochés : label A. blanci, la pluie dans `extra_labels`."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].input("léonard").run()
    _tick(at, "pluie")
    _tick(at, "A. blanci")
    _button(at, "Envoyer ▶").click().run()
    assert not at.exception
    [(_, label, conditions)] = _labels(app_config)
    assert label == "blanci"
    assert json.loads(conditions)["extra_labels"] == ["rain"]


def test_a_recording_is_split_into_windows_annotated_on_the_same_page(app_config):
    """Enregistrement entier (12 s) découpé en fenêtres de 3 s, calées sur le candidat (3 s) :
    4 fenêtres, la première annotée est celle du candidat, puis la suivante, sans changer
    de candidat."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].input("léonard").run()
    next(r for r in at.sidebar.radio if r.label == "Étendue affichée").set_value("whole").run()
    next(c for c in at.sidebar.checkbox if c.label == "Découper en fenêtres").check().run()
    assert not at.exception
    window = next(s for s in at.selectbox if s.label == "Fenêtre")
    assert len(window.options) == 4
    assert window.value == 1
    _button(at, "Envoyer ▶").click().run()
    _tick(at, "oiseau")
    _button(at, "Envoyer ▶").click().run()
    assert not at.exception
    assert _caption(at).startswith("Candidat 1 / 1")
    assert [(o, label) for o, label, _ in _labels(app_config)] == [
        (3.0, "background"),
        (6.0, "bird"),
    ]
