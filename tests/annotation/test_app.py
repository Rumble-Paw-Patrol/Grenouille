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


def _window_mode(at):
    next(c for c in at.sidebar.checkbox if c.label.startswith("Annoter par fenêtres")).check()
    at.run()


def _one_window(at):
    """Fenêtres plus longues que l'enregistrement : le découpage se réduit à la fenêtre du
    candidat."""
    next(n for n in at.sidebar.number_input if n.label.startswith("Longueur")).set_value(30.0)
    at.run()


def _multiclass(at):
    next(c for c in at.sidebar.checkbox if c.label.startswith("Multi-classe")).check()
    at.run()


def _keys(at):
    """Clés de l'état de session : `filtered_state` jusqu'à Streamlit 1.59 (Windows, uv.lock),
    `at.session_state` lui-même ensuite (1.64 sous Linux), qui n'a plus `filtered_state`."""
    return getattr(at.session_state, "filtered_state", at.session_state)


def _draw(at, intervals):
    """Intervalles « tracés » : la valeur que renverrait le spectrogramme."""
    key = next(k for k in _keys(at) if k.startswith("iv::"))
    at.session_state[key] = intervals


def _spans(cfg):
    con = connect(cfg["paths"]["db"])
    spans = con.execute(
        "SELECT span_id, start_s, end_s, other_label, classes, annotator, source, species "
        "FROM spans ORDER BY span_id"
    ).fetchall()
    intervals = con.execute(
        "SELECT span_id, start_s, end_s, label FROM intervals ORDER BY interval_id"
    ).fetchall()
    return [tuple(r) for r in spans], [tuple(r) for r in intervals]


def _caption(at):
    return next(c.value for c in at.caption if c.value.startswith("Candidat "))


def test_intervals_are_drawn_on_the_whole_recording_and_saved(app_config):
    """Méthode par défaut : l'enregistrement entier, des intervalles tracés, les autres
    classes cochées, puis « Envoyer l'extrait » : un span et ses intervalles."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "CDR" in at.subheader[0].value
    at.sidebar.text_input[0].input("léonard").run()
    assert any("Aucun intervalle" in c.value for c in at.caption)
    _multiclass(at)
    _draw(at, [[4.0, 5.5, "blanci"], [8.0, 8.4, "blanci_uncertain"]])
    _tick(at, "oiseau")
    next(t for t in at.text_input if t.label.startswith("Espèce")).input("Fourmilier tacheté")
    _button(at, "Envoyer l'extrait ▶").click().run()
    assert not at.exception
    spans, intervals = _spans(app_config)
    [(span_id, start, end, other, classes, annotator, source, species)] = spans
    assert (start, end, other, annotator, source, species) == (
        0.0,
        12.0,
        "bird",
        "léonard",
        "random",
        "Fourmilier tacheté",
    )
    assert json.loads(classes) == ["bird", "blanci", "blanci_uncertain"]
    assert intervals == [(span_id, 4.0, 5.5, "blanci"), (span_id, 8.0, 8.4, "blanci_uncertain")]
    assert any("File terminée" in s.value for s in at.success)
    assert "déjà écouté : A. blanci" in _caption(at)  # le candidat (3–6 s) touche 4–5,5 s


def test_a_plan_extract_is_listened_and_saved_alone_then_the_next_one_opens(app_config, tmp_path):
    """File tirée par plan (n° 196) : l'extrait est la fenêtre du candidat, pas
    l'enregistrement entier ; l'envoyer ouvre le candidat suivant."""
    reports = tmp_path / "data" / "reports"
    queue = pd.read_csv(reports / "candidats_test.csv")
    plan = pd.concat([queue, queue]).assign(offset_s=[2.0, 7.0], dur_s=4.0, source="plan")
    plan.to_csv(reports / "candidats_test.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].input("léonard").run()
    assert _caption(at).startswith("Candidat 1 / 2")
    _draw(at, [[3.0, 4.5, "blanci"]])
    _button(at, "Envoyer l'extrait ▶").click().run()
    assert not at.exception
    spans, intervals = _spans(app_config)
    [(span_id, start, end, _, _, _, source, _)] = spans
    assert (start, end, source) == (2.0, 6.0, "plan")
    assert intervals == [(span_id, 3.0, 4.5, "blanci")]
    assert _caption(at).startswith("Candidat 2 / 2")


def test_blanci_ticked_without_interval_is_refused(app_config):
    """Une case A. blanci n'existe pas dans le formulaire : ce sont les intervalles qui le
    disent ; sans intervalle, l'extrait est enregistré négatif."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].input("léonard").run()
    assert not any(c.label == "A. blanci" for c in at.main.checkbox)
    _button(at, "Envoyer l'extrait ▶").click().run()
    spans, intervals = _spans(app_config)
    assert [s[3] for s in spans] == ["background"] and intervals == []


def test_window_mode_saves_a_label(app_config):
    """Méthode par fenêtres (suspendue) : classe, espèce, commentaire, puis « Envoyer »."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].input("léonard").run()
    _window_mode(at)
    _multiclass(at)
    _tick(at, "oiseau")
    next(t for t in at.text_input if t.label.startswith("Espèce")).input("Fourmilier tacheté")
    next(t for t in at.text_input if t.label.startswith("Commentaire")).input("chant lointain")
    _button(at, "Envoyer ▶").click().run()
    assert not at.exception
    con = connect(app_config["paths"]["db"])
    rows = con.execute("SELECT label, annotator, source, species FROM labels").fetchall()
    assert ("bird", "léonard", "random", "Fourmilier tacheté") in [tuple(r) for r in rows]


def test_sending_without_an_annotator_is_refused(app_config):
    at = AppTest.from_file(APP, default_timeout=60).run()
    _button(at, "Envoyer l'extrait ▶").click().run()
    assert any("annotateur" in e.value for e in at.error)
    assert _spans(app_config) == ([], [])


def test_a_queue_is_drawn_in_the_app_and_opened(app_config):
    """Mode de sélection « fenêtres au hasard » : la file est écrite puis ouverte sur place."""
    from pathlib import Path

    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox(key="mode").set_value("random").run()
    at.sidebar.number_input(key="n::random").set_value(1).run()
    _button(at, "Générer la file").click().run()
    assert not at.exception
    queues = list(Path(app_config["paths"]["reports"]).glob("files/random_*/candidats.csv"))
    assert len(queues) == 1
    assert "Fenêtres au hasard" in (queues[0].parent / "LISEZMOI.md").read_text(encoding="utf-8")
    assert at.sidebar.selectbox(key="mode").value == "existing"
    assert "CDR" in at.subheader[0].value


def test_modes_needing_an_encoder_say_so(app_config):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox(key="mode").set_value("map").run()
    assert not at.exception
    assert any("encodeur" in i.value.lower() for i in at.info)


def test_without_multiclass_only_blanci_is_asked(app_config):
    """Multi-classe décoché (défaut) : ni autres classes ni espèce dans le formulaire, et
    l'extrait dit que les autres classes n'ont pas été cherchées."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].input("léonard").run()
    assert not any(c.label == "oiseau" for c in at.main.checkbox)
    assert not any(t.label.startswith("Espèce entendue") for t in at.text_input)
    _draw(at, [[4.0, 5.5, "blanci"], [8.0, 9.0, "false_friend"]])
    next(t for t in at.text_input if t.label.startswith("Espèce du faux ami")).input("Adenomera")
    _button(at, "Envoyer l'extrait ▶").click().run()
    con = connect(app_config["paths"]["db"])
    [(other, species, conditions)] = con.execute(
        "SELECT other_label, species, conditions FROM spans"
    ).fetchall()
    assert (other, species) == ("background", "Adenomera")  # le faux ami a son intervalle
    assert json.loads(conditions)["multiclass"] is False
    assert [tuple(r) for r in con.execute("SELECT label FROM intervals")] == [
        ("blanci",),
        ("false_friend",),
    ]


@pytest.mark.parametrize("n_candidates", [3])
def test_numbering_follows_the_whole_queue_and_going_back_works(app_config):
    """Après « 1 / 3 » vient « 2 / 3 » (et non « 1 / 2 ») ; ◀ ▶ sautent les candidats déjà
    écoutés si la case est cochée, vont au voisin sinon ; on revient sur un candidat déjà
    écouté et on le corrige."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].input("léonard").run()
    _window_mode(at)
    _one_window(at)
    assert _caption(at).startswith("Candidat 1 / 3")
    assert not any(b.label.startswith("Prochain jamais") for b in at.button)
    _button(at, "Envoyer ▶").click().run()
    assert not at.exception
    assert _caption(at).startswith("Candidat 2 / 3")
    assert _button(at, "◀ Candidat précédent").disabled  # rien de jamais écouté avant
    next(c for c in at.sidebar.checkbox if c.label.startswith("Sauter")).uncheck().run()
    _button(at, "◀ Candidat précédent").click().run()
    assert _caption(at).startswith("Candidat 1 / 3")
    assert "déjà écouté : rien" in _caption(at)
    _tick(at, "A. blanci ?")
    _button(at, "Envoyer ▶").click().run()
    assert _caption(at).startswith("Candidat 2 / 3")  # le voisin
    _button(at, "Envoyer ▶").click().run()
    assert _caption(at).startswith("Candidat 3 / 3")
    next(c for c in at.sidebar.checkbox if c.label.startswith("Sauter")).check().run()
    assert _button(at, "◀ Candidat précédent").disabled
    strip = at.sidebar.selectbox(key=next(k for k in _keys(at) if k.startswith("strip::")))
    strip.set_value(0).run()
    assert _caption(at).startswith("Candidat 1 / 3")
    _button(at, "Candidat suivant ▶").click().run()
    assert _caption(at).startswith("Candidat 3 / 3")  # le 2e est déjà écouté
    assert [(o, label) for o, label, _ in _labels(app_config)] == [
        (3.0, "background"),
        (3.0, "blanci_uncertain"),
        (6.0, "background"),
    ]


def test_several_classes_are_saved_together(app_config):
    """Méthode par fenêtres, A. blanci et pluie cochés : label A. blanci, la pluie dans
    `extra_labels`."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].input("léonard").run()
    _window_mode(at)
    _one_window(at)
    _multiclass(at)
    _tick(at, "pluie")
    _tick(at, "A. blanci")
    _button(at, "Envoyer ▶").click().run()
    assert not at.exception
    [(_, label, conditions)] = _labels(app_config)
    assert label == "blanci"
    assert json.loads(conditions)["extra_labels"] == ["rain"]


def test_a_recording_is_split_into_windows_annotated_on_the_same_page(app_config):
    """Méthode par fenêtres sur l'enregistrement entier (12 s), fenêtres de 3 s calées sur le
    candidat (3 s) : 4 fenêtres, la première annotée est celle du candidat, puis la suivante,
    sans changer de candidat."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].input("léonard").run()
    _window_mode(at)
    assert not at.exception
    window = next(s for s in at.selectbox if s.label == "Fenêtre")
    assert len(window.options) == 4
    assert window.value == 1
    _button(at, "Envoyer ▶").click().run()
    _tick(at, "A. blanci")
    _button(at, "Envoyer ▶").click().run()
    assert not at.exception
    assert _caption(at).startswith("Candidat 1 / 1")
    assert [(o, label) for o, label, _ in _labels(app_config)] == [
        (3.0, "background"),
        (6.0, "blanci"),
    ]


def test_a_moved_disk_is_found_by_pasting_its_folder(app_config, tmp_path):
    """Disque introuvable : le poste le dit et propose de coller le dossier, retenu ensuite."""
    import shutil

    from blanci.core import locate

    moved = tmp_path / "ailleurs"
    shutil.move(app_config["paths"]["raw"], moved)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("introuvable" in e.value for e in at.error)
    field = next(t for t in at.text_input if t.label == "Dossier des enregistrements")
    field.input(str(moved)).run()
    assert not at.exception and not at.error
    locate._FOUND.clear()  # nouvelle session : le dossier collé a été retenu à côté de la base
    again = AppTest.from_file(APP, default_timeout=60).run()
    assert not again.exception and not again.error
