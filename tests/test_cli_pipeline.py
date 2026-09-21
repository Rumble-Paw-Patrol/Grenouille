"""Chaîne complète en ligne de commande : embed → benchmark → train → score → queue → evaluate."""

import numpy as np
import pytest
import soundfile as sf
import yaml
from typer.testing import CliRunner

from blanci.cli import app
from blanci.db import connect
from blanci.encoders.base import BaseEncoder

SR = 16000
WINDOW_S = 3.0
DURATION_S = 12.0
runner = CliRunner()


def write_soundscape(path, tone_hz=None, seed=0):
    """Fond large bande, avec ou sans tonale.

    Un sinus pur serait classé « micro dans sac » par le contrôle qualité (pas d'énergie
    au-dessus de 2 kHz) et exclu de l'extraction : un paysage sonore est large bande.
    """
    rng = np.random.default_rng(seed)
    n = int(SR * DURATION_S)
    x = rng.normal(0, 0.05, n)
    if tone_hz is not None:
        x += 0.2 * np.sin(2 * np.pi * tone_hz * np.arange(n) / SR)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x.astype(np.float32), SR, subtype="PCM_16")
    return path


class ToyEncoder(BaseEncoder):
    """Encodeur factice : l'énergie en bande 4–6 kHz suffit à séparer le corpus de test."""

    name = "toy"
    version = "1"
    sample_rate = SR
    window_s = WINDOW_S
    dim = 8
    has_tokens = False

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(batch, axis=1))
        bands = np.array_split(spectrum, self.dim, axis=1)
        return np.stack([np.log1p(b.mean(axis=1)) for b in bands], axis=1).astype(np.float32)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Un dépôt jouet : audio, config, et `toy` branché comme encodeur disponible."""
    from blanci import encoders

    monkeypatch.setattr(encoders, "get_encoder", lambda name, cfg: ToyEncoder(), raising=True)
    import blanci.cli as cli_module

    monkeypatch.setattr(cli_module, "get_encoder", lambda name, cfg: ToyEncoder(), raising=True)

    raw = tmp_path / "raw"
    seed = 0
    for site, mics in (("mataroni", ["M1", "M2", "M3"]), ("tresor", ["T1", "T2"])):
        for mic in mics:
            for day in range(4):
                # Le premier jour de chaque micro porte une tonale à 4,75 kHz : le « chant ».
                path = raw / "2026" / site / mic / f"{mic}_20260{2 + day}10_100000.wav"
                write_soundscape(path, tone_hz=4750.0 if day == 0 else None, seed=seed)
                seed += 1

    config = tmp_path / "blanci.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "raw": str(raw),
                    "db": str(tmp_path / "db" / "blanci.sqlite"),
                    "embeddings": str(tmp_path / "embeddings"),
                    "label_imports": str(tmp_path / "imports"),
                    "models": str(tmp_path / "models"),
                    "frozen_test": str(tmp_path / "frozen"),
                    "reports": str(tmp_path / "reports"),
                },
                "benchmark": {"n_boot": 20, "negatives_per_positive": 4},
                "head": {"n_splits": 3, "C_grid": [1.0], "seed": 0},
                "active": {"batch_recordings": 5, "mix": [0.6, 0.2, 0.2]},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path, config


def run(config, *args):
    result = runner.invoke(app, ["--config", str(config), *args])
    assert result.exit_code == 0, (
        f"{' '.join(args)} a échoué :\n{result.output}\n{result.exception}"
    )
    return result.output


def label_positives(config, tmp_path):
    """Annote les fenêtres du premier jour de chaque micro, comme l'expert le ferait."""
    con = connect(tmp_path / "db" / "blanci.sqlite")
    rows = con.execute(
        "SELECT w.window_id FROM windows w JOIN recordings r USING (recording_id) "
        "WHERE r.path LIKE '%20260210%'"
    ).fetchall()
    assert rows, "aucune fenêtre du jour positif"
    for row in rows:
        run(config, "label", row["window_id"], "--label", "blanci_solo", "--source", "import")
    return len(rows)


# --- Inventaire et extraction --------------------------------------------------------------


def test_ingest_then_embed(workspace):
    tmp_path, config = workspace
    output = run(config, "ingest", "--dataset", "2026", "--no-hash")
    assert "20 ajoutés" in output

    output = run(config, "embed", "--encoder", "toy")
    assert "toy-1" in output and "fenêtres/s" in output
    assert (tmp_path / "embeddings" / "toy-1").exists()


def test_embed_is_resumable(workspace):
    tmp_path, config = workspace
    run(config, "ingest", "--dataset", "2026", "--no-hash")
    run(config, "embed", "--encoder", "toy")
    output = run(config, "embed", "--encoder", "toy")
    assert "0 encodés" in output and "20 déjà faits" in output


def test_embed_can_target_a_site(workspace):
    tmp_path, config = workspace
    run(config, "ingest", "--dataset", "2026", "--no-hash")
    output = run(config, "embed", "--encoder", "toy", "--site", "tresor")
    assert "8 enregistrements à traiter" in output


def test_embed_without_matching_recordings_fails_cleanly(workspace):
    tmp_path, config = workspace
    run(config, "ingest", "--dataset", "2026", "--no-hash")
    result = runner.invoke(
        app, ["--config", str(config), "embed", "--encoder", "toy", "--site", "molokoi"]
    )
    assert result.exit_code == 1
    assert "aucun enregistrement" in result.output


# --- Chaîne d'apprentissage -----------------------------------------------------------------


@pytest.fixture
def embedded(workspace):
    tmp_path, config = workspace
    run(config, "ingest", "--dataset", "2026", "--no-hash")
    run(config, "embed", "--encoder", "toy")
    label_positives(config, tmp_path)
    return tmp_path, config


def test_train_reports_threshold_and_saves_the_head(embedded):
    tmp_path, config = embedded
    output = run(config, "train", "--encoder", "toy-1")
    assert "tête toy-1 v1" in output and "seuil" in output
    assert (tmp_path / "models" / "head" / "toy-1-v1" / "manifest.json").exists()


def test_score_writes_decisions_and_ranks_points(embedded):
    tmp_path, config = embedded
    run(config, "train", "--encoder", "toy-1")
    output = run(config, "score", "--encoder", "toy-1")
    assert "fenêtres scorées" in output
    assert (tmp_path / "reports" / "points_toy-1_v1.csv").exists()

    con = connect(tmp_path / "db" / "blanci.sqlite")
    assert con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 20


def test_queue_writes_a_csv(embedded):
    tmp_path, config = embedded
    run(config, "train", "--encoder", "toy-1")
    run(config, "score", "--encoder", "toy-1")
    output = run(config, "queue", "--encoder", "toy-1", "--n", "5")
    assert "5 enregistrements" in output
    assert (tmp_path / "reports" / "queue_toy-1.csv").exists()


def test_queue_rejects_a_malformed_mix(embedded):
    tmp_path, config = embedded
    run(config, "train", "--encoder", "toy-1")
    run(config, "score", "--encoder", "toy-1")
    result = runner.invoke(
        app,
        ["--config", str(config), "queue", "--encoder", "toy-1", "--mix", "0.6,0.4"],
    )
    assert result.exit_code != 0
    assert "trois proportions" in result.output


def test_search_lists_candidates(embedded):
    tmp_path, config = embedded
    output = run(config, "search", "--encoder", "toy-1", "--k", "10")
    assert "10 candidats" in output
    assert (tmp_path / "reports" / "search_toy-1.csv").exists()


def test_benchmark_writes_the_report(embedded):
    tmp_path, config = embedded
    output = run(config, "benchmark", "--encoders", "toy-1")
    assert "Niveau window" in output and "Niveau recording" in output
    assert (tmp_path / "reports" / "benchmark.md").exists()


def test_evaluate_grouped_by_mic(embedded):
    tmp_path, config = embedded
    output = run(config, "evaluate", "--encoder", "toy-1")
    assert "plis groupés par micro" in output
    assert "AP" in output and "rappel à P≥0.1" in output


def test_evaluate_holds_out_a_site(embedded):
    """Le protocole du §6 niveau 2 : entraîner ailleurs, mesurer sur le site tenu à l'écart."""
    tmp_path, config = embedded
    output = run(config, "evaluate", "--encoder", "toy-1", "--holdout", "tresor")
    assert "entraînement hors tresor" in output


def test_evaluate_rejects_an_unknown_level(embedded):
    tmp_path, config = embedded
    result = runner.invoke(
        app, ["--config", str(config), "evaluate", "--encoder", "toy-1", "--level", "point"]
    )
    assert result.exit_code != 0
    assert "window ou recording" in result.output


# --- Annotation en ligne de commande ---------------------------------------------------------


def test_label_command_appends(embedded):
    tmp_path, config = embedded
    con = connect(tmp_path / "db" / "blanci.sqlite")
    window_id = con.execute("SELECT window_id FROM windows LIMIT 1").fetchone()["window_id"]
    before = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    output = run(config, "label", window_id, "--label", "bird", "--source", "active")
    assert "ajouté" in output
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == before + 1


def test_status_summarizes_the_database(embedded):
    tmp_path, config = embedded
    output = run(config, "status")
    assert "mataroni" in output and "tresor" in output
    assert "blanci_solo" in output
