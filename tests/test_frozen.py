"""Jeu gelé (§6) : versionné, en lecture seule, jamais entraîné, seulement jugé."""

import os

import pandas as pd
import pytest
from typer.testing import CliRunner

from blanci.cli import app
from blanci.dataset import training_set
from blanci.db import connect
from blanci.frozen import freeze, frozen_recordings, frozen_versions
from blanci.workbench import save_answer
from tests.test_cli_pipeline import DURATION_S, embedded, run, workspace  # noqa: F401

runner = CliRunner()


def _listen_in_full(tmp_path, mic="T1"):
    """Écoute en entier des enregistrements d'un micro : label d'enregistrement entier."""
    con = connect(tmp_path / "db" / "blanci.sqlite")
    rows = con.execute(
        "SELECT recording_id, path FROM recordings WHERE mic_id = ? ORDER BY path", (mic,)
    ).fetchall()
    for r in rows:
        label = "blanci" if "20260210" in r["path"] else "background"
        candidate = {
            "recording_id": r["recording_id"],
            "offset_s": 0.0,
            "dur_s": DURATION_S,
            "source": "audit",
            "reason": "jeu_gele",
        }
        save_answer(con, candidate, label, "léonard")
    queue = tmp_path / "reports" / "candidats_gele.csv"
    queue.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"recording_id": [r["recording_id"] for r in rows]}).to_csv(queue, index=False)
    return con, queue, {r["recording_id"] for r in rows}


def _cfg(config):
    from blanci.config import load_config

    return load_config(config)


def test_freeze_writes_a_read_only_version_and_never_overwrites(embedded):  # noqa: F811
    tmp_path, config = embedded
    con, queue, ids = _listen_in_full(tmp_path)
    cfg = _cfg(config)
    path, report = freeze(con, cfg, queue, "v1")
    assert report["n_recordings"] == 4 and report["positive_labels_withdrawn"] >= 1
    assert not os.access(path, os.W_OK)
    assert frozen_recordings(cfg) == ids and list(frozen_versions(cfg)) == ["v1"]
    with pytest.raises(FileExistsError):
        freeze(con, cfg, queue, "v1")


def test_frozen_recordings_leave_the_training_set(embedded):  # noqa: F811
    tmp_path, config = embedded
    con, queue, ids = _listen_in_full(tmp_path)
    grid = pd.read_sql_query("SELECT window_id, recording_id, offset_s FROM windows", con)
    grid = grid.assign(dur_s=3.0)
    everything = training_set(con, grid, per_positive=2)
    assert set(everything["recording_id"]) & ids
    kept = training_set(con, grid, per_positive=2, exclude_recordings=ids)
    assert not set(kept["recording_id"]) & ids  # ni labels, ni négatifs appariés
    # `row` pointe toujours dans la grille complète.
    assert (grid.loc[kept["row"], "window_id"].to_numpy() == kept["window_id"].to_numpy()).all()


def test_a_head_is_judged_on_the_frozen_set_only_if_trained_after_the_freeze(embedded):  # noqa: F811
    tmp_path, config = embedded
    _, queue, _ = _listen_in_full(tmp_path)
    run(config, "train", "--encoder", "toy-1")  # avant le gel
    run(config, "freeze", str(queue), "--version", "v1")
    refused = runner.invoke(
        app, ["--config", str(config), "evaluate", "--encoder", "toy-1", "--frozen", "v1"]
    )
    assert refused.exit_code != 0 and "avant le gel" in str(refused.exception)

    run(config, "train", "--encoder", "toy-1")  # après le gel : v2, sans le jeu gelé
    output = run(config, "evaluate", "--encoder", "toy-1", "--frozen", "last")
    assert "jeu gelé v1" in output and "fausses alarmes par heure" in output
    assert "4 enregistrements écoutés en entier" in output
