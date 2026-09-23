"""Réentraînement sans intervention avec adoption (M5) et courbes d'activité (M4)."""

import numpy as np
import pandas as pd
import pytest

import blanci.service as service
from blanci.activity import (
    curve_correlation,
    daily_probability,
    diel_index,
    reference_checks,
)
from blanci.config import load_config
from blanci.db import connect
from tests.test_cli_pipeline import embedded, run, workspace  # noqa: F401
from tests.test_frozen import _listen_in_full

REFERENCE = load_config()["activity"]["reference"]


def _db(tmp_path):
    return connect(tmp_path / "db" / "blanci.sqlite")


# --- Réentraînement et adoption -------------------------------------------------------------


def test_without_frozen_set_nothing_is_adopted(embedded):  # noqa: F811
    tmp_path, config = embedded
    output = run(config, "retrain", "--encoder", "toy-1")
    assert "NON ADOPTÉE" in output and "aucun jeu gelé" in output
    assert service.adopted_version(_db(tmp_path), "toy-1") is None
    output = run(config, "retrain", "--encoder", "toy-1", "--force")
    assert "ADOPTÉE" in output and service.adopted_version(_db(tmp_path), "toy-1") == "v2"


def test_first_head_judged_on_frozen_set_is_adopted_then_compared(embedded):  # noqa: F811
    tmp_path, config = embedded
    _, queue, _ = _listen_in_full(tmp_path)
    run(config, "freeze", str(queue), "--version", "v1")
    first = run(config, "retrain", "--encoder", "toy-1")
    assert "ADOPTÉE" in first and "aucune tête adoptée" in first
    second = run(config, "retrain", "--encoder", "toy-1")  # mêmes labels : même qualité
    assert "nouvelle v2 sur le jeu gelé" in second and "adoptée v1 sur le jeu gelé" in second
    assert "ADOPTÉE : pas moins bonne que v1" in second
    assert service.adopted_version(_db(tmp_path), "toy-1") == "v2"


def test_a_worse_head_is_not_adopted(embedded, monkeypatch):  # noqa: F811
    tmp_path, config = embedded
    _, queue, _ = _listen_in_full(tmp_path)
    run(config, "freeze", str(queue), "--version", "v1")
    run(config, "retrain", "--encoder", "toy-1")  # v1 adoptée

    real = service.evaluate_frozen

    def degraded(con, encoder_id, cfg, version="latest", frozen_version=None):
        out = real(con, encoder_id, cfg, version, frozen_version)
        if out["head_version"] == "v2":  # la nouvelle tête perd 0,1 d'AP
            out["recording"] = out["recording"] | {"ap": out["recording"]["ap"] - 0.1}
        return out

    monkeypatch.setattr(service, "evaluate_frozen", degraded)
    output = run(config, "retrain", "--encoder", "toy-1")
    assert "NON ADOPTÉE : moins bonne que v1 sur ap_recording" in output
    assert service.adopted_version(_db(tmp_path), "toy-1") == "v1"


def test_score_uses_the_adopted_head_not_the_latest(embedded):  # noqa: F811
    tmp_path, config = embedded
    run(config, "retrain", "--encoder", "toy-1", "--force")  # v1 adoptée
    run(config, "train", "--encoder", "toy-1")  # v2 entraînée, jamais jugée
    run(config, "score", "--encoder", "toy-1")
    versions = {r[0] for r in _db(tmp_path).execute("SELECT head_version FROM decisions")}
    assert versions == {"v1"}


# --- Courbes d'activité ---------------------------------------------------------------------


def synthetic_detections(peaks=True, season=True, seed=0):
    """Un an, 5 h–20 h, deux sites : activité aux heures de pic et de janvier à avril."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in pd.date_range("2023-11-25", "2024-11-24", freq="D"):
        for hour in range(5, 20):
            p = 0.05
            if (not peaks or hour in (7, 8, 15, 16)) and (not season or day.month in (1, 2, 3, 4)):
                p = 0.8
            for site in ("Molokoi", "Tresor"):
                rows.append(
                    {
                        "site": site,
                        "hour": hour,
                        "day": day.strftime("%Y-%m-%d"),
                        "month": day.strftime("%Y-%m"),
                        "detected": rng.random() < p,
                    }
                )
    return pd.DataFrame(rows)


def test_published_patterns_are_found_when_present():
    detections = synthetic_detections()
    diel, seasonal = diel_index(detections), daily_probability(detections)
    assert set(diel["hour"]) == set(range(5, 20)) and (diel["lo"] <= diel["index"]).all()
    checks = reference_checks(diel, seasonal, REFERENCE)
    assert checks["peaks_found"].all() and checks["season_found"].all()
    assert (checks["prob_low_months"] < 0.8).all()


def test_flat_activity_misses_the_peaks():
    detections = synthetic_detections(peaks=False)
    checks = reference_checks(diel_index(detections), daily_probability(detections), REFERENCE)
    assert not checks["peaks_found"].any() and checks["season_found"].all()
    assert checks["peak_ratio"].between(0.8, 1.25).all()


def test_correlation_with_a_digitised_reference():
    diel = diel_index(synthetic_detections())
    ref = pd.DataFrame({"hour": range(5, 20)})
    ref["value"] = ref["hour"].isin([7, 8, 15, 16]).astype(float)
    corr = curve_correlation(diel, ref, "hour", "index")
    assert (corr["pearson_r"] > 0.8).all() and (corr["points"] == 15).all()
    flat = curve_correlation(diel_index(synthetic_detections(peaks=False)), ref, "hour", "index")
    assert (flat["pearson_r"] < 0.5).all()


def test_daily_probability_counts_days_not_recordings():
    detections = pd.DataFrame(
        {
            "site": ["A"] * 4,
            "hour": [7, 8, 7, 8],
            "day": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "month": ["2024-01"] * 4,
            "detected": [True, True, False, False],
        }
    )
    row = daily_probability(detections).iloc[0]
    assert row["days"] == 2 and row["days_detected"] == 1 and row["probability"] == 0.5


def test_activity_command_writes_curves_and_checks(embedded):  # noqa: F811
    tmp_path, config = embedded
    run(config, "train", "--encoder", "toy-1")
    run(config, "score", "--encoder", "toy-1")
    ref = tmp_path / "ref_heures.csv"
    pd.DataFrame({"hour": [9, 10, 11], "value": [0.1, 0.5, 0.2]}).to_csv(ref, index=False)
    output = run(config, "activity", "--encoder", "toy-1", "--reference-hours", str(ref))
    assert "rapport" in output
    reports = tmp_path / "reports"
    for suffix in ("horaire.csv", "mensuel.csv", "controles.csv", "correlations.csv"):
        assert (reports / f"activite_toy-1_v1_{suffix}").exists()
    assert "Courbes d'activité" in (reports / "activite_toy-1_v1.md").read_text(encoding="utf-8")


def test_activity_needs_decisions(embedded):  # noqa: F811
    tmp_path, config = embedded
    run(config, "train", "--encoder", "toy-1")
    with pytest.raises(ValueError, match="blanci score"):
        service.activity_curves(_db(tmp_path), "toy-1", load_config(config))
