"""Relevé des encodeurs (§2) : débit, mémoire, projection sur une campagne."""

import numpy as np
import pandas as pd
import pytest

from blanci.encoders.base import BaseEncoder
from blanci.throughput import (
    CAMPAIGN_HOURS,
    machine_description,
    measure_encoder,
    measure_in_subprocess,
    peak_memory_mb,
    write_throughput_report,
)


class SlowToy(BaseEncoder):
    name, version, sample_rate, window_s, dim, has_tokens = "toy", "1", 16_000, 3.0, 4, False

    def _forward(self, batch):
        return np.stack([batch.mean(1), batch.std(1), batch.min(1), batch.max(1)], axis=1)


def test_measure_reports_speed_and_projects_a_campaign():
    row = measure_encoder(SlowToy(), n_windows=8, hop_ratio=0.5)
    assert row["dim"] == 4 and row["finite"] and not row["has_tokens"]
    # temps réel = fenêtres/s × pas (1,5 s) ; campagne = 575 h / temps réel
    assert row["realtime_factor"] == pytest.approx(row["windows_per_s"] * 1.5)
    assert row["campaign_h"] == pytest.approx(CAMPAIGN_HOURS / row["realtime_factor"])
    assert row["campaign_peak_hours_h"] == pytest.approx(row["campaign_h"] / 3.6)


def test_peak_memory_is_positive():
    assert peak_memory_mb() > 10


def test_an_unknown_encoder_gives_an_error_row_not_a_crash():
    row = measure_in_subprocess("inconnu", None, n_windows=2, timeout_s=300)
    assert row["encoder"] == "inconnu" and "encodeur inconnu" in row["error"]


def test_report_lists_every_encoder(tmp_path):
    rows = [measure_encoder(SlowToy(), n_windows=4), {"encoder": "perch_v2", "error": "boum"}]
    paths = write_throughput_report(rows, tmp_path, machine_description())
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "toy" in text and "perch_v2" in text and "boum" in text


def test_a_new_measure_updates_its_row_and_keeps_the_others(tmp_path):
    first = measure_encoder(SlowToy(), n_windows=4)
    other = first | {"encoder": "autre", "windows_per_s": 99.0}
    write_throughput_report([first, other], tmp_path, "test")
    again = first | {"windows_per_s": 1.0}
    write_throughput_report([again], tmp_path, "test")
    table = pd.read_csv(tmp_path / "debit.csv").set_index("encoder")
    assert sorted(table.index) == ["autre", "toy"]
    assert table.loc["toy", "windows_per_s"] == 1.0 and table.loc["autre", "windows_per_s"] == 99.0
