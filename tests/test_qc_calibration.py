"""Calibration du contrôle qualité : seuils proposés sans jamais signaler A. blanci."""

import numpy as np
import pandas as pd
import soundfile as sf

from blanci.config import load_config
from blanci.qc_calibration import calibration_indices, suggest_thresholds


def _indices(rows):
    """rows : (recording_id, group, hf_ratio, flatness)."""
    return pd.DataFrame(
        [
            {
                "recording_id": rid,
                "group": group,
                "r_hf_ratio": hf,
                "r_flatness_1_10k": flat,
                "r_rms_dbfs": -40.0,
                "r_clip_fraction": 0.0,
            }
            for rid, group, hf, flat in rows
        ]
    )


def test_suggests_a_threshold_between_bagged_mics_and_blanci():
    thresholds = load_config()["qc"]
    indices = _indices(
        [
            ("sac1", "in_bag", 0.01, 0.1),
            ("sac2", "in_bag", 0.04, 0.1),  # manqué par le seuil actuel (0,02)
            ("b1", "blanci", 0.30, 0.2),
            ("b2", "blanci", 0.10, 0.2),
            ("o1", "other", 0.05, 0.2),
        ]
    )
    table = suggest_thresholds(indices, thresholds).set_index("flag")
    row = table.loc["in_bag"]
    assert row["targets_flagged_now"] == 1 and row["blanci_flagged_now"] == 0
    assert row["suggested"] == (0.04 + 0.10) / 2
    assert row["targets_flagged_suggested"] == 2


def test_never_proposes_to_flag_a_recording_with_blanci():
    """Chevauchement : le seuil s'arrête au positif le plus bas, quitte à rater des cibles."""
    indices = _indices(
        [("sac", "in_bag", 0.08, 0.1), ("b1", "blanci", 0.05, 0.2), ("b2", "blanci", 0.3, 0.2)]
    )
    row = suggest_thresholds(indices, load_config()["qc"]).set_index("flag").loc["in_bag"]
    assert row["suggested"] == 0.05 and row["targets_flagged_suggested"] == 0
    assert "chevauchement" in row["reason"]


def test_rain_and_blanci_in_the_same_recording_stays_protected():
    indices = _indices(
        [("r", "rain", 0.3, 0.9), ("r", "blanci", 0.3, 0.9), ("b", "blanci", 0.3, 0.2)]
    )
    row = suggest_thresholds(indices, load_config()["qc"]).set_index("flag").loc["rain"]
    assert row["targets"] == 0 and row["blanci_recordings"] == 2


def test_calibration_indices_read_each_recording_once(tmp_path):
    sr = 16_000
    rng = np.random.default_rng(0)
    sf.write(tmp_path / "a.wav", rng.normal(0, 0.05, (sr * 6, 2)).astype(np.float32), sr)
    windows = pd.DataFrame(
        {
            "window_id": ["w1", "w2"],
            "recording_id": ["a", "a"],
            "path": ["a.wav", "a.wav"],
            "offset_s": [0.0, 3.0],
            "dur_s": [3.0, 3.0],
            "group": ["blanci", "other"],
            "label": ["blanci", "bird"],
        }
    )
    out = calibration_indices(windows, tmp_path, channel=0)
    assert len(out) == 2 and out["r_hf_ratio"].nunique() == 1
    assert out["w_hf_ratio"].between(0, 1).all()
