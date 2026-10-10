"""Calibration du contrôle qualité : seuils proposés sans jamais signaler A. blanci."""

import numpy as np
import pandas as pd
import soundfile as sf

from blanci.core.config import load_config
from blanci.evaluation.qc_calibration import calibration_indices, suggest_thresholds
from blanci.inputs.qc import _welch, qc_indices


def test_welch_by_chunks_matches_scipy():
    from scipy.signal import welch

    rng = np.random.default_rng(0)
    for n in (48000 * 7 + 13, 4096, 3000, 500):  # long, deux segments, court, plus court qu'un
        x = rng.normal(0, 0.1, n).astype(np.float32)
        freqs, psd = _welch(x, 48000)
        ref_freqs, ref = welch(x, fs=48000, nperseg=min(2048, n))
        np.testing.assert_allclose(freqs, ref_freqs)
        np.testing.assert_allclose(psd, ref, rtol=1e-5)
    assert 0 < qc_indices(x, 48000)["hf_ratio"] < 1


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
    thresholds = load_config()["qc"] | {"in_bag_hf_ratio": 0.02}  # seuil d'avant le n° 81
    indices = _indices(
        [
            ("sac1", "in_bag", 0.01, 0.1),
            ("sac2", "in_bag", 0.04, 0.1),  # manqué par un seuil à 0,02
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


def test_bag_and_blanci_in_the_same_recording_stays_protected():
    indices = _indices(
        [("r", "in_bag", 0.3, 0.9), ("r", "blanci", 0.3, 0.9), ("b", "blanci", 0.3, 0.2)]
    )
    row = suggest_thresholds(indices, load_config()["qc"]).set_index("flag").loc["in_bag"]
    assert row["targets"] == 0 and row["blanci_recordings"] == 2


def test_audio_qc_only_flags_broken_or_irrelevant_recordings():
    """Ni pluie ni saturation : le contrôle audio n'écarte que silencieux et micro dans sac."""
    from blanci.inputs.qc import AUDIO_FLAGS, qc_flags

    saturated_noise = np.clip(np.random.default_rng(0).normal(0, 2, 48000), -1, 1)
    flags = qc_flags(qc_indices(saturated_noise.astype(np.float32), 48000), load_config()["qc"])
    assert set(flags) == {"silent", "in_bag", "indices"} == {*AUDIO_FLAGS, "indices"}
    assert flags["indices"]["clip_fraction"] > 0.1 and not flags["silent"]


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


def test_in_bag_flag_follows_the_run_rule_of_the_qc_module():
    from blanci.evaluation.qc_calibration import current_flags

    thresholds = load_config()["qc"] | {
        "in_bag_hf_ratio": 0.02,
        "in_bag_min_run": 3,
        "in_bag_max_gap_min": 60.0,
    }
    rows = [  # trois sacs consécutifs d'un micro, un grave isolé d'un autre
        ("a1", "in_bag", 0.01, 0.1),
        ("a2", "in_bag", 0.01, 0.1),
        ("a3", "in_bag", 0.01, 0.1),
        ("iso", "other", 0.01, 0.1),
    ]
    indices = _indices(rows).assign(
        dataset="d",
        site="s",
        mic_id=["m1", "m1", "m1", "m2"],
        start_utc=["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z", "2026-01-01T01:00:00Z"]
        + ["2026-01-01T00:00:00Z"],
    )
    flags = current_flags(indices, thresholds).set_index("recording_id")["in_bag"]
    assert flags.to_dict() == {"a1": True, "a2": True, "a3": True, "iso": False}
    row = suggest_thresholds(indices, thresholds).query("flag == 'in_bag'").iloc[0]
    assert row["targets_flagged_now"] == 3 and row["blanci_flagged_now"] == 0
