"""Calibration du contrôle qualité : seuils proposés sans jamais signaler A. blanci."""

import json

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


def _recording_db(tmp_path, durations):
    from blanci.core.db import connect

    con = connect(tmp_path / "db.sqlite")
    for rid, duration in durations.items():
        con.execute(
            "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc,"
            " duration_s) VALUES (?, ?, 'd', 's', 'm', '2026-01-01T00:00:00Z', ?)",
            (rid, f"{rid}.wav", duration),
        )
    con.commit()
    return con


def test_listened_recording_without_windows_counts_in_calibration(tmp_path):
    """`append_span` ne remplit pas `windows` : la grille des extraits vient des extraits."""
    from blanci.evaluation.qc_calibration import labelled_windows
    from blanci.service import append_span

    con = _recording_db(tmp_path, {"a": 12.0, "b": 12.0})
    assert con.execute("SELECT COUNT(*) FROM windows").fetchone()[0] == 0
    append_span(con, "a", 0.0, 12.0, [(3.0, 6.0, "blanci")], "background", "audit")
    append_span(con, "b", 0.0, 12.0, [(3.0, 6.0, "false_friend")], "rain", "audit")
    df = labelled_windows(con)
    assert set(df["recording_id"]) == {"a", "b"}
    assert set(df.loc[df["recording_id"] == "a", "group"]) == {"blanci", "other"}
    assert "rain" in set(df.loc[df["recording_id"] == "b", "group"])
    assert df["path"].notna().all()


def test_window_label_and_interval_in_conflict_are_dropped_from_calibration(tmp_path):
    """Règle de `_merge_labelled` : désaccord positif / négatif, fenêtre écartée ; accord,
    le label de fenêtre prime (ici ses conditions de pluie)."""
    from blanci.annotation.workbench import save_answer
    from blanci.evaluation.qc_calibration import labelled_windows
    from blanci.service import append_span

    con = _recording_db(tmp_path, {"a": 12.0})

    def label(offset, name, **kw):
        cand = {"recording_id": "a", "offset_s": offset, "dur_s": 3.0, "source": "random"}
        save_answer(con, cand, name, "léonard", **kw)

    label(3.0, "bird")  # conflit avec l'intervalle positif
    label(6.0, "blanci")  # conflit avec un extrait sans intervalle
    label(9.0, "blanci")  # d'accord avec l'intervalle
    append_span(
        con, "a", 0.0, 12.0, [(3.0, 6.0, "blanci"), (9.5, 11.5, "blanci")], "background", "audit"
    )
    df = labelled_windows(con)
    offsets = set(df.loc[df["label"] != "", "offset_s"])
    assert 3.0 not in offsets and 6.0 not in offsets
    assert df.loc[df["offset_s"] == 9.0, "label"].tolist() == ["blanci"]


def test_in_bag_suggested_count_follows_the_run_rule():
    """Le compte au seuil proposé suit la règle des suites, comme « aux seuils actuels » :
    deux sacs isolés (suite de 2 < 3) ne sont pas signalés même sous le seuil proposé."""
    thresholds = load_config()["qc"] | {
        "in_bag_hf_ratio": 0.02,
        "in_bag_min_run": 3,
        "in_bag_max_gap_min": 60.0,
    }
    rows = [
        ("s1", "in_bag", 0.04, 0.1),
        ("s2", "in_bag", 0.04, 0.1),
        ("b1", "blanci", 0.30, 0.2),
    ]
    indices = _indices(rows).assign(
        dataset="d",
        site="s",
        mic_id="m",
        start_utc=["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z", "2026-01-01T05:00:00Z"],
    )
    row = suggest_thresholds(indices, thresholds).query("flag == 'in_bag'").iloc[0]
    assert row["suggested"] == (0.04 + 0.30) / 2
    assert row["targets_flagged_suggested"] == 0
    longer = indices.copy()
    longer.loc[len(longer)] = ["s3", "in_bag", 0.04, 0.1, -40.0, 0.0, "d", "s", "m",
                               "2026-01-01T01:00:00Z"]  # fmt: skip
    row = suggest_thresholds(longer, thresholds).query("flag == 'in_bag'").iloc[0]
    assert row["targets_flagged_suggested"] == 3


def test_series_attr_reaches_suggest_thresholds(tmp_path):
    """Chemin de `qc-calibrate` : `attrs["series"]` passe de `labelled_windows` à
    `calibration_indices` puis `suggest_thresholds`, et les voisins non étiquetés complètent
    la suite d'`in_bag`."""
    from blanci.evaluation.qc_calibration import labelled_windows

    sr = 16_000
    con = _recording_db(tmp_path, {"a": 6.0, "n1": 6.0, "n2": 6.0})
    for i, rid in enumerate(("a", "n1", "n2")):
        qc = {"indices": {"hf_ratio": 0.01, "rms_dbfs": -40.0}}
        con.execute(
            "UPDATE recordings SET start_utc = ?, qc_flags = ? WHERE recording_id = ?",
            (f"2026-01-01T0{i}:00:00Z", json.dumps(qc), rid),
        )
    con.execute("INSERT INTO windows VALUES ('wa', 'a', 0.0, 3.0)")
    con.execute(
        "INSERT INTO labels (window_id, label, source, created_at)"
        " VALUES ('wa', 'artefact_in_bag', 'random', 'now')"
    )
    con.commit()
    sf.write(tmp_path / "a.wav", np.random.default_rng(0).normal(0, 0.05, sr * 6), sr)
    windows = labelled_windows(con)
    assert set(windows.attrs["series"]["recording_id"]) == {"a", "n1", "n2"}
    indices = calibration_indices(windows, tmp_path)
    assert set(indices.attrs["series"]["recording_id"]) == {"a", "n1", "n2"}
    thresholds = load_config()["qc"] | {
        "in_bag_hf_ratio": 1.0,
        "in_bag_min_run": 3,
        "in_bag_max_gap_min": 90.0,
    }
    row = suggest_thresholds(indices, thresholds).query("flag == 'in_bag'").iloc[0]
    assert row["targets"] == 1 and row["targets_flagged_now"] == 1  # a + n1 + n2 : suite de 3
    alone = indices.copy()
    alone.attrs = {}
    row = suggest_thresholds(alone, thresholds).query("flag == 'in_bag'").iloc[0]
    assert row["targets_flagged_now"] == 0  # sans les voisins, la suite n'y est pas
