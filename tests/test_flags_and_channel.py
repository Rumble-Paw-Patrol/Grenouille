"""Canal audio lu et drapeaux d'inventaire (durée anormale, hors relevé)."""

import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from blanci.audio import load_audio
from blanci.db import connect, recording_id_for
from blanci.embed import select_recordings
from blanci.qc import apply_metadata_flags, metadata_flags

THRESHOLDS = {"expected_duration_s": 120.0, "duration_tolerance_s": 1.0, "campaign_gap_days": 7}

# --- Canal audio ----------------------------------------------------------------------------


@pytest.fixture
def stereo(tmp_path):
    """Deux micros distincts : canal 0 à faible niveau, canal 1 quatre fois plus fort."""
    rng = np.random.default_rng(0)
    left = 0.05 * rng.normal(size=16000).astype(np.float32)
    right = 0.2 * rng.normal(size=16000).astype(np.float32)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([left, right], axis=1), 16000, subtype="FLOAT")
    return path, left, right


def test_channel_selects_one_microphone(stereo):
    path, left, right = stereo
    assert np.allclose(load_audio(path, 0)[0], left)
    assert np.allclose(load_audio(path, 1)[0], right)


def test_mean_mixes_both_microphones(stereo):
    path, left, right = stereo
    assert np.allclose(load_audio(path, "mean")[0], (left + right) / 2)


def test_missing_channel_is_an_error(stereo):
    with pytest.raises(ValueError, match="canal 2 absent"):
        load_audio(stereo[0], 2)


def test_mono_file_ignores_the_channel(tmp_path):
    path = tmp_path / "mono.wav"
    sf.write(path, np.ones(100, dtype=np.float32) * 0.5, 16000, subtype="FLOAT")
    assert np.allclose(load_audio(path, 1)[0], 0.5)


def test_default_config_reads_the_low_gain_microphone():
    from blanci.config import load_config

    assert load_config()["audio"]["channel"] == 0


# --- Drapeaux d'inventaire ------------------------------------------------------------------


def recording(name, start, duration=120.0, site="Mataroni", mic="2LA03021", dataset="2026"):
    return {
        "recording_id": name,
        "dataset": dataset,
        "site": site,
        "mic_id": mic,
        "start_utc": start,
        "duration_s": duration,
    }


def campaign(mic="2LA03021", site="Mataroni", day0=6, days=7):
    """Série normale : 2 enregistrements par jour pendant une semaine."""
    return [
        recording(f"{mic}_{d}_{h}", f"2026-01-{day0 + d:02d}T{h:02d}:00:00Z", mic=mic, site=site)
        for d in range(days)
        for h in (10, 19)
    ]


def flags_of(rows):
    return metadata_flags(pd.DataFrame(rows), THRESHOLDS).set_index("recording_id")


def test_regular_campaign_is_not_flagged():
    flags = flags_of(campaign())
    assert not flags["duration_off"].any() and not flags["off_campaign"].any()


@pytest.mark.parametrize("duration, flagged", [(4.9, True), (828.9, True), (119.5, False)])
def test_duration_outside_the_schedule(duration, flagged):
    flags = flags_of(campaign() + [recording("essai", "2026-01-08T12:00:00Z", duration)])
    assert bool(flags.at["essai", "duration_off"]) is flagged


def test_test_recording_months_before_is_off_campaign():
    """Le test de juillet 2024 resté sur la carte (2LA03021_20240719_081751)."""
    flags = flags_of(campaign() + [recording("juillet", "2024-07-19T08:17:51Z")])
    assert flags.at["juillet", "off_campaign"]
    assert flags["off_campaign"].sum() == 1


def test_leftover_weeks_before_the_survey_is_off_campaign():
    """Enregistrement du 17 décembre sur une carte posée à Mataroni le 6 janvier."""
    flags = flags_of(campaign() + [recording("decembre", "2025-12-17T22:00:00Z")])
    assert flags.at["decembre", "off_campaign"]


def test_continuous_deployment_across_surveys_stays_one_block():
    """Phénologie 2023-2024 : un micro posé des mois, relevé tous les deux mois."""
    rows = [
        recording(
            f"j{d}", (pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=d)).isoformat()
        )
        for d in range(0, 120)
        if not 45 <= d < 48  # trois jours sans enregistrement le temps du relevé
    ]
    assert not flags_of(rows)["off_campaign"].any()


def test_same_recorder_on_two_sites_is_two_series():
    """2LA03550 : Mataroni en janvier, RNRT en février. Deux séries, aucune hors relevé."""
    rows = campaign(site="Mataroni") + [
        recording(f"rnrt_{d}", f"2026-02-{9 + d:02d}T10:00:00Z", site="RNRT") for d in range(7)
    ]
    assert not flags_of(rows)["off_campaign"].any()


def test_undated_recordings_are_not_flagged_off_campaign():
    flags = flags_of(campaign() + [recording("sans_date", None)])
    assert not flags.at["sans_date", "off_campaign"]


# --- Écriture dans la base ------------------------------------------------------------------


@pytest.fixture
def con(tmp_path):
    con = connect(tmp_path / "db.sqlite")
    rows = campaign() + [
        recording("2LA03021_20240719_081751", "2024-07-19T08:17:51Z", 4.9),
    ]
    con.executemany(
        "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
        "duration_s, sample_rate, channels, qc_flags) VALUES (?, ?, ?, ?, ?, ?, ?, 48000, 2, ?)",
        [
            (
                recording_id_for(r["recording_id"]),
                f"x/{r['recording_id']}.wav",
                r["dataset"],
                r["site"],
                r["mic_id"],
                r["start_utc"],
                r["duration_s"],
                json.dumps({"rain": True}) if i == 0 else None,
            )
            for i, r in enumerate(rows)
        ],
    )
    con.commit()
    return con


def test_flags_are_written_without_losing_audio_qc(con):
    counts = apply_metadata_flags(con, THRESHOLDS)
    assert counts == {"duration_off": 1, "off_campaign": 1}
    first = json.loads(con.execute("SELECT qc_flags FROM recordings LIMIT 1").fetchone()[0])
    assert first["rain"] is True and first["off_campaign"] is False


def test_flagged_recordings_are_never_selected_for_encoding(con):
    apply_metadata_flags(con, THRESHOLDS)
    selected = select_recordings(con)
    assert len(selected) == len(campaign())
    assert not selected["path"].str.contains("20240719").any()


def test_flagging_is_idempotent_and_deletes_nothing(con):
    before = con.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
    first = apply_metadata_flags(con, THRESHOLDS)
    assert apply_metadata_flags(con, THRESHOLDS) == first
    assert con.execute("SELECT COUNT(*) FROM recordings").fetchone()[0] == before
