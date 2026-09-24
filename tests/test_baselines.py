"""Baselines sans encodeur (§3) : seuillage spectral, onsets, rythme, template matching."""

import json

import numpy as np
import pytest
import soundfile as sf

from blanci.baselines import (
    BASELINES,
    evaluation_windows,
    note_patch,
    oof_template_scores,
    run_baselines,
    template_scores,
    window_features,
    write_baseline_report,
)
from blanci.dataset import benchmark_recordings
from blanci.db import connect, window_id_for
from blanci.ingest import ingest
from blanci.service import append_label

SR = 24_000
DURATION_S = 12.0
IOI_S = 1.4  # intervalle moyen entre notes d'A. blanci


def soundscape(seed, notes=False, clicks=False, note_hz=4900.0, level=0.3):
    """Fond large bande ; notes de 0,1 s en bande toutes les 1,4 s ; ou claquements large bande."""
    rng = np.random.default_rng(seed)
    n = int(SR * DURATION_S)
    x = rng.normal(0, 0.02, n)
    note = np.hanning(int(0.1 * SR))
    t = np.arange(len(note)) / SR
    for start in np.arange(0.4, DURATION_S - 0.2, IOI_S):
        i = int(start * SR)
        if notes:
            x[i : i + len(note)] += level * note * np.sin(2 * np.pi * note_hz * t)
        if clicks:
            x[i : i + len(note)] += level * note * rng.normal(0, 1.0, len(note))
    return x.astype(np.float32)


def test_a_note_train_scores_above_background():
    cfg_signal = _signal_cfg()
    frog = window_features(soundscape(0, notes=True)[: 3 * SR], SR, cfg_signal)
    background = window_features(soundscape(1)[: 3 * SR], SR, cfg_signal)
    for name in ("band_energy", "band_contrast", "notes", "rhythm"):
        assert frog.fixed[name] > background.fixed[name], name
    assert frog.fixed["notes"] >= 2 and frog.fixed["rhythm"] >= 1


def test_broadband_clicks_raise_the_band_energy_but_not_the_contrast():
    """Un son large bande (pluie, claquement) monte dans la bande de la note ET autour : le
    contraste l'ignore, l'énergie seule s'y laisse prendre."""
    cfg_signal = _signal_cfg()
    frog = window_features(soundscape(0, notes=True)[: 3 * SR], SR, cfg_signal)
    clicks = window_features(soundscape(2, clicks=True)[: 3 * SR], SR, cfg_signal)
    assert clicks.fixed["band_energy"] > 5
    assert frog.fixed["band_contrast"] > clicks.fixed["band_contrast"] + 5


def test_features_do_not_depend_on_the_recorder_sample_rate():
    """Tout est rééchantillonné à 24 kHz : un gabarit appris à 48 kHz s'applique à 32 kHz."""
    from blanci.audio import resample

    cfg_signal = _signal_cfg()
    x = soundscape(0, notes=True)[: 3 * SR]
    a = window_features(resample(x, SR, 48_000), 48_000, cfg_signal)
    b = window_features(resample(x, SR, 32_000), 32_000, cfg_signal)
    assert a.spec_db.shape == b.spec_db.shape


def test_template_matches_a_note_better_than_clicks():
    cfg_signal = _signal_cfg()
    frog = window_features(soundscape(0, notes=True)[: 3 * SR], SR, cfg_signal).spec_db
    other_frog = window_features(soundscape(5, notes=True)[: 3 * SR], SR, cfg_signal).spec_db
    clicks = window_features(soundscape(2, clicks=True)[: 3 * SR], SR, cfg_signal).spec_db
    width = 20
    template = note_patch(frog, width)
    scores = template_scores([other_frog, clicks], template, width)
    assert scores[0] > scores[1] + 0.2


def test_template_learns_only_from_annotated_positives_of_other_mics():
    """Scores hors-pli : un micro n'est jamais scoré par un gabarit appris sur lui, et un
    négatif présumé n'est jamais pris pour gabarit."""
    cfg_signal = _signal_cfg()
    specs, y, presumed, groups = [], [], [], []
    for mic in range(4):
        for k in range(3):
            specs.append(
                window_features(
                    soundscape(10 * mic + k, notes=True)[: 3 * SR], SR, cfg_signal
                ).spec_db
            )
            y.append(1)
            presumed.append(False)
            groups.append(f"m{mic}")
        for k in range(3):
            specs.append(
                window_features(soundscape(100 + 10 * mic + k)[: 3 * SR], SR, cfg_signal).spec_db
            )
            y.append(0)
            presumed.append(True)
            groups.append(f"m{mic}")
    y, presumed, groups = np.array(y), np.array(presumed), np.array(groups)
    out = oof_template_scores(specs, y, presumed, groups, width=20, n_splits=4)
    for values in out.values():
        assert np.isfinite(values).all()
        assert values[y == 1].min() > values[y == 0].max()


# --- De bout en bout, sur des fichiers --------------------------------------------------------


def _signal_cfg():
    from blanci.config import load_config

    return load_config()["signal"]


@pytest.fixture
def corpus(tmp_path, cfg):
    """4 micros × 4 jours à 10 h, stéréo. Premier jour : chant (3 fenêtres annotées) ; autres
    jours : fond seul, candidats aux négatifs appariés. Un jour hors créneau (15 h) n'en est pas."""
    raw = tmp_path / "raw"
    cfg["paths"]["raw"] = str(raw)
    cfg["qc"]["expected_duration_s"] = DURATION_S
    cfg["benchmark"] |= {"n_boot": 20, "negatives_per_positive": 6}
    cfg["head"]["n_splits"] = 4
    seed = 0
    for mic in ("M1", "M2", "M3", "M4"):
        for day, hour in ((10, 10), (11, 10), (12, 10), (13, 10), (14, 15)):
            left = soundscape(seed, notes=(day == 10))
            right = soundscape(seed + 1000, notes=(day == 10), level=0.1)
            path = raw / "2026" / "mataroni" / mic / f"{mic}_202602{day}_{hour:02d}0000.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, np.stack([left, right], axis=1), SR, subtype="PCM_16")
            seed += 1
    con = connect(tmp_path / "db.sqlite")
    ingest(con, raw, "2026", cfg, run_qc=False, hash_file=False)
    rows = con.execute("SELECT recording_id FROM recordings WHERE path LIKE '%20260210%'")
    for (rid,) in rows.fetchall():
        for offset in (0.0, 3.0, 6.0):
            wid = window_id_for(rid, offset)
            con.execute(
                "INSERT INTO windows (window_id, recording_id, offset_s, dur_s) VALUES (?,?,?,3.0)",
                (wid, rid, offset),
            )
            append_label(con, wid, "blanci", "import")
    con.commit()
    return con, cfg, raw


def test_benchmark_recordings_keeps_labelled_and_same_slot_candidates(corpus):
    con, cfg, _ = corpus
    subset = benchmark_recordings(con, 30, -3)
    assert (subset["role"] == "labelled").sum() == 4
    assert (subset["role"] == "paired_candidate").sum() == 12  # 3 jours × 4 micros, à 10 h
    assert not subset["path"].str.contains("_20260214_").any()  # 15 h : hors créneau


def test_benchmark_recordings_skips_flagged_candidates(corpus):
    con, cfg, _ = corpus
    flagged = con.execute("SELECT recording_id FROM recordings WHERE path LIKE '%M1_20260211%'")
    rid = flagged.fetchone()[0]
    con.execute(
        "UPDATE recordings SET qc_flags = ? WHERE recording_id = ?",
        (json.dumps({"duration_off": True}), rid),
    )
    subset = benchmark_recordings(con, 30, -3)
    assert rid not in set(subset["recording_id"])


def test_evaluation_windows_mix_annotations_and_paired_negatives(corpus):
    con, cfg, _ = corpus
    windows = evaluation_windows(con, cfg)
    assert windows["y"].sum() == 12
    negatives = windows[windows["y"] == 0]
    assert len(negatives) == 4 * 6 and negatives["presumed"].all()
    assert not negatives["path"].str.contains("20260210").any()
    assert windows["point"].nunique() == 4


def test_run_baselines_scores_every_baseline_on_each_channel(corpus, tmp_path):
    con, cfg, raw = corpus
    before = {p: p.stat().st_mtime_ns for p in raw.rglob("*.wav")}
    table, scores = run_baselines(con, cfg, raw, channels=(0, 1), progress_every=0)
    assert set(table["baseline"]) == set(BASELINES)
    assert set(table["channel"]) == {0, 1}
    assert set(table["level"]) == {"window", "recording"}
    assert len(table) == len(BASELINES) * 2 * 2
    best = table[(table["baseline"] == "band_contrast") & (table["level"] == "window")]
    assert (best["ap"] > 0.9).all()
    assert len(scores) == len(BASELINES) * 2 * (12 + 24)
    # L'audio est lu, jamais écrit.
    assert {p: p.stat().st_mtime_ns for p in raw.rglob("*.wav")} == before

    paths = write_baseline_report(table, scores, tmp_path / "reports")
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "go/no-go" in text and "template_max" in text
