"""Module séquentiel en amont : seuillage spectral et portes de rythme (DECISIONS n° 90, 103)."""

import numpy as np
import pandas as pd
import pytest

from blanci.config import load_config
from blanci.db import connect, recording_id_for
from blanci.embed import embed_recordings
from blanci.encoders import _with_transforms
from blanci.encoders.base import BaseEncoder
from blanci.encoders.upstream import UpstreamEncoder
from blanci.head import oof_scores
from blanci.sequential import (
    GATED_SCORE,
    Upstream,
    apply_gate,
    bandpass,
    denoise,
    gate_mask,
    gate_sweep,
    gate_values,
    upstream_from_cfg,
)
from blanci.store import EmbeddingStore, gated_mask

SR = 32_000
SIGNAL = load_config()["signal"]


def tone(hz, seconds=1.0, amplitude=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def song(duration_s=5.0, noise=0.01, seed=0, with_notes=True):
    """Notes de 0,094 s à 4,75 kHz toutes les 1,414 s, dans du bruit blanc."""
    rng = np.random.default_rng(seed)
    wav = rng.normal(0, noise, int(SR * duration_s))
    if with_notes:
        envelope = np.hanning(int(SR * 0.094))
        note = envelope * np.sin(2 * np.pi * 4750 * np.arange(len(envelope)) / SR)
        for start in np.arange(0.4, duration_s - 0.1, 1.414):
            i = int(start * SR)
            wav[i : i + len(note)] += 0.3 * note
    return wav.astype(np.float32)


# --- Transformations --------------------------------------------------------------------------


def test_bandpass_keeps_the_note_band_and_removes_the_rest():
    note, low, high = tone(4750), tone(800), tone(12_000)
    assert np.std(bandpass(note, SR, (3000, 7000))) > 0.95 * np.std(note)
    assert np.std(bandpass(low, SR, (3000, 7000))) < 0.02 * np.std(low)
    assert np.std(bandpass(high, SR, (3000, 7000))) < 0.02 * np.std(high)


def test_bandpass_works_on_a_batch_of_windows():
    batch = np.stack([tone(4750), tone(800)])
    out = bandpass(batch, SR, (3000, 7000))
    assert out.shape == batch.shape and np.std(out[1]) < 0.02


def test_denoise_lowers_the_stationary_background_more_than_the_notes():
    rng = np.random.default_rng(1)
    background = rng.normal(0, 0.05, SR * 3).astype(np.float32)
    burst = np.zeros_like(background)
    burst[SR : SR + 3200] = 0.5 * np.sin(2 * np.pi * 4750 * np.arange(3200) / SR)
    cleaned = denoise(background + burst, SR)
    quiet = slice(0, SR // 2)
    note = slice(SR, SR + 3200)
    before = np.std((background + burst)[note]) / np.std(background[quiet])
    after = np.std(cleaned[note]) / np.std(cleaned[quiet])
    assert after > 2 * before  # le rapport note / fond s'améliore nettement
    assert cleaned.shape == background.shape


# --- Réglages -----------------------------------------------------------------------------------


def test_default_config_leaves_everything_off():
    upstream = upstream_from_cfg(load_config())
    assert not upstream.active and upstream.transforms == {} and upstream.gates == {}


def test_command_line_choice_replaces_the_switches():
    upstream = upstream_from_cfg(load_config(), "bandpass,notes")
    assert list(upstream.transforms) == ["bandpass"]
    assert upstream.transforms["bandpass"]["band_hz"] == [3000, 7000]
    assert upstream.gates == {"notes": 1.0}
    assert upstream.transform_tag() == "bp3-7k"
    assert upstream.gate_tag() == "g-notes1-all"
    assert not upstream_from_cfg(load_config(), "none").active


def test_config_switches_and_thresholds_are_read():
    cfg = load_config()
    cfg["sequential"]["upstream"]["denoise"]["enabled"] = True
    cfg["sequential"]["upstream"]["gates"]["band_energy"] = {"enabled": True, "min_db": 9.0}
    cfg["sequential"]["upstream"]["gates"]["combine"] = "any"
    upstream = upstream_from_cfg(cfg)
    assert list(upstream.transforms) == ["denoise"]
    assert upstream.gates == {"band_energy": 9.0} and upstream.combine == "any"


def test_unknown_feature_is_refused():
    with pytest.raises(ValueError, match="inconnues"):
        upstream_from_cfg(load_config(), "bandpass,magic")


# --- Portes ---------------------------------------------------------------------------------------


def test_gate_mask_all_any_and_missing_values():
    values = pd.DataFrame({"notes": [0, 2, np.nan, 3], "rhythm": [0, 0, 1, 2]})
    gates = {"notes": 1, "rhythm": 1}
    assert gate_mask(values, gates, "all").tolist() == [False, False, True, True]
    assert gate_mask(values, gates, "any").tolist() == [False, True, True, True]
    assert gate_mask(values, {}).all()


def test_gate_values_see_the_notes_and_their_rhythm():
    values = gate_values(np.stack([song(), song(with_notes=False)]), SR, SIGNAL)
    assert values.loc[0, "notes"] >= 3 and values.loc[0, "rhythm"] >= 2
    assert values.loc[1, "notes"] == 0 and values.loc[1, "rhythm"] == 0
    assert values.loc[0, "band_energy"] > values.loc[1, "band_energy"]


def test_gate_sweep_counts_lost_positives_and_stopped_negatives():
    values = pd.DataFrame(
        {
            "band_energy": [10, 2, 1, 0],
            "band_contrast": [5, 5, 0, 0],
            "notes": [3, 0, 0, 0],
            "rhythm": [2, 0, 0, 0],
        }
    )
    y = np.array([1, 1, 0, 0])
    recordings = np.array(["a", "b", "c", "d"])
    sweep = gate_sweep(values, y, recordings, {"notes": (1,), "band_contrast": (3.0,)})
    notes = sweep[sweep["gate"] == "notes"].iloc[0]
    assert notes["neg_stopped"] == 1.0 and notes["pos_windows_lost"] == 1
    assert notes["recall_ceiling"] == 0.5
    contrast = sweep[sweep["gate"] == "band_contrast"].iloc[0]
    assert contrast["recall_ceiling"] == 1.0 and contrast["neg_stopped"] == 1.0


def test_a_positive_recording_is_lost_only_if_all_its_positive_windows_are_stopped():
    values = pd.DataFrame({"notes": [0, 2], "band_energy": 0, "band_contrast": 0, "rhythm": 0})
    sweep = gate_sweep(values, np.array([1, 1]), np.array(["a", "a"]), {"notes": (1,)})
    assert sweep.loc[0, "recall_ceiling"] == 1.0 and sweep.loc[0, "pos_windows_lost"] == 1


def test_gated_windows_get_the_lowest_score():
    assert apply_gate(np.array([0.2, 3.0]), np.array([True, False])).tolist() == [0.2, GATED_SCORE]


def test_oof_scores_never_learn_from_gated_windows():
    rng = np.random.default_rng(0)
    y = np.array([1, 0] * 20)
    X = rng.normal(0, 1, (40, 4)) + y[:, None]
    groups = np.repeat(np.arange(5), 8)
    gated = np.zeros(40, dtype=bool)
    gated[:6] = True
    out = oof_scores(X, y, groups, n_splits=5, method="prototype", gated=gated)
    assert (out.values[:6] == GATED_SCORE).all() and np.isfinite(out.values[6:]).all()
    assert (out.values[6:] > GATED_SCORE).all()


# --- Encodeur transformé et portes à l'encodage ---------------------------------------------------


class HighEnergy(BaseEncoder):
    """Encodeur factice : puissance au-dessus de 9 kHz, par FFT."""

    name, version, sample_rate, window_s, dim, has_tokens = "fake", "1", SR, 1.0, 2, False

    def _forward(self, batch):
        spectrum = np.abs(np.fft.rfft(batch, axis=1)) ** 2
        freqs = np.fft.rfftfreq(batch.shape[1], 1 / SR)
        high = spectrum[:, freqs > 9000].sum(axis=1)
        band = spectrum[:, (freqs > 4000) & (freqs < 5500)].sum(axis=1)
        return np.log1p(np.stack([high, band], axis=1))  # tient en float16 dans le stock


def test_upstream_encoder_hears_only_the_band_and_has_its_own_name():
    upstream = Upstream({"bandpass": {"band_hz": [3000, 7000]}})
    inner = HighEnergy()
    wrapped = UpstreamEncoder(inner, upstream)
    assert wrapped.name == "fake+bp3-7k" and wrapped.dim == 2
    x = (tone(4750) + tone(12_000))[None, :]
    assert wrapped.embed(x, SR)[0, 0] < 0.5 * inner.embed(x, SR)[0, 0]  # log : ~×10⁻⁴
    assert wrapped.embed(x, SR)[0, 1] > 0.95 * inner.embed(x, SR)[0, 1]


def test_config_entry_transforms_wrap_the_encoder():
    cfg = load_config()
    cfg["encoders"]["models"]["fake_bp"] = {"backend": "x", "transforms": {"bandpass": {}}}
    wrapped = _with_transforms(HighEnergy(), "fake_bp", cfg, None)
    assert wrapped.name == "fake+bp3-7k"
    assert _with_transforms(HighEnergy(), "fake", cfg, upstream_from_cfg(cfg)).name == "fake"


def test_embed_with_gates_skips_windows_without_notes(tmp_path):
    import soundfile as sf

    raw, store_root = tmp_path / "raw", tmp_path / "emb"
    con = connect(tmp_path / "db.sqlite")
    wav = np.concatenate([song(4.0), song(4.0, with_notes=False, seed=3)])
    path = raw / "2026/mataroni/M1/a.wav"
    path.parent.mkdir(parents=True)
    sf.write(path, wav, SR)
    rid = recording_id_for("2026/mataroni/M1/a.wav")
    con.execute(
        "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
        "duration_s, sample_rate, channels, qc_flags) VALUES (?, ?, '2026', 'mataroni', 'M1', "
        "'2026-02-10T13:00:00Z', 8.0, ?, 1, '{}')",
        (rid, "2026/mataroni/M1/a.wav", SR),
    )
    con.commit()
    recordings = pd.read_sql_query("SELECT * FROM recordings", con)
    upstream = Upstream(gates={"notes": 1}, signal_cfg=SIGNAL)
    report = embed_recordings(con, HighEnergy(), recordings, raw, store_root, gates=upstream)
    assert report.encoder_id == "fake-1+g-notes1-all"
    meta, emb = EmbeddingStore(store_root, report.encoder_id).load()
    gated = gated_mask(meta)
    assert report.gated == gated.sum() > 0 and (~gated).sum() > 0
    assert (emb[gated] == 0).all()
    late = meta["offset_s"].to_numpy() >= 5.0  # dernières fenêtres : fond seul
    assert gated[late].all() and not gated[meta["offset_s"].to_numpy() == 0.0].any()


def test_store_keeps_the_gated_column_across_writes(tmp_path):
    store = EmbeddingStore(tmp_path, "fake-1")
    first = pd.DataFrame({"window_id": ["a", "b"], "recording_id": "r", "offset_s": [0.0, 1.0]})
    store.write(first, np.ones((2, 2)), "2026", "s", "202602")
    second = first.iloc[:1].assign(window_id="c", gated=True)
    store.write(second, np.zeros((1, 2)), "2026", "s", "202602")
    meta, _ = store.load()
    assert dict(zip(meta["window_id"], gated_mask(meta), strict=True)) == {
        "a": False,
        "b": False,
        "c": True,
    }


def test_rhythm_gates_count_the_recording_onsets():
    """Notes et rythme se comptent sur les débuts de notes de l'enregistrement, les mêmes que
    le module en parallèle et la chaîne de décision (DECISIONS n° 103)."""
    windows = np.stack([song(with_notes=False), song(with_notes=False, seed=2)])
    onsets = np.array([0.4, 1.8, 3.2, 7.0])  # trois notes au rythme d'A. blanci, puis une
    values = gate_values(
        windows, SR, SIGNAL, offsets_s=np.array([0.0, 5.0]), onsets=onsets, dur_s=5.0
    )
    assert values.loc[0, "notes"] == 3 and values.loc[0, "rhythm"] == 2
    assert values.loc[1, "notes"] == 1 and values.loc[1, "rhythm"] == 0


def test_old_prefilter_section_is_still_read():
    cfg = load_config()
    cfg.pop("sequential")
    cfg["prefilter"] = {"gates": {"notes": {"enabled": True, "min_count": 2}}}
    assert upstream_from_cfg(cfg).gates == {"notes": 2.0}


def test_decision_chain_uses_the_rhythm_gates_or_notes_by_default():
    from blanci.sequential import onset_gates

    cfg = load_config()
    assert onset_gates(cfg) == ({"notes": 1.0}, "all")
    cfg["sequential"]["upstream"]["gates"]["rhythm"] = {"enabled": True, "min_count": 2}
    cfg["sequential"]["upstream"]["gates"]["band_energy"]["enabled"] = True
    assert onset_gates(cfg) == ({"rhythm": 2.0}, "all")  # les portes spectrales : à l'encodage
