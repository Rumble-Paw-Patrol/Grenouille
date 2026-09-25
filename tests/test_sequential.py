import numpy as np
import pytest

from blanci.sequential import (
    band_envelope_db,
    detect_onsets,
    persistence_features,
    rhythm_features,
    sequential_features,
)

# Structure du chant d'A. blanci (§1, Fouquet et al.) : note de 0,090–0,103 s à 4,48–5,41 kHz,
# intervalle entre notes de 1,200–1,906 s (moyenne 1,414 s).
SR = 32000
NOTE_S = 0.094
IOI_S = 1.414
FREQ_HZ = 4750.0


def blanci_song(duration_s=12.0, ioi_s=IOI_S, freq_hz=FREQ_HZ, noise=0.01, seed=0):
    """Bouffées tonales régulières noyées dans du bruit blanc : un chant d'A. blanci idéalisé."""
    rng = np.random.default_rng(seed)
    n = int(SR * duration_s)
    wav = rng.normal(0, noise, n)
    envelope = np.hanning(int(SR * NOTE_S))  # attaque et chute douces, comme une vraie note
    t = np.arange(len(envelope)) / SR
    note = envelope * np.sin(2 * np.pi * freq_hz * t)
    starts = np.arange(0.5, duration_s - NOTE_S, ioi_s)
    for start in starts:
        i = int(start * SR)
        wav[i : i + len(note)] += note
    return wav.astype(np.float64), starts


# --- Enveloppe et onsets ----------------------------------------------------------------------


def test_envelope_rises_on_the_notes():
    wav, starts = blanci_song()
    env = band_envelope_db(wav, SR, (4400, 5500))
    on_note = env[int((starts[0] + NOTE_S / 2) * SR)]
    between = env[int((starts[0] + IOI_S / 2) * SR)]
    assert on_note > between + 20


def test_envelope_ignores_energy_outside_the_band():
    """Une tonale à 1 kHz ne doit pas lever l'enveloppe de la bande 4,4–5,5 kHz."""
    t = np.arange(int(SR * 2.0)) / SR
    out_of_band = np.sin(2 * np.pi * 1000 * t)
    in_band = np.sin(2 * np.pi * FREQ_HZ * t)
    assert band_envelope_db(in_band, SR, (4400, 5500)).mean() > (
        band_envelope_db(out_of_band, SR, (4400, 5500)).mean() + 40
    )


def test_onsets_recover_the_song_rhythm():
    """Le test de référence : des notes toutes les 1,414 s doivent redonner cet IOI."""
    wav, starts = blanci_song()
    onsets = detect_onsets(wav, SR)
    assert len(onsets) == pytest.approx(len(starts), abs=1)
    assert float(np.median(np.diff(onsets))) == pytest.approx(IOI_S, abs=0.02)


def test_onsets_land_on_the_notes():
    wav, starts = blanci_song()
    onsets = detect_onsets(wav, SR)
    for onset in onsets:
        assert np.min(np.abs(starts - onset)) < 0.03


def test_onsets_reject_notes_that_are_too_long():
    """Filtre de durée : une tonale continue en bande n'est pas une note d'A. blanci."""
    t = np.arange(int(SR * 5.0)) / SR
    continuous = np.sin(2 * np.pi * FREQ_HZ * t)
    assert len(detect_onsets(continuous, SR)) == 0


def test_onsets_are_silent_on_noise_alone():
    noise = np.random.default_rng(0).normal(0, 0.01, int(SR * 10))
    assert len(detect_onsets(noise, SR)) <= 2


def test_onsets_survive_a_noisier_recording():
    """« malgré la pluie » : le seuil médiane + k·MAD est relatif au fond de l'enregistrement."""
    wav, starts = blanci_song(noise=0.05)
    onsets = detect_onsets(wav, SR)
    assert len(onsets) >= len(starts) - 1


# --- Rythme -----------------------------------------------------------------------------------


def test_rhythm_flags_blanci_intervals():
    _, starts = blanci_song()
    features = rhythm_features(starts, duration_s=12.0)
    assert features["ioi_median_s"] == pytest.approx(IOI_S, abs=1e-6)
    assert features["frac_ioi_blanci"] == 1.0
    assert features["frac_ioi_short"] == 0.0
    assert features["ioi_cv"] == pytest.approx(0.0, abs=1e-9)


def test_rhythm_flags_a_chorus_as_short_intervals():
    """H21 : plusieurs chanteurs → des IOI courts et une densité d'onsets élevée."""
    chorus = np.sort(np.concatenate([np.arange(0, 12, 1.414) + shift for shift in (0.0, 0.3, 0.7)]))
    features = rhythm_features(chorus, duration_s=12.0)
    assert features["frac_ioi_short"] > 0.5
    assert (
        features["onset_rate_hz"] > rhythm_features(np.arange(0, 12, 1.414), 12.0)["onset_rate_hz"]
    )


def test_rhythm_is_defined_without_onsets():
    features = rhythm_features(np.array([]), duration_s=120.0)
    assert features["onset_rate_hz"] == 0.0
    assert np.isnan(features["ioi_median_s"])
    assert features["frac_ioi_blanci"] == 0.0


# --- Persistance ------------------------------------------------------------------------------


def test_persistence_sees_a_continuous_song():
    """Chant continu (H20) : presque toutes les fenêtres positives, une seule longue série."""
    scores = np.full(40, 0.9)
    features = persistence_features(scores, threshold=0.5)
    assert features["frac_windows"] == 1.0
    assert features["longest_run"] == 40
    assert features["n_isolated"] == 0


def test_persistence_isolates_a_single_window():
    """Une détection isolée : le signal d'un faux positif probable (§1, file « suspect »)."""
    scores = np.zeros(40)
    scores[7] = 0.9
    features = persistence_features(scores, threshold=0.5)
    assert features["n_positive"] == 1
    assert features["longest_run"] == 1
    assert features["n_isolated"] == 1


def test_persistence_counts_runs_not_windows():
    scores = np.array([1, 1, 1, 0, 0, 1, 0, 0, 1, 1], dtype=float)
    features = persistence_features(scores, threshold=0.5)
    assert features["n_positive"] == 6
    assert features["longest_run"] == 3
    assert features["n_isolated"] == 1


def test_persistence_span_uses_offsets_when_given():
    scores = np.array([1, 0, 0, 0, 1], dtype=float)
    offsets = np.array([0.0, 1.5, 3.0, 4.5, 6.0])
    assert persistence_features(scores, 0.5, offsets)["span"] == pytest.approx(6.0)
    assert persistence_features(scores, 0.5)["span"] == pytest.approx(4.0)  # en indices


def test_persistence_is_defined_on_an_empty_recording():
    features = persistence_features(np.array([]), threshold=0.5)
    assert features["frac_windows"] == 0.0 and features["n_positive"] == 0


# --- Assemblage -------------------------------------------------------------------------------


def test_sequential_features_merge_rhythm_and_persistence():
    wav, _ = blanci_song()
    onsets = detect_onsets(wav, SR)
    features = sequential_features(onsets, np.full(40, 0.9), threshold=0.5, duration_s=12.0)
    assert set(features) >= {"ioi_median_s", "frac_ioi_blanci", "frac_windows", "longest_run"}
    assert features["ioi_median_s"] == pytest.approx(IOI_S, abs=0.02)
    assert features["frac_windows"] == 1.0


def test_note_snr_tracks_the_note_level():
    """Plus la note est forte par rapport au fond, plus le RSB estimé est élevé."""
    from blanci.sequential import note_snr_db

    sr = 24_000
    rng = np.random.default_rng(0)
    note = np.hanning(int(0.1 * sr)) * np.sin(2 * np.pi * 4900 * np.arange(int(0.1 * sr)) / sr)

    def song(level):
        x = rng.normal(0, 0.02, 3 * sr)
        for start in (0.4, 1.8):
            i = int(start * sr)
            x[i : i + len(note)] += level * note
        return x

    weak, strong = note_snr_db(song(0.08), sr), note_snr_db(song(0.5), sr)
    assert 0 < weak < strong
    assert np.isnan(note_snr_db(rng.normal(0, 0.02, 3 * sr), sr))  # pas de note


# --- Débuts de notes stockés, rythme par fenêtre, persistance par enregistrement --------------


def test_onsets_round_trip_through_the_database(tmp_path):
    from blanci.db import connect
    from blanci.sequential import load_onsets, store_onsets

    con = connect(tmp_path / "db.sqlite")
    con.execute(
        "INSERT INTO recordings (recording_id, path, dataset) VALUES ('r1', 'a.wav', '2026')"
    )
    store_onsets(con, "r1", np.array([0.5, 1.9, 3.3]), 0)
    assert load_onsets(con)["r1"].tolist() == [0.5, 1.9, 3.3]
    assert load_onsets(con, {"autre"}) == {}


def test_window_rhythm_counts_only_notes_inside_the_window():
    import pandas as pd

    from blanci.sequential import window_rhythm

    onsets = {"r1": np.array([0.5, 1.9, 3.3, 4.7, 10.0])}
    windows = pd.DataFrame(
        {"recording_id": ["r1", "r1", "r2"], "offset_s": [0.0, 6.0, 0.0], "dur_s": [5.0, 5.0, 5.0]}
    )
    rhythm = window_rhythm(windows, onsets)
    assert rhythm.loc[0, "onset_rate_hz"] == pytest.approx(4 / 5)
    assert rhythm.loc[0, "frac_ioi_blanci"] == 1.0  # IOI de 1,4 s : A. blanci
    assert rhythm.loc[1, "onset_rate_hz"] == pytest.approx(1 / 5)  # la note à 10 s
    assert np.isnan(rhythm.loc[2, "onset_rate_hz"])  # pas de débuts de notes calculés


def test_recording_persistence_uses_every_window_of_the_recording():
    import pandas as pd

    from blanci.sequential import recording_persistence

    scores = pd.DataFrame(
        {
            "recording_id": ["a"] * 6 + ["b"] * 3,
            "offset_s": [0, 1.5, 3, 4.5, 6, 7.5, 0, 1.5, 3],
            "score": [1, 1, 1, -1, 1, -1, -1, 2, -1],
        }
    )
    table = recording_persistence(scores, threshold=0.0)
    assert table.loc["a", "frac_windows"] == pytest.approx(4 / 6)
    assert table.loc["a", "longest_run"] == 3 and table.loc["a", "n_isolated"] == 1
    assert table.loc["b", "n_positive"] == 1 and table.loc["b", "n_isolated"] == 1


# --- Faux négatifs suspects : fenêtres négatives encadrées de positives (DECISIONS n° 102) --------


def test_surrounded_needs_a_positive_on_both_sides_within_the_radius():
    from blanci.sequential import surrounded_by_positives

    positives = np.array([10.0, 20.0])
    centers = np.array([15.0, 5.0, 25.0, 15.0])
    assert surrounded_by_positives(centers, positives, 6.0).tolist() == [True, False, False, True]
    assert not surrounded_by_positives(np.array([15.0]), positives, 4.0).any()  # trop loin
    assert not surrounded_by_positives(np.array([15.0]), np.array([]), 6.0).any()


def test_persistence_counts_the_gaps_inside_a_song():
    """Miroir des détections isolées : un trou dans une série positive est compté."""
    scores = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    offsets = np.arange(len(scores)) * 1.5
    features = persistence_features(scores, 0.5, offsets, gap_radius_s=3.0)
    assert features["n_gaps"] == 1
    assert persistence_features(scores, 0.5)["n_gaps"] == 1  # sans décalages : 2 fenêtres
