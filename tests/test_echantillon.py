"""Banc d'essai sur l'échantillon versionné (vrai son, DECISIONS n° 113)."""

from pathlib import Path

import numpy as np
import pytest

from blanci.config import load_config
from blanci.echantillon import SAMPLE_DIR, cut_window, load_labels, run

ROOT = Path(__file__).resolve().parents[1] / SAMPLE_DIR
pytestmark = pytest.mark.skipif(not (ROOT / "labels.csv").exists(), reason="échantillon absent")


class BandEnergies:
    """Encodeur de substitution : log-énergie de 32 bandes de 250 Hz (0–8 kHz), sans poids."""

    window_s, sample_rate, name = 3.0, 48_000, "bandes"

    def embed(self, wav, sr):
        spectrum = np.abs(np.fft.rfft(wav, axis=1)) ** 2
        freqs = np.fft.rfftfreq(wav.shape[1], 1 / sr)
        edges = np.arange(0, 8001, 250)
        bands = [
            spectrum[:, (freqs >= a) & (freqs < b)].sum(axis=1)
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        ]
        return np.log10(np.stack(bands, axis=1) + 1e-12).astype(np.float32)

    def embed_tokens(self, wav, sr):
        return None


def test_labels_and_windows_come_from_the_real_clips():
    df = load_labels(ROOT)
    assert len(df) == 65 and df["y"].sum() == 10  # 66 clips, « uncertain » écarté
    row = df.iloc[0]
    center = row.fenetre_debut_dans_clip_s + row.fenetre_dur_s / 2
    wav, sr = cut_window(ROOT / row.fichier, center, 5.0)
    assert sr == 48_000 and wav.shape == (240_000,) and np.abs(wav).max() > 0


def test_sample_bench_runs_heads_regularizations_and_losses(monkeypatch, tmp_path):
    import blanci.encoders

    monkeypatch.setattr(blanci.encoders, "get_encoder", lambda name, cfg: BandEnergies())
    cfg = load_config()
    cfg["benchmark"]["n_boot"] = 20
    cfg["head"]["C_grid"] = [0.1, 1.0]
    methods = ["prototype", "logistic", "logistic+R19", "logistic+R19+R21", "loss:gce"]
    out = run(cfg, "bandes", methods, ROOT, tmp_path)
    assert set(out["table"]["head"]) == set(methods)
    assert out["n_pos"] == 10 and out["n_mics_pos"] == 5 and out["n_splits"] == 5
    assert out["table"]["ap"].between(0, 1).all()
    assert (tmp_path / "bandes.npz").exists()  # cache : le second passage ne réencode pas
    again = run(cfg, "bandes", ["logistic"], ROOT, tmp_path)
    assert again["dim"] == 32
