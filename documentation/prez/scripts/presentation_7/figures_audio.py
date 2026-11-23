# ruff: noqa: E501
"""Enveloppe, spectrogramme et extrait audio d'un chant net d'A. blanci (Molokoi, lecture seule).

    uv run python documentation/prez/scripts/presentation_7/figures_audio.py

Enregistrement : Molokoi F (SMA14163), 26/02/2024 16 h 30 UTC, notes A. blanci annotées
(qualité A) à 25,1 / 28,9 / 32,0 / 34,5 / 39,6 / 43,6 / 46,8 / 50,5 s. Piste lue : celle du
manifeste du corpus (`annotations/corpus_20261120/corpus.csv`). Écrit `envelope.npz`
(énergie crête par 0,5 s, large bande et bande 4,4-5,5 kHz), `spectro.png` et `extrait.wav`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, spectrogram

from blanci.core.audio import load_audio
from blanci.core.config import LOCAL_CONFIG, load_config
from blanci.core.locate import locate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ICI = Path(__file__).resolve().parents[4] / "rapports" / "presentation_7"  # données, jamais committées
ICI.mkdir(parents=True, exist_ok=True)
REL = "Projet Phénologie blanci/Troisième relevé_avril 2024/Molokoi 11042024/SM F_SMA14163/Data/SMA14163_20240226_163000.wav"
EXTRAIT = (24.0, 34.0)

cfg = load_config(LOCAL_CONFIG)
chemin = locate(REL, Path(cfg["paths"]["raw"]), renamed=cfg["audio"]["renamed"])
corpus = pd.read_csv("annotations/corpus_20261120/corpus.csv")
piste = 0
print("fichier :", chemin)
extr = pd.read_csv("annotations/extraits_annotes.csv").query("path == @REL")
if len(extr):
    ligne = corpus[corpus["recording_id"] == extr["recording_id"].iloc[0]]
    if len(ligne):
        piste = int(ligne["piste"].iloc[0])
        print("recording_id", extr["recording_id"].iloc[0], "piste", piste)
wav, sr = load_audio(chemin, piste)
print(sr, len(wav) / sr, "s")

pas = int(0.5 * sr)
n = len(wav) // pas
large = np.abs(wav[: n * pas]).reshape(n, pas).max(axis=1)
sos = butter(6, [4400, 5500], btype="band", fs=sr, output="sos")
bande = np.abs(sosfiltfilt(sos, wav))[: n * pas].reshape(n, pas).max(axis=1)
np.savez(ICI / "envelope.npz", large=large, bande=bande, pas_s=0.5)

a, b = (int(t * sr) for t in EXTRAIT)
clip = wav[a:b] / (np.abs(wav[a:b]).max() + 1e-9) * 0.9
sf.write(ICI / "extrait.wav", clip, sr)
f, t, S = spectrogram(wav[a:b], fs=sr, nperseg=1024, noverlap=768)
S = 10 * np.log10(S + 1e-12)
fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
garde = f <= 12000
ax.pcolormesh(t, f[garde] / 1000, S[garde], shading="auto", cmap="magma",
              vmin=np.percentile(S[garde], 20), vmax=np.percentile(S[garde], 99.7))
ax.set_ylabel("fréquence (kHz)", color="#5B6B63")
ax.set_xlabel("temps (s)", color="#5B6B63")
ax.tick_params(colors="#5B6B63")
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(ICI / "spectro.png", transparent=True)

fig, axes = plt.subplots(2, 1, figsize=(12, 4))
axes[0].bar(np.arange(n) * 0.5, large, width=0.45)
axes[1].bar(np.arange(n) * 0.5, bande, width=0.45, color="g")
fig.savefig(ICI / "envelope_controle.png", dpi=70)
print("écrit")
