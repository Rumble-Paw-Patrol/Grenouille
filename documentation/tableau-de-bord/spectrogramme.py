"""Bandeau du tableau de bord : spectrogramme d'un vrai chant d'A. blanci, centré sur sa bande,
et le même extrait à écouter.

    uv run python documentation/tableau-de-bord/spectrogramme.py

Lit Molokoi SMA14163, 26/02/2024 à 16 h 30 locales, de 27 à 45 s (canal 0, 48 kHz), sur le
disque des enregistrements (`paths.raw` de la configuration, lecture seule : le disque doit
être branché), et écrit `spectrogramme.jpg` et `chant.mp3` (versionnés, publiés à côté de la
page ; le poste d'annotation les reprend dans son bandeau). Fenêtre 3–6 kHz : la bande du chant
(4,4–5,5 kHz) au milieu, avec ce qui chante juste au-dessus et au-dessous (choix de Léonard,
07/10/2026).

Le son : les mêmes 18 s, que la tête de lecture du bandeau balaie en même temps qu'on les
entend (bouton « Écouter le chant »). Passe-haut à 2 kHz (le vent et les grondements, sous tout
ce que montre l'image), crête ramenée à -1 dBFS pour qu'on entende le chant sans monter le
volume, fondus de 20 ms aux deux bouts (la page le joue en boucle, sans claquement).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.colors import PowerNorm
from scipy.signal import butter, sosfilt, stft

from blanci.core.config import config_path, default_user_config, load_config

ICI = Path(__file__).resolve().parent
SOURCE = (
    "Projet Phénologie blanci/Troisième relevé_avril 2024/Molokoi 11042024/SM F_SMA14163/Data/"
    "SMA14163_20240226_163000.wav"
)
DEBUT_S, FIN_S = 27.0, 45.0
SORTIE = ICI / "spectrogramme.jpg"
SON = ICI / "chant.mp3"
PASSE_HAUT_HZ = 2000
BANDE = (3000, 6000)
# Réglages du poste d'annotation (viewer.py) : dynamique en dB sous le maximum, et contraste,
# courbe en puissance d'exposant 1 + 0,35 × contraste sur l'échelle (Léonard, 07/10/2026).
DYNAMIQUE_DB = 70
CONTRASTE = 8


def ecrire_son(wav: np.ndarray, sr: int) -> None:
    """Le même extrait en MP3 (lu par tous les navigateurs ; libsndfile 1.1 ou plus)."""
    if "MP3" not in sf.available_formats():
        raise SystemExit("libsndfile sans MP3 : mettre à jour soundfile (0.12 ou plus)")
    son = sosfilt(butter(4, PASSE_HAUT_HZ, "highpass", fs=sr, output="sos"), wav - wav.mean())
    son *= 10 ** (-1 / 20) / max(float(np.max(np.abs(son))), 1e-9)
    rampe = np.linspace(0, 1, round(0.02 * sr))
    son[: rampe.size] *= rampe
    son[-rampe.size :] *= rampe[::-1]
    sf.write(SON, son.astype(np.float32), sr, format="MP3")


def main() -> None:
    with sf.SoundFile(config_path(load_config(default_user_config()), "raw") / SOURCE) as f:
        sr = f.samplerate
        f.seek(round(DEBUT_S * sr))
        wav = f.read(round((FIN_S - DEBUT_S) * sr), always_2d=True)[:, 0]
    n = 1024
    f, t, z = stft(wav, fs=sr, nperseg=n, noverlap=n * 7 // 8)
    s = 20 * np.log10(np.abs(z) + 1e-9)
    garde = (f >= BANDE[0]) & (f <= BANDE[1])
    vmax = np.percentile(s[garde], 99.7)
    fig = plt.figure(figsize=(14, 2))
    ax = fig.add_axes([0, 0, 1, 1])
    norme = PowerNorm(1 + 0.35 * CONTRASTE, vmin=vmax - DYNAMIQUE_DB, vmax=vmax, clip=True)
    ax.pcolormesh(t, f[garde], s[garde], shading="gouraud", cmap="magma", norm=norme)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(*BANDE)
    ax.set_axis_off()
    fig.savefig(SORTIE, dpi=100, pil_kwargs={"quality": 82})
    plt.close(fig)
    print(f"{SORTIE.name} : {t[-1]:.1f} s, {BANDE[0] / 1000:g}–{BANDE[1] / 1000:g} kHz")
    ecrire_son(wav, sr)
    print(f"{SON.name} : {len(wav) / sr:.1f} s, {SON.stat().st_size / 1e3:.0f} ko")


if __name__ == "__main__":
    main()
