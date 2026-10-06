"""Bandeau du tableau de bord : spectrogramme d'un vrai chant d'A. blanci, centré sur sa bande.

    uv run --group app python documentation/tableau-de-bord/spectrogramme.py

Lit l'extrait de 10 s de Mataroni de la présentation de suivi n° 2
(`documentation/prez/presentation-suivi-2/audio/blanci_net.wav`, canal 0, 24 kHz) et écrit
`spectrogramme.jpg` (versionné, publié à côté de la page). Fenêtre 3,5–6,3 kHz : la bande du chant
(4,4–5,5 kHz) au milieu, avec ce qui chante juste au-dessus et au-dessous.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft

ICI = Path(__file__).resolve().parent
SOURCE = ICI.parents[0] / "prez" / "presentation-suivi-2" / "audio" / "blanci_net.wav"
SORTIE = ICI / "spectrogramme.jpg"
BANDE = (3500, 6300)


def main() -> None:
    wav, sr = sf.read(SOURCE)
    if wav.ndim > 1:
        wav = wav[:, 0]
    n = 1024
    f, t, z = stft(wav, fs=sr, nperseg=n, noverlap=n * 7 // 8)
    s = 20 * np.log10(np.abs(z) + 1e-9)
    garde = (f >= BANDE[0]) & (f <= BANDE[1])
    vmax = np.percentile(s[garde], 99.7)
    fig = plt.figure(figsize=(14, 2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh(
        t, f[garde], s[garde], shading="gouraud", cmap="magma", vmin=vmax - 50, vmax=vmax
    )
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(*BANDE)
    ax.set_axis_off()
    fig.savefig(SORTIE, dpi=100, pil_kwargs={"quality": 82})
    plt.close(fig)
    print(f"{SORTIE.name} : {t[-1]:.1f} s, {BANDE[0] / 1000:g}–{BANDE[1] / 1000:g} kHz")


if __name__ == "__main__":
    main()
