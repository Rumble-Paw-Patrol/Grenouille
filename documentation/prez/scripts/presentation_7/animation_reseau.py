# ruff: noqa: E501
"""Animation du réseau de neurones (GIF en boucle) et spectrogramme sur fond sombre, pour la diapo
« dans BirdNET » de la prez PNA v3. Écrit dans `rapports/presentation_7/` :
- `reseau.gif` : un schéma de réseau (5 couches) où des impulsions lumineuses traversent les
  connexions de gauche à droite et allument les neurones au passage. Schéma pédagogique : BirdNET
  est un réseau convolutif bien plus grand ; seul le principe (des couches qui s'activent l'une
  après l'autre) est montré ;
- `fenetre_spectro_sombre.png` : le spectrogramme de la fenêtre 24-27 s (bande 0-15 kHz de
  BirdNET), sans axes, pour l'entrée du réseau.

    uv run python documentation/prez/scripts/presentation_7/animation_reseau.py
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy.signal import spectrogram

from blanci.core.audio import load_audio
from blanci.core.config import LOCAL_CONFIG, load_config
from blanci.core.locate import locate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
ICI = ROOT / "rapports" / "presentation_7"
DARK = "#0E2A24"
ARETE, NEURONE, BORD = "#24564B", "#173D34", "#4E8C7C"
OR, CYAN = np.array([233, 180, 76]) / 255, np.array([156, 224, 208]) / 255
REL = "Projet Phénologie blanci/Troisième relevé_avril 2024/Molokoi 11042024/SM F_SMA14163/Data/SMA14163_20240226_163000.wav"

# ---------------------------------------------------------------- spectrogramme sombre
cfg = load_config(LOCAL_CONFIG)
wav, sr = load_audio(locate(REL, Path(cfg["paths"]["raw"]), renamed=cfg["audio"]["renamed"]), 0)
f, t, S = spectrogram(wav[24 * sr : 27 * sr], fs=sr, nperseg=1024, noverlap=896)
S = 10 * np.log10(S + 1e-12)
g = f <= 15000
fig = plt.figure(figsize=(3.2, 3.0), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.pcolormesh(t, f[g], S[g], shading="auto", cmap="magma", vmin=np.percentile(S[g], 20), vmax=np.percentile(S[g], 99.8))
ax.set_axis_off()
fig.savefig(ICI / "fenetre_spectro_sombre.png", facecolor=DARK)
plt.close(fig)

# ---------------------------------------------------------------- réseau
rng = np.random.default_rng(3)
COUCHES = [6, 9, 9, 7, 5]
XS = np.linspace(0.07, 0.93, len(COUCHES))
pos = [np.column_stack([np.full(n, x), np.linspace(0.5 - 0.085 * (n - 1) / 2, 0.5 + 0.085 * (n - 1) / 2, n)]) for n, x in zip(COUCHES, XS, strict=True)]
aretes = [(l_, i, j) for l_ in range(len(COUCHES) - 1) for i in range(COUCHES[l_]) for j in range(COUCHES[l_ + 1])]
actives = [a for a in aretes if rng.random() < 0.35]
force = {(l_, i): rng.uniform(0.35, 1.0) for l_ in range(len(COUCHES)) for i in range(COUCHES[l_])}
NF, VAGUES = 64, (0.0, 0.5)  # 64 images par boucle, deux vagues décalées d'une demi-boucle
D = 1.0 / (len(COUCHES) - 1) / 2  # durée d'un passage de couche, en fraction de boucle


def image(k):
    phi = k / NF
    fig = plt.figure(figsize=(6.0, 3.6), dpi=130, facecolor=DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor(DARK)
    for l_, i, j in aretes:
        (x1, y1), (x2, y2) = pos[l_][i], pos[l_ + 1][j]
        ax.plot([x1, x2], [y1, y2], color=ARETE, lw=0.6, zorder=1)
    for y in pos[0][:, 1]:
        ax.annotate("", (XS[0] - 0.012, y), (0.0, y), arrowprops={"arrowstyle": "-|>", "color": BORD, "lw": 1})
    for y in pos[-1][:, 1]:
        ax.annotate("", (1.0, y), (XS[-1] + 0.012, y), arrowprops={"arrowstyle": "-|>", "color": BORD, "lw": 1})
    act = {key: 0.0 for key in force}
    for v in VAGUES:
        ph = (phi - v) % 1.0
        for l_ in range(len(COUCHES)):
            d = ph - l_ * D  # la vague atteint la couche l à l·D
            if -0.02 < d < 0.18:
                for i in range(COUCHES[l_]):
                    act[(l_, i)] = max(act[(l_, i)], force[(l_, i)] * np.exp(-((d - 0.02) / 0.05) ** 2))
        for l_, i, j in actives:
            s = (ph - l_ * D) / D
            if 0 <= s <= 1:
                (x1, y1), (x2, y2) = pos[l_][i], pos[l_ + 1][j]
                x, y = x1 + (x2 - x1) * s, y1 + (y2 - y1) * s
                ax.plot([x1, x], [y1, y], color=OR, lw=1.6, alpha=0.55 * force[(l_, i)], zorder=2)
                ax.scatter([x], [y], s=220, color=OR, alpha=0.35, zorder=3, linewidths=0)
                ax.scatter([x], [y], s=42, color="#FFF4D6", zorder=4, linewidths=0)
    for l_ in range(len(COUCHES)):
        for i, (x, y) in enumerate(pos[l_]):
            a_ = act[(l_, i)]
            coul = (1 - a_) * np.array([0x17, 0x3D, 0x34]) / 255 + a_ * CYAN
            if a_ > 0.05:
                ax.scatter([x], [y], s=900 * a_, color=CYAN, alpha=0.3 * a_, zorder=5, linewidths=0)
            ax.scatter([x], [y], s=170, color=coul, edgecolors=BORD, linewidths=1.2, zorder=6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=DARK)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


images = [image(k) for k in range(NF)]
mosaique = Image.new("RGB", (images[0].width * 4, images[0].height))
for k, im in enumerate(images[2 : NF // 2 : NF // 8][:4]):
    mosaique.paste(im, (k * images[0].width, 0))
pal = mosaique.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
images = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
images[0].save(ICI / "reseau.gif", save_all=True, append_images=images[1:], duration=45, loop=0, disposal=2, optimize=False)
print("écrit", ICI / "reseau.gif", (ICI / "reseau.gif").stat().st_size // 1024, "Ko")
