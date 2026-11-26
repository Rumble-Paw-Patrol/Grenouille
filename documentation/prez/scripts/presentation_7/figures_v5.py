# ruff: noqa: E501
"""Figures de la prez PNA v5 (lecture seule), dans `rapports/presentation_7/`.

    uv run python documentation/prez/scripts/presentation_7/figures_v5.py

- `carte_guyane.png` : la Guyane (contour IGN de la région, `carte/region-guyane.geojson`, dépôt
  public gregoiredavid/france-geojson, téléchargé à part) et les trois sites dont la position est
  connue (Kaw, Molokoï, Trésor : moyenne des points de pose de `data/reports/carte_points_ecoute.html`),
  en vert si A. blanci a été entendu dans mes écoutes (fenêtres annotées BirdNET). Ces sites sont
  publics (sentiers balisés) ; les points de pose eux-mêmes ne sont pas montrés ;
- `tete_apprend.gif` : l'entraînement de la tête (donnees.json, « entrainement » : itérations de
  L-BFGS depuis des poids nuls, le solveur et l'objectif de scikit-learn). À chaque passe : les 16
  poids affichés (vert : pousse vers A. blanci, rouge : vers autre chose), le score de notre
  fenêtre, et les scores de 140 fenêtres annotées (70 avec chant, 70 sans) qui se séparent. Les
  lignes sont alignées sur les 17 lignes du vecteur de la diapo.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

from blanci.core.config import LOCAL_CONFIG, load_config

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
ICI = ROOT / "rapports" / "presentation_7"
D = json.loads((ICI / "donnees.json").read_text(encoding="utf-8"))
INK, MUTED, SAGE, OR = "#1B2A24", "#5B6B63", "#A7BFB2", "#E9B44C"
POS, NEG, GRIS = "#1E6B45", "#C8553D", "#B9C3BD"
DARK = "#0E2A24"
plt.rcParams.update({"font.family": ["Calibri", "DejaVu Sans"], "font.size": 12})
cfg = load_config(LOCAL_CONFIG)


def fr(x, d=2, signe=False):
    s = f"{x:+.{d}f}" if signe else f"{x:.{d}f}"
    return s.replace(".", ",").replace("-", "−")


# ---------------------------------------------------------------- carte de la Guyane
geo = json.loads((ICI / "carte" / "region-guyane.geojson").read_text(encoding="utf-8"))
geom = geo["features"][0]["geometry"] if "features" in geo else geo["geometry"]
polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
COS = np.cos(np.deg2rad(4.0))  # équirectangulaire à 4° N : 1 unité = 111,3 km dans les deux sens

html = (ROOT / "data" / "reports" / "carte_points_ecoute.html").read_text(encoding="utf-8")
marques = pd.DataFrame(json.loads(re.search(r"const D = (\{.*?\});\s*\n", html, re.S).group(1))["marques"])
a = np.load(ICI / "birdnet_annotees.npz")
con = sqlite3.connect(f"file:{Path(cfg['paths']['db']).as_posix()}?mode=ro", uri=True)
rec = pd.read_sql_query("SELECT recording_id, site, mic_id FROM recordings", con).set_index("recording_id")
ann = pd.DataFrame({"recording_id": a["recording_id"], "y": a["y"]}).join(rec, on="recording_id")
par_site = ann.groupby("site")["y"].max()
sites = marques.groupby("site").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("micro", "size"))
sites["etat"] = [int(par_site.get(s_, -1)) for s_ in sites.index]
print(sites)

fig = plt.figure(figsize=(3.7, 4.15), dpi=250)
ax = fig.add_axes([0.0, 0.11, 1.0, 0.89])
for poly in polys:
    ext = np.array(poly[0])
    ax.add_patch(matplotlib.patches.Polygon(np.c_[ext[:, 0] * COS, ext[:, 1]], closed=True, fc="#DCE8DF", ec="#7FA08F", lw=0.8, zorder=1))
ax.set_xlim(-54.75 * COS, -51.45 * COS)
ax.set_ylim(2.0, 6.0)
ax.set_aspect("equal")
ax.set_axis_off()
for nom, lon, lat in (("Océan\nAtlantique", -52.0, 5.55), ("Suriname", -54.55, 3.6), ("Brésil", -51.9, 3.05)):
    ax.text(lon * COS, lat, nom, fontsize=9.5, color="#8A968F", style="italic", ha="center", va="center", linespacing=1.0)
ax.plot([-52.326 * COS], [4.935], marker="s", ms=4.5, color=INK, zorder=4)
ax.annotate("Cayenne", (-52.326 * COS, 4.935), xytext=(-5, 4), textcoords="offset points", ha="right", va="bottom", fontsize=10, color=INK)
NOMS = {"Tresor": "Trésor", "Molokoi": "Molokoï"}
DECAL = {"Tresor": (7, 8, "left"), "Kaw": (8, -5, "left"), "Molokoi": (-7, -11, "right")}
for site, r in sites.iterrows():
    coul = POS if r.etat == 1 else NEG if r.etat == 0 else GRIS
    ax.scatter([r.lon * COS], [r.lat], s=70, color=coul, edgecolors="white", linewidths=1.3, zorder=5)
    dx, dy, ha = DECAL.get(site, (8, 0, "left"))
    ax.annotate(NOMS.get(site, site), (r.lon * COS, r.lat), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=11, fontweight="bold", color=INK, zorder=6)
L = 50 / 111.32  # barre d'échelle de 50 km
x0b, y0b = -52.55 * COS, 2.2
ax.plot([x0b, x0b + L], [y0b, y0b], color=INK, lw=2.5, solid_capstyle="butt")
ax.text(x0b + L / 2, y0b + 0.07, "50 km", ha="center", va="bottom", fontsize=9.5, color=INK)
ax.annotate("N", (-54.35 * COS, 5.85), xytext=(-54.35 * COS, 5.45), ha="center", va="top", fontsize=11, fontweight="bold", color=INK, arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.3})
h = [matplotlib.lines.Line2D([], [], ls="", marker="o", ms=8.5, color=POS, mec="white", label="A. blanci entendu (mes écoutes)")]
if (sites.etat == 0).any():
    h.append(matplotlib.lines.Line2D([], [], ls="", marker="o", ms=8.5, color=NEG, mec="white", label="écouté, pas entendu"))
fig.legend(handles=h, loc="lower center", fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3)
fig.savefig(ICI / "carte_guyane.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- l'entraînement de la tête
E = D["entrainement"]
W = np.array(E["poids"])  # (itérations + 1, 16)
SH = np.array(E["score_hero"])
SE = np.array(E["scores_ech"])  # (itérations + 1, 140)
YE = np.array(E["y_ech"])
NIT = len(W) - 1
WMAX = float(np.abs(W[-1]).max())
VY0, VH, VW = 1.95, 0.255, 1.35  # le vecteur de la diapo (deck.py)
GX0, GY0, GWI, GHI = 2.72, 1.7, 5.95, 5.2  # place du GIF sur la diapo, en pouces
VERT_C, ROUGE_C, CASE0 = np.array([0x5C, 0xC9, 0x8B]) / 255, np.array([0xE0, 0x7A, 0x5F]) / 255, np.array([0x1F, 0x3D, 0x36]) / 255
XW = 0.55  # colonne des poids, dans le GIF
SIGX, SIGY = 2.75, VY0 - GY0 + 8.5 * VH
jit = np.random.default_rng(0).uniform(-0.09, 0.09, len(YE))


def image(it):
    w, sh, se = W[it], SH[it], SE[it]
    fig = plt.figure(figsize=(GWI, GHI), dpi=160, facecolor=DARK)  # ≥ 20 px par corps : en dessous, matplotlib dessine invisibles les bitmaps intégrés de Calibri
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, GWI)
    ax.set_ylim(GHI, 0)
    ax.set_axis_off()
    ax.text(0.27, SIGY, "×", color="white", fontsize=26, ha="center", va="center")
    lignes = [r for r in range(17) if r != 8]
    for j, r in enumerate(lignes):
        y = VY0 - GY0 + r * VH
        t = min(1.0, abs(w[j]) / WMAX)
        base = VERT_C if w[j] >= 0 else ROUGE_C
        fond = (1 - t) * CASE0 + t * base
        ax.add_patch(matplotlib.patches.Rectangle((XW, y), VW, VH, fc=fond, ec="#3D5F56", lw=0.6))
        ax.text(XW + VW / 2, y + VH / 2, fr(w[j], 2, signe=abs(w[j]) >= 0.005), color=DARK if t > 0.55 else "white", fontsize=12.5, ha="center", va="center", fontweight="bold" if t > 0.55 else "normal")
        ax.plot([XW + VW + 0.1, SIGX - 0.3], [y + VH / 2, SIGY], color="#5E8479", lw=0.7, alpha=0.6, zorder=1)
    ax.text(XW + VW / 2, VY0 - GY0 + 8.5 * VH, "⋮", color="#9CCFC4", fontsize=16, ha="center", va="center")
    bas = VY0 - GY0 + 17 * VH
    for xx, sgn in ((XW - 0.08, 1), (XW + VW + 0.06, -1)):  # crochets, comme ceux de la signature
        ax.plot([xx + 0.01, xx + 0.01], [VY0 - GY0 - 0.04, bas + 0.04], color="white", lw=1.4)
        for yy in (VY0 - GY0 - 0.03, bas + 0.03):
            ax.plot([xx + 0.01, xx + 0.01 + 0.08 * sgn], [yy, yy], color="white", lw=1.4)
    ax.text(XW + VW / 2, bas + 0.27, "les poids appris", color=SAGE, fontsize=12, ha="center", va="center")
    ax.add_patch(matplotlib.patches.Circle((SIGX, SIGY), 0.3, fc="#173D34", ec="#9CCFC4", lw=2, zorder=3))
    ax.text(SIGX, SIGY, "Σ", color="white", fontsize=21, ha="center", va="center", fontweight="bold", zorder=4)
    ax.annotate("", (3.45, SIGY), (SIGX + 0.33, SIGY), arrowprops={"arrowstyle": "-|>", "color": "#9CCFC4", "lw": 2})
    # compteur de passes et score de notre fenêtre
    ax.text(3.6, 0.42, f"passe {it}", color="white", fontsize=16, fontweight="bold", va="center")
    ax.text(3.6, 0.75, f"{E['n_fenetres']:,} fenêtres d'apprentissage".replace(",", " "), color=SAGE, fontsize=11.5, va="center")
    ax.text(3.6, SIGY - 0.5, "score de notre fenêtre", color=SAGE, fontsize=12, va="center")
    ax.text(3.6, SIGY + 0.05, fr(sh, 2), color=OR, fontsize=34, fontweight="bold", va="center", family="Cambria")
    # mes fenêtres annotées : leurs scores se séparent
    x0, x1, yt = 3.55, 5.75, 3.55
    ax.text(x0, yt, "mes fenêtres annotées", color=SAGE, fontsize=11.5, va="center")
    for k, (lab, yy, coul) in enumerate((("avec chant", yt + 0.42, VERT_C), ("sans chant", yt + 0.82, ROUGE_C))):
        m = YE == (1 - k)
        ax.scatter(x0 + (x1 - x0) * se[m], yy + jit[m], s=11, color=coul, alpha=0.8, linewidths=0, zorder=3)
        ax.text(x0 - 0.06, yy, lab, color="white", fontsize=10, ha="right", va="center")
    ax.plot([x0, x1], [yt + 1.12, yt + 1.12], color="#5E8479", lw=1)
    for v, lab in ((0, "0"), (0.5, "0,5"), (1, "1")):
        ax.plot([x0 + (x1 - x0) * v] * 2, [yt + 1.08, yt + 1.16], color="#5E8479", lw=1)
        ax.text(x0 + (x1 - x0) * v, yt + 1.3, lab, color=SAGE, fontsize=10, ha="center", va="center")
    ax.plot([x0 + (x1 - x0) * 0.5] * 2, [yt + 0.22, yt + 1.05], color="#5E8479", lw=0.8, ls=(0, (3, 3)))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=DARK)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# images : passes 0 à 12 une à une, puis de plus en plus espacées jusqu'à la dernière
pas = sorted(set(list(range(min(13, NIT + 1))) + np.unique(np.geomspace(13, NIT, 30).round().astype(int)).tolist()))
pas = [p for p in pas if p <= NIT]
durees = [1200] + [140] * (len(pas) - 2) + [3500]
images = [image(p) for p in pas]
mosaique = Image.new("RGB", (images[0].width * 4, images[0].height))
for k, im in enumerate([images[0], images[3], images[len(images) // 2], images[-1]]):
    mosaique.paste(im, (k * images[0].width, 0))
pal = mosaique.quantize(colors=200, method=Image.Quantize.MEDIANCUT)
images_p = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
images_p[0].save(ICI / "tete_apprend.gif", save_all=True, append_images=images_p[1:], duration=durees, loop=0, disposal=2)
images[-1].save(ICI / "tete_apprend_fin.png")
images[3].save(ICI / "tete_apprend_debut.png")
print("écrit carte_guyane.png, tete_apprend.gif :", len(pas), "images, passes", pas[:5], "…", pas[-3:], "; score final de notre fenêtre", SH[-1])
