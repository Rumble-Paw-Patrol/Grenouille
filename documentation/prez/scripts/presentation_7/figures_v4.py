# ruff: noqa: E501
"""Figures de la prez PNA v4 (lecture seule), dans `rapports/presentation_7/`.

    uv run python documentation/prez/scripts/presentation_7/figures_v4.py

- `spectro_grand.png` : les 10 s de Molokoï de la diapo 3, sur toute la largeur, notes d'A. blanci
  annotées repérées par un triangle doré ;
- `fenetre_spectro_plein.png` : le spectrogramme de la fenêtre 24-27 s, sans cadre sur la note ;
- `courbe_v4.png` : le rejeu d'apprentissage actif (donnees.json), étiquettes directes ;
- `carte_presence.png` : points d'écoute dont la position de pose est connue
  (`data/reports/carte_points_ecoute.html`), en vert si A. blanci a été entendu dans mes écoutes
  (fenêtres annotées BirdNET, `birdnet_annotees.npz`), en rouge si des fenêtres y ont été écoutées
  sans l'entendre, en gris si aucune. Pas de fond de carte ni de coordonnées affichées : une espèce
  protégée, et la présentation part dans un dépôt public ;
- `tete.gif` : la tête de classification, un seul neurone : les 16 nombres affichés de la signature
  entrent par des connexions dont la couleur et l'épaisseur suivent le poids appris (vert : pousse
  vers A. blanci, rouge : vers autre chose), des impulsions les parcourent, le neurone s'allume,
  le score sort. Les entrées sont alignées sur les 17 lignes du vecteur de la diapo.
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
from scipy.signal import spectrogram

from blanci.core.audio import load_audio
from blanci.core.config import LOCAL_CONFIG, load_config
from blanci.core.locate import locate

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
ICI = ROOT / "rapports" / "presentation_7"
D = json.loads((ICI / "donnees.json").read_text(encoding="utf-8"))
REL = "Projet Phénologie blanci/Troisième relevé_avril 2024/Molokoi 11042024/SM F_SMA14163/Data/SMA14163_20240226_163000.wav"
INK, MUTED, GRID, SAGE, OR = "#1B2A24", "#5B6B63", "#E4EEE8", "#A7BFB2", "#E9B44C"
POS, NEG, GRIS = "#1E6B45", "#C8553D", "#B9C3BD"
DARK = "#0E2A24"
plt.rcParams.update({"font.family": ["Calibri", "DejaVu Sans"], "font.size": 13, "axes.edgecolor": SAGE, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": MUTED, "axes.spines.top": False, "axes.spines.right": False})
virgule = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
cfg = load_config(LOCAL_CONFIG)

# ---------------------------------------------------------------- spectrogrammes
wav, sr = load_audio(locate(REL, Path(cfg["paths"]["raw"]), renamed=cfg["audio"]["renamed"]), 0)
iv = pd.read_csv(ROOT / "annotations" / "intervalles_annotes.csv")
iv = iv[(iv.valable == True) & (iv.path == REL) & iv.label.isin(["blanci", "blanci_chorus"])]  # noqa: E712


def spectro(ax, t0, t1, fmax):
    f, t, S = spectrogram(wav[int(t0 * sr) : int(t1 * sr)], fs=sr, nperseg=1024, noverlap=896)
    S = 10 * np.log10(S + 1e-12)
    g = f <= fmax
    ax.pcolormesh(t, f[g] / 1000, S[g], shading="auto", cmap="magma", vmin=np.percentile(S[g], 20), vmax=np.percentile(S[g], 99.7), rasterized=True)


fig, ax = plt.subplots(figsize=(12.1, 4.7), dpi=170)
spectro(ax, 24.0, 34.0, 12000)
notes = iv[(iv.start_s >= 24) & (iv.end_s <= 34)]
for s_, e_ in zip(notes.start_s - 24, notes.end_s - 24, strict=True):
    ax.plot([(s_ + e_) / 2], [5.95], marker="v", ms=13, color=OR, mec="white", mew=1.2, clip_on=False)
ax.text(0.02, 6.3, "A. blanci", color=OR, fontsize=15, fontweight="bold", va="bottom", style="italic")
ax.set_ylabel("fréquence (kHz)")
ax.set_xlabel("temps (s)")
ax.xaxis.set_major_formatter(virgule)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(ICI / "spectro_grand.png", transparent=True)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=200)
spectro(ax, 24.0, 27.0, 15000)
ax.set_ylabel("fréquence (kHz)")
ax.set_xlabel("temps dans la fenêtre (s)")
ax.xaxis.set_major_formatter(virgule)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(ICI / "fenetre_spectro_plein.png", transparent=True)
plt.close(fig)

# ---------------------------------------------------------------- courbe du rejeu
SIM = D["simulation"]
cv = SIM["courbes"]
fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=200)
ax.grid(axis="y", color=GRID, lw=0.8)
for mode, coul, mk, nom in (("hasard", "#8C9690", "s", "au hasard"), ("actif", POS, "o", "file 60-20-20")):
    c = cv[mode]
    ax.fill_between(c["n"], c["min"], c["max"], color=coul, alpha=0.13, lw=0)
    ax.plot(c["n"], c["moy"], "-", color=coul, lw=2.2, marker=mk, ms=6, mec="white", mew=1)
    ax.text(c["n"][-1] + 10, c["moy"][-1] + (0.035 if mode == "actif" else -0.035), nom, color=INK, fontsize=12.5, va="center", fontweight="bold" if mode == "actif" else "normal",
            bbox={"fc": "white", "ec": "none", "pad": 1.5})
ap_tout, prev = SIM["ap_toutes_annotations"], SIM["prevalence_eval"]
ax.axhline(ap_tout, color=MUTED, ls=":", lw=1.3)
ax.text(cv["actif"]["n"][0], ap_tout + 0.015, f"avec toutes mes annotations ({SIM['n_train']:,} fenêtres) : {ap_tout:.2f}".replace(",", " ").replace(".", ","), color=MUTED, fontsize=11, va="bottom")
ax.axhline(prev, color=NEG, ls="--", lw=1.2)
ax.text(cv["actif"]["n"][0], prev + 0.015, f"un modèle qui répondrait au hasard : {prev:.2f}".replace(".", ","), color=NEG, fontsize=11, va="bottom")
ax.set_xlim(cv["actif"]["n"][0] - 10, cv["actif"]["n"][-1] + 110)
ax.set_xticks([n for n in range(200, cv["actif"]["n"][-1] + 1, 100)])
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(virgule)
ax.set_xlabel("fenêtres annotées pour l'entraînement")
ax.set_ylabel("précision moyenne")
fig.tight_layout()
fig.savefig(ICI / "courbe_v4.png", transparent=True)
plt.close(fig)

# ---------------------------------------------------------------- carte de présence
html = (ROOT / "data" / "reports" / "carte_points_ecoute.html").read_text(encoding="utf-8")
marques = pd.DataFrame(json.loads(re.search(r"const D = (\{.*?\});\s*\n", html, re.S).group(1))["marques"])
a = np.load(ICI / "birdnet_annotees.npz")
con = sqlite3.connect(f"file:{Path(cfg['paths']['db']).as_posix()}?mode=ro", uri=True)
rec = pd.read_sql_query("SELECT recording_id, site, mic_id FROM recordings", con).set_index("recording_id")
ann = pd.DataFrame({"recording_id": a["recording_id"], "y": a["y"]}).join(rec, on="recording_id")
par_micro = ann.groupby(["site", "mic_id"])["y"].max()
marques["etat"] = [par_micro.get((s, m), -1) for s, m in zip(marques.site, marques.micro, strict=True)]
lat0 = np.deg2rad(marques.lat.mean())
x = (marques.lon - marques.lon.mean()) * 111.32 * np.cos(lat0)  # km, autour du centre
y = (marques.lat - marques.lat.mean()) * 110.57
cx_, cy_ = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
demi = max(x.max() - x.min(), y.max() - y.min()) / 2 * 1.3
fig = plt.figure(figsize=(4.6, 5.1), dpi=200)
ax = fig.add_axes([0.02, 0.2, 0.96, 0.78])
ax.set_facecolor("#F4F7F2")
ax.set_xlim(cx_ - demi, cx_ + demi)
ax.set_ylim(cy_ - demi, cy_ + demi)
for et, coul, z in ((-1, GRIS, 1), (0, NEG, 2), (1, POS, 3)):
    m = marques.etat == et
    ax.scatter(x[m], y[m], s=150, color=coul, edgecolors="white", linewidths=1.5, zorder=z)
NOMS = {"Tresor": "Trésor", "Molokoi": "Molokoï"}
for site, g in marques.assign(x=x, y=y).groupby("site"):
    n_ = len(g)
    ax.annotate(f"{NOMS.get(site, site)} ({n_} point{'s' if n_ > 1 else ''})", (g.x.mean(), g.y.mean()), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=11, color=INK)
L = 20  # barre d'échelle, km
x0b, y0b = cx_ - demi * 0.9, cy_ - demi * 0.88
ax.plot([x0b, x0b + L], [y0b, y0b], color=INK, lw=3, solid_capstyle="butt")
ax.text(x0b + L / 2, y0b + demi * 0.04, f"{L} km", ha="center", fontsize=10, color=INK)
xn, yn = cx_ + demi * 0.85, cy_ + demi * 0.62
ax.annotate("N", (xn, yn + demi * 0.22), xytext=(xn, yn), ha="center", va="top", fontsize=12, fontweight="bold", color=INK, arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.5})
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for s_ in ax.spines.values():
    s_.set_visible(True)
    s_.set_color(SAGE)
h = [matplotlib.lines.Line2D([], [], ls="", marker="o", ms=10, color=c, mec="white", label=lab) for c, lab in ((POS, "A. blanci entendu"), (NEG, "écouté, pas entendu"), (GRIS, "pas encore écouté"))]
fig.legend(handles=h, loc="lower center", ncol=1, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.savefig(ICI / "carte_presence.png", transparent=False, facecolor="white")
plt.close(fig)
cnt = marques.etat.value_counts().to_dict()
print("carte :", len(marques), "points ;", {"entendu": cnt.get(1, 0), "pas entendu": cnt.get(0, 0), "pas écouté": cnt.get(-1, 0)})

# ---------------------------------------------------------------- la tête : un neurone
V = D["vecteur"]
PDS = np.array(V["poids"])
PMAX = np.abs(PDS).max()
LIGNES = 17  # 8 nombres, « ⋮ », 8 nombres
ys = [1 - (r + 0.5) / LIGNES for r in range(LIGNES)]
entrees = [r for r in range(LIGNES) if r != 8]
XO, YO = 0.78, 0.5
VERT_C, ROUGE_C, CYAN = "#5CC98B", "#E07A5F", np.array([156, 224, 208]) / 255
NF = 60


def image(k):
    phi = k / NF
    fig = plt.figure(figsize=(6.05, 4.335), dpi=120, facecolor=DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    for j, r in enumerate(entrees):
        p = PDS[j]
        coul = VERT_C if p >= 0 else ROUGE_C
        ax.plot([0.03, XO], [ys[r], YO], color=coul, lw=0.6 + 4.2 * abs(p) / PMAX, alpha=0.35 + 0.5 * abs(p) / PMAX, solid_capstyle="round", zorder=1)
        ax.scatter([0.03], [ys[r]], s=40, color="#2C5A4F", edgecolors="#9CCFC4", linewidths=1, zorder=3)
        ax.text(0.075, ys[r] + 0.012, ("+" if p > 0 else "") + f"{p:.1f}".replace(".", ","), color=coul, fontsize=9, va="center", zorder=4)
        s = (phi - 0.04 * (j % 4)) / 0.5  # les impulsions partent presque ensemble et rejoignent le neurone
        if 0 <= s <= 1:
            px, py = 0.03 + (XO - 0.03) * s, ys[r] + (YO - ys[r]) * s
            ax.scatter([px], [py], s=150, color=OR, alpha=0.35, linewidths=0, zorder=5)
            ax.scatter([px], [py], s=30, color="#FFF4D6", linewidths=0, zorder=6)
    ax.text(0.03, ys[8], "⋮", color="#9CCFC4", fontsize=16, ha="center", va="center")
    act = float(np.exp(-(((phi - 0.62) / 0.12) ** 2)))
    if act > 0.03:
        ax.scatter([XO], [YO], s=4200 * act, color=CYAN, alpha=0.28 * act, linewidths=0, zorder=6)
    coul = (1 - act) * np.array([0x17, 0x3D, 0x34]) / 255 + act * CYAN
    ax.scatter([XO], [YO], s=1500, color=coul, edgecolors="#9CCFC4", linewidths=2, zorder=7)
    ax.text(XO, YO, "Σ", color="white" if act < 0.5 else DARK, fontsize=22, ha="center", va="center", zorder=8, fontweight="bold")
    ax.annotate("", (1.0, YO), (XO + 0.07, YO), arrowprops={"arrowstyle": "-|>", "color": "#9CCFC4", "lw": 2}, zorder=2)
    s = (phi - 0.68) / 0.3
    if 0 <= s <= 1:
        px = XO + 0.07 + (0.97 - XO - 0.07) * s
        ax.scatter([px], [YO], s=200, color=OR, alpha=0.4, linewidths=0, zorder=9)
        ax.scatter([px], [YO], s=40, color="#FFF4D6", linewidths=0, zorder=10)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=DARK)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


images = [image(k) for k in range(NF)]
mosaique = Image.new("RGB", (images[0].width * 4, images[0].height))
for k, im in enumerate([images[10], images[25], images[37], images[50]]):
    mosaique.paste(im, (k * images[0].width, 0))
pal = mosaique.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
images = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
images[0].save(ICI / "tete.gif", save_all=True, append_images=images[1:], duration=50, loop=0, disposal=2)
images[37].convert("RGB").save(ICI / "tete_controle.png")
print("écrit spectro_grand, fenetre_spectro_plein, courbe_v4, carte_presence, tete.gif")
