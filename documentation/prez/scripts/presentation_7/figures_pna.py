# ruff: noqa: E501
"""Figures et chiffres de la prez PNA v2 (lecture seule), dans `rapports/presentation_7/`.

    uv run python documentation/prez/scripts/presentation_7/figures_pna.py

- `inv_sites_pna.png` : heures par lieu, relevés empilés sur leur lieu (`lieux` de
  documentation/tableau-de-bord/structure.yaml : Gabaret et Pai → Gabaret), même tracé que la
  figure de la prez 4, base d'aujourd'hui (relevés du 23/11 compris) ;
- `onde.npz` : forme d'onde de l'enregistrement de Molokoï (minimum et maximum du signal par
  tranche de 0,2 s sur 120 s ; par tranche de 10 ms dans la fenêtre BirdNET 24-27 s) ;
- `fenetre_spectro.png` : spectrogramme de la fenêtre 24-27 s sur la bande de BirdNET (0-15 kHz) ;
- `chiffres_pna.json` : corpus du 23/11 (enregistrements, heures, sites, points d'écoute) et part
  du chant dans une fenêtre de 3 s : pour chaque fenêtre de la grille BirdNET (pas de 3 s) qui
  contient au moins une note isolée annotée (intervalle A. blanci de moins de 1 s, valable),
  durée des notes dans la fenêtre / 3 s ; part de la bande 4,4-5,5 kHz dans 0-15 kHz.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib
import matplotlib.patches
import matplotlib.ticker
import numpy as np
import pandas as pd
import yaml
from scipy.signal import spectrogram

from blanci.core.audio import load_audio
from blanci.core.config import LOCAL_CONFIG, load_config
from blanci.core.locate import locate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
ICI = ROOT / "rapports" / "presentation_7"
ICI.mkdir(parents=True, exist_ok=True)
REL = "Projet Phénologie blanci/Troisième relevé_avril 2024/Molokoi 11042024/SM F_SMA14163/Data/SMA14163_20240226_163000.wav"
FEN = (24.0, 27.0)
BANDE, FMAX = (4400, 5500), 15000
VERT, OR, INK, MUTED, GRID, SAGE = "#2A7F72", "#E9B44C", "#1B2A24", "#5B6B63", "#E4EEE8", "#A7BFB2"
NOMS = {"Tresor": "Trésor", "Molokoi": "Molokoï", "PatawaEst": "Patawa Est", "PatawaOuest": "Patawa Ouest"}
plt.rcParams.update({
    "font.family": ["Calibri", "DejaVu Sans"], "font.size": 12, "axes.edgecolor": SAGE,
    "xtick.color": MUTED, "ytick.color": INK, "axes.labelcolor": MUTED, "xtick.labelsize": 12,
    "ytick.labelsize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
})
cfg = load_config(LOCAL_CONFIG)
OUT: dict = {}

# ---------------------------------------------------------------- corpus du 23/11
c = pd.read_csv(ROOT / "annotations" / "corpus_20261123" / "corpus.csv", low_memory=False)
OUT["corpus"] = {
    "enregistrements": int(len(c)), "heures": round(float(c.duration_s.sum() / 3600), 1),
    "sites": int(c.site.nunique()), "points": int(c.groupby(["dataset", "site", "mic_id"]).ngroups),
    "positifs": int(c["positif"].astype(str).str.lower().eq("true").sum()),
    "piste_1": int((c["piste"] == 1).sum()),
}

# ---------------------------------------------------------------- inventaire par lieu
con = sqlite3.connect(f"file:{Path(cfg['paths']['db']).as_posix()}?mode=ro", uri=True)
rec = pd.read_sql_query("SELECT recording_id, dataset, site, mic_id, duration_s FROM recordings WHERE dataset IN ('2023', '2026')", con)
lieux = yaml.safe_load((ROOT / "documentation" / "tableau-de-bord" / "structure.yaml").read_text(encoding="utf-8"))["lieux"]
rec["site"] = rec.site.map(lambda s: NOMS.get(s, s))
rec["lieu"] = rec.site.map(lambda s: lieux.get(s, s))
OUT["inventaire"] = {"enregistrements": int(len(rec)), "heures": round(float(rec.duration_s.sum() / 3600), 1), "enregistreurs": int(rec.mic_id.nunique())}
h = rec.groupby(["lieu", "dataset", "site"]).agg(h=("duration_s", lambda s: s.sum() / 3600), n=("recording_id", "size")).reset_index()
tot = h.groupby("lieu").agg(h=("h", "sum"), n=("n", "sum")).sort_values("h")
fig, ax = plt.subplots(figsize=(7.6, 4.0))
for y, lieu in enumerate(tot.index):
    x = 0.0
    petits, x_petits = [], None
    for _, p in h[h.lieu == lieu].sort_values(["dataset", "h"], ascending=[True, False]).iterrows():
        col = VERT if p.dataset == "2023" else OR
        ax.barh(y, p.h, left=x, color=col, height=0.62, edgecolor="white", linewidth=1.5)
        nom = "suivi 2023-24" if p.dataset == "2023" else p.site
        if p.h >= 150:
            ax.text(x + p.h / 2, y, f"{nom}\n{p.h:.0f} h", ha="center", va="center", fontsize=10.5, color="white" if p.dataset == "2023" else INK)
        else:
            petits.append(f"{nom} {p.h:.0f} h")
            x_petits = x if x_petits is None else x_petits
        x += p.h
    if petits:
        dessous = y == 0  # la barre du bas : étiquette en dessous, sinon elle chevauche la barre voisine
        ax.annotate(" · ".join(petits), ((x_petits + x) / 2, y + (-0.31 if dessous else 0.31)), xytext=(x_petits + (20 if dessous else -60), y + (-0.55 if dessous else 0.5)),
                    fontsize=10.5, color=INK, ha="left", va="top" if dessous else "bottom", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.text(x + 15, y, f"{x:,.0f} h · {tot.loc[lieu, 'n']:,.0f} enr.".replace(",", " "), va="center", fontsize=11.5, color=INK, fontweight="bold")
ax.set_yticks(range(len(tot)), tot.index)
ax.set_xlim(0, tot.h.max() * 1.45)
ax.set_ylim(-0.95, len(tot) - 0.25)
ax.set_xlabel("heures d'enregistrement")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.bar(0, 0, color=VERT, label="phénologie 2023-2024 (un an)")
ax.bar(0, 0, color=OR, label="relevés 2025-2026")
ax.legend(frameon=False, fontsize=11, loc="lower right")
fig.tight_layout()
fig.savefig(ICI / "inv_sites_pna.png", dpi=250)
plt.close(fig)

# ---------------------------------------------------------------- forme d'onde et spectrogramme
_iv = pd.read_csv(ROOT / "annotations" / "intervalles_annotes.csv")
_iv = _iv[(_iv.valable == True) & (_iv.path == REL) & _iv.label.isin(["blanci", "blanci_chorus"]) & (_iv.start_s >= FEN[0]) & (_iv.end_s <= FEN[1])]  # noqa: E712
NOTES_FEN = list(zip(_iv.start_s, _iv.end_s, strict=True))
chemin = locate(REL, Path(cfg["paths"]["raw"]), renamed=cfg["audio"]["renamed"])
wav, sr = load_audio(chemin, 0)  # piste 0 : celle du manifeste pour cet enregistrement


def tranches(sig, pas):
    n = len(sig) // pas
    t = sig[: n * pas].reshape(n, pas)
    return t.min(axis=1), t.max(axis=1)


a, b = (int(t * sr) for t in FEN)
lo, hi = tranches(wav, int(0.2 * sr))
flo, fhi = tranches(wav[a:b], int(0.01 * sr))
np.savez(ICI / "onde.npz", lo=lo, hi=hi, pas=0.2, flo=flo, fhi=fhi, fpas=0.01, fen=np.array(FEN))
f, t, S = spectrogram(wav[a:b], fs=sr, nperseg=1024, noverlap=896)
S = 10 * np.log10(S + 1e-12)
g = f <= FMAX
fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=200)
ax.pcolormesh(t, f[g] / 1000, S[g], shading="auto", cmap="magma", vmin=np.percentile(S[g], 20), vmax=np.percentile(S[g], 99.8))
ax.set_ylabel("fréquence (kHz)")
ax.set_xlabel("temps dans la fenêtre (s)")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
for n0, n1 in [(s_ - FEN[0], e_ - FEN[0]) for s_, e_ in NOTES_FEN]:
    ax.add_patch(matplotlib.patches.Rectangle((n0 - 0.04, BANDE[0] / 1000 - 0.25), n1 - n0 + 0.08, (BANDE[1] - BANDE[0]) / 1000 + 0.5, fill=False, ec=OR, lw=2.5))
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(ICI / "fenetre_spectro.png", transparent=True)
OUT["spectro_axes"] = {"fmax_khz": FMAX / 1000, "duree_s": FEN[1] - FEN[0]}
plt.close(fig)

# ---------------------------------------------------------------- part du chant dans une fenêtre
iv = pd.read_csv(ROOT / "annotations" / "intervalles_annotes.csv")
iv = iv[(iv.valable == True) & iv.label.isin(["blanci", "blanci_chorus"])]  # noqa: E712
notes = iv[(iv.end_s - iv.start_s) < 1.0]
parts = []
for _, g_ in notes.groupby("path"):
    for w0 in np.arange(0, 120, 3.0):
        ov = (np.minimum(g_.end_s, w0 + 3) - np.maximum(g_.start_s, w0)).clip(lower=0).sum()
        if ov > 0:
            parts.append(ov / 3)
parts = np.array(parts)
bande = (BANDE[1] - BANDE[0]) / FMAX
OUT["chant"] = {
    "n_notes": int(len(notes)), "n_fenetres": int(len(parts)), "duree_note_mediane_s": round(float((notes.end_s - notes.start_s).median()), 2),
    "temps_p25": round(float(np.percentile(parts, 25)), 3), "temps_mediane": round(float(np.median(parts)), 3), "temps_p75": round(float(np.percentile(parts, 75)), 3),
    "temps_p10": round(float(np.percentile(parts, 10)), 3), "temps_p90": round(float(np.percentile(parts, 90)), 3),
    "bande": round(bande, 3), "surface_mediane": round(float(np.median(parts) * bande), 4),
}
hero = notes[(notes.path == REL) & (notes.start_s >= FEN[0]) & (notes.end_s <= FEN[1])]
OUT["note_hero"] = [[round(float(s) - FEN[0], 2), round(float(e) - FEN[0], 2)] for s, e in zip(hero.start_s, hero.end_s, strict=True)]
(ICI / "chiffres_pna.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(OUT, ensure_ascii=False, indent=1))
