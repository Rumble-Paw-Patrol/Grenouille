"""Figure : l'écart de spectre entre les deux pistes d'un même enregistrement, par point.

Suite de la diapo 20 de la présentation n° 4 (« 164 points : un micro sourd se voit tout de
suite »), qui classait les points par l'écart de niveau RMS global entre pistes, moins l'écart
réglé. Ici, le même écart bande par bande (tiers d'octave, 100 Hz – 20 kHz) :
d(f) = niveau piste 0 − niveau piste 1 − (gain 0 − gain 1). Le paysage sonore s'annule dans la
différence ; un micro qui faiblit, s'étouffe ou crépite d'un seul côté la fait sortir de 0.

- à gauche : les 164 points classés par leur pire épisode de deux semaines (médiane glissante
  sur 14 jours de l'écart de plus grand module entre 1 et 12 kHz), ±6 dB en tirets, les points
  au-delà nommés ;
- à droite : d(f) des points au-delà du seuil pendant cet épisode, bande par bande, rangés par
  l'écart moyen sur 1-12 kHz (le pire écart en étiquette).

Le profil de Kaw D avant, pendant et après sa perte (`spectral_gap_profile`) n'est plus tracé :
la diapo 20 montrait déjà la perte dans le temps.

Chaque chiffre sort de `blanci/exploration/inventaire.py` (`spectral_gaps`,
`spectral_gap_by_point`), sur les mesures de
`scripts/controle_qualite/spectre_pistes.py` (`paths.reports/pistes/spectre_pistes.csv`) et
l'échantillon des pistes (`echantillon_pistes.csv`, gains lus dans l'en-tête de chaque fichier).

    uv run python documentation/prez/scripts/figure_spectre_pistes.py [--config local.yaml]

Écrit `documentation/prez/figures/inventaire_qc/qc_spectre_pistes.png` et, à côté des mesures,
`paths.reports/pistes/spectre_pistes_points.csv` (une ligne par point, celle de la figure).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from blanci.core.config import config_path, default_user_config, load_config  # noqa: E402
from blanci.exploration.inventaire import (  # noqa: E402
    spectral_gap_by_point,
    spectral_gaps,
    third_octave_bands,
)

FIG = ROOT / "documentation" / "prez" / "figures" / "inventaire_qc"
INK, MUTED, GRID, SAGE = "#1B2A24", "#5B6B63", "#E4EEE8", "#A7BFB2"
ROUGE, VERT = "#C8553D", "#2A7F72"
NOMS = {"Tresor": "Trésor", "Molokoi": "Molokoï", "PatawaEst": "Patawa Est",
        "PatawaOuest": "Patawa Ouest"}
SEUIL = 6.0  # dB, comme la diapo 20 : la moitié de l'écart réglé de 12 dB
# Divergente : rouge = piste 0 plus faible que réglé, vert = plus forte, gris clair au milieu.
DIVERGE = LinearSegmentedColormap.from_list("ecart", [ROUGE, "#F3F1EC", VERT])

plt.rcParams.update({
    "font.family": ["Calibri", "DejaVu Sans"], "font.size": 14, "axes.edgecolor": SAGE,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 13,
    "ytick.labelsize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
})


def charger(cfg: dict) -> pd.DataFrame:
    pistes = config_path(cfg, "reports") / "pistes"
    sample = pd.read_csv(pistes / "echantillon_pistes.csv")
    levels = pd.read_csv(pistes / "spectre_pistes.csv")
    if "erreur" in levels:
        levels = levels[levels["erreur"].isna()].drop(columns="erreur")
    return spectral_gaps(levels, sample, cfg["recorder"]["filename_utc_offset_h"])


def nom(r) -> str:
    return f"{NOMS.get(r.site, r.site)} {r.mic_id}  ({r.extreme_db:+.0f} dB)".replace("-", "−")


def titre(ax, texte: str, sous_titre: str, pad: float = 50) -> None:
    ax.set_title(texte, loc="left", fontsize=17, color=INK, fontweight="bold", pad=pad)
    ax.text(0, 1.015, sous_titre, transform=ax.transAxes, fontsize=13, color=MUTED, va="bottom")


def figure(pts: pd.DataFrame, out: Path) -> None:
    bands = third_octave_bands(100, 20000)
    cols = [f"d_{n}" for n in bands["name"]]
    pts = pts.reset_index(drop=True).assign(rang=lambda d: np.arange(len(d)))
    inside = np.flatnonzero((bands["name"] >= 1000) & (bands["name"] <= 10000))
    # Carte : rangée par l'écart moyen sur 1-12 kHz (dégradé lisible), le pire écart en étiquette
    suspects = (pts[pts.extreme_db.abs() > SEUIL]
                .assign(moyen=lambda d: d[[cols[i] for i in inside]].mean(axis=1))
                .sort_values("moyen"))
    n, k = len(pts), len(suspects)

    fig = plt.figure(figsize=(17, max(8.5, 2.2 + 0.24 * k)))
    # colonnes : classement, place des noms de lignes, carte, échelle
    grid = fig.add_gridspec(1, 4, width_ratios=[0.85, 0.66, 1.2, 0.035], wspace=0.08)

    # 1. Classement des points, sans nom : les noms sont sur les lignes de la carte
    ax = fig.add_subplot(grid[0])
    couleur = np.where(pts.extreme_db.abs() > SEUIL, ROUGE, SAGE)
    ax.scatter(pts.extreme_db, pts.rang, c=couleur, s=34, zorder=3, linewidths=0)
    ax.axvline(0, color=INK, lw=0.9)
    for s in (-SEUIL, SEUIL):
        ax.axvline(s, color=MUTED, lw=1, ls="--")
    lim = max(30.0, float(np.ceil(pts.extreme_db.abs().max() / 5) * 5) + 2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-2, n + 1)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    bas, haut = int((pts.extreme_db < -SEUIL).sum()), int((pts.extreme_db > SEUIL).sum())
    ax.text(SEUIL + 1.5, bas + 12, f"{bas} points :\npiste 0\ntrop faible", fontsize=14,
            color=ROUGE, va="bottom")
    ax.text(-SEUIL - 1.5, n - haut / 2 - 2, f"{haut} points :\npiste 0 trop forte", fontsize=14,
            color=ROUGE, ha="right", va="top")
    ax.set_xlabel("pire écart entre les deux micros,\nmoins l'écart réglé (dB, 1-12 kHz)\n"
                  f"tirets : ±{SEUIL:.0f} dB")
    titre(ax, f"{k} points sur {n} au-delà de ±{SEUIL:.0f} dB",
          "un point = un enregistreur à un site,\nclassé par son pire épisode de 14 jours")

    # 2. Carte de chaleur des seuls points suspects, nommés, même ordre
    ax = fig.add_subplot(grid[2])
    im = ax.imshow(suspects[cols].to_numpy(dtype=float), aspect="auto", origin="lower",
                   cmap=DIVERGE, vmin=-20, vmax=20, interpolation="nearest")
    ax.grid(False)
    ticks = [i for i, b in enumerate(bands["name"]) if b in (100, 250, 500, 1000, 2000, 4000,
                                                              8000, 16000)]
    ax.set_xticks(ticks, [f"{b / 1000:g}".replace(".", ",") for b in bands["name"].iloc[ticks]])
    for x in (inside[0] - 0.5, inside[-1] + 0.5):
        ax.axvline(x, color=INK, lw=1.2, ls=":")
    noms = [f"{nom(r)}, {r.episode_start:%d/%m/%y}" for r in suspects.itertuples()]
    ax.set_yticks(range(k), noms, fontsize=12, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("fréquence (kHz) ; pointillés : 1-12 kHz\n"
                  "rouge : piste 0 plus faible que réglé ; vert : plus forte")
    titre(ax, "Les points suspects : à quelle fréquence",
          "d(f) pendant le pire épisode (date de début), rangés par\n"
          "l'écart moyen sur 1-12 kHz ; entre parenthèses : le pire écart")
    cb = fig.colorbar(im, cax=fig.add_subplot(grid[3]), ticks=[-20, -10, -6, 0, 6, 10, 20])
    cb.set_label("dB", color=MUTED)
    cb.outline.set_visible(False)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config or default_user_config())
    gaps = charger(cfg)
    pts = spectral_gap_by_point(gaps)
    pts.to_csv(config_path(cfg, "reports") / "pistes" / "spectre_pistes_points.csv", index=False)
    FIG.mkdir(parents=True, exist_ok=True)
    figure(pts, FIG / "qc_spectre_pistes.png")
    print(f"{len(gaps)} enregistrements, {len(pts)} points ; au-delà de ±{SEUIL:.0f} dB :")
    show = ["dataset", "site", "mic_id", "extreme_db", "episode_start", "episode_end", "days"]
    print(pts[pts.extreme_db.abs() > SEUIL][show].to_string(index=False))


if __name__ == "__main__":
    main()
