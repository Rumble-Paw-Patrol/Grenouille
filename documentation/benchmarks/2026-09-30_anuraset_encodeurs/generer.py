"""Figures et tableau du rapport 08, recalculés depuis `donnees/*.csv` (rien d'autre à lancer).

    uv run --group notebook python documentation/benchmarks/2026-09-30_anuraset_encodeurs/generer.py

Couleurs : palette de référence du skill dataviz (catégorielle en ordre fixe, validée pour 6
séries ; bleu/orange pour logistique/meilleure tête), formes différentes en plus de la couleur.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ICI = Path(__file__).resolve().parent
DONNEES = ICI / "donnees"
FIGURES = ICI / "figures"
sys.path.insert(0, str(ICI.parents[1] / "tableaux"))
import generer as tableaux  # noqa: E402

SURFACE, ENCRE, ENCRE_2, FILET, GRIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#b9b8b1"
BLEU, ORANGE = "#2a78d6", "#eb6834"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARQUES = ["o", "s", "D", "^", "v", "P"]
ESPECES = ["DENMIN", "PITAZU", "PHYCUV", "LEPLAT", "BOAFAB"]
REFERENCE = ("perch_v2", "logistic")
COURBE = [  # encodeurs de la figure 4 (ordre fixe des couleurs)
    "perch_v2",
    "naturebeats",
    "esp_aves2_sl_beats_bio",
    "perch_bird",
    "birdnet_v3",
    "birdmae_base",
]
K_ORDRE, K_NOMS = [0, 1, 2, 5, 10, 20, -1], ["0", "1", "2", "5", "10", "20", "tout"]
LIBRE = {"oui": "libre", "sous réserve": "libre ?", "non": "non libre", "non relevée": "?"}
COURT = {"esp_aves2_": "aves2 ", "_ssl_all": "", "logistic+R37=glmm": "logistic+R37"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": FILET,
        "axes.labelcolor": ENCRE_2,
        "xtick.color": ENCRE_2,
        "ytick.color": ENCRE_2,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def court(text: str) -> str:
    for a, b in COURT.items():
        text = text.replace(a, b)
    return text


def load() -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_csv(p) for p in DONNEES.glob("*.csv")}


def _axes(ax, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=FILET, linewidth=0.6)
    ax.set_axisbelow(True)


def tableau(d: dict[str, pd.DataFrame]) -> None:
    """Une ligne par encodeur, sa meilleure tête : AP moyenne par site · AP poolée (minute)."""
    t = d["transfert"][d["transfert"]["level"] == "minute"]
    c = d["comparaisons"]
    enc = d["encodeurs"].set_index("encoder")
    best = d["meilleures"]
    lignes = [(REFERENCE[0], REFERENCE[1])] + [
        (e, h) for e, h in zip(best["encoder"], best["best_head"], strict=True)
    ]
    rows = []
    for e, h in dict.fromkeys(lignes):
        ref = (e, h) == REFERENCE
        label = f"{court(e)}  ·  {court(h)}"
        row: list = [tableaux.C(label, "ref") if ref else label]
        row.append(LIBRE.get(enc.loc[e, "libre"], "?") if e in enc.index else "?")
        values = []
        for sp in ESPECES:
            r = t[(t["encoder"] == e) & (t["head"] == h) & (t["species"] == sp)]
            if not len(r):
                row.append("—")
                continue
            r = r.iloc[0]
            values.append(r["ap_site"])
            text = f"{r['ap_site']:.2f} · {r['ap']:.2f}".replace(".", ",")
            k = c[(c["encoder"] == e) & (c["head"] == h) & (c["species"] == sp)]
            tag = "ref" if ref else None
            if len(k) and bool(k["significant_holm"].iloc[0]) and abs(k["diff"].iloc[0]) >= 0.02:
                tag = "mieux" if k["diff"].iloc[0] > 0 else "moins"
            row.append(tableaux.C(text, tag) if tag else text)
        row.append(f"{np.mean(values):.2f}".replace(".", ",") if len(values) == 5 else "—")
        rows.append(row)
    table = tableaux.Tableau(
        "encodeurs_08",
        "Tous les encodeurs, un site neuf à la fois : AP moyenne par site · AP poolée (minute)",
        "Meilleure tête de chaque encodeur, choisie après coup sur la moyenne des cinq espèces."
        " Couleur : écart apparié à perch_v2 + logistique (AP par site) significatif après Holm"
        " et d'au moins 0,02 ; enregistrements tirés dans chaque site (optimiste).",
        [
            tableaux.Section(
                [
                    tableaux.Colonne("Encodeur · tête", 2.6, code=True),
                    tableaux.Colonne("Licence", 0.8),
                ]
                + [tableaux.Colonne(sp, 1.0) for sp in ESPECES]
                + [tableaux.Colonne("Moyenne", 0.8)],
                rows,
            )
        ],
        [
            "Moyenne : AP par site, cinq espèces. rcl_fs_bsed : témoin non passé, apprentissage"
            " plafonné (fiche). Licence : n° 156 (« ? » : non relevée)."
        ],
        largeur=2000,
    )
    tableaux.verifier(table)
    tableaux.rendre(table).quantize(colors=64).save(
        FIGURES / "1_tableau_encodeurs.png", optimize=True
    )


def figure_tete(d: dict[str, pd.DataFrame]) -> None:
    """Logistique contre meilleure tête, par encodeur (AP moyenne par site, minute)."""
    best = d["meilleures"].sort_values("ap_site_best")
    fig, ax = plt.subplots(figsize=(8.5, 0.3 * len(best) + 1.2))
    y = np.arange(len(best))
    for yi, (a, b) in zip(
        y, zip(best["ap_site_logistic"], best["ap_site_best"], strict=True), strict=True
    ):
        ax.plot([a, b], [yi, yi], color=FILET, linewidth=2.5, zorder=1)
    ax.scatter(
        best["ap_site_logistic"],
        y,
        s=40,
        color=BLEU,
        marker="o",
        zorder=2,
        edgecolor=SURFACE,
        label="logistique",
    )
    ax.scatter(
        best["ap_site_best"],
        y,
        s=46,
        color=ORANGE,
        marker="D",
        zorder=3,
        edgecolor=SURFACE,
        label="meilleure tête (après coup)",
    )
    for yi, (x, h) in enumerate(zip(best["ap_site_best"], best["best_head"], strict=True)):
        if h != "logistic":
            ax.text(x + 0.01, yi, court(h), va="center", fontsize=7.5, color=ENCRE_2)
    ax.set_yticks(y, [court(e) for e in best["encoder"]], fontsize=8.5, family="DejaVu Sans Mono")
    ax.set_xlabel("AP moyenne par site (minute), moyenne des cinq espèces")
    ax.set_xlim(0.3, 1.0)
    _axes(ax, "x")
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "2_tete_adaptee.png", dpi=150)
    plt.close(fig)


DECALAGE = {  # étiquettes voisines écartées à la main
    "perch_v2": (7, 7, "left"),
    "esp_aves2_sl_beats_bio": (-7, 6, "right"),
    "esp_aves2_sl_beats_all": (-7, -11, "right"),
    "birdmae_large": (6, -12, "left"),
    "esp_aves2_sl_eat_bio_ssl_all": (6, 6, "left"),
}


def figure_cout(d: dict[str, pd.DataFrame]) -> None:
    """Meilleure AP contre vitesse d'encodage ; plein : libre, creux : non libre ou inconnu."""
    enc = d["encodeurs"].set_index("encoder")
    best = d["meilleures"].set_index("encoder")
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for e, r in best.iterrows():
        if e not in enc.index:
            continue
        speed = enc.loc[e, "fenetres_par_s"] * enc.loc[e, "fenetre_s"]
        if not np.isfinite(speed):  # débit pas encore relevé (fiche à venir)
            continue
        free = enc.loc[e, "libre"] in ("oui", "sous réserve")
        top = r["ap_site_best"] >= 0.65 or free
        ax.scatter(
            speed,
            r["ap_site_best"],
            s=60 if top else 34,
            color=ORANGE if free else (BLEU if top else GRIS),
            facecolor=(ORANGE if free else SURFACE),
            edgecolor=ORANGE if free else (BLEU if top else GRIS),
            linewidth=1.6,
            zorder=3 if top else 2,
        )
        if top:
            dx, dy, ha = DECALAGE.get(e, (6, 3, "left"))
            ax.annotate(
                court(e),
                (speed, r["ap_site_best"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                fontsize=8,
                color=ENCRE,
            )
    ax.set_xscale("log")
    ticks = [2, 5, 10, 20, 50, 100, 200]
    ax.set_xticks(ticks, [f"×{t}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("vitesse d'encodage, fois le temps réel (CPU 4 cœurs du cloud, échelle log)")
    ax.set_ylabel("meilleure AP moyenne par site (minute)")
    ax.set_ylim(0.35, 0.9)
    _axes(ax)
    ax.scatter([], [], s=50, color=ORANGE, label="libre (n° 156)")
    ax.scatter(
        [],
        [],
        s=50,
        facecolor=SURFACE,
        edgecolor=BLEU,
        linewidth=1.6,
        label="non libre ou licence absente",
    )
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "3_cout_qualite_licence.png", dpi=150)
    plt.close(fig)


def figure_courbe(d: dict[str, pd.DataFrame]) -> None:
    """Courbe d'amorçage (logistique, AP par fenêtre sur le site cible, moyenne des cibles)."""
    curve = d["courbe"]
    part = curve[curve["head"] == "logistic"]
    mean = part.groupby(["encoder", "k"])["ap_window"].mean()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(K_ORDRE))
    for enc, color, marker in zip(COURBE, SERIES, MARQUES, strict=True):
        if enc not in mean.index.get_level_values(0):
            continue
        values = mean.loc[enc].reindex(K_ORDRE).to_numpy()
        ax.plot(
            x, values, color=color, marker=marker, linewidth=2, markersize=6.5, label=court(enc)
        )
    ax.set_xticks(x, K_NOMS)
    ax.set_xlabel("enregistrements positifs du site annotés (k)")
    ax.set_ylabel("AP sur le site cible (fenêtre), logistique")
    ax.set_ylim(0.4, 1.0)
    _axes(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "4_courbe_amorcage.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    d = load()
    tableau(d)
    figure_tete(d)
    figure_cout(d)
    figure_courbe(d)
    print(sorted(p.name for p in FIGURES.iterdir()))


if __name__ == "__main__":
    main()
