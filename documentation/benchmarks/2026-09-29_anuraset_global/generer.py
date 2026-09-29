"""Figures et tableaux du rapport 07, recalculés depuis `donnees/*.csv` (rien d'autre à lancer).

    uv run --group notebook python documentation/benchmarks/2026-09-29_anuraset_global/generer.py

Couleurs : palette catégorielle de référence du skill dataviz (6 encodeurs, ordre fixe, validée :
séparation CVD ≥ 9), avec des marqueurs différents et des étiquettes directes en plus.
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

SURFACE, ENCRE, ENCRE_2, FILET = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
ENCODEURS = {  # ordre fixe : couleur, marqueur
    "perch_v2": ("#2a78d6", "o"),
    "perch_bird": ("#eb6834", "s"),
    "birdnet": ("#1baf7a", "D"),
    "birdmae_base": ("#eda100", "^"),
    "birdmae_huge": ("#e87ba4", "v"),
    "protoclr": ("#008300", "P"),
}
MEILLEURE = {  # tête fixée d'avance : meilleur rang moyen dans le benchmark de l'encodeur
    "perch_v2": "logistic+R37=glmm",
    "perch_bird": "logistic+R37=glmm",
    "birdnet": "logistic+R37=glmm",
    "birdmae_base": "lda_shrunk",
    "birdmae_huge": "lda_shrunk",
    "protoclr": "lda_shrunk",
}
ESPECES = ["DENMIN", "PITAZU", "PHYCUV", "LEPLAT", "BOAFAB"]
REFERENCE = ("perch_v2", "logistic")
K_ORDRE = [0, 1, 2, 5, 10, 20, -1]
K_NOMS = ["0", "1", "2", "5", "10", "20", "tout"]
TETES_COURBE = {
    "logistic": ("#2a78d6", "o"),
    "logistic+R37=glmm": ("#eb6834", "s"),
    "lda_shrunk": ("#1baf7a", "D"),
    "logistic+R20": ("#eda100", "^"),
    "prototype": ("#e87ba4", "v"),
}

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


def load() -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_csv(p) for p in DONNEES.glob("*.csv")}


def _axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=FILET, linewidth=0.6)
    ax.set_axisbelow(True)


def tableau_encodeurs(transfert: pd.DataFrame, comparaisons: pd.DataFrame) -> None:
    """AP poolée · AP moyenne par site, niveau minute ; deux lignes par encodeur (logistique,
    meilleure version) ; couleur : écart apparié à perch_v2 + logistique (AP par site) qui
    survit à Holm et atteint 0,02."""
    t = transfert[transfert["level"] == "minute"]
    c = comparaisons[comparaisons["level"] == "minute"]
    rows = []
    for enc in ENCODEURS:
        for head in dict.fromkeys(["logistic", MEILLEURE[enc]]):
            ref = (enc, head) == REFERENCE
            label = f"{enc}  ·  {head}"
            row: list = [tableaux.C(label, "ref") if ref else label]
            for sp in ESPECES:
                r = t[(t["encoder"] == enc) & (t["head"] == head) & (t["species"] == sp)].iloc[0]
                text = f"{r['ap']:.2f} · {r['ap_site']:.2f}".replace(".", ",")
                k = c[(c["encoder"] == enc) & (c["head"] == head) & (c["species"] == sp)]
                tag = "ref" if ref else None
                if (
                    len(k)
                    and bool(k["significant_holm"].iloc[0])
                    and abs(k["diff"].iloc[0]) >= 0.02
                ):
                    tag = "mieux" if k["diff"].iloc[0] > 0 else "moins"
                row.append(tableaux.C(text, tag) if tag else text)
            rows.append(row)
    table = tableaux.Tableau(
        "encodeurs",
        "Six encodeurs, un site neuf à la fois : AP poolée · AP moyenne par site (minute)",
        "Minute : score = max des fenêtres de l'enregistrement, mêmes 1 599 enregistrements pour"
        " tous. Tête apprise sur les autres sites, toutes les minutes du site jugées. Couleur :"
        " écart apparié à perch_v2 + logistique (AP par site) significatif après Holm et d'au"
        " moins 0,02 ; enregistrements tirés dans chaque site (optimiste : sites non tirés).",
        [
            tableaux.Section(
                [tableaux.Colonne("Encodeur · tête", 2.2, code=True)]
                + [tableaux.Colonne(sp, 1.0) for sp in ESPECES],
                rows,
            )
        ],
        [
            "Meilleure version : tête au meilleur rang moyen dans le benchmark de l'encodeur"
            " (02 à 06), fixée avant ce benchmark."
        ],
        largeur=1800,
    )
    tableaux.verifier(table)
    image = tableaux.rendre(table).quantize(colors=48)
    image.save(FIGURES / "1_tableau_encodeurs.png", optimize=True)


def figure_fenetres(transfert: pd.DataFrame) -> None:
    """AP moyenne par site au niveau fenêtre (grille propre à chaque encodeur), meilleure
    version, espèce par espèce."""
    t = transfert[transfert["level"] == "fenetre"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(ESPECES))
    width = 0.13
    for i, (enc, (color, marker)) in enumerate(ENCODEURS.items()):
        part = t[(t["encoder"] == enc) & (t["head"] == MEILLEURE[enc])].set_index("species")
        values = part.reindex(ESPECES)["ap_site"].to_numpy()
        ax.scatter(
            x + (i - 2.5) * width,
            values,
            s=46,
            color=color,
            marker=marker,
            edgecolor=SURFACE,
            linewidth=1,
            label=enc,
            zorder=2,
        )
    ax.set_xticks(x, ESPECES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AP moyenne par site (fenêtre)")
    _axes(ax)
    ax.legend(ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "2_fenetres.png", dpi=150)
    plt.close(fig)


def figure_cout(transfert: pd.DataFrame, encodeurs: pd.DataFrame) -> None:
    """Qualité (AP moyenne par site, minute, moyenne des espèces) contre vitesse d'encodage."""
    t = transfert[transfert["level"] == "minute"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for enc, (color, marker) in ENCODEURS.items():
        e = encodeurs.set_index("encoder").loc[enc]
        ap = t[(t["encoder"] == enc) & (t["head"] == MEILLEURE[enc])]["ap_site"].mean()
        ax.scatter(
            e["temps_reel"], ap, s=70, color=color, marker=marker, edgecolor=SURFACE, zorder=2
        )
        ax.annotate(
            f"{enc} ({int(e['dim'])} d)",
            (e["temps_reel"], ap),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=8.5,
            color=ENCRE,
        )
    ax.set_xscale("log")
    ax.set_xlabel("vitesse d'encodage, fois le temps réel (CPU 4 cœurs, échelle log)")
    ax.set_ylabel("AP moyenne par site (minute)")
    ax.set_ylim(0, 1)
    _axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "3_cout_qualite.png", dpi=150)
    plt.close(fig)


def figure_courbe(courbe: pd.DataFrame, level: str, encodeur_tetes: str) -> None:
    """AP sur la moitié test du site cible selon le nombre k d'enregistrements positifs du
    site ajoutés à l'entraînement (« tout » : toute l'autre moitié). À gauche, la meilleure
    version de chaque encodeur ; à droite, les têtes d'un encodeur."""
    column = f"ap_{level}"
    mean = courbe.groupby(["encoder", "head", "k"])[column].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)
    x = np.arange(len(K_ORDRE))
    for enc, (color, marker) in ENCODEURS.items():
        if (enc, MEILLEURE[enc]) not in mean.index.droplevel("k"):
            continue
        values = mean.loc[(enc, MEILLEURE[enc])].reindex(K_ORDRE).to_numpy()
        axes[0].plot(x, values, color=color, marker=marker, linewidth=2, markersize=7, label=enc)
    for head, (color, marker) in TETES_COURBE.items():
        values = mean.loc[(encodeur_tetes, head)].reindex(K_ORDRE).to_numpy()
        axes[1].plot(x, values, color=color, marker=marker, linewidth=2, markersize=7, label=head)
    axes[0].set_title("meilleure version de chaque encodeur", fontsize=10, color=ENCRE)
    axes[1].set_title(f"têtes, avec {encodeur_tetes}", fontsize=10, color=ENCRE)
    axes[0].set_ylabel(f"AP sur le site cible ({'fenêtre' if level == 'window' else 'minute'})")
    for ax in axes:
        ax.set_xticks(x, K_NOMS)
        ax.set_xlabel("enregistrements positifs du site annotés (k)")
        ax.set_ylim(0, 1)
        _axes(ax)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "4_courbe_amorcage.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    d = load()
    tableau_encodeurs(d["transfert"], d["comparaisons"])
    figure_fenetres(d["transfert"])
    figure_cout(d["transfert"], d["encodeurs"])
    figure_courbe(d["courbe"], "window", "perch_v2")
    print(sorted(p.name for p in FIGURES.iterdir()))


if __name__ == "__main__":
    main()
