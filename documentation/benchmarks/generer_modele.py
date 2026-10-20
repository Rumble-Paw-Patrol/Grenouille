"""Figures et tableaux du rapport, recalculés depuis `donnees/*.csv` (rien d'autre à lancer).

    uv run --group notebook python \
        documentation/benchmarks/__DOSSIER__/generer.py

Couleurs : palette de référence du skill dataviz (catégorielle pour les sites, rampe bleue pour
les AP, bleu/orange pour AP poolée / AP par site, formes différentes en plus de la couleur).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ICI = Path(__file__).resolve().parent
DONNEES = ICI / "donnees"
FIGURES = ICI / "figures"
sys.path.insert(0, str(ICI.parents[1] / "tableaux"))
import generer as tableaux  # noqa: E402

SURFACE, ENCRE, ENCRE_2, FILET = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SITES = {"INCT4": "#2a78d6", "INCT17": "#eb6834", "INCT20955": "#1baf7a", "INCT41": "#eda100"}
POOLEE, PAR_SITE = "#2a78d6", "#eb6834"
BLEUS = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ESPECES = ["DENMIN", "PITAZU", "PHYCUV", "LEPLAT", "BOAFAB"]  # de la plus dure à la plus facile
REFERENCE = "logistic"
ENCODEUR = "__ENCODEUR__"
FENETRE_S = "__FENETRE__"  # durée des fenêtres de l'encodeur
CAS_AMORCAGE = [("PITAZU", "INCT41"), ("LEPLAT", "INCT4")]  # sites presque vides

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": FILET,
        "axes.labelcolor": ENCRE_2,
        "xtick.color": ENCRE_2,
        "ytick.color": ENCRE,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def site_key(site: str) -> str:
    return site.replace("INCT04", "INCT4")


def load() -> dict[str, pd.DataFrame]:
    out = {p.stem: pd.read_csv(p) for p in DONNEES.glob("*.csv")}
    for key, col in (("sites", "held_out_site"), ("amorcage", "site")):
        out[key][col] = out[key][col].map(site_key)
    return out


def head_order(tetes: pd.DataFrame) -> list[str]:
    """Têtes par rang moyen de l'AP poolée (fenêtres) sur les espèces, la meilleure en haut."""
    w = tetes[tetes["level"] == "window"].copy()
    w["rang"] = w.groupby("species")["ap"].rank(ascending=False)
    return w.groupby("head")["rang"].mean().sort_values().index.tolist()


def figure_positifs(sites: pd.DataFrame) -> None:
    """Fenêtres positives de chaque espèce, par site : où l'espèce chante, et combien."""
    counts = (
        sites[sites["head"] == REFERENCE]
        .pivot_table(index="species", columns="held_out_site", values="n_pos", aggfunc="first")
        .reindex(index=ESPECES[::-1], columns=list(SITES))
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    left = np.zeros(len(counts))
    for site, color in SITES.items():
        values = counts[site].to_numpy()
        ax.barh(
            counts.index,
            values,
            left=left,
            color=color,
            edgecolor=SURFACE,
            linewidth=2,
            height=0.62,
            label=site,
        )
        for i, v in enumerate(values):
            if v >= 60:
                ax.text(
                    left[i] + v / 2,
                    i,
                    f"{int(v)}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=ENCRE,
                )
        left += values
    for i, total in enumerate(left):
        ax.text(total + 25, i, f"{int(total)}", va="center", fontsize=9, color=ENCRE_2)
    ax.set_xlabel(f"fenêtres positives ({FENETRE_S} s)")
    ax.set_xlim(0, left.max() * 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    ax.grid(axis="x", color=FILET, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "1_positifs_par_site.png", dpi=150)
    plt.close(fig)


def figure_ap_par_site(sites: pd.DataFrame, order: list[str]) -> None:
    """AP de chaque tête sur chaque site tenu à l'écart (sites où l'espèce chante)."""
    s = sites[sites["n_pos"] > 0]
    columns = [
        (sp, site)
        for sp in ESPECES
        for site in SITES
        if ((s["species"] == sp) & (s["held_out_site"] == site)).any()
    ]
    grid = np.array(
        [
            [
                s[(s["species"] == sp) & (s["held_out_site"] == site) & (s["head"] == h)][
                    "ap"
                ].iloc[0]
                for sp, site in columns
            ]
            for h in order
        ]
    )
    n_pos = [
        int(s[(s["species"] == sp) & (s["held_out_site"] == site)]["n_pos"].iloc[0])
        for sp, site in columns
    ]
    cmap = LinearSegmentedColormap.from_list("bleus", BLEUS)
    fig, ax = plt.subplots(figsize=(1.0 + 0.72 * len(columns), 0.36 * len(order) + 1.8))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if v > 0.62 else ENCRE,
            )
    ax.set_yticks(range(len(order)), order, fontsize=9, family="DejaVu Sans Mono")
    labels = [f"{sp}\n{site}\n{n} pos." for (sp, site), n in zip(columns, n_pos, strict=True)]
    ax.set_xticks(range(len(columns)), labels, fontsize=7.5)
    ax.xaxis.tick_top()
    for j in range(1, len(columns)):  # séparation entre espèces
        if columns[j][0] != columns[j - 1][0]:
            ax.axvline(j - 0.5, color=SURFACE, linewidth=4)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "2_ap_par_site.png", dpi=150)
    plt.close(fig)


def figure_poolee_site(tetes: pd.DataFrame, order: list[str]) -> None:
    """AP poolée (tous les sites ensemble) contre AP moyenne par site, tête par tête."""
    w = tetes[tetes["level"] == "window"]
    fig, axes = plt.subplots(1, len(ESPECES), figsize=(13, 0.3 * len(order) + 1.6), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, sp in zip(axes, ESPECES, strict=True):
        part = w[w["species"] == sp].set_index("head").reindex(order)
        for yi, (a, b) in zip(y, zip(part["ap"], part["ap_fold_mean"], strict=True), strict=True):
            ax.plot([a, b], [yi, yi], color=FILET, linewidth=2, zorder=1)
        ax.scatter(
            part["ap"],
            y,
            s=36,
            color=POOLEE,
            marker="o",
            zorder=2,
            edgecolor=SURFACE,
            linewidth=1,
            label="AP poolée",
        )
        ax.scatter(
            part["ap_fold_mean"],
            y,
            s=40,
            color=PAR_SITE,
            marker="D",
            zorder=2,
            edgecolor=SURFACE,
            linewidth=1,
            label="AP moyenne par site",
        )
        ax.set_title(sp, fontsize=10, color=ENCRE)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.grid(axis="x", color=FILET, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, order, fontsize=9, family="DejaVu Sans Mono")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "3_poolee_contre_site.png", dpi=150)
    plt.close(fig)


def tableau_tetes(tetes: pd.DataFrame, comparaisons: pd.DataFrame, order: list[str]) -> None:
    """Tableau PNG : AP poolée · AP par site, en vert/rouge si l'écart apparié à la
    logistique survit à Holm (toutes espèces et têtes du niveau fenêtre) et atteint 0,02."""
    w = tetes[tetes["level"] == "window"]
    c = comparaisons[comparaisons["level"] == "window"]
    rows = []
    for h in order:
        row: list = [tableaux.C(h, "ref") if h == REFERENCE else h]
        for sp in ESPECES:
            r = w[(w["species"] == sp) & (w["head"] == h)].iloc[0]
            text = f"{r['ap']:.2f} · {r['ap_fold_mean']:.2f}".replace(".", ",")
            k = c[(c["species"] == sp) & (c["head"] == h)]
            tag = "ref" if h == REFERENCE else None
            if len(k) and bool(k["significant_holm"].iloc[0]) and abs(k["diff"].iloc[0]) >= 0.02:
                tag = "mieux" if k["diff"].iloc[0] > 0 else "moins"
            row.append(tableaux.C(text, tag) if tag else text)
        rows.append(row)
    table = tableaux.Tableau(
        "tetes",
        "Têtes jugées un site à la fois : AP poolée · AP moyenne par site",
        f"{ENCODEUR}, fenêtres de {FENETRE_S} s, 20 négatifs par positif et par site."
        " Couleur : écart apparié"
        " à la logistique (AP poolée) significatif après Holm et d'au moins 0,02 ; bootstrap par"
        " enregistrement, donc optimiste (les sites ne sont pas tirés).",
        [
            tableaux.Section(
                [tableaux.Colonne("Tête", 1.3, code=True)]
                + [tableaux.Colonne(sp, 1.0) for sp in ESPECES],
                rows,
            )
        ],
        ["Espèces de la plus dure à la plus facile. Têtes par rang moyen de l'AP poolée."],
        largeur=1700,
    )
    tableaux.verifier(table)
    image = tableaux.rendre(table).quantize(colors=48)
    image.save(FIGURES / "4_tableau_tetes.png", optimize=True)


def tableau_amorcage(amorcage: pd.DataFrame, order: list[str]) -> None:
    """Tableau PNG : amorcer un site presque vide. AP du site · rappel à précision 0,5 au seuil
    choisi sur les autres sites (entre parenthèses : seuil choisi sur place, optimiste)."""
    rows = []
    for h in order:
        row: list = [tableaux.C(h, "ref") if h == REFERENCE else h]
        for sp, site in CAS_AMORCAGE:
            r = amorcage[(amorcage["species"] == sp) & (amorcage["site"] == site)]
            r = r[r["head"] == h].iloc[0]
            prec = r["precision_seuil_ailleurs"]
            prec = "–" if pd.isna(prec) else f"{prec:.2f}"
            text = (
                f"{r['ap_site']:.2f} · {r['recall_seuil_ailleurs']:.2f} / {prec} "
                f"({r['recall_seuil_sur_place']:.2f})"
            ).replace(".", ",")
            row.append(tableaux.C(text, "ref") if h == REFERENCE else text)
        rows.append(row)
    n_pos = {
        (sp, site): int(
            amorcage[(amorcage["species"] == sp) & (amorcage["site"] == site)]["n_pos"].iloc[0]
        )
        for sp, site in CAS_AMORCAGE
    }
    table = tableaux.Tableau(
        "amorcage",
        "Amorcer un site presque vide",
        f"{ENCODEUR}. AP du site · rappel / précision obtenus sur le site au seuil de précision"
        " 0,5 choisi sur les autres sites (– : aucune alerte). Entre parenthèses : rappel au"
        " seuil choisi sur place (oracle).",
        [
            tableaux.Section(
                [tableaux.Colonne("Tête", 1.3, code=True)]
                + [
                    tableaux.Colonne(f"{sp} · {site} ({n_pos[(sp, site)]} pos.)", 1.4)
                    for sp, site in CAS_AMORCAGE
                ],
                rows,
            )
        ],
        ["Têtes par rang moyen de l'AP poolée (figure 2)."],
        largeur=1100,
    )
    tableaux.verifier(table)
    image = tableaux.rendre(table).quantize(colors=48)
    image.save(FIGURES / "5_amorcage.png", optimize=True)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    d = load()
    order = head_order(d["tetes"])
    figure_positifs(d["sites"])
    figure_ap_par_site(d["sites"], order)
    figure_poolee_site(d["tetes"], order)
    tableau_tetes(d["tetes"], d["comparaisons"], order)
    tableau_amorcage(d["amorcage"], order)
    print(sorted(p.name for p in FIGURES.iterdir()))


if __name__ == "__main__":
    main()
