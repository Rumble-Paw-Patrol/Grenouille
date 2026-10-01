"""Inventaire et déséquilibre des annotations, par site et par micro, lus dans la base locale.

La base (`data/db/blanci.sqlite`) n'est pas dans git : la présentation n'a que les totaux par
site (DECISIONS n° 35, 49). Ce script produit le détail, à glisser dans la diapo « Les
annotations » : tableaux CSV et deux figures (enregistrements par site ; fenêtres annotées
positives/négatives par micro).

    uv run --group notebook python \
        documentation/prez/presentation-suivi-2/inventaire_annotations.py [data/db/blanci.sqlite]

Base ouverte en lecture seule. Le dernier label d'une fenêtre fait foi (labels en ajout seul).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from blanci.labels import POSITIVE_LABELS

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "inventaire"
DB = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "db" / "blanci.sqlite"

INK, MUTED, POS, NEG = "#1C2B24", "#5E6E66", "#2E8B57", "#B8C4BD"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    rec = pd.read_sql("SELECT dataset, site, mic_id, duration_s, qc_flags FROM recordings", con)
    inv = (
        rec.groupby(["dataset", "site"])
        .agg(
            enregistrements=("mic_id", "size"),
            micros=("mic_id", "nunique"),
            heures=("duration_s", lambda s: s.sum() / 3600),
        )
        .round(1)
        .reset_index()
    )
    inv.to_csv(OUT / "enregistrements_par_site.csv", index=False)

    lab = pd.read_sql(
        """
        SELECT l.window_id, l.label, l.label_id, r.site, r.mic_id, r.recording_id
        FROM labels l JOIN windows w USING (window_id) JOIN recordings r USING (recording_id)
        """,
        con,
    )
    lab = lab.sort_values("label_id").groupby("window_id").tail(1)  # dernier label
    lab["classe"] = lab["label"].isin(POSITIVE_LABELS).map({True: "positive", False: "négative"})
    par_micro = (
        lab.pivot_table(
            index=["site", "mic_id"],
            columns="classe",
            values="window_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=["positive", "négative"], fill_value=0)
        .reset_index()
    )
    rec_pos = (
        lab[lab.classe == "positive"]
        .groupby(["site", "mic_id"])
        .recording_id.nunique()
        .rename("enregistrements_positifs")
    )
    par_micro = par_micro.merge(rec_pos, on=["site", "mic_id"], how="left").fillna(0)
    par_micro = par_micro.sort_values(["positive", "négative"], ascending=False)
    par_micro.to_csv(OUT / "annotations_par_micro.csv", index=False)
    par_site = par_micro.groupby("site")[["positive", "négative", "enregistrements_positifs"]].sum()
    par_site.to_csv(OUT / "annotations_par_site.csv")

    # Figure : fenêtres annotées par micro (barres empilées horizontales)
    d = par_micro.head(40)
    labels = d["site"] + " · " + d["mic_id"].astype(str)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(d) + 1)))
    ax.barh(labels, d["positive"], color=POS, label="positives", edgecolor="white", linewidth=1)
    ax.barh(
        labels,
        d["négative"],
        left=d["positive"],
        color=NEG,
        label="négatives",
        edgecolor="white",
        linewidth=1,
    )
    ax.invert_yaxis()
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="x", color="#E3E8E5", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Fenêtres annotées par micro (40 premiers)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "annotations_par_micro.png", dpi=200)
    plt.close(fig)

    print(inv.to_string(index=False))
    print()
    print(par_site.to_string())
    print(f"\nSorties : {OUT}")


if __name__ == "__main__":
    main()
