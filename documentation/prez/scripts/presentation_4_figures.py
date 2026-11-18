"""Figures de la présentation n° 4 (inventaire et contrôle qualité), diapos 3, 4 et 5 bis.

Lit la base locale en lecture seule (`paths.db` de la config utilisateur) et écrit dans
`documentation/prez/figures/inventaire_qc/` :

- `inv_sites.png` (diapo 3, gauche) : heures par site, les relevés 2025-2026 empilés sur le site
  dont ils sont un sous-site (CDR, Patawa Est, Patawa Ouest → Kaw ; RNRT → Trésor, comme
  `lieux` dans `documentation/tableau-de-bord/structure.yaml`) ;
- `inv_programmes.png` (diapo 3, droite) : enregistrements selon le programme du micro
  (5 h → 19 h 30 ou 6 h → 18 h 30), lu dans les données comme dans le notebook 05 (§ 4) ;
- `inv_calendrier.png` (diapo 4) : un trait par jour enregistré, axe coupé entre les deux jeux ;
- `inv_base.png` (diapo 5 bis) : vraies lignes de la table recordings, vue d'explorateur.

Chaque figure est dessinée à la taille de sa place sur la diapo : 1 pt de police = 1 pt affiché.

    uv run python documentation/prez/scripts/presentation_4_figures.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from blanci.core.config import config_path, default_user_config, load_config  # noqa: E402
from blanci.inputs.qc import parse_flags  # noqa: E402

FIG = ROOT / "documentation" / "prez" / "figures" / "inventaire_qc"
VERT, OR, INK, MUTED, GRID, SAGE = "#2A7F72", "#E9B44C", "#1B2A24", "#5B6B63", "#E4EEE8", "#A7BFB2"
NOMS = {"Tresor": "Trésor", "Molokoi": "Molokoï", "PatawaEst": "Patawa Est",
        "PatawaOuest": "Patawa Ouest"}

plt.rcParams.update({
    "font.family": ["Calibri", "DejaVu Sans"], "font.size": 12, "axes.edgecolor": SAGE,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK, "xtick.labelsize": 11,
    "ytick.labelsize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
})


def milliers(n: float) -> str:
    return f"{n:,.0f}".replace(",", " ")


def charger() -> pd.DataFrame:
    cfg = load_config(default_user_config())
    con = sqlite3.connect(f"file:{config_path(cfg, 'db').as_posix()}?mode=ro", uri=True)
    rec = pd.read_sql_query(
        "SELECT recording_id, dataset, site, mic_id, start_utc, duration_s, qc_flags "
        "FROM recordings WHERE dataset IN ('2023', '2026')", con)
    con.close()
    lieux = yaml.safe_load((ROOT / "documentation" / "tableau-de-bord" / "structure.yaml")
                           .read_text(encoding="utf-8"))["lieux"]
    rec["site"] = rec.site.map(lambda s: NOMS.get(s, s))
    rec["lieu"] = rec.site.map(lambda s: lieux.get(s, s))
    rec["fl"] = rec.qc_flags.map(parse_flags)
    offset = pd.Timedelta(hours=cfg["recorder"]["filename_utc_offset_h"])
    rec["local"] = pd.to_datetime(rec.start_utc, utc=True).dt.tz_convert(None) + offset
    rec["day"] = rec.local.dt.normalize()
    rec["campagne"] = ~rec.fl.map(lambda f: bool(f.get("off_campaign")))
    rec["normal"] = (rec.duration_s - cfg["qc"]["expected_duration_s"]).abs() \
        <= cfg["qc"]["duration_tolerance_s"]
    rec["programme"] = programmes(rec)
    return rec


def programmes(rec: pd.DataFrame) -> list[str]:
    """Programme de chaque point (jeu, site, micro) : créneaux présents sur au moins la moitié
    de ses jours intérieurs (ni pose ni relève), comme dans le notebook 05, § 4."""
    ok = rec[rec.campagne & rec.normal].assign(slot=lambda x: x.local.dt.strftime("%H:%M"))
    days = ok.groupby(["dataset", "site", "mic_id", "day"]).slot.agg(frozenset).reset_index()
    bord = days.groupby(["dataset", "site", "mic_id"]).day.transform
    days = days[(days.day != bord("min")) & (days.day != bord("max"))]
    prog = {}
    for key, g in days.groupby(["dataset", "site", "mic_id"]):
        freq = pd.Series([s for x in g.slot for s in x]).value_counts() / len(g)
        v = sorted(freq[freq >= 0.5].index)
        if v:
            prog[key] = f"{v[0]}-{v[-1]}"
    return [prog.get(k, "?") for k in zip(rec.dataset, rec.site, rec.mic_id, strict=True)]


def fig_sites(rec: pd.DataFrame) -> None:
    h = rec.groupby(["lieu", "dataset", "site"]).agg(
        h=("duration_s", lambda s: s.sum() / 3600), n=("recording_id", "size"),
        micros=("mic_id", "nunique")).reset_index()
    tot = h.groupby("lieu").agg(h=("h", "sum"), n=("n", "sum")).sort_values("h")
    fig, ax = plt.subplots(figsize=(6.9, 3.55))
    for y, lieu in enumerate(tot.index):
        x = 0.0
        parts = h[h.lieu == lieu].sort_values(["dataset", "h"], ascending=[True, False])
        petits, x_petits = [], None
        for _, p in parts.iterrows():
            c = VERT if p.dataset == "2023" else OR
            ax.barh(y, p.h, left=x, color=c, height=0.62, edgecolor="white", linewidth=1.5)
            nom = "suivi 2023-24" if p.dataset == "2023" else p.site
            if p.h >= 150:
                ax.text(x + p.h / 2, y, f"{nom}\n{p.h:.0f} h", ha="center", va="center",
                        fontsize=10.5, color="white" if p.dataset == "2023" else INK)
            else:
                petits.append(f"{nom} {p.h:.0f} h")
                x_petits = x if x_petits is None else x_petits
            x += p.h
        if petits:
            ax.annotate(" · ".join(petits), ((x_petits + x) / 2, y + 0.31),
                        xytext=(x_petits - 60, y + 0.5), fontsize=10.5, color=INK, ha="left",
                        va="bottom", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
        ax.text(x + 15, y, f"{x:,.0f} h · {milliers(tot.loc[lieu, 'n'])} enr.".replace(",", " "),
                va="center", fontsize=11.5, color=INK, fontweight="bold")
    ax.set_yticks(range(len(tot)), tot.index)
    ax.set_xlim(0, tot.h.max() * 1.42)
    ax.set_ylim(-0.5, len(tot) - 0.25)
    ax.set_xlabel("heures d'enregistrement")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.bar(0, 0, color=VERT, label="phénologie 2023-2024 (un an)")
    ax.bar(0, 0, color=OR, label="relevés 2025-2026 (une semaine)")
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    ax.set_title("Heures enregistrées par site (relevés empilés sur leur site)", loc="left",
                 fontsize=14, color=INK, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "inv_sites.png", dpi=200)
    plt.close(fig)


def fig_programmes(rec: pd.DataFrame) -> None:
    progs = {"05:00-19:30": "5 h → 19 h 30\n30 par jour",
             "06:00-18:30": "6 h → 18 h 30\n26 par jour"}
    g = rec[rec.programme.isin(progs)].groupby(["programme", "dataset", "site"]).agg(
        n=("recording_id", "size"), micros=("mic_id", "nunique")).reset_index()
    fig, ax = plt.subplots(figsize=(5.0, 3.55))
    ymax = g.groupby("programme").n.sum().max()
    for x, prog in enumerate(progs):
        parts = g[g.programme == prog].sort_values(["dataset", "n"], ascending=[True, False])
        y, petits = 0, []
        for _, p in parts.iterrows():
            c = VERT if p.dataset == "2023" else OR
            ax.bar(x, p.n, bottom=y, color=c, width=0.6, edgecolor="white", linewidth=1.5)
            if p.n >= 4000:
                nom = p.site + ("\n2023-24" if p.dataset == "2023" else "")
                ax.text(x, y + p.n / 2, nom, ha="center", va="center", fontsize=10.5,
                        color="white" if p.dataset == "2023" else INK)
            else:
                petits.append(f"{p.site} ({p.micros} micro{'s' if p.micros > 1 else ''})")
            y += p.n
        ax.text(x, y + ymax * 0.02, milliers(y), ha="center", va="bottom", fontsize=12,
                color=INK, fontweight="bold")
        if petits:
            ax.annotate("+ " + "\n+ ".join(petits), (x + 0.3, y - ymax * 0.01),
                        xytext=(x + 0.36, y - ymax * 0.02), fontsize=9.5, color=MUTED,
                        va="top", ha="left")
    ax.set_xticks(range(len(progs)), list(progs.values()), fontsize=11.5, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.45, 1.95)
    ax.set_ylim(0, ymax * 1.1)
    ax.set_ylabel("enregistrements de 2 min")
    ax.yaxis.set_major_formatter(lambda v, _: milliers(v))
    ax.tick_params(axis="y", labelsize=11, labelcolor=MUTED)
    ax.grid(axis="x", visible=False)
    ax.set_title("Enregistrements selon le programme du micro", loc="left", fontsize=14,
                 color=INK, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "inv_programmes.png", dpi=200)
    plt.close(fig)


def fig_calendrier(rec: pd.DataFrame) -> None:
    jours = rec[rec.campagne].groupby(["lieu", "dataset", "site"]).day.unique()
    ordre = ["Mataroni", "Molokoï", "Trésor", "Kaw"]
    fenetres = [(pd.Timestamp("2023-10-15"), pd.Timestamp("2024-12-15")),
                (pd.Timestamp("2025-12-12"), pd.Timestamp("2026-03-12"))]
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 3.3), sharey=True,
                             gridspec_kw={"width_ratios": [3, 2], "wspace": 0.04})
    for ax, (a, b) in zip(axes, fenetres, strict=True):
        for (lieu, jeu, site), d in jours.items():
            y = ordre.index(lieu)
            c = VERT if jeu == "2023" else OR
            ax.broken_barh([(pd.Timestamp(t), pd.Timedelta(days=1)) for t in d], (y - 0.3, 0.6),
                           color=c)
            if jeu == "2026" and a <= pd.Timestamp(min(d)) <= b:
                debut, fin = pd.Timestamp(min(d)), pd.Timestamp(max(d)) + pd.Timedelta(days=1)
                ha = {"CDR": "right", "Patawa Ouest": "left"}.get(site, "center")
                xt = {"right": fin, "left": debut, "center": debut + (fin - debut) / 2}[ha]
                ax.text(xt, y + 0.36, site, ha=ha, va="bottom", fontsize=11, color=INK)
        ax.set_xlim(a, b)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(ordre)), ordre)
    axes[0].set_ylim(-0.6, len(ordre) - 0.25)
    axes[0].xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%Y"))
    axes[1].xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%Y"))
    axes[1].spines["left"].set_visible(False)
    axes[0].set_title("Phénologie 2023-2024 : un an sans interruption", loc="left", fontsize=13,
                      color=VERT, fontweight="bold")
    axes[1].set_title("Relevés 2025-2026 : une semaine chacun", loc="left", fontsize=13,
                      color="#B07D1A", fontweight="bold")
    for ax, cote in ((axes[0], 1), (axes[1], 0)):  # marques de coupure de l'axe
        ax.plot([cote], [0], transform=ax.transAxes, marker=[(-1, -2), (1, 2)], ms=10,
                color=SAGE, mew=1.2, clip_on=False, ls="none")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.12)
    fig.savefig(FIG / "inv_calendrier.png", dpi=200)
    plt.close(fig)


def fig_base() -> None:
    """Vue de la table recordings (diapo 5 bis) : les vraies lignes du micro 2LA04185 le matin du
    25/12/2025, dessinées comme dans un explorateur de base SQLite."""
    cfg = load_config(default_user_config())
    base = config_path(cfg, "db")
    con = sqlite3.connect(f"file:{base.as_posix()}?mode=ro", uri=True)
    cols = ["recording_id", "path", "dataset", "site", "mic_id", "start_utc", "duration_s",
            "sample_rate", "channels", "sha256", "qc_flags"]
    lignes = con.execute(
        f"SELECT {', '.join(cols)} FROM recordings WHERE mic_id = '2LA04185' "
        "AND start_utc >= '2025-12-25T07:00:00Z' ORDER BY start_utc LIMIT 10").fetchall()
    n = con.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
    con.close()

    def court(c, v):
        if v is None:
            return "NULL"
        v = str(v)
        if c == "path":
            return "…/" + "/".join(v.split("/")[-3:])
        if c == "qc_flags":
            return v[:34] + "…"
        return v

    larg = {"recording_id": 17, "path": 45, "dataset": 8, "site": 5, "mic_id": 9,
            "start_utc": 21, "duration_s": 11, "sample_rate": 12, "channels": 9, "sha256": 7,
            "qc_flags": 36}
    total = sum(larg.values()) + 4  # + colonne des numéros de ligne
    fig = plt.figure(figsize=(12.13, 3.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total)
    ax.set_ylim(len(lignes) + 3.2, 0)
    ax.axis("off")
    police = {"family": "Consolas", "size": 8.6}
    # barre d'onglets et en-tête de colonnes
    ax.add_patch(plt.Rectangle((0, 0), total, 1.1, color="#E9ECEF"))
    ax.text(0.6, 0.6, f"blanci.sqlite  ›  Parcourir les données  ›  Table : recordings"
            f"   ({milliers(n)} lignes)", va="center", fontsize=10, color=INK,
            fontweight="bold")
    ax.add_patch(plt.Rectangle((0, 1.1), total, 1.0, color="#F6F7F8", ec="#C9CED3", lw=0.6))
    x = 4
    for c in cols:
        ax.text(x + 0.5, 1.6, c, va="center", fontdict={**police, "weight": "bold"},
                color=INK)
        ax.plot([x, x], [1.1, len(lignes) + 2.1], color="#D5D9DD", lw=0.6)
        x += larg[c]
    for i, ligne in enumerate(lignes):
        y = 2.1 + i
        cible = ligne[1].endswith("2LA04185_20251225_050000.wav")
        fond = "#FFF3CD" if cible else ("#FFFFFF" if i % 2 == 0 else "#F8F9FA")
        ax.add_patch(plt.Rectangle((0, y), total, 1, color=fond, zorder=0))
        ax.text(3.4, y + 0.5, str(i + 1), ha="right", va="center", fontdict=police,
                color=MUTED)
        x = 4
        for c, v in zip(cols, ligne, strict=True):
            ax.text(x + 0.5, y + 0.5, court(c, v), va="center", fontdict=police,
                    color="#9AA0A6" if v is None else INK, clip_on=True)
            x += larg[c]
        ax.plot([0, total], [y + 1, y + 1], color="#E3E6E9", lw=0.5)
    ax.add_patch(plt.Rectangle((0, 0), total, len(lignes) + 2.1, fill=False, ec="#C9CED3",
                               lw=1))
    ax.text(0.6, len(lignes) + 2.7, "Le micro 2LA04185 à CDR, le matin du 25 décembre 2025 "
            "(heures en UTC) ; surligné, le fichier de la diapo précédente.", va="center",
            fontsize=10, color=MUTED)
    fig.savefig(FIG / "inv_base.png", dpi=200)
    plt.close(fig)


def main() -> None:
    fig_base()
    rec = charger()
    print(f"{len(rec)} enregistrements, {rec.duration_s.sum() / 3600:.1f} h, "
          f"{rec.mic_id.nunique()} enregistreurs, {rec.groupby(['site', 'mic_id']).ngroups} points")
    print(rec.groupby(["programme", "dataset", "site"]).agg(
        n=("recording_id", "size"), micros=("mic_id", "nunique")).to_string())
    fig_sites(rec)
    fig_programmes(rec)
    fig_calendrier(rec)


if __name__ == "__main__":
    main()
