"""Figures de la présentation n° 4 (inventaire et contrôle qualité), diapos 3 à 7, 9 et 11.

Lit la base locale en lecture seule (`paths.db` de la config utilisateur) et écrit dans
`documentation/prez/figures/inventaire_qc/` :

- `inv_sites.png` (diapo 3, gauche) : heures par site, les relevés 2025-2026 empilés sur le site
  dont ils sont un sous-site (CDR, Patawa Est, Patawa Ouest → Kaw ; RNRT → Trésor, comme
  `lieux` dans `documentation/tableau-de-bord/structure.yaml`) ;
- `inv_programmes.png` (diapo 3, droite) : enregistrements selon le programme du micro
  (5 h → 19 h 30 ou 6 h → 18 h 30), lu dans les données comme dans le notebook 05 (§ 4) ;
- `inv_calendrier.png` (diapo 4) : un trait par jour enregistré, axe coupé entre les deux jeux ;
- `inv_base.png` (diapo 5 bis) : vraies lignes de la table recordings, vue d'explorateur ;
- `inv_gains_reglage.png`, `inv_gains_mesure.png` (diapo 6) : gain réglé (barres) et écart de niveau
  mesuré entre les deux pistes (histogramme), mêmes couleurs ; l'histogramme se refait depuis
  `paths.reports/pistes/echantillon_pistes.csv`, sinon il est découpé dans `inv_gains.png` ;
- `inv_completude_p2.png` (diapo 7) : complétude du programme, 2023-2024 par station et 2025-2026
  par site, même tracé ;
- `mol_niveaux.png`, `mol_spectro_1..4.png` (diapo 9) : niveau des deux micros de Molokoï le
  03/04/2024 à l'heure du nom et à celle de l'en-tête, avec la corrélation ; spectrogrammes lus
  sur le disque BLANCI_2023-24 s'il est branché, sinon découpés dans
  `sources_paquet_2/molokoi_03avril.png` ;
- `qc_ligne_drapeau.png` (diapo 11) : une vraie ligne de la base qui porte un drapeau.

Chaque figure est dessinée à la taille de sa place sur la diapo : 1 pt de police = 1 pt affiché.

    uv run python documentation/prez/scripts/presentation_4_figures.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from blanci.core.config import config_path, default_user_config, load_config  # noqa: E402
from blanci.inputs.qc import parse_flags  # noqa: E402

FIG = ROOT / "documentation" / "prez" / "figures" / "inventaire_qc"
VERT, OR, INK, MUTED, GRID, SAGE = "#2A7F72", "#E9B44C", "#1B2A24", "#5B6B63", "#E4EEE8", "#A7BFB2"
# Couleurs du notebook 05 (gains, complétude, horloge) : le réglage et l'observation les partagent.
G_VERT, G_ORANGE, ROUGE = "#2E8B57", "#D9822B", "#C8553D"
SOURCES = ROOT / "documentation" / "prez" / "presentation_4_paquets" / "sources_paquet_2"
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
    # Jours où le programme a tourné (fichiers de 120 s) : les tests de réglage des 30 et
    # 31/12/2025, faits au bureau à moins de 7 jours de la pose du 06/01, ne sont pas hors
    # relevé et dessinaient un faux bloc de Mataroni avant un faux trou.
    jours = rec[rec.campagne & rec.normal].groupby(["lieu", "dataset", "site"]).day.unique()
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

# --------------------------------------------------------------------------- diapo 6 : gains
# Points par réglage lus dans la dernière exécution du notebook 05 avec les disques (3 607
# enregistrements, 164 points). Le gain vit dans l'en-tête GUANO des WAV, pas dans la base : sans
# `echantillon_pistes.csv` (disque F:), on ne peut pas le recompter ici.
GAINS_NOTEBOOK = {"6 / 18 dB": 143, "18 / 6 dB": 21}


def fig_gains() -> str:
    """Diapo 6 : à gauche le réglage (barres verticales), à droite l'écart mesuré, mêmes couleurs.
    Renvoie « csv » si tout vient de l'échantillon des deux pistes, « partiel » sinon."""
    cfg = load_config(default_user_config())
    csv = config_path(cfg, "reports") / "pistes" / "echantillon_pistes.csv"
    S = None
    if csv.is_file():
        S = pd.read_csv(csv)
        if "erreur" in S:
            S = S[S["erreur"].isna()]
        S = S.dropna(subset=["gain_0", "gain_1", "level_diff_db"]).copy()
        S["config"] = [f"{int(a)} / {int(b)} dB" for a, b in zip(S.gain_0, S.gain_1, strict=True)]
        S["ecart_reglage"] = S.gain_0 - S.gain_1
        points = S.groupby(["dataset", "site", "mic_id"]).config.first().value_counts().to_dict()
    else:
        points = GAINS_NOTEBOOK
    couleur = {"6 / 18 dB": G_VERT, "18 / 6 dB": G_ORANGE}
    ecart = {"6 / 18 dB": "−12 dB", "18 / 6 dB": "+12 dB"}
    fig, ax = plt.subplots(figsize=(5.48, 3.35))
    for x, cle in enumerate(couleur):
        v = points[cle]
        ax.bar(x, v, color=couleur[cle], alpha=0.85, width=0.62)
        ax.text(x, v + 2, str(v), ha="center", va="bottom", fontsize=14, color=INK,
                fontweight="bold")
        ax.text(x, v / 2 if v > 60 else v + 38, f"écart réglé\n{ecart[cle]}", ha="center",
                va="center", fontsize=12, color="white" if v > 60 else INK)
    ax.set_xticks(range(2), list(couleur), fontsize=13, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 172)
    ax.set_ylabel("points (site, micro)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Réglage : gain piste 0 / piste 1", loc="left", fontsize=13, color=INK,
                 fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "inv_gains_reglage.png", dpi=200)
    plt.close(fig)

    if S is not None:
        fig, ax = plt.subplots(figsize=(6.45, 3.35))
        bins = np.arange(-30, 30.5, 1)
        for cle in couleur:
            g = S[S.config == cle]
            ax.hist(g.level_diff_db, bins=bins, color=couleur[cle], alpha=0.85,
                    label=f"réglage {ecart[cle]} ({len(g)} enr.)")
            ax.axvline(g.ecart_reglage.iloc[0], color=couleur[cle], lw=1, ls="--")
        ax.set_xlabel("niveau piste 0 − niveau piste 1, mesuré (dB)")
        ax.set_ylabel("enregistrements")
        ax.legend(frameon=False, fontsize=11)
        ax.set_title("Observation : écart mesuré (tirets : écart réglé)", loc="left", fontsize=13,
                     color=INK, fontweight="bold", pad=10)
        fig.tight_layout()
        fig.savefig(FIG / "inv_gains_mesure.png", dpi=200)
        plt.close(fig)
        return "csv"
    # Sans l'échantillon : l'histogramme du notebook 05 (déjà vert et orange), découpé à droite.
    ancien = Image.open(FIG / "inv_gains.png")
    ancien.crop((831, 0, ancien.width, ancien.height)).save(FIG / "inv_gains_mesure.png")
    return "partiel"


# --------------------------------------------------------------------------- diapo 7 : complétude
def jours_micros(rec: pd.DataFrame) -> pd.DataFrame:
    """Un jour-micro : enregistrements de durée normale reçus, programme attendu (créneaux présents
    sur au moins la moitié des jours intérieurs du point), complétude, bord (pose ou relève)."""
    ok = rec[rec.campagne & rec.normal].assign(slot=lambda x: x.local.dt.strftime("%H:%M"))
    days = ok.groupby(["dataset", "site", "mic_id", "day"]).agg(
        n=("recording_id", "size"), slots=("slot", frozenset)).reset_index()
    bord = days.groupby(["dataset", "site", "mic_id"]).day.transform
    days["bord"] = (days.day == bord("min")) | (days.day == bord("max"))
    prog = {}
    for key, g in days[~days.bord].groupby(["dataset", "site", "mic_id"]):
        freq = pd.Series([s for x in g.slots for s in x]).value_counts() / len(g)
        prog[key] = len(freq[freq >= 0.5])
    days["attendu"] = [prog.get(k, np.nan) or np.nan
                       for k in zip(days.dataset, days.site, days.mic_id, strict=True)]
    days["completude"] = days.n / days.attendu
    return days


def fig_completude(rec: pd.DataFrame) -> dict:
    """Diapo 7, 12,13 × 3,2 po : à gauche 2023-2024 par station (comme dans le notebook 05, § 4),
    à droite 2025-2026 par site, même tracé : un carré par jour, vert = part du programme reçue,
    rouge = micro sans aucun enregistrement entre la pose et la relève."""
    cfg = load_config(default_user_config())
    days = jours_micros(rec)
    inner = days[~days.bord]
    station = {(site, mic): f"{site}/{st}" for site, sts in cfg["plan"]["stations_2023"].items()
               for st, mics in sts.items() for mic in mics}
    inverse = {v: k for k, v in NOMS.items()}  # « Trésor » (rec) → « Tresor » (config)
    d23 = days[days.dataset == "2023"].assign(station=lambda x: [
        station.get((inverse.get(a, a), b), f"{a}/?")
        for a, b in zip(x.site, x.mic_id, strict=True)])
    stations = sorted(d23.station.unique())
    rouge, cm = "#C0492F", plt.get_cmap("Greens")
    h1, h2 = 0.15, 0.127  # même épaisseur à l'écran (5 et 6 lignes sur la même hauteur)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.13, 5.0),
                                 gridspec_kw={"width_ratios": [1.7, 1]})

    def trous(vides):
        """Séries de jours vides consécutifs : [(premier, dernier), ...]."""
        runs, debut, prec = [], None, None
        for d in vides:
            if prec is None or d - prec > pd.Timedelta(days=1):
                if debut is not None:
                    runs.append((debut, prec))
                debut = d
            prec = d
        if debut is not None:
            runs.append((debut, prec))
        return runs

    for y, st in enumerate(stations):
        g = d23[d23.station == st]
        jours = g.groupby("day").completude.mean().clip(0, 1)
        a1.broken_barh([(d, pd.Timedelta(days=1)) for d in jours.index], (y - h1 / 2, h1),
                       color=[cm(v) for v in jours.values])
        vides = pd.date_range(g.day.min(), g.day.max(), freq="D").difference(g.day.unique())
        a1.broken_barh([(d, pd.Timedelta(days=1)) for d in vides], (y - h1 / 2, h1), color=rouge)
        for mic, gm in g.groupby("mic_id"):
            a1.text(gm.day.min(), y + 0.2, mic, fontsize=10, color=MUTED)
        for d0, d1 in trous(vides):
            n = (d1 - d0).days + 1
            tard = d0 > pd.Timestamp("2024-06-01")  # texte aligné à droite pour ne pas déborder
            a1.text(d1 + pd.Timedelta(days=1) if tard else d0, y - 0.2,
                    f"aucun enregistrement du {d0:%d/%m} au {d1:%d/%m} ({n} j)",
                    fontsize=10.5, color=rouge, va="top", ha="right" if tard else "left")
    a1.set_yticks(range(len(stations)), stations, fontsize=12, color=INK)
    a1.set_ylim(-0.6, len(stations) - 0.3)
    a1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    a1.tick_params(axis="x", labelsize=11)
    a1.set_title("Phénologie 2023-2024, par station", loc="left", fontsize=13, color=INK,
                 fontweight="bold", pad=26)
    a1.text(0, 1.03, "un carré par jour : vert, part du programme reçue ; rouge, rien",
            fontsize=11, color=MUTED, transform=a1.transAxes, va="bottom")
    a1.grid(axis="y", visible=False)

    d26 = days[days.dataset == "2026"]
    sites = list(d26.groupby("site").day.min().sort_values().index)  # chronologique, bas → haut
    for y, site in enumerate(sites):
        gs = d26[d26.site == site]
        pose = gs.day.min()
        moy = gs.groupby("day").completude.mean().clip(0, 1)
        a2.broken_barh([(d, 1) for d in (moy.index - pose).days], (y - h2 / 2, h2),
                       color=[cm(v) for v in moy.values])
        a2.text(0, y + 0.2, f"{site} · {gs.mic_id.nunique()} micros · pose le {pose:%d/%m}",
                fontsize=10, color=MUTED)
        manque = {}  # jour → micros sans aucun enregistrement entre la pose et la relève
        for mic, gm in gs.groupby("mic_id"):
            vides = pd.date_range(gm.day.min(), gm.day.max(), freq="D").difference(gm.day.unique())
            for v in vides:
                manque.setdefault(v, []).append(mic)
        for v, mics in manque.items():
            a2.broken_barh([((v - pose).days, 1)], (y - h2 / 2, h2), color=rouge)
            a2.text((v - pose).days, y - 0.2, f"{len(mics)} micros sur {gs.mic_id.nunique()} sans "
                    f"rien le {v:%d/%m}", fontsize=10.5, color=rouge, va="top")
    a2.set_yticks(range(len(sites)), sites, fontsize=12, color=INK)
    a2.set_ylim(-0.6, len(sites) - 0.3)
    a2.set_xlim(-0.5, 13)
    a2.set_xlabel("jours depuis la pose", fontsize=11)
    a2.tick_params(axis="x", labelsize=11)
    a2.set_title("Relevés 2025-2026, par site", loc="left", fontsize=13, color=INK,
                 fontweight="bold", pad=26)
    a2.text(0, 1.03, "un carré par jour : vert, part reçue ; rouge, rien",
            fontsize=11, color=MUTED, transform=a2.transAxes, va="bottom")
    a2.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=cm(1.0), label="programme complet (100 %)"),
                        Patch(color=cm(0.45), label="partiel : jour de pose ou de relève"),
                        Patch(color=rouge, label="aucun enregistrement ce jour-là")],
               loc="lower center", ncol=3, frameon=False, fontsize=11, handlelength=1.6,
               columnspacing=2.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.84, bottom=0.2, wspace=0.2)
    fig.savefig(FIG / "inv_completude_p2.png", dpi=200)
    plt.close(fig)
    return {"jours_micros": len(inner), "complets": float((inner.completude >= 1).mean()),
            "partiels": int(((inner.completude < 1) & (inner.completude > 0)).sum())}


# --------------------------------------------------------------------------- diapo 9 : Molokoï
def molokoi_niveaux() -> tuple[pd.DataFrame, dict[float, float]]:
    """Niveau (RMS de la piste 0) des enregistrements de Molokoï, avec l'heure de l'en-tête GUANO
    avant correction (`t_entete`), et la corrélation de SMA14636 corrigé avec SMA14163 selon le
    décalage appliqué (profil journalier retiré, comme dans le notebook 05, § 6)."""
    cfg = load_config(default_user_config())
    con = sqlite3.connect(f"file:{config_path(cfg, 'db').as_posix()}?mode=ro", uri=True)
    mol = pd.read_sql_query(
        "SELECT mic_id, start_utc, qc_flags FROM recordings WHERE site = 'Molokoi'", con)
    con.close()
    fl = mol.qc_flags.map(parse_flags)
    mol["rms"] = fl.map(lambda f: f.get("indices", {}).get("rms_dbfs"))
    mol["corrige"] = fl.map(lambda f: bool(f.get("clock_corrected")))
    mol["t"] = pd.to_datetime(mol.start_utc, utc=True).dt.tz_convert(None)
    mol["local"] = mol.t + pd.Timedelta(hours=cfg["recorder"]["filename_utc_offset_h"])
    mol = mol.dropna(subset=["rms"]).copy()
    # La correction a reculé ces enregistrements d'une heure : l'en-tête dit une heure de plus.
    mol["t_entete"] = mol.t + pd.to_timedelta(mol.corrige.astype(int), unit="h")
    mol["local_entete"] = mol.local + pd.to_timedelta(mol.corrige.astype(int), unit="h")
    mol["resid"] = mol.rms - mol.groupby(["mic_id", mol.local.dt.hour]).rms.transform("median")
    a = mol[mol.corrige].set_index("t_entete").resid
    b = mol[mol.mic_id == "SMA14163"].set_index("t").resid
    r = {}
    for lag in np.arange(-3, 3.25, 0.5):
        decale = a.copy()
        decale.index = decale.index + pd.Timedelta(hours=lag)
        j = pd.concat([decale.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
        r[float(lag)] = float(j.a.corr(j.b))
    return mol, r


def fig_molokoi_niveaux() -> dict:
    """Diapo 9, haut : les deux micros voisins le 03/04/2024, à l'heure du nom (corrigée) et à
    celle de l'en-tête GUANO, avec la corrélation sur les 1 521 enregistrements corrigés."""
    mol, r = molokoi_niveaux()
    jour = (pd.Timestamp("2024-04-03 08:00"), pd.Timestamp("2024-04-03 15:00"))
    sain = mol[mol.mic_id == "SMA14163"].set_index("local").rms.sort_index()
    sourd = mol[mol.mic_id == "SMA14636"]
    cas = [("À l'heure du nom de fichier (corrigée)", sourd.set_index("local").rms.sort_index(),
            r[-1.0]),
           ("À l'heure de l'en-tête GUANO (avant correction)",
            sourd.set_index("local_entete").rms.sort_index(), r[0.0])]
    fig, axes = plt.subplots(1, 2, figsize=(12.13, 2.45), sharey=True)
    for ax, (titre, serie, corr) in zip(axes, cas, strict=True):
        ax.plot(sain[jour[0]:jour[1]].index, sain[jour[0]:jour[1]].values, "o-", color=VERT, ms=5,
                lw=1.8, label="SMA14163 (voisin, horloge saine)")
        ax.plot(serie[jour[0]:jour[1]].index, serie[jour[0]:jour[1]].values, "s--", color=ROUGE,
                ms=5, lw=1.8, label="SMA14636")
        ax.axvline(pd.Timestamp("2024-04-03 11:00"), color=MUTED, lw=0.9, ls=":")
        ax.set_title(titre, loc="left", fontsize=13, color=INK, fontweight="bold", pad=6)
        ax.text(0.015, 0.96, f"corrélation {corr:.2f}".replace(".", ","), transform=ax.transAxes,
                ha="left", va="top", fontsize=13, color=INK, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": SAGE})
        ax.set_ylim(-70, -12)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", labelsize=10.5)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_xlabel("heure de Guyane, 03/04/2024", fontsize=11)
    axes[0].set_ylabel("niveau (dBFS, piste 0)", fontsize=11)
    axes[0].legend(frameon=False, fontsize=10.5, loc="lower center", ncol=2, columnspacing=1.2,
                   handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIG / "mol_niveaux.png", dpi=200)
    plt.close(fig)
    return {"corr_nom": r[-1.0], "corr_entete": r[0.0], "meilleur": max(r, key=r.get)}


SPECTRO = [("SMA14163", "SM F_SMA14163/Data/SMA14163_20240403_103000.wav", (75, 493)),
           ("SMA14163", "SM F_SMA14163/Data/SMA14163_20240403_110000.wav", (713, 493)),
           ("SMA14636", "SM E_SMA14636/Data/SMA14636_20240403_103000.wav", (75, 831)),
           ("SMA14636", "SM E_SMA14636/Data/SMA14636_20240403_110000.wav", (713, 831))]


def fig_molokoi_spectros(racine: Path | None = None) -> str:
    """Diapo 9, bas : les 4 spectrogrammes (0-12 kHz, 2 min, piste 0). Lus sur le disque si
    `racine` (dossier « Molokoi 11042024 » du disque BLANCI_2023-24) existe ; sinon découpés dans
    la figure du 06/11 (`sources_paquet_2/molokoi_03avril.png`, panneaux de 558 × 234 px)."""
    racine = racine or Path("D:/audio/R3_2024-04_avril/Molokoi 11042024")
    if racine.is_dir():
        import soundfile as sf
        from scipy.signal import spectrogram
        for i, (_, chemin, _) in enumerate(SPECTRO, 1):
            x, sr = sf.read(racine / chemin, dtype="float32", always_2d=True)
            f, tt, S = spectrogram(x[:, 0], sr, nperseg=1024, noverlap=512)
            fig = plt.figure(figsize=(2.95, 1.24))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.pcolormesh(tt, f / 1000, 10 * np.log10(S + 1e-14), shading="auto", vmin=-130,
                          vmax=-60, cmap="magma")
            ax.set_ylim(0, 12)
            ax.axis("off")
            fig.savefig(FIG / f"mol_spectro_{i}.png", dpi=200)
            plt.close(fig)
        return "disque"
    source = Image.open(SOURCES / "molokoi_03avril.png").convert("RGB")
    for i, (_, _, (x, y)) in enumerate(SPECTRO, 1):
        source.crop((x, y, x + 558, y + 234)).save(FIG / f"mol_spectro_{i}.png")
    return "png d'origine"


# --------------------------------------------------------------------------- diapo 11 : une ligne
def fig_ligne_drapeau() -> None:
    """Diapo 11 : une vraie ligne de la table recordings qui porte un drapeau (Mataroni SMA13417,
    06/01/2026, 104 s : `duration_off`), dessinée comme `fig_base`, `qc_flags` en entier."""
    cfg = load_config(default_user_config())
    con = sqlite3.connect(f"file:{config_path(cfg, 'db').as_posix()}?mode=ro", uri=True)
    cols = ["recording_id", "site", "mic_id", "start_utc", "duration_s", "qc_flags"]
    ligne = con.execute(f"SELECT {', '.join(cols)} FROM recordings WHERE recording_id = ?",
                        ("9e453c33941c80ce",)).fetchone()
    n = con.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
    con.close()
    assert ligne[4] == 104.0 and json.loads(ligne[5]).get("duration_off") is True, ligne
    morceaux, cur = [], ""
    for mot in json.dumps(json.loads(ligne[5])).split(", "):
        if cur and len(cur) + len(mot) > 58:
            morceaux.append(cur + ",")
            cur = mot
        else:
            cur = f"{cur}, {mot}" if cur else mot
    morceaux.append(cur)
    larg = {"recording_id": 1.65, "site": 1.0, "mic_id": 1.1, "start_utc": 2.1, "duration_s": 1.1}
    L, H = 12.13, 1.55
    fig = plt.figure(figsize=(L, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, L)
    ax.set_ylim(H, 0)
    ax.axis("off")
    police = {"family": "Consolas", "size": 11}
    ax.add_patch(plt.Rectangle((0, 0), L, 0.34, color="#E9ECEF"))
    ax.text(0.12, 0.17, f"blanci.sqlite  ›  Parcourir les données  ›  Table : recordings   "
            f"({milliers(n)} lignes)", va="center", fontsize=11, color=INK, fontweight="bold")
    ax.add_patch(plt.Rectangle((0, 0.34), L, 0.34, color="#F6F7F8", ec="#C9CED3", lw=0.6))
    ax.add_patch(plt.Rectangle((0, 0.68), L, H - 0.68, color="#FFF3CD", zorder=0))
    x = 0.12
    for c, v in zip(cols, ligne, strict=True):
        ax.text(x, 0.51, c, va="center", fontdict={**police, "weight": "bold"}, color=INK)
        ax.plot([x - 0.08, x - 0.08], [0.34, H], color="#D5D9DD", lw=0.6)
        if c == "qc_flags":
            for k, m in enumerate(morceaux):
                ax.text(x, 0.88 + 0.25 * k, m, va="center", fontdict=police, color=INK)
        else:
            ax.text(x, 0.88, str(v), va="center", fontdict=police, color=INK)
        x += larg.get(c, 0)
    ax.add_patch(plt.Rectangle((0, 0), L, H, fill=False, ec="#C9CED3", lw=1))
    fig.savefig(FIG / "qc_ligne_drapeau.png", dpi=200)
    plt.close(fig)


def fig_recap_criteres() -> None:
    """Paquet 5, bilan du contrôle qualité, 12,13 × 3,5 po. Chiffres de la base au 18/11/2026
    (jeux 2023 et 2026, 96 310 enregistrements) : 680 écartés = 506 son étouffé (399 par le calcul,
    107 à l'écoute) + 172 durée anormale (dont 86 aussi hors relevé) + 2 hors relevé seul.
    Les doublons (3 673 copies) et les 2 fichiers illisibles ne sont jamais entrés dans la base.
    À droite, ce qui est mesuré sans rien écarter de plus : micro sourd 1 871 (Kaw D, encodés sur
    la piste sourde), durées coupées 34 (déjà dans les durées anormales), bascules 11 (CDR)."""
    ecartent = [("Son étouffé", 506, ROUGE, "dont 399 par le calcul, 107 à l'écoute"),
                ("Durée anormale", 172, ROUGE, "dont 86 aussi hors relevé"),
                ("Hors relevé, seul", 2, ROUGE, None),
                ("Silence", 0, ROUGE, None),
                ("Doublons", None, ROUGE, "3 673 copies, jamais inventoriées"),
                ("Fichiers illisibles", None, ROUGE, "2, jamais inventoriés")]
    mesures = [("Micro sourd (Kaw D, 2024)", 1871, G_ORANGE, "enr. encodés sur la piste sourde"),
               ("Durées coupées\n(programme interrompu)", 34, G_ORANGE, "dans les 172"),
               ("Bascule en plein enregistrement", 11, G_ORANGE, "à CDR"),
               ("Clics et parasites", None, G_ORANGE, "indices non concluants (insectes)"),
               ("Trous numériques\n(échantillons figés)", None, G_ORANGE,
                "aucun sur 3 607 mesurés")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.13, 3.5),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    def panneau(ax, lignes, xmax, titre, sous_titre):
        n = len(lignes)
        for y, (_nom, v, c, note) in enumerate(lignes):
            pos = n - 1 - y
            if v is None:
                ax.text(xmax * 0.012, pos, note, va="center", fontsize=11, color=MUTED,
                        style="italic")
                continue
            if v == 0:
                ax.text(xmax * 0.012, pos, "aucun aujourd'hui", va="center", fontsize=11,
                        color=MUTED, style="italic")
                continue
            coupe = v > xmax
            dedans = coupe or v > xmax * 0.6
            ax.barh(pos, min(v, xmax * 0.97), height=0.62, color=c)
            if dedans:
                if coupe:
                    ax.plot([xmax * 0.9, xmax * 0.93], [pos - 0.34, pos + 0.34], color="white",
                            lw=3)
                texte = f"{v:,}".replace(",", " ") + (f"  {note}" if note else "")
                ax.text(xmax * 0.03, pos, texte, va="center", fontsize=11.5, color="white",
                        fontweight="bold")
            else:
                ax.text(v + xmax * 0.012, pos, f"{v}" + (f"  {note}" if note else ""),
                        va="center", fontsize=11.5, color=INK, fontweight="bold")
        ax.set_yticks(range(n), [ligne[0] for ligne in lignes][::-1], fontsize=12, color=INK)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(0, xmax)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_title(titre, loc="left", fontsize=13, color=INK, fontweight="bold", pad=22)
        ax.text(0, 1.03, sous_titre, fontsize=11, color=MUTED, transform=ax.transAxes, va="bottom")
        ax.grid(axis="y", visible=False)

    panneau(a1, ecartent, 600, "Les 680 écartés, par critère",
            "506 + 172 + 2 = 680 enregistrements")
    panneau(a2, mesures, 60, "Mesuré, à décider", "enregistrements ; barre coupée")
    fig.subplots_adjust(left=0.2, right=0.985, top=0.8, bottom=0.1, wspace=0.62)
    fig.savefig(FIG / "qc_recap_criteres.png", dpi=200)
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
    print("gains :", fig_gains())
    print("complétude :", fig_completude(rec))
    print("horloge de Molokoï :", fig_molokoi_niveaux())
    print("spectrogrammes :", fig_molokoi_spectros())
    fig_ligne_drapeau()
    fig_recap_criteres()


if __name__ == "__main__":
    main()
