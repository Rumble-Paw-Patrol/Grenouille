"""Figures de la présentation n° 5 v2 (annotation).

Lit en lecture seule la base locale (`paths.db`), les CSV de `annotations/` et l'audio des
disques branchés (BLANCI_2023-24 / BLANCI_2025-26), et écrit dans
`documentation/prez/figures/annotation/` :

- `fen_label.png` (diapo 10) : une fenêtre prend le label de l'intervalle qu'elle croise,
  A. blanci l'emporte sur le faux ami (Kaw C, 19/09/2024 5 h, extrait 88) ;
- `fen_deux_notes.png` (diapo 11) : deux notes à 3,07 s l'une de l'autre, deux intervalles ;
  la même fenêtre de 3 s est négative, et deviendrait positive avec un seul intervalle
  (Molokoï F, 09/03/2024 8 h 30, extrait 77) ;
- `fen_bord.png` (diapo 12) : une fenêtre qui n'a que 0,18 s du chant est un bord, écartée
  (Mataroni, extrait du 4363c2e6417c7a85) ;
- `strates.png` (diapo 7) : part de chaque tranche horaire dans le programme, les lots et le test ;
- `duree_chants.png` (diapo 17) : durée des intervalles d'A. blanci ;
- `precision_prevalence.png` (diapo 18) : précision d'un même détecteur selon la prévalence ;
- `site_heure.png` (diapo 19) : extraits écoutés et positifs par site et tranche horaire.

Les labels de fenêtres sont ceux que rend `interval_labels` (règle du code, seuils actuels).
Chaque figure est dessinée à la taille de sa place sur la diapo (1 pt = 1 pt affiché).

    uv run python documentation/prez/scripts/presentation_5_figures.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib.patches import Rectangle
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from blanci.core.config import config_path, default_user_config, load_config  # noqa: E402
from blanci.core.locate import locate  # noqa: E402
from blanci.inputs.dataset import current_intervals, interval_labels  # noqa: E402

FIG = ROOT / "documentation" / "prez" / "figures" / "annotation"
ANN = ROOT / "annotations"
INK, MUTED, GRID, SAGE, PALE = "#1B2A24", "#5B6B63", "#E4EEE8", "#A7BFB2", "#F3F7F4"
VERT, ORANGE, BLEU, ROUGE, VIOLET = "#2E8B57", "#D9822B", "#3E7CB1", "#C0492F", "#8A6BBE"
TEAL, GOLD = "#2A7F72", "#E9B44C"
NOMS = {"Tresor": "Trésor", "Molokoi": "Molokoï", "PatawaEst": "Patawa Est",
        "PatawaOuest": "Patawa Ouest"}
TRANCHES = [("aube", "aube\n5–7 h"), ("pic_matin", "pic du matin\n7–9 h"),
            ("journee", "journée\n9–15 h"), ("pic_soir", "pic du soir\n15–17 h"),
            ("soir", "soir\n17–20 h")]
# Couleur et nom de chaque label de fenêtre.
STYLE = {
    "blanci": (VERT, "A. blanci"), "blanci_chorus": (VERT, "A. blanci"),
    "false_friend": (ROUGE, "faux ami"), "background": ("#D5DED9", "négative"),
    "edge": (GOLD, "bord : écartée"), "blanci_uncertain": (VIOLET, "incertaine"),
}

plt.rcParams.update({
    "font.family": ["Calibri", "DejaVu Sans"], "font.size": 12, "axes.edgecolor": SAGE,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK, "xtick.labelsize": 11,
    "ytick.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True,
})


def connexion() -> tuple[sqlite3.Connection, dict]:
    cfg = load_config(default_user_config())
    con = sqlite3.connect(f"file:{config_path(cfg, 'db').as_posix()}?mode=ro", uri=True)
    return con, cfg


def extraits_intervalles(con: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extraits et intervalles de la base, au format de `interval_labels` (colonnes lues
    une à une : la requête ne dépend pas des colonnes ajoutées par les migrations)."""
    sp = pd.read_sql_query(
        "SELECT span_id, recording_id, start_s, end_s, other_label, quality, conditions "
        "FROM spans ORDER BY span_id", con)
    sp["comment"] = sp.conditions.map(lambda c: (json.loads(c) if c else {}).get("comment"))
    sp = sp.drop(columns="conditions")
    iv = pd.read_sql_query(
        "SELECT i.span_id, s.recording_id, i.start_s, i.end_s, i.label, i.quality "
        "FROM intervals i JOIN spans s USING (span_id) ORDER BY i.interval_id", con)
    return sp, iv


def audio(con, cfg, rid: str, t0: float, t1: float) -> tuple[np.ndarray, int]:
    rel = con.execute("SELECT path FROM recordings WHERE recording_id = ?", (rid,)).fetchone()[0]
    roots = [Path(f"{d}:/audio") for d in "EFD"] + [Path(f"{d}:/") for d in "EFD"]
    full = locate(rel, config_path(cfg, "raw"), roots, cfg["audio"].get("renamed"))
    if full is None:
        raise FileNotFoundError(f"{rel} : brancher le disque qui le porte")
    sr = sf.info(full).samplerate
    x, sr = sf.read(full, start=int(t0 * sr), stop=int(t1 * sr), always_2d=True)
    return x[:, 0], sr


def spectro(ax, x, sr, t0, fmin=2.0, fmax=8.0):
    f, t, s = spectrogram(x, sr, nperseg=1024, noverlap=896)
    db = 10 * np.log10(s + 1e-12)
    keep = (f >= fmin * 1000) & (f <= fmax * 1000)
    db = db[keep]
    # Contraste calé sur la bande : le fond à 40 % (noir), les notes au-dessus de 99,8 %.
    vmin, vmax = np.percentile(db, 40), np.percentile(db, 99.8)
    ax.pcolormesh(t0 + t, f[keep] / 1000, db, shading="auto", cmap="magma",
                  vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlim(t0, t0 + len(x) / sr)
    ax.set_ylim(fmin, fmax)
    ax.set_ylabel("kHz")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def intervalles(ax, iv, t0, t1, fmin=2.0, fmax=8.0, lw=2.2, ls="-"):
    for _, i in iv.iterrows():
        if i.end_s < t0 or i.start_s > t1:
            continue
        c = VERT if i.label in ("blanci", "blanci_chorus") else ROUGE
        ax.add_patch(Rectangle((i.start_s, fmin + 0.08), i.end_s - i.start_s, fmax - fmin - 0.16,
                               fill=False, ec=c, lw=lw, ls=ls, zorder=3))


def fenetres(ax, lab, t0, t1, dur=3.0, highlight=()):
    """Fenêtres de 3 s (pas de 1,5 s) sur deux rangées, colorées par leur label."""
    keep = (lab.offset_s >= t0 - 1e-6) & (lab.offset_s + dur <= t1 + 1e-6)
    lab = lab[keep].sort_values("offset_s")
    for _, w in lab.iterrows():
        row = int(round(w.offset_s / 1.5)) % 2
        c, nom = STYLE.get(w.label, ("#D5DED9", w.label))
        hi = round(w.offset_s, 2) in highlight
        ax.add_patch(Rectangle((w.offset_s + 0.04, 1 - row - 0.92), dur - 0.08, 0.84, fc=c,
                               ec=INK if hi else "white", lw=2.4 if hi else 1.0,
                               alpha=1.0 if w.label != "background" else 0.9))
        dark = w.label in ("blanci", "blanci_chorus", "false_friend", "blanci_uncertain")
        ax.text(w.offset_s + dur / 2, 1 - row - 0.5, nom, ha="center", va="center", fontsize=11,
                color="white" if dark else INK, fontweight="bold" if hi else "normal")
    ax.set_xlim(t0, t1)
    ax.set_ylim(-1.05, 1.0)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("secondes dans l'enregistrement de 2 min")


def grille(rid: str, offsets) -> pd.DataFrame:
    return pd.DataFrame({"window_id": [f"{rid}:{o:.2f}" for o in offsets], "recording_id": rid,
                         "offset_s": list(offsets), "dur_s": 3.0})


def fig_fenetre_label(con, cfg, sp, iv) -> None:
    rid, t0, t1 = "bb05fb14b1d029c5", 78.0, 91.5
    cur = current_intervals(sp, iv)
    lab = interval_labels(sp, iv, grille(rid, np.arange(0, 117.01, 1.5)))
    x, sr = audio(con, cfg, rid, t0, t1)
    fig, (a, b) = plt.subplots(2, 1, figsize=(12.1, 4.3), height_ratios=[2.1, 1], sharex=True)
    spectro(a, x, sr, t0)
    intervalles(a, cur[cur.recording_id == rid], t0, t1)
    a.text(80.65, 8.12, "A. blanci", color=VERT, ha="center", va="bottom", fontsize=12,
           fontweight="bold")
    a.text(82.85, 8.12, "faux ami", color=ROUGE, ha="center", va="bottom", fontsize=12,
           fontweight="bold")
    fenetres(b, lab[lab.recording_id == rid], t0, t1, highlight=(81.0, 82.5))
    b.text(t1 + 0.1, 0.5, "fenêtres de 3 s\n(BirdNET,\npas de 1,5 s)", fontsize=10.5,
           color=MUTED, va="center")
    fig.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.14, hspace=0.08)
    fig.savefig(FIG / "fen_label.png", dpi=200)
    plt.close(fig)


def fig_deux_notes(con, cfg, sp, iv) -> dict:
    rid, t0, t1 = "8b00a67c54d752f0", 89.0, 98.5
    cur = current_intervals(sp, iv)
    mine = cur[cur.recording_id == rid]
    w = (92.28,)  # une fenêtre de 3 s entièrement dans le silence entre les deux notes
    juste = interval_labels(sp, iv, grille(rid, w))
    # Le même extrait avec un seul intervalle couvrant les deux notes (ce qu'il ne faut pas faire).
    sid = int(mine.span_id.iloc[0])
    merged = iv[~((iv.span_id == sid) & iv.start_s.between(92.0, 95.4))]
    merged = pd.concat([merged, pd.DataFrame([{
        "span_id": sid, "recording_id": rid, "start_s": 92.08, "end_s": 101.21,
        "label": "blanci", "quality": "A"}])], ignore_index=True)
    faux = interval_labels(sp, merged, grille(rid, w))
    x, sr = audio(con, cfg, rid, t0, t1)
    fig, axes = plt.subplots(2, 1, figsize=(8.25, 4.75), sharex=True)
    for ax, ivs, lab, titre in (
        (axes[0], mine, juste, "Tracé réel : deux intervalles"),
        (axes[1], merged[merged.recording_id == rid], faux, "Si l'on traçait un seul intervalle"),
    ):
        spectro(ax, x, sr, t0)
        intervalles(ax, ivs, t0, t1, lw=2.4, ls="-" if ax is axes[0] else "--")
        lbl = lab.label.iloc[0]
        c, nom = STYLE[lbl]
        verdict = "négative : juste" if lbl == "background" else f"{nom} : à tort"
        ax.add_patch(Rectangle((w[0], 2.15), 3.0, 1.25, fc=c, ec="white", lw=1.5, zorder=4))
        ax.text(w[0] + 1.5, 2.78, f"fenêtre de 3 s\n{verdict}", ha="center", va="center",
                color="white" if lbl != "background" else INK, fontsize=11, fontweight="bold",
                zorder=5, linespacing=1.1)
        ax.set_xlim(t0, t1)
        ax.set_title(titre, loc="left", fontsize=13, color=INK, pad=4)
    axes[0].annotate("", xy=(95.33, 7.3), xytext=(92.26, 7.3),
                     arrowprops=dict(arrowstyle="<->", color="white", lw=1.6))
    axes[0].text(93.8, 7.45, "3,07 s sans chant", color="white", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
    axes[1].set_xlabel("secondes dans l'enregistrement de 2 min")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.11, hspace=0.3)
    fig.savefig(FIG / "fen_deux_notes.png", dpi=200)
    plt.close(fig)
    return {"juste": juste.label.iloc[0], "un_seul": faux.label.iloc[0]}


def fig_bord(con, cfg, sp, iv) -> dict:
    rid, t0, t1 = "4363c2e6417c7a85", 27.0, 37.5
    cur = current_intervals(sp, iv)
    mine = cur[cur.recording_id == rid]
    lab = interval_labels(sp, iv, grille(rid, np.arange(0, 117.01, 1.5)))
    x, sr = audio(con, cfg, rid, t0, t1)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.25, 4.75), height_ratios=[1.9, 1], sharex=True)
    spectro(a, x, sr, t0)
    intervalles(a, mine, t0, t1)
    a.axvspan(31.5, 31.68, color=GOLD, alpha=0.55, zorder=4)
    a.annotate("0,18 s en commun avec\nla fenêtre 31,5–34,5 s,\nmoins que les 0,2 s exigées",
               xy=(31.68, 6.6), xytext=(32.4, 6.6), color="white", fontsize=11.5,
               fontweight="bold", va="center",
               bbox=dict(boxstyle="round,pad=0.3", fc="#0E2A24", ec="none", alpha=0.85),
               arrowprops=dict(arrowstyle="->", color="white", lw=1.4))
    fenetres(b, lab[lab.recording_id == rid], t0, t1, highlight=(31.5,))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.12, hspace=0.06)
    fig.savefig(FIG / "fen_bord.png", dpi=200)
    plt.close(fig)
    i = mine[(mine.start_s < 31.7) & (mine.end_s > 31.5)].iloc[0]
    return {"intervalle": (i.start_s, i.end_s), "commun": round(i.end_s - 31.5, 2)}


def fig_strates() -> dict:
    # demi-heures du programme de 5 h à 19 h 30
    prog = {"aube": 4, "pic_matin": 4, "journee": 12, "pic_soir": 4, "soir": 6}
    parts = {"programme (5 h → 19 h 30)": pd.Series(prog) / sum(prog.values())}
    for f, nom in (("lot1", "lot 1 + lot 2"), ("test_v1", "jeu de test v1")):
        c = pd.concat([pd.read_csv(ANN / "files_ecoute" / q / "candidats.csv")
                       for q in (["lot1", "lot2"] if f == "lot1" else [f])])
        c = c.drop_duplicates("recording_id")
        parts[nom] = c.strate.str.split("/").str[-1].value_counts(normalize=True)
    df = pd.DataFrame(parts).reindex([k for k, _ in TRANCHES]).fillna(0)
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    xs = np.arange(len(df))
    cols = ["#C9D6CF", VERT, ORANGE]
    for k, (nom, col) in enumerate(zip(df.columns, cols, strict=True)):
        bars = ax.bar(xs + (k - 1) * 0.27, df[nom] * 100, 0.25, color=col, label=nom,
                      edgecolor="white", lw=1)
        if k:
            for b, v in zip(bars, df[nom], strict=True):
                ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.6, f"{v * 100:.0f}",
                        ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(xs, [n for _, n in TRANCHES], fontsize=10.5)
    ax.set_ylabel("% des enregistrements tirés")
    ax.yaxis.grid(True, color=GRID)
    ax.legend(frameon=False, fontsize=10.5, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(FIG / "strates.png", dpi=200)
    plt.close(fig)
    pics = df.loc[["pic_matin", "pic_soir"]].sum()
    return {k: round(v * 100) for k, v in pics.items()}


def courants() -> pd.DataFrame:
    ex = pd.read_csv(ANN / "extraits_annotes.csv")
    cur = ex[ex.remplace_par.isna()].copy()
    cand = pd.concat([pd.read_csv(ANN / "files_ecoute" / f / "candidats.csv").assign(file=f)
                      for f in ("lot1", "lot2", "test_v1")]).drop_duplicates("recording_id")
    cur = cur.merge(cand[["recording_id", "file", "niveau", "prob_tirage", "point"]],
                    on="recording_id", how="left")
    cur["pos"] = cur.classes.fillna("").map(
        lambda s: any(c.strip() in ("blanci", "blanci_chorus") for c in s.split(",")))
    return cur


def fig_durees() -> dict:
    iv = pd.read_csv(ANN / "intervalles_annotes.csv")
    iv = iv[(iv.valable.astype(str) == "True") & iv.label.isin(["blanci", "blanci_chorus"])]
    d = (iv.end_s - iv.start_s).clip(lower=0.03)
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    bins = np.logspace(np.log10(0.03), np.log10(130), 32)
    ax.hist(d, bins=bins, color=VERT, edgecolor="white", lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100], ["0,1", "0,3", "1", "3", "10", "30", "100"])
    ax.axvline(d.median(), color=INK, lw=1.2, ls="--")
    ax.text(d.median() * 1.12, ax.get_ylim()[1] * 0.92, "médiane\n0,33 s", fontsize=10.5,
            color=INK, va="top")
    ax.set_xlabel("durée d'un intervalle d'A. blanci (s, échelle log)")
    ax.set_ylabel("intervalles")
    ax.yaxis.grid(True, color=GRID)
    fig.tight_layout()
    fig.savefig(FIG / "duree_chants.png", dpi=200)
    plt.close(fig)
    return {"n": len(d), "mediane": round(float(d.median()), 2),
            "q75": round(float(d.quantile(0.75)), 2)}


def prevalences(cur: pd.DataFrame, n_boot: int = 4000) -> pd.DataFrame:
    """Prévalence par niveau du test, par enregistrement de 2 min : brute, et pondérée par
    1 / probabilité de tirage (Horvitz-Thompson), intervalle à 95 % par bootstrap."""
    t = cur[cur.file == "test_v1"].groupby("recording_id").agg(
        pos=("pos", "max"), p=("prob_tirage", "first"), niveau=("niveau", "first"))
    rng = np.random.default_rng(0)
    rows = []
    for niv, g in t.groupby("niveau"):
        w, y = 1 / g.p.to_numpy(), g.pos.to_numpy(float)
        est = (w * y).sum() / w.sum()
        k = rng.integers(0, len(g), size=(n_boot, len(g)))
        boot = (w[k] * y[k]).sum(1) / w[k].sum(1)
        rows.append((niv, len(g), int(y.sum()), y.mean(), est, *np.percentile(boot, [2.5, 97.5])))
    return pd.DataFrame(rows, columns=["niveau", "n", "pos", "brute", "ponderee", "bas", "haut"])


def fig_precision(prev: pd.DataFrame) -> dict:
    pi = np.logspace(-3, np.log10(0.6), 300)
    se = 0.9
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    out = {}
    for sp, col, ls in ((0.99, VERT, "-"), (0.95, ORANGE, "-")):
        ppv = se * pi / (se * pi + (1 - sp) * (1 - pi))
        ax.plot(pi * 100, ppv * 100, color=col, lw=2.2, ls=ls,
                label=f"rappel 0,90 ; {round((1 - sp) * 100)} % de fausses alarmes "
                "sur les négatifs")
    for niv, nom in (("niveau1", "niveau 1"), ("niveau3", "niveau 3")):
        r = prev[prev.niveau == niv].iloc[0]
        p = r.ponderee
        ax.axvline(p * 100, color=MUTED, lw=1, ls=":")
        for sp in (0.99, 0.95):
            v = se * p / (se * p + (1 - sp) * (1 - p))
            out[(niv, sp)] = round(v * 100)
            ax.plot(p * 100, v * 100, "o", ms=7, color=VERT if sp == 0.99 else ORANGE,
                    mec="white", mew=1.5, zorder=4)
            ax.text(p * 100 * 1.08, v * 100 - 2, f"{v * 100:.0f} %", fontsize=10.5, color=INK,
                    va="top")
        ax.text(p * 100, 101, f"{nom}\n{p * 100:.1f} %".replace(".", ","), ha="center",
                va="bottom", fontsize=10.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10, 50], ["0,1 %", "1 %", "10 %", "50 %"])
    ax.set_xlim(0.1, 60)
    ax.set_ylim(0, 100)
    ax.set_xlabel("prévalence : part des enregistrements avec A. blanci (log)")
    ax.set_ylabel("précision (% d'alertes justes)")
    ax.yaxis.grid(True, color=GRID)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "precision_prevalence.png", dpi=200)
    plt.close(fig)
    return out


def fig_site_heure(cur: pd.DataFrame) -> pd.DataFrame:
    c = cur[cur.source == "plan"].copy()
    h = (pd.to_datetime(c.start_utc) + pd.to_timedelta(c.start_s, unit="s")
         - pd.Timedelta(hours=3))
    hh = h.dt.hour + h.dt.minute / 60
    c["tranche"] = pd.cut(hh, [0, 7, 9, 15, 17, 24], right=False,
                          labels=[k for k, _ in TRANCHES])
    order = ["Kaw", "Molokoi", "Tresor", "Mataroni", "CDR", "PatawaEst", "PatawaOuest", "RNRT"]
    n = c.pivot_table(index="site", columns="tranche", values="pos", aggfunc="size",
                      observed=False).reindex(order).fillna(0)
    p = c.pivot_table(index="site", columns="tranche", values="pos", aggfunc="sum",
                      observed=False).reindex(order).fillna(0)
    frac = (p / n.replace(0, np.nan)).to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    cmap = plt.get_cmap("Greens")
    ax.imshow(np.nan_to_num(frac), cmap=cmap, vmin=0, vmax=0.6, aspect="auto")
    for i in range(len(order)):
        for j in range(len(TRANCHES)):
            v = frac[i, j]
            ax.text(j, i, f"{int(p.iat[i, j])} / {int(n.iat[i, j])}", ha="center", va="center",
                    fontsize=11, color="white" if v == v and v > 0.35 else INK)
    ax.set_xticks(range(len(TRANCHES)), [nm for _, nm in TRANCHES], fontsize=10.5)
    ax.set_yticks(range(len(order)),
                  [("2023 · " if s in ("Kaw", "Molokoi", "Tresor") else "2026 · ") + NOMS.get(s, s)
                   for s in order], fontsize=11)
    ax.axhline(2.5, color="white", lw=4)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("extraits avec A. blanci / extraits écoutés (lots et test)", loc="left",
                 fontsize=11.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "site_heure.png", dpi=200)
    plt.close(fig)
    return p.sum(1).to_frame("pos").join(n.sum(1).rename("n"))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    con, cfg = connexion()
    sp, iv = extraits_intervalles(con)
    fig_fenetre_label(con, cfg, sp, iv)
    print("deux notes :", fig_deux_notes(con, cfg, sp, iv))
    print("bord :", fig_bord(con, cfg, sp, iv))
    print("pics (%) :", fig_strates())
    print("durées :", fig_durees())
    cur = courants()
    prev = prevalences(cur)
    print(prev.round(4).to_string())
    print("précision :", fig_precision(prev))
    print(fig_site_heure(cur).to_string())


if __name__ == "__main__":
    main()
