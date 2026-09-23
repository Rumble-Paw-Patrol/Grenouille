"""Courbes d'activité (§1, §6 niveau 3, jalon M4) : reproduire les patrons publiés.

Validation sans étiquettes : notre détecteur, appliqué aux enregistrements de 2023, doit
retrouver les patrons de Courtois et al. 2025 — pics 7–9 h et 15–17 h, activité forte de
janvier à avril, quasi nulle de juillet à octobre, Kaw_B faible. Un désaccord signale un
problème de généralisation (autre année, autres capteurs) ou un artefact (§11, risque 6).

Indices, par site ou par micro, à partir des décisions par enregistrement (`blanci score`) :
- indice horaire : part des enregistrements détectés à chaque heure locale ; l'effort
  (nombre d'enregistrements de l'heure) est le dénominateur, un créneau peu enregistré ne
  pèse donc pas moins ;
- probabilité journalière, par mois : part des jours avec au moins un enregistrement
  détecté, parmi les jours enregistrés ;
- intervalles de Wilson à 95 % sur chaque proportion.

Un enregistrement « suspect » (détection isolée, §1) ne compte pas comme détecté, sauf
demande contraire : l'indice se lit alors comme une borne haute.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blanci.evaluate import wilson_interval

DETECTED = ("positive", "verified_positive")


def detections_table(
    con: sqlite3.Connection,
    encoder_id: str,
    head_version: str,
    threshold_id: str,
    utc_offset_h: float,
    dataset: str | None = None,
    include_suspect: bool = False,
) -> pd.DataFrame:
    """Une ligne par enregistrement décidé : site, micro, heure et jour locaux, détecté."""
    df = pd.read_sql_query(
        """SELECT d.recording_id, d.status, r.dataset, r.site, r.mic_id, r.start_utc
           FROM decisions d JOIN recordings r USING (recording_id)
           WHERE d.encoder_id = ? AND d.head_version = ? AND d.threshold_id = ?
             AND (? IS NULL OR r.dataset = ?)""",
        con,
        params=(encoder_id, head_version, threshold_id, dataset, dataset),
    )
    if df.empty:
        raise ValueError(
            f"aucune décision pour {encoder_id} {head_version} ({threshold_id})"
            + (f" sur le jeu {dataset}" if dataset else "")
            + " : lancer `blanci score`"
        )
    df = df.dropna(subset=["start_utc"])  # sans date, ni heure ni jour
    detected = DETECTED + (("suspect",) if include_suspect else ())
    local = pd.to_datetime(df["start_utc"], utc=True) + pd.Timedelta(hours=utc_offset_h)
    return df.assign(
        detected=df["status"].isin(detected),
        hour=local.dt.hour,
        day=local.dt.strftime("%Y-%m-%d"),
        month=local.dt.strftime("%Y-%m"),
    )


def _with_wilson(frame: pd.DataFrame, k: str, n: str, value: str) -> pd.DataFrame:
    bounds = [wilson_interval(int(a), int(b)) for a, b in zip(frame[k], frame[n], strict=True)]
    return frame.assign(
        **{value: frame[k] / frame[n]},
        lo=[b[0] for b in bounds],
        hi=[b[1] for b in bounds],
    )


def diel_index(detections: pd.DataFrame, by: str = "site") -> pd.DataFrame:
    """Indice horaire : by, hour, n (enregistrements), detected, index, lo, hi."""
    table = detections.groupby([by, "hour"])["detected"].agg(n="size", detected="sum").reset_index()
    return _with_wilson(table, "detected", "n", "index")


def daily_probability(detections: pd.DataFrame, by: str = "site") -> pd.DataFrame:
    """Probabilité journalière par mois : by, month, days, days_detected, probability, lo, hi.

    Un jour compte pour un groupe (site ou micro) s'il y a au moins un enregistrement ; il est
    détecté si au moins un de ses enregistrements l'est.
    """
    days = detections.groupby([by, "month", "day"])["detected"].any().reset_index()
    table = (
        days.groupby([by, "month"])["detected"].agg(days="size", days_detected="sum").reset_index()
    )
    return _with_wilson(table, "days_detected", "days", "probability")


def _hours(ranges: list[list[int]]) -> set[int]:
    return {h for a, b in ranges for h in range(a, b)}


def _pooled(frame: pd.DataFrame, mask: pd.Series, k: str, n: str) -> tuple[float, float, float]:
    """Proportion regroupée (somme des k / somme des n) et son intervalle de Wilson."""
    hits, total = int(frame.loc[mask, k].sum()), int(frame.loc[mask, n].sum())
    lo, hi = wilson_interval(hits, total)
    return (hits / total if total else np.nan), lo, hi


def reference_checks(
    diel: pd.DataFrame, seasonal: pd.DataFrame, reference: dict[str, Any], by: str = "site"
) -> pd.DataFrame:
    """Confronte chaque courbe aux patrons publiés (config `activity.reference`).

    - pics : part des enregistrements détectés aux heures de pic contre aux autres heures ;
    - saison : part des jours avec détection les mois forts contre les mois creux.
    Un patron est retrouvé quand les deux intervalles de Wilson sont disjoints, dans le bon
    sens : un simple rapport > 1 se lèverait au hasard sur une activité plate.
    """
    peaks = _hours(reference["peak_hours"])
    high, low = set(reference["high_months"]), set(reference["low_months"])
    rows = []
    for group in sorted(set(diel[by]) | set(seasonal[by])):
        d = diel[diel[by] == group]
        s = seasonal[seasonal[by] == group]
        in_peak = d["hour"].isin(peaks)
        peak, peak_lo, _ = _pooled(d, in_peak, "detected", "n")
        off, _, off_hi = _pooled(d, ~in_peak, "detected", "n")
        months = s["month"].str[5:].astype(int)
        strong, strong_lo, _ = _pooled(s, months.isin(high), "days_detected", "days")
        weak, _, weak_hi = _pooled(s, months.isin(low), "days_detected", "days")
        rows.append(
            {
                by: group,
                "recordings": int(d["n"].sum()),
                "index_peak_hours": peak,
                "index_other_hours": off,
                "peak_ratio": peak / off if off > 0 else np.inf if peak > 0 else np.nan,
                "peaks_found": bool(peak_lo > off_hi),
                "prob_high_months": strong,
                "prob_low_months": weak,
                "season_ratio": strong / weak if weak > 0 else np.inf if strong > 0 else np.nan,
                "season_found": bool(strong_lo > weak_hi),
            }
        )
    return pd.DataFrame(rows)


def curve_correlation(
    ours: pd.DataFrame, reference: pd.DataFrame, key: str, value: str, by: str = "site"
) -> pd.DataFrame:
    """Corrélation de Pearson entre notre courbe et une courbe de référence numérisée.

    `reference` : colonnes `key` (hour ou month) et `value`, plus `by` (site) si la courbe
    est donnée par site ; sans colonne `by`, elle vaut pour tous. Mois : numéro 1–12 ou
    « AAAA-MM ».
    """
    ref = reference.copy()
    if key == "month":
        ref["_k"] = ref["month"].astype(str).str[-2:].astype(int)
        ours = ours.assign(_k=ours["month"].str[5:].astype(int))
    else:
        ref["_k"] = ref[key].astype(int)
        ours = ours.assign(_k=ours[key].astype(int))
    rows = []
    for group, part in ours.groupby(by):
        this = ref[ref[by] == group] if by in ref else ref
        mean = part.groupby("_k")[value].mean()  # un mois peut revenir deux années
        merged = pd.concat([mean.rename("ours"), this.groupby("_k")["value"].mean()], axis=1)
        merged = merged.dropna()
        r = merged["ours"].corr(merged["value"]) if len(merged) >= 3 else np.nan
        rows.append({by: group, "curve": key, "points": len(merged), "pearson_r": r})
    return pd.DataFrame(rows)


def plot_activity(diel: pd.DataFrame, seasonal: pd.DataFrame, path: Path, by: str = "site"):
    """Deux panneaux : indice horaire, probabilité journalière par mois (bandes de Wilson)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))
    for group, part in diel.groupby(by):
        line = left.plot(part["hour"], part["index"], marker="o", label=str(group))[0]
        left.fill_between(part["hour"], part["lo"], part["hi"], alpha=0.15, color=line.get_color())
    left.set(xlabel="heure locale", ylabel="part des enregistrements détectés", ylim=(0, 1))
    left.set_title("Indice horaire")
    months = sorted(seasonal["month"].unique())
    position = {m: i for i, m in enumerate(months)}
    for group, part in seasonal.groupby(by):
        x = part["month"].map(position)
        line = right.plot(x, part["probability"], marker="o", label=str(group))[0]
        right.fill_between(x, part["lo"], part["hi"], alpha=0.15, color=line.get_color())
    right.set_xticks(range(len(months)), months, rotation=60, ha="right")
    right.set(ylabel="part des jours avec détection", ylim=(0, 1))
    right.set_title("Probabilité journalière, par mois")
    right.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_activity_report(
    directory: Path,
    stem: str,
    diel: pd.DataFrame,
    seasonal: pd.DataFrame,
    checks: pd.DataFrame,
    correlations: pd.DataFrame | None,
    min_correlation: float,
    description: str,
) -> list[Path]:
    """CSV des courbes et des contrôles, figure, résumé Markdown."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in (("horaire", diel), ("mensuel", seasonal), ("controles", checks)):
        path = directory / f"{stem}_{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    try:
        paths.append(plot_activity(diel, seasonal, directory / f"{stem}.png"))
    except ImportError:  # matplotlib : groupe `app`
        pass
    lines = [f"# Courbes d'activité — {description}", ""]
    if not checks.empty:
        lines += ["| groupe | enregistrements | pics / autres heures | mois forts / creux | "
                  "mois creux | pics | saison |", "|---|---|---|---|---|---|---|"]  # fmt: skip
        key = checks.columns[0]
        for r in checks.itertuples(index=False):
            row = r._asdict()
            lines.append(
                f"| {row[key]} | {row['recordings']} | {row['peak_ratio']:.2f} | "
                f"{row['season_ratio']:.2f} | {row['prob_low_months']:.2f} | "
                f"{'oui' if row['peaks_found'] else 'NON'} | "
                f"{'oui' if row['season_found'] else 'NON'} |"
            )
    if correlations is not None and not correlations.empty:
        path = directory / f"{stem}_correlations.csv"
        correlations.to_csv(path, index=False)
        paths.append(path)
        lines += ["", f"Corrélation avec la référence (alerte sous {min_correlation}) :", ""]
        lines += [
            f"- {r.iloc[0]} ({r['curve']}) : r = {r['pearson_r']:.2f} sur {r['points']} points"
            + (" **ALERTE**" if r["pearson_r"] < min_correlation else "")
            for _, r in correlations.iterrows()
        ]
    report = directory / f"{stem}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [report, *paths]
