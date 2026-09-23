"""Benchmark des encodeurs (§2) : mêmes labels, mêmes plis, sondes légères sur embeddings gelés.

Sondes : kNN cosinus, prototype simple, prototype différentiel, régression logistique L2 (§3).
L'écart entre les deux prototypes dit combien l'embedding capte le fond sonore partagé.
Chaque sonde est évaluée en scores hors-pli, groupés par point (site/micro) : un micro ne se
retrouve jamais des deux côtés d'un pli (§6). Chaque encodeur est jugé au niveau fenêtre et au
niveau enregistrement, avec AP (IC bootstrap par enregistrement) et rappel aux précisions plancher
(IC de Wilson en enregistrements).

Les encodeurs se départagent sur les mêmes enregistrements par bootstrap apparié : « A meilleur
que B » seulement si l'intervalle exclut zéro (§6). Un écart d'AP < 0,1 est une égalité, que
départagent alors licence, vitesse et prise en main — colonnes renseignées à la main.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from blanci.dataset import embedded_training_set
from blanci.db import encoder_params
from blanci.evaluate import average_precision, evaluate, paired_bootstrap, to_recordings
from blanci.head import oof_scores
from blanci.index import l2_normalize
from blanci.store import EmbeddingStore

PROBES = ("knn", "simple_prototype", "prototype", "logistic")
LEVELS = ("window", "recording")


def knn_top1(X: np.ndarray, y: np.ndarray, recordings: np.ndarray, chunk_rows: int = 4096) -> float:
    """Part des positifs dont le plus proche voisin cosinus est positif (§2).

    Les fenêtres du même enregistrement sont exclues du voisinage : voisines dans le temps,
    elles partagent le fond sonore et donneraient un score de fuite.
    """
    Xn = l2_normalize(X)
    y = np.asarray(y).astype(int)
    recordings = np.asarray(recordings)
    positives = np.flatnonzero(y == 1)
    if not len(positives):
        return float("nan")
    hits = 0
    for start in range(0, len(positives), chunk_rows):
        rows = positives[start : start + chunk_rows]
        sims = Xn[rows] @ Xn.T
        sims[recordings[rows][:, None] == recordings[None, :]] = -np.inf
        hits += int((y[sims.argmax(axis=1)] == 1).sum())
    return hits / len(positives)


def probe_table(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    recordings: np.ndarray,
    n_splits: int = 5,
    C_grid: list[float] | None = None,
    precisions: tuple[float, ...] = (0.1, 0.5),
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Une ligne par (sonde, niveau) ; renvoie aussi les scores hors-pli de chaque sonde."""
    rows, scores = [], {}
    for probe in PROBES:
        oof = oof_scores(X, y, groups, n_splits=n_splits, method=probe, C_grid=C_grid, seed=seed)
        scores[probe] = oof.values
        for level in LEVELS:
            metrics = evaluate(
                oof.values,
                y,
                recordings,
                level=level,
                precisions=precisions,
                n_boot=n_boot,
                seed=seed,
            )
            rows.append({"probe": probe, **metrics})
    return pd.DataFrame(rows), scores


def benchmark_encoder(
    con: sqlite3.Connection,
    encoder_id: str,
    store_root: Path,
    cfg: dict,
    filters: dict | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Évalue un encodeur. Renvoie (tableau, scores logistiques hors-pli, labels, enregistrements).

    Les trois derniers servent aux comparaisons appariées entre encodeurs.
    """
    params = encoder_params(con, encoder_id)
    bench, head = cfg["benchmark"], cfg["head"]
    data, X = embedded_training_set(
        con,
        EmbeddingStore(store_root, encoder_id),
        params["window_s"],
        per_positive=bench["negatives_per_positive"],
        slot_tolerance_min=bench["slot_tolerance_min"],
        utc_offset_h=cfg["recorder"]["filename_utc_offset_h"],
        seed=head["seed"],
        filters=filters,
    )
    if data["y"].nunique() < 2:
        raise ValueError(f"{encoder_id} : une seule classe dans le jeu étiqueté")

    y = data["y"].to_numpy()
    groups = data["point"].to_numpy()
    recordings = data["recording_id"].to_numpy()

    table, scores = probe_table(
        X,
        y,
        groups,
        recordings,
        n_splits=head["n_splits"],
        C_grid=head["C_grid"],
        precisions=tuple(bench["precisions"]),
        n_boot=bench["n_boot"],
        seed=head["seed"],
    )
    last_run = params.get("last_run", {})
    table.insert(0, "encoder_id", encoder_id)
    table["dim"] = params["dim"]
    table["has_tokens"] = params["has_tokens"]
    table["window_s"] = params["window_s"]
    table["knn_top1"] = knn_top1(X, y, recordings)
    table["windows_per_s"] = last_run.get("windows_per_s", float("nan"))
    table["realtime_factor"] = last_run.get("realtime_factor", float("nan"))
    table["n_mics"] = len(np.unique(groups))
    return table, scores["logistic"], y, recordings


def run_benchmark(
    con: sqlite3.Connection,
    encoder_ids: list[str],
    store_root: Path,
    cfg: dict,
    filters: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark complet : (tableau par encodeur et sonde, comparaisons appariées)."""
    tables, per_encoder = [], {}
    for encoder_id in encoder_ids:
        table, scores, y, recordings = benchmark_encoder(con, encoder_id, store_root, cfg, filters)
        tables.append(table)
        per_encoder[encoder_id] = (scores, y, recordings)
    results = pd.concat(tables, ignore_index=True)
    results = results.sort_values(["level", "ap"], ascending=[True, False], kind="stable")
    return results.reset_index(drop=True), compare_encoders(per_encoder, cfg)


def compare_encoders(
    per_encoder: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], cfg: dict
) -> pd.DataFrame:
    """Bootstrap apparié entre encodeurs, au niveau enregistrement (sonde logistique).

    Le niveau enregistrement est le seul comparable : deux encodeurs de fenêtres différentes
    n'ont pas la même grille, mais ils voient les mêmes enregistrements.
    """
    folded = {}
    for encoder_id, (scores, y, recordings) in per_encoder.items():
        rec = to_recordings(scores, y, recordings).set_index("recording_id")
        folded[encoder_id] = rec
    names = list(folded)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = folded[a].index.intersection(folded[b].index)
            if not len(shared):
                continue
            left, right = folded[a].loc[shared], folded[b].loc[shared]
            result = paired_bootstrap(
                left["y"].to_numpy(),
                left["score"].to_numpy(),
                right["score"].to_numpy(),
                shared.to_numpy(),
                n_boot=cfg["benchmark"]["n_boot"],
                seed=cfg["head"]["seed"],
            )
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "n_recordings": len(shared),
                    "ap_a": average_precision(left["y"].to_numpy(), left["score"].to_numpy()),
                    "ap_b": average_precision(right["y"].to_numpy(), right["score"].to_numpy()),
                    **result,
                }
            )
    return pd.DataFrame(rows)


# --- Rapports ---------------------------------------------------------------------------------

REPORT_COLUMNS = [
    "encoder_id",
    "probe",
    "level",
    "n_pos",
    "n_neg",
    "n_mics",
    "ap",
    "ap_lo",
    "ap_hi",
    "recall@p0.1",
    "recall@p0.5",
    "knn_top1",
    "dim",
    "has_tokens",
    "window_s",
    "windows_per_s",
    "realtime_factor",
]


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Tableau markdown sans dépendance supplémentaire (pandas.to_markdown exige tabulate)."""
    if df.empty:
        return "_(vide)_\n"

    def cell(value) -> str:
        if isinstance(value, float | np.floating):
            return "—" if np.isnan(value) else floatfmt.format(value)
        return str(value)

    header = list(df.columns)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in header) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    results: pd.DataFrame, comparisons: pd.DataFrame, reports_dir: Path, stem: str = "benchmark"
) -> dict[str, Path]:
    """Écrit le tableau en CSV (complet) et en markdown (colonnes du §2)."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": reports_dir / f"{stem}.csv",
        "comparisons_csv": reports_dir / f"{stem}_comparaisons.csv",
        "markdown": reports_dir / f"{stem}.md",
    }
    results.to_csv(paths["csv"], index=False)
    comparisons.to_csv(paths["comparisons_csv"], index=False)

    shown = [c for c in REPORT_COLUMNS if c in results.columns]
    text = ["# Benchmark des encodeurs (§2)", ""]
    for level in LEVELS:
        part = results[results["level"] == level]
        if part.empty:
            continue
        text += [f"## Niveau {level}", "", to_markdown(part[shown]), ""]
    text += [
        "## Comparaisons appariées (sonde logistique, niveau enregistrement)",
        "",
        "« A meilleur que B » seulement si l'intervalle exclut zéro (§6).",
        "",
        to_markdown(comparisons),
        "",
        "Colonnes à renseigner à la main : licence, prise en main, projection i5-1145G7.",
    ]
    paths["markdown"].write_text("\n".join(text), encoding="utf-8")
    return paths
