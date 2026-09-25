"""Benchmark complet des modèles (DECISIONS n° 98) : toutes les sources, mêmes enregistrements.

Sources : tout ce que le stock de scores hors-pli contient (`blanci sources`) — encodeurs ×
têtes (`benchmark`, `heads`), baselines (`baselines`), fusions (`fusion-bench`), ensembles
(`ensemble`), détecteurs (`detector-bench`) — plus des sources externes sans apprentissage,
rangées à la demande sur les fenêtres des baselines (`external_source`) : Blancinet
(`import-detections`), logits des congénères de Perch.

Tout se compare au niveau enregistrement (score maximal de ses fenêtres) : une fenêtre de 3 s
et une de 5 s ne se comparent pas, deux enregistrements si. Par défaut, sur les seuls
enregistrements évalués par **toutes** les sources retenues (mêmes positifs, mêmes négatifs) ;
chacune sur les siens avec `common=False`.

Colonnes : AP [IC bootstrap], rappel aux précisions plancher [Wilson], rappel par site au seuil
de précision plancher, et pour les encodeurs le coût (dimension, fenêtre, débit mesuré à
l'encodage). Comparaison appariée de chaque source à la référence (la meilleure, ou celle
demandée) : « meilleur » seulement si l'intervalle exclut zéro (§6). Une source dont l'empreinte
des labels n'est plus celle d'aujourd'hui est signalée (`up_to_date`) : à relancer.

Réserves : Blancinet a peut-être été entraîné sur ces mêmes enregistrements de Mataroni (à
confirmer avec Biophonia) ; son score est alors optimiste. Licence et prise en main : colonnes à
remplir à la main (§2).
"""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from blanci.evaluate import (
    average_precision,
    bootstrap_ci,
    paired_bootstrap,
    recall_at_precision,
    wilson_interval,
)
from blanci.fusion import project_scores
from blanci.oof import (
    labels_fingerprint,
    load_oof,
    oof_frame,
    recording_scores,
    save_oof,
)


def external_source(con: sqlite3.Connection, cfg: dict, model: str, name: str | None = None) -> str:
    """Range un détecteur externe ou sans apprentissage dans le stock hors-pli, sur les fenêtres
    des baselines : Blancinet (`model` = « blancinet », scores de ses détections) ou les logits
    des congénères (`model` = « <encodeur perch>:logit »). Une fenêtre sans détection qui la
    recouvre reçoit le score le plus bas du modèle (il ne l'a pas remontée)."""
    from blanci.baselines import evaluation_windows
    from blanci.dataset import folds_for

    windows = evaluation_windows(con, cfg).reset_index(drop=True)
    pattern = f"{model}:%" if model.endswith(":logit") else model
    operator = "LIKE" if model.endswith(":logit") else "="
    scores = pd.read_sql_query(
        "SELECT w.recording_id, w.offset_s, w.dur_s, MAX(s.score) AS score FROM scores s "
        f"JOIN windows w USING (window_id) WHERE s.model_id {operator} ? GROUP BY s.window_id",
        con,
        params=(pattern,),
    )
    if scores.empty:
        raise ValueError(f"aucun score pour {model!r} dans la base")
    values = project_scores(windows, scores)
    values = np.where(np.isnan(values), float(scores["score"].min()) - 1.0, values)
    source = name or f"external/{model.replace(':logit', '/congeners')}"
    save_oof(
        cfg,
        oof_frame(
            source, "external", windows, values, folds_for(con, cfg), labels_fingerprint(con, cfg)
        ),
    )
    return source


def _encoder_costs(con: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Dimension, fenêtre et débit mesuré de chaque stock d'encodeur (table models)."""
    import json

    out = {}
    for model_id, params in con.execute(
        "SELECT model_id, params_json FROM models WHERE kind = 'encoder'"
    ):
        p = json.loads(params or "{}")
        last = p.get("last_run", {}) or {}
        out[model_id] = {
            "dim": p.get("dim"),
            "window_s": p.get("window_s"),
            "windows_per_s": last.get("windows_per_s"),
            "realtime_factor": last.get("realtime_factor"),
        }
    return out


def run_full_benchmark(
    con: sqlite3.Connection,
    cfg: dict,
    sources: list[str] | None = None,
    reference: str | None = None,
    common: bool = True,
) -> dict[str, Any]:
    """Tableau de toutes les sources, comparaisons à la référence, rappel par site."""
    table = load_oof(cfg, sources)
    if table.empty:
        raise ValueError("aucun score hors-pli enregistré (benchmark, heads, baselines…)")
    table = table.dropna(subset=["score"])
    per_recording = recording_scores(table)
    site_of = table.groupby("recording_id")["site"].first()
    names = sorted(per_recording["source"].unique())
    if common:
        counts = per_recording.groupby("recording_id")["source"].nunique()
        shared = set(counts[counts == len(names)].index)
        if not shared:
            raise ValueError("aucun enregistrement évalué par toutes les sources (common=False ?)")
        per_recording = per_recording[per_recording["recording_id"].isin(shared)]
    current = labels_fingerprint(con, cfg)
    fingerprints = table.groupby("source")["fingerprint"].first()
    kinds = table.groupby("source")["kind"].first()
    costs = _encoder_costs(con)
    bench, seed = cfg["benchmark"], cfg["head"]["seed"]
    floor = min(bench["precisions"])

    rows, by_site = [], []
    for name in names:
        part = per_recording[per_recording["source"] == name]
        y, s, units = (
            part["y"].to_numpy(),
            part["score"].to_numpy(),
            part["recording_id"].to_numpy(),
        )
        lo, hi = bootstrap_ci(y, s, units, n_boot=bench["n_boot"], seed=seed)
        row: dict[str, Any] = {
            "source": name,
            "kind": kinds[name],
            "n_recordings": len(part),
            "n_pos": int(y.sum()),
            "ap": average_precision(y, s),
            "ap_lo": lo,
            "ap_hi": hi,
        }
        for p in bench["precisions"]:
            recall, threshold = recall_at_precision(y, s, p)
            k = int(((s >= threshold) & (y == 1)).sum()) if np.isfinite(threshold) else 0
            row[f"recall@p{p}"] = recall
            row[f"recall@p{p}_lo"], row[f"recall@p{p}_hi"] = wilson_interval(k, int(y.sum()))
            if p == floor:
                hits = part.assign(hit=(s >= threshold), site=part["recording_id"].map(site_of))
                for site, group in hits[hits["y"] == 1].groupby("site"):
                    by_site.append(
                        {
                            "source": name,
                            "site": site,
                            "n_pos": len(group),
                            "recall": float(group["hit"].mean()),
                        }
                    )
        encoder = name.split("/")[0]
        row |= costs.get(encoder, {})
        row["up_to_date"] = fingerprints[name] == current
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("ap", ascending=False, kind="stable")
    reference = reference or str(result.iloc[0]["source"])
    if reference not in names:
        raise ValueError(f"référence inconnue : {reference!r}")
    comparisons = []
    ref = per_recording[per_recording["source"] == reference].set_index("recording_id")
    for name in names:
        if name == reference:
            continue
        part = per_recording[per_recording["source"] == name].set_index("recording_id")
        shared = part.index.intersection(ref.index)
        if not len(shared):
            continue
        comparisons.append(
            {
                "source": name,
                "reference": reference,
                "n_recordings": len(shared),
                **paired_bootstrap(
                    part.loc[shared, "y"].to_numpy(),
                    part.loc[shared, "score"].to_numpy(),
                    ref.loc[shared, "score"].to_numpy(),
                    shared.to_numpy(),
                    n_boot=bench["n_boot"],
                    seed=seed,
                ),
            }
        )
    site_table = pd.DataFrame(by_site)
    if not site_table.empty:
        site_table = site_table.pivot_table(index="source", columns="site", values="recall")
    return {
        "table": result.reset_index(drop=True),
        "comparisons": pd.DataFrame(comparisons),
        "by_site": site_table,
        "reference": reference,
        "n_common_recordings": int(per_recording["recording_id"].nunique()),
    }
