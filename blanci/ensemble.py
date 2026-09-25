"""Ensemble de modèles (DECISIONS n° 96) : combiner deux modèles ou plus.

Trois façons, du plus simple au plus riche :

1. **Combinaison tardive, par enregistrement** (`run_ensemble`) : n'importe quelles sources du
   stock de scores hors-pli (`blanci/oof.py`) — encodeur × tête, baseline, fusion, détecteur —
   ramenées au score maximal par enregistrement, sur les enregistrements évalués par toutes.
   Méthodes de `fusion.FUSION_METHODS` (moyenne, rangs, OU, ET, poids appris ou cherchés),
   apprises pli par pli sur les plis communs. Les grilles différentes (3 s, 5 s) ne gênent pas :
   tout le monde voit les mêmes enregistrements.
2. **Combinaison par fenêtre** : c'est la fusion à N entrées (`blanci/stacking.py`,
   `fusion.sources: [head:<encodeur>]`) — la tête d'un autre encodeur est ramenée sur la grille
   de l'encodeur principal, et la fusion décide fenêtre par fenêtre (production possible).
3. **Concaténation des embeddings** (`concat_benchmark`) : un vecteur par fenêtre, bout à bout
   (chaque bloc normalisé), pour des encodeurs de même grille (même fenêtre, même pas). Coûte
   les deux encodages ; la tête voit tout à la fois.

Un ensemble n'est jugé que contre sa meilleure source seule, par bootstrap apparié (§6) : un
ensemble qui n'y gagne pas ne vaut pas le double calcul.
"""

from __future__ import annotations

import sqlite3
import warnings
from typing import Any

import numpy as np
import pandas as pd

from blanci.config import config_path
from blanci.db import encoder_params
from blanci.evaluate import evaluate, paired_bootstrap
from blanci.fusion import FUSION_METHODS, fusion_model_oof
from blanci.index import l2_normalize
from blanci.oof import labels_fingerprint, load_oof, oof_frame, recording_scores, save_oof
from blanci.store import EmbeddingStore


def recording_table(cfg: dict, sources: list[str], allow_mixed: bool = False) -> pd.DataFrame:
    """Une ligne par enregistrement évalué par **toutes** les sources : recording_id, point,
    fold, y, puis une colonne de score par source (maximum de ses fenêtres)."""
    if len(sources) < 2:
        raise ValueError("un ensemble demande au moins deux sources")
    table = load_oof(cfg, sources)
    fingerprints = table.groupby("source")["fingerprint"].first()
    if fingerprints.nunique() > 1 and not allow_mixed:
        raise ValueError(
            "sources calculées sur des labels différents (empreintes "
            f"{dict(fingerprints)}) : relancer leurs benchmarks, ou allow_mixed"
        )
    folds = table.groupby("recording_id")["fold"].max()
    scores = recording_scores(table.dropna(subset=["score"]))
    wide = scores.pivot_table(index="recording_id", columns="source", values="score")
    wide = wide[sources].dropna()
    labels = scores.groupby("recording_id").agg(y=("y", "max"), point=("point", "first"))
    out = labels.join(wide, how="inner").join(folds.rename("fold"))
    return out.reset_index()


def run_ensemble(
    con: sqlite3.Connection,
    cfg: dict,
    sources: list[str],
    methods: list[str] | None = None,
    allow_mixed: bool = False,
) -> dict[str, Any]:
    """Chaque méthode de combinaison, hors-pli, contre la meilleure source seule."""
    methods = methods or ["mean", "rank_mean", "max", "logistic", "weight_grid"]
    unknown = sorted(set(methods) - set(FUSION_METHODS))
    if unknown:
        raise ValueError(f"méthodes inconnues : {unknown} (connues : {FUSION_METHODS})")
    table = recording_table(cfg, sources, allow_mixed)
    if table["y"].nunique() < 2:
        raise ValueError("une seule classe sur les enregistrements communs aux sources")
    y = table["y"].to_numpy(dtype=int)
    units = table["recording_id"].to_numpy()
    groups = table["point"].astype(str).to_numpy()
    assignment = dict(zip(groups, table["fold"].astype(int), strict=True))
    X = table[sources].to_numpy(dtype=float)
    bench, seed = cfg["benchmark"], cfg["head"]["seed"]

    def metrics(values: np.ndarray) -> dict[str, float]:
        m = evaluate(
            values, y, units, "recording", tuple(bench["precisions"]), bench["n_boot"], seed
        )
        return {k: m[k] for k in ("n_pos", "n_neg", "ap", "ap_lo", "ap_hi", "recall@p0.1")}

    rows = [{"model": s, "kind": "source", **metrics(X[:, j])} for j, s in enumerate(sources)]
    best = max(rows, key=lambda r: r["ap"])
    best_values = X[:, sources.index(best["model"])]
    comparisons, combined = [], {}
    fingerprint = labels_fingerprint(con, cfg)
    for method in methods:
        options = {"weights": dict.fromkeys(sources, 1 / len(sources))}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = fusion_model_oof(
                method, X, y, groups, list(sources), assignment=assignment, seed=seed, **options
            ).values
        name = f"ensemble:{method}({'+'.join(sources)})"
        combined[name] = values
        rows.append({"model": name, "kind": "ensemble", **metrics(values)})
        comparisons.append(
            {
                "ensemble": name,
                "best_source": best["model"],
                **paired_bootstrap(
                    y, values, best_values, units, n_boot=bench["n_boot"], seed=seed
                ),
            }
        )
        rec = table.assign(
            window_id=table["recording_id"] + ":enregistrement", offset_s=0.0, presumed=False
        )
        save_oof(cfg, oof_frame(name, "ensemble", rec, values, assignment, fingerprint, np.nan))
    return {
        "table": pd.DataFrame(rows).sort_values("ap", ascending=False).reset_index(drop=True),
        "comparisons": pd.DataFrame(comparisons),
        "n_recordings": len(table),
    }


# --- Concaténation des embeddings ---------------------------------------------------------------


def concat_training_set(
    con: sqlite3.Connection, cfg: dict, encoder_ids: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fenêtres évaluées du premier encodeur, embeddings de tous bout à bout (chaque bloc
    normalisé L2 : aucun encodeur ne pèse par sa seule échelle). Même grille exigée."""
    from blanci.head_benchmark import benchmark_data

    if len(encoder_ids) < 2:
        raise ValueError("une concaténation demande au moins deux encodeurs")
    grids = {
        eid: (encoder_params(con, eid)["window_s"], encoder_params(con, eid)["hop_s"])
        for eid in encoder_ids
    }
    if len(set(grids.values())) > 1:
        raise ValueError(
            f"grilles différentes {grids} : concaténer demande la même fenêtre et le même pas ; "
            "combiner plutôt par enregistrement (run_ensemble) ou par fenêtre (fusion.sources)"
        )
    data, X, _ = benchmark_data(con, cfg, encoder_ids[0])
    blocks = [l2_normalize(X)]
    wanted = data["window_id"].tolist()
    for eid in encoder_ids[1:]:
        meta, emb = EmbeddingStore(config_path(cfg, "embeddings"), eid).load()
        index = pd.Series(np.arange(len(meta)), index=meta["window_id"])
        index = index[~index.index.duplicated()]
        missing = [w for w in wanted if w not in index.index]
        if missing:
            raise ValueError(f"{eid} n'a pas encodé {len(missing)} fenêtres de {encoder_ids[0]}")
        blocks.append(l2_normalize(emb[index.loc[wanted].to_numpy()].astype(np.float32)))
    return data, np.concatenate(blocks, axis=1)


def concat_benchmark(
    con: sqlite3.Connection,
    cfg: dict,
    encoder_ids: list[str],
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Têtes sur les embeddings concaténés, mêmes plis que tous les autres modèles ; scores
    hors-pli enregistrés sous `<a>+<b>/<tête>`."""
    from blanci.dataset import folds_for
    from blanci.head import oof_scores

    methods = methods or ["logistic", "prototype"]
    data, X = concat_training_set(con, cfg, encoder_ids)
    head_cfg, bench = cfg["head"], cfg["benchmark"]
    y, groups = data["y"].to_numpy(), data["point"].to_numpy()
    recordings = data["recording_id"].to_numpy()
    assignment = folds_for(con, cfg)
    fingerprint = labels_fingerprint(con, cfg)
    name = "+".join(encoder_ids)
    rows = []
    for method in methods:
        oof = oof_scores(
            X,
            y,
            groups,
            n_splits=head_cfg["n_splits"],
            method=method,
            C_grid=head_cfg["C_grid"],
            seed=head_cfg["seed"],
            gated=data["gated"].to_numpy(),
            assignment=assignment,
        )
        save_oof(
            cfg,
            oof_frame(
                f"{name}/{method}", "encoder_head", data, oof.values, assignment, fingerprint
            ),
        )
        for level in ("window", "recording"):
            rows.append(
                {
                    "encoders": name,
                    "head": method,
                    **evaluate(
                        oof.values,
                        y,
                        recordings,
                        level,
                        tuple(bench["precisions"]),
                        bench["n_boot"],
                        head_cfg["seed"],
                    ),
                }
            )
    return pd.DataFrame(rows)
