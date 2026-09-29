"""Amorçage d'un site peu annoté : AP et rappel à précision 0,5 de chaque site tenu à l'écart,
le seuil choisi sur les scores hors-pli des **autres** sites (ce qu'on aurait en arrivant sur un
site nouveau) ou sur le site lui-même (oracle). Refait le tirage de `run_anuraset_heads` (même
graine, une espèce par processus : mêmes fenêtres que la campagne ; l'AP par site doit
retomber sur `donnees/sites.csv`). Écrit `donnees/amorcage.csv`. Lent (têtes réapprises) :
`generer.py` n'en a pas besoin.

    OMP_NUM_THREADS=1 uv run python \
        documentation/benchmarks/2026-09-29_anuraset_birdmae_base/amorcage.py birdmae_base PITAZU

Un processus à la fois (le CSV est relu puis réécrit).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from blanci.anuraset import (
    _encoder_windows,
    read_strong_labels,
    read_weak_labels,
    species_rows,
    weak_only_files,
    window_labels,
)
from blanci.config import config_path, load_config, project_path
from blanci.db import connect
from blanci.evaluate import average_precision, recall_at_precision
from blanci.head import calibration_options, oof_scores
from blanci.head_benchmark import _inputs
from blanci.regularization import (
    Context,
    domain_statistics,
    needs_domain,
    needs_pool,
    regularizer_for,
)

ICI = Path(__file__).resolve().parent
ENCODEURS = {  # nom → (stock, base, dossier des embeddings)
    "birdmae_base": ("birdmae_base-bacpipe1.3.5@o0", None, None),
    # perch_v2 : stock et base de la branche `donnees-anuraset` (benchmark 01), extraits sous
    # data/perch_ref/ (git archive origin/donnees-anuraset data/db data/embeddings_anuraset)
    "perch_v2": (
        "perch_v2-bacpipe1.3.5@o0",
        "data/perch_ref/data/db/anuraset.sqlite",
        "data/perch_ref/data/embeddings_anuraset",
    ),
}
CAS = {"PITAZU": "INCT41", "LEPLAT": "INCT4"}  # site peu annoté de chaque espèce
TETES = ["logistic", "prototype", "logistic+R37", "logistic+R37=glmm", "logistic+R18=64"]
PRECISION = 0.5


def seuil(y: np.ndarray, s: np.ndarray) -> float:
    return recall_at_precision(y, s, PRECISION)[1]


def une_espece(nom: str, sp: str) -> list[dict]:
    encodeur, db, stock = ENCODEURS[nom]
    cfg = load_config(Path("config/anuraset.yaml"))
    if db:
        cfg["paths"]["db"], cfg["paths"]["embeddings"] = db, stock
    cfg["regularization"]["R37"]["glmm_grid"] = [0.3, 1.0, 3.0]
    acfg, head_cfg = cfg["anuraset"], cfg["head"]
    con = connect(config_path(cfg, "db"))
    calls = read_strong_labels(project_path(acfg["labels"]))
    weak = read_weak_labels(acfg.get("weak_labels"))
    rng = np.random.default_rng(head_cfg["seed"])
    meta, emb = _encoder_windows(con, cfg, encodeur)
    sites = meta["site"].astype(str).to_numpy()
    domain = domain_statistics(emb, sites, "site") if needs_domain(TETES) else None
    y_all = window_labels(meta, calls, sp, unsure_files=weak_only_files(weak, calls, sp))
    rows = species_rows(meta, y_all, sp, acfg["negatives_per_positive"], rng, encodeur)
    X, y = emb[rows].astype(np.float32), y_all[rows].astype(int)
    groups = sites[rows]
    context = Context(
        groups,
        domain=domain,
        domain_rows=None if domain is None else domain.rows(groups),
        classes=np.where(y == 1, "blanci", "fond").astype(object),
    )
    assert not needs_pool(TETES)
    out = []
    for spec in TETES:
        name, base, regularizer = regularizer_for(spec, cfg, context)
        method, inputs = _inputs(base, X, None)
        s = oof_scores(
            inputs,
            y,
            groups,
            n_splits=len(np.unique(groups)),
            method=method,
            C_grid=head_cfg["C_grid"],
            seed=head_cfg["seed"],
            regularizer=regularizer,
            **calibration_options(cfg),
        ).values
        for site in np.unique(groups):
            ici, ailleurs = groups == site, groups != site
            t = seuil(y[ailleurs], s[ailleurs])
            dessus = s[ici] >= t
            oracle, _ = recall_at_precision(y[ici], s[ici], PRECISION)
            out.append(
                {
                    "encoder": nom,
                    "species": sp,
                    "head": name,
                    "site": site,
                    "peu_annote": site == CAS[sp],
                    "n_pos": int(y[ici].sum()),
                    "n_neg": int((1 - y[ici]).sum()),
                    "ap": average_precision(y[ici], s[ici]),
                    "seuil_ailleurs": t,
                    "rappel_seuil_ailleurs": float(y[ici][dessus].sum() / max(y[ici].sum(), 1)),
                    "precision_seuil_ailleurs": (
                        float(y[ici][dessus].mean()) if dessus.any() else float("nan")
                    ),
                    "rappel_oracle": oracle,
                }
            )
        print(sp, name, flush=True)
    return out


def main() -> None:
    nom, especes = sys.argv[1], sys.argv[2:] or list(CAS)
    frames = [pd.DataFrame(une_espece(nom, sp)) for sp in especes]
    path = ICI / "donnees" / "amorcage.csv"
    new = pd.concat(frames, ignore_index=True)
    if path.exists():
        old = pd.read_csv(path)
        kept = old[~(old["species"].isin(especes) & (old["encoder"] == nom))]
        new = pd.concat([kept, new], ignore_index=True)
    new.to_csv(path, index=False)
    print(path)


if __name__ == "__main__":
    main()
