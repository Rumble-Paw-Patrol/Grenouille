"""Benchmark global (07) : un encodeur, une espèce, un processus.

1. Transfert : un pli par site ; tête apprise sur les autres sites (tous les positifs, 20
   négatifs par positif et par site), puis **toutes** les fenêtres du site tenu à l'écart
   scorées. Scores gardés pour le niveau minute (max des fenêtres de chaque enregistrement),
   commun à tous les encodeurs quelle que soit leur fenêtre.
2. Courbe d'amorçage : pour chaque site où l'espèce a au moins `MIN_POS_REC` enregistrements
   positifs, moitié des enregistrements en test, l'autre en réserve ; k enregistrements positifs
   de la réserve (et des négatifs en proportion) ajoutés à l'entraînement ; `None` = toute la
   réserve (régime courant, site déjà annoté). Mêmes moitiés pour tous les encodeurs.

Usage : global_bench.py <encodeur> <ESPECE> <sortie> [--curve]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

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
from blanci.evaluate import average_precision, to_recordings
from blanci.head import choose_C, fit_and_score
from blanci.head_benchmark import _inputs
from blanci.regularization import Context, domain_statistics, needs_domain, regularizer_for

TRANSFER = ["logistic", "logistic+R37=glmm", "lda_shrunk"]
CURVE = ["logistic", "logistic+R37=glmm", "lda_shrunk", "logistic+R20", "prototype"]
K = [0, 1, 2, 5, 10, 20, None]
REPEATS = 2
MIN_POS_REC = 30
SPECIES = ["DENMIN", "PITAZU", "PHYCUV", "LEPLAT", "BOAFAB"]
SITES = ["INCT17", "INCT20955", "INCT4", "INCT41"]

name, sp, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
curve = "--curve" in sys.argv
out.mkdir(parents=True, exist_ok=True)
encoder_id = f"{name}-bacpipe1.3.5@o0"
cfg = load_config(Path("config/anuraset.yaml"))
cfg["regularization"]["R37"]["glmm_grid"] = [0.3, 1.0, 3.0]  # comme les benchmarks 01 à 06
acfg, head_cfg = cfg["anuraset"], cfg["head"]
seed, C_grid = head_cfg["seed"], head_cfg["C_grid"]
t0 = time.time()

con = connect(config_path(cfg, "db"))
calls = read_strong_labels(project_path(acfg["labels"]))
weak = read_weak_labels(acfg.get("weak_labels"))
meta, emb = _encoder_windows(con, cfg, encoder_id)
X_all = emb.astype(np.float32)
sites = meta["site"].astype(str).str.replace("INCT04", "INCT4").to_numpy()
recordings = meta["recording_id"].to_numpy()
unsure = weak_only_files(weak, calls, sp)
y_all = window_labels(meta, calls, sp, unsure_files=unsure)
known = ~np.isnan(y_all)
y = np.nan_to_num(y_all).astype(int)  # lignes inconnues : jamais apprises, jamais jugées
rows = species_rows(
    meta,
    y_all,
    sp,
    acfg["negatives_per_positive"],
    np.random.default_rng([seed, SPECIES.index(sp)]),
)
methods = sorted(set(TRANSFER) | (set(CURVE) if curve else set()))
domain = domain_statistics(X_all, sites, "site") if needs_domain(methods) else None
context = Context(
    sites,
    domain=domain,
    domain_rows=None if domain is None else domain.rows(sites),
    classes=np.where(y == 1, "blanci", "fond").astype(object),
)
heads = {spec: regularizer_for(spec, cfg, context) for spec in methods}

# Label par enregistrement (minute) : 1 si un chant daté de l'espèce, 0 sinon ; fichiers où
# les labels faibles la signalent sans chant daté : écartés.
rec_table = pd.DataFrame({"recording_id": recordings, "file_key": meta["file_key"], "site": sites})
rec_table = rec_table.drop_duplicates("recording_id").set_index("recording_id")
positive_files = set(calls.loc[calls["species"] == sp, "file_key"])
rec_table["y"] = np.where(
    rec_table["file_key"].isin(unsure),
    np.nan,
    rec_table["file_key"].isin(positive_files).astype(float),
)


def score(spec: str, train: np.ndarray, test: np.ndarray, C=None) -> np.ndarray:
    _, base, regularizer = heads[spec]
    method, inputs = _inputs(base, X_all, None)
    n_groups = len(np.unique(sites[train]))
    return fit_and_score(
        method,
        inputs,
        y,
        sites,
        train,
        test,
        C=C,
        C_grid=C_grid,
        n_splits=max(2, n_groups),
        seed=seed,
        regularizer=regularizer,
    )


def minute_ap(values: np.ndarray, test: np.ndarray) -> float:
    rec = to_recordings(values, np.zeros(len(test)), recordings[test])
    labels = rec_table.loc[rec["recording_id"], "y"].to_numpy()
    ok = ~np.isnan(labels)
    return average_precision(labels[ok], rec["score"].to_numpy()[ok])


with threadpool_limits(limits=1, user_api="blas"):
    # 1. Transfert, un pli par site : toutes les fenêtres du site tenu à l'écart.
    transfer = np.full((len(TRANSFER), len(y)), np.nan, dtype=np.float32)
    for site in SITES:
        test = np.flatnonzero(sites == site)
        train = rows[sites[rows] != site]
        for i, spec in enumerate(TRANSFER):
            transfer[i, test] = score(spec, train, test)
    np.savez_compressed(
        out / f"{name}_{sp}_transfert.npz",
        heads=np.array(TRANSFER),
        scores=transfer,
        y=y_all,
        site=sites,
        recording_id=recordings,
        train_rows=rows,
    )
    print(f"transfert {name} {sp} {time.time() - t0:.0f} s", flush=True)

    # 2. Courbe d'amorçage.
    runs = []
    for site in SITES if curve else []:
        mine = rec_table[rec_table["site"] == site].sort_index()
        pos = mine.index[mine["y"] == 1].tolist()
        neg = mine.index[mine["y"] == 0].tolist()
        if len(pos) < MIN_POS_REC:
            continue
        others = rows[sites[rows] != site]
        C_of = {}
        for spec in CURVE:
            _, base, regularizer = heads[spec]
            method, inputs = _inputs(base, X_all, None)
            C_of[spec] = choose_C(method, inputs, y, sites, others, C_grid, 3, seed, regularizer)
        for r in range(REPEATS):
            rng = np.random.default_rng([seed, r, SPECIES.index(sp), SITES.index(site)])
            pos_r, neg_r = list(rng.permutation(pos)), list(rng.permutation(neg))
            cut_p, cut_n = (len(pos_r) + 1) // 2, (len(neg_r) + 1) // 2
            test_pos, pool_pos = pos_r[:cut_p], pos_r[cut_p:]
            test_neg, pool_neg = neg_r[:cut_n], neg_r[cut_n:]
            test = np.flatnonzero(np.isin(recordings, test_pos + test_neg))
            judged = test[known[test]]
            for k in K:
                k_pos = len(pool_pos) if k is None else k
                n_neg = round(len(pool_neg) * k_pos / len(pool_pos))
                support = pool_pos[:k_pos] + pool_neg[:n_neg]
                extra = np.flatnonzero(np.isin(recordings, support) & known)
                train = np.concatenate([others, extra])
                for spec in CURVE:
                    values = score(spec, train, test, C=C_of[spec])
                    in_judged = known[test]
                    runs.append(
                        {
                            "encoder": name,
                            "species": sp,
                            "site": site,
                            "repeat": r,
                            "k": -1 if k is None else k,
                            "k_pos": k_pos,
                            "k_neg": n_neg,
                            "head": spec,
                            "ap_window": average_precision(y[judged], values[in_judged]),
                            "ap_minute": minute_ap(values, test),
                            "n_test_pos_windows": int(y[judged].sum()),
                            "n_test_pos_rec": len(test_pos),
                            "n_test_rec": len(test_pos) + len(test_neg),
                        }
                    )
        print(f"courbe {name} {sp} {site} {time.time() - t0:.0f} s", flush=True)
    if runs:
        pd.DataFrame(runs).to_csv(out / f"{name}_{sp}_courbe.csv", index=False)

(out / f"{name}_{sp}_duree_s.txt").write_text(f"{time.time() - t0:.0f}")
print("FINI", name, sp, f"{time.time() - t0:.0f} s", flush=True)
