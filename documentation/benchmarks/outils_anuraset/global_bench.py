"""Benchmark global (07) : un encodeur, une espèce, un processus.

1. Transfert : un pli par site ; tête apprise sur les autres sites (tous les positifs, 20
   négatifs par positif et par site), puis **toutes** les fenêtres du site tenu à l'écart
   scorées. Scores gardés pour le niveau minute (max des fenêtres de chaque enregistrement),
   commun à tous les encodeurs quelle que soit leur fenêtre.
2. Courbe d'amorçage : pour chaque site où l'espèce a au moins `MIN_POS_REC` enregistrements
   positifs, moitié des enregistrements en test, l'autre en réserve ; k enregistrements positifs
   de la réserve (et des négatifs en proportion) ajoutés à l'entraînement ; `None` = toute la
   réserve (régime courant, site déjà annoté). Mêmes moitiés pour tous les encodeurs.

Options (n° 151, tête adaptée à la sortie de l'encodeur) : `--tokens` ajoute au transfert les
têtes sur jetons (`TOKEN_HEADS`, jetons de `jetons.py`) ; `--native` ajoute le classifieur de
l'encodeur lui-même, sans entraînement (probabilités rangées dans `scores` par l'encodage,
`logit_classes` : birdnet_v3). Jetons lourds : 2 processus en parallèle au plus.

Usage : global_bench.py <encodeur> <ESPECE> <sortie> [--curve] [--tokens] [--native]
(`<sortie>/<encodeur>_<ESPECE>.delegue` présent : tâche sautée, confiée à une autre machine.)
"""

import sys
import time
from dataclasses import replace
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
from blanci.regularization import Context, needs_domain, regularizer_for
from blanci.regularization.windows import _Accumulator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jetons import charger, stock_of  # noqa: E402

TRANSFER = ["logistic", "logistic+R37=glmm", "lda_shrunk", "simple_prototype"]
TOKEN_HEADS = ["attentive", "logistic:max", "proto_probe"]
# Classe du classifieur d'origine de chaque espèce (étiquettes de birdnet_v3 ; PITAZU absente).
NATIVE = {
    "DENMIN": "Dendropsophus minutus",
    "LEPLAT": "Leptodactylus latrans",
    "PHYCUV": "Physalaemus cuvieri",
    "BOAFAB": "Hypsiboas faber",
}
CURVE = ["logistic", "logistic+R37=glmm", "lda_shrunk", "logistic+R20", "prototype"]
CHUNK = 30000
TRAIN_CAP = 60000  # rcl_fs_bsed : 262 000 × 2 048 ne tient pas en 15 Go
K = [0, 1, 2, 5, 10, 20, None]
REPEATS = 2
MIN_POS_REC = 30
SPECIES = ["DENMIN", "PITAZU", "PHYCUV", "LEPLAT", "BOAFAB"]
SITES = ["INCT17", "INCT20955", "INCT4", "INCT41"]

name, sp, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
curve = "--curve" in sys.argv
out.mkdir(parents=True, exist_ok=True)
if (out / f"{name}_{sp}.delegue").exists():  # confiée à une autre machine
    sys.exit(f"délégué : {name} {sp}")
cfg = load_config(Path("config/anuraset.yaml"))
encoder_id = stock_of(name, config_path(cfg, "embeddings"))  # bacpipe ou avex
cfg["regularization"]["R37"]["glmm_grid"] = [0.3, 1.0, 3.0]  # comme les benchmarks 01 à 06
acfg, head_cfg = cfg["anuraset"], cfg["head"]
seed, C_grid = head_cfg["seed"], head_cfg["C_grid"]
t0 = time.time()

con = connect(config_path(cfg, "db"))
calls = read_strong_labels(project_path(acfg["labels"]))
weak = read_weak_labels(acfg.get("weak_labels"))
meta, emb = _encoder_windows(con, cfg, encoder_id)
# float16 tel que stocké (converti par paquets dans `score`) : 480 000 × 2048 en float32 = 4 Go
X_all = emb.astype(np.float32) if curve else emb
del emb  # 480 000 trames × 2048 : la mémoire compte
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
domain = None
if needs_domain(methods):  # par paquets : un site entier en float64 ne tient pas (rcl_fs_bsed)
    acc = _Accumulator()
    for i in range(0, len(X_all), 20000):
        acc.add(X_all[i : i + 20000], sites[i : i + 20000])
    domain = acc.stats("site")
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
    """Tête apprise sur `train`, scores de `test` par paquets : un site peut peser 370 000
    fenêtres (rcl_fs_bsed) et les régularisations copient la matrice qu'on leur donne."""
    _, base, regularizer = heads[spec]
    n_groups = len(np.unique(sites[train]))

    def restricted(rows: np.ndarray):
        reg = regularizer
        if reg is not None:
            reg = replace(reg, context=reg.context.subset(rows))
        method, inputs = _inputs(base, X_all[rows].astype(np.float32), None)
        return method, inputs, reg

    if C is None:
        method, inputs, reg = restricted(train)
        C = choose_C(
            method, inputs, y[train], sites[train], np.arange(len(train)), C_grid,
            max(2, n_groups), seed, reg,
        )
    values = np.empty(len(test), dtype=np.float32)
    for a in range(0, len(test), CHUNK):
        part = test[a : a + CHUNK]
        sub = np.union1d(train, part)
        method, inputs, reg = restricted(sub)
        values[a : a + CHUNK] = fit_and_score(
            method,
            inputs,
            y[sub],
            sites[sub],
            np.searchsorted(sub, train),
            np.searchsorted(sub, part),
            C=C,
            C_grid=C_grid,
            n_splits=max(2, n_groups),
            seed=seed,
            regularizer=reg,
        )
    return values


def minute_ap(values: np.ndarray, test: np.ndarray) -> float:
    rec = to_recordings(values, np.zeros(len(test)), recordings[test])
    labels = rec_table.loc[rec["recording_id"], "y"].to_numpy()
    ok = ~np.isnan(labels)
    return average_precision(labels[ok], rec["score"].to_numpy()[ok])


def native_scores() -> np.ndarray | None:
    """Probabilités du classifieur de l'encodeur pour l'espèce, alignées sur les fenêtres."""
    if sp not in NATIVE:
        return None
    rows_db = con.execute(
        "SELECT window_id, score FROM scores WHERE model_id = ?",
        (f"{encoder_id}:logit:{NATIVE[sp]}",),
    ).fetchall()
    if not rows_db:
        raise SystemExit(f"--native : aucun score {NATIVE[sp]} pour {encoder_id} (logit_classes)")
    by_window = dict(rows_db)
    values = meta["window_id"].map(by_window).to_numpy(dtype=np.float32)
    if np.isnan(values).mean() > 0.01:
        raise SystemExit(f"--native : {np.isnan(values).mean():.0%} des fenêtres sans score")
    return values


token_heads = TOKEN_HEADS if "--tokens" in sys.argv else []
tokens = charger(encoder_id, meta["window_id"].tolist()) if token_heads else None
native = native_scores() if "--native" in sys.argv else None

with threadpool_limits(limits=1, user_api="blas"):
    # 1. Transfert, un pli par site : toutes les fenêtres du site tenu à l'écart.
    names = TRANSFER + token_heads + (["classifieur_origine"] if native is not None else [])
    transfer = np.full((len(names), len(y)), np.nan, dtype=np.float32)
    for site in SITES:
        test = np.flatnonzero(sites == site)
        train = rows[sites[rows] != site]
        if len(train) > TRAIN_CAP:  # apprentissage plafonné (mémoire) : tous les positifs
            rng = np.random.default_rng([seed, SPECIES.index(sp), SITES.index(site)])
            pos, neg = train[y[train] == 1], train[y[train] == 0]
            keep = max(TRAIN_CAP - len(pos), 0)
            train = np.sort(np.r_[pos, rng.choice(neg, min(keep, len(neg)), replace=False)])
            print(f"  {site} : apprentissage plafonné à {len(train)} fenêtres", flush=True)
        for i, spec in enumerate(TRANSFER):
            transfer[i, test] = score(spec, train, test)
        for j, spec in enumerate(token_heads):  # sans régularisation, C par plis internes
            method, inputs = _inputs(spec, X_all, tokens)
            transfer[len(TRANSFER) + j, test] = fit_and_score(
                method,
                inputs,
                y,
                sites,
                train,
                test,
                C_grid=C_grid,
                n_splits=max(2, len(np.unique(sites[train]))),
                seed=seed,
            )
        print(f"  pli {site} {time.time() - t0:.0f} s", flush=True)
    if native is not None:
        transfer[-1] = native
    np.savez_compressed(
        out / f"{name}_{sp}_transfert.npz",
        heads=np.array(names),
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
