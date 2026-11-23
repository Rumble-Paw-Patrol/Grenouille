# ruff: noqa: E501
"""Extrait des embeddings Perch v2 pour le prototype de la prez PNA (lecture seule).

    uv run python documentation/prez/scripts/presentation_7/extraire.py

Lit la base (mode=ro) et le stock perch_v2-bacpipe1.3.5 partition par partition ; écrit
`echantillon.npz` à côté du script : un tirage au hasard de fenêtres ouvertes (≈ 600 par site)
et toutes les fenêtres annotées, avec site, label et embedding.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from blanci.core.config import LOCAL_CONFIG, load_config
from blanci.inputs.dataset import current_labels
from blanci.inputs.labels import POSITIVE_LABELS
from blanci.service import store_for

ICI = Path(__file__).resolve().parents[4] / "rapports" / "presentation_7"  # données, jamais committées
ICI.mkdir(parents=True, exist_ok=True)
ENC = "perch_v2-bacpipe1.3.5"
PAR_SITE = 600

cfg = load_config(LOCAL_CONFIG)
con = sqlite3.connect(f"file:{Path(cfg['paths']['db']).as_posix()}?mode=ro", uri=True)
labels = current_labels(con)
print(labels["label"].value_counts().to_string())
lab = labels.set_index("window_id")["label"]
rec_site = pd.read_sql_query("SELECT recording_id, site FROM recordings", con).set_index(
    "recording_id"
)["site"]

cfg["paths"]["embeddings"] = str(Path("data/embeddings").resolve())  # stock Perch : F:data
store = store_for(cfg, ENC)
paths = list(store.fragments())
print(len(paths), "partitions ; exemple :", paths[0])
rng = np.random.default_rng(0)
tailles = {}
for p in paths:
    m = store.read_meta(p)
    s = m["recording_id"].map(rec_site).iloc[0]
    tailles[p] = (s, len(m))
par_site = pd.Series({p: v[1] for p, v in tailles.items()}).groupby(
    pd.Series({p: v[0] for p, v in tailles.items()})
).sum()
print(par_site.to_string())

metas, embs = [], []
for p in paths:
    meta, emb = store.read(p)
    site, n = tailles[p]
    ouverte = ~meta["gated"].to_numpy() if "gated" in meta else np.ones(len(meta), bool)
    k = int(round(PAR_SITE * n / par_site[site]))
    tirees = np.zeros(len(meta), bool)
    cand = np.flatnonzero(ouverte)
    if k and len(cand):
        tirees[rng.choice(cand, size=min(k, len(cand)), replace=False)] = True
    annotees = meta["window_id"].isin(lab.index).to_numpy() & ouverte
    garde = tirees | annotees
    if garde.any():
        metas.append(meta[garde].assign(tiree=tirees[garde], site=site))
        embs.append(emb[garde].astype(np.float32))

meta = pd.concat(metas, ignore_index=True)
X = np.concatenate(embs)
meta["label"] = meta["window_id"].map(lab)
meta["y"] = np.where(meta["label"].isin(POSITIVE_LABELS), 1, np.where(meta["label"].isna(), -1, 0))
print(meta.groupby(["site", "y"]).size().unstack(fill_value=0).to_string())
np.savez_compressed(
    ICI / "echantillon.npz",
    X=X,
    site=meta["site"].to_numpy(str),
    y=meta["y"].to_numpy(),
    label=meta["label"].fillna("").to_numpy(str),
    tiree=meta["tiree"].to_numpy(),
    recording_id=meta["recording_id"].to_numpy(str),
    window_id=meta["window_id"].to_numpy(),
)
print("écrit", X.shape)
