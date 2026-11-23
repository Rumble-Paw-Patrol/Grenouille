# ruff: noqa: E501
"""Fenêtres Perch v2 annotées + signatures moyennes d'enregistrements (lecture seule).

    uv run python documentation/prez/scripts/presentation_7/extraire_labels.py

Une passe sur les partitions du stock perch_v2-bacpipe1.3.5 :
- `annotees.npz` : les fenêtres étiquetées par `inputs.dataset.training_set` (labels importés
  et intervalles du poste d'annotation, fenêtres douteuses écartées, sans négatifs présumés),
  avec y, site, point d'écoute (micro), enregistrement, décalage ;
- `enregistrements.npz` : pour 80 enregistrements tirés au hasard par site, la moyenne des
  embeddings de leurs fenêtres ouvertes (signature d'enregistrement), avec site, micro, heure.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from blanci.core.config import LOCAL_CONFIG, load_config
from blanci.inputs.dataset import training_set
from blanci.service import store_for

ICI = Path(__file__).resolve().parents[4] / "rapports" / "presentation_7"  # données, jamais committées
ICI.mkdir(parents=True, exist_ok=True)
ENC, WIN, PAR_SITE = "perch_v2-bacpipe1.3.5", 5.0, 80

cfg = load_config(LOCAL_CONFIG)
con = sqlite3.connect(f"file:{Path(cfg['paths']['db']).as_posix()}?mode=ro", uri=True)
recs = pd.read_sql_query("SELECT recording_id, site, mic_id, start_utc FROM recordings", con)
recs = recs.set_index("recording_id")
cfg["paths"]["embeddings"] = str(Path("data/embeddings").resolve())  # stock Perch : F:\data
store = store_for(cfg, ENC)
paths = list(store.fragments())

# tirage des enregistrements pour les signatures moyennes, parmi ceux du stock
metas = {p: store.read_meta(p) for p in paths}
dans_stock = pd.concat([m[["recording_id"]] for m in metas.values()]).drop_duplicates()
dans_stock["site"] = dans_stock["recording_id"].map(recs["site"])
rng = np.random.default_rng(0)
tires = set()
for _site, g in dans_stock.groupby("site"):
    tires |= set(rng.choice(g["recording_id"].to_numpy(), min(PAR_SITE, len(g)), replace=False))

lab_parts, lab_embs, rec_rows, rec_embs = [], [], [], []
for k, p in enumerate(paths):
    meta, emb = store.read(p)
    ouverte = ~meta["gated"].to_numpy() if "gated" in meta else np.ones(len(meta), bool)
    data = training_set(con, meta.assign(dur_s=WIN), per_positive=0)
    if len(data):
        data = data[ouverte[data["row"].to_numpy()]]
        lab_parts.append(data)
        lab_embs.append(emb[data["row"].to_numpy()].astype(np.float32))
    sel = meta["recording_id"].isin(tires).to_numpy() & ouverte
    if sel.any():
        sub = meta[sel]
        X = emb[sel].astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        for rec, idx in sub.groupby("recording_id").indices.items():
            rec_rows.append(rec)
            rec_embs.append(X[idx].mean(axis=0))
    print(f"{k + 1}/{len(paths)} {p.parent.name}/{p.stem} : {len(data)} annotées", flush=True)

lab = pd.concat(lab_parts, ignore_index=True)
XL = np.concatenate(lab_embs)
lab["mic_id"] = lab["recording_id"].map(recs["mic_id"])
print(lab.groupby(["site", "y"]).size().unstack(fill_value=0).to_string())
print("enregistrements positifs :", lab.loc[lab.y == 1, "recording_id"].nunique())
np.savez_compressed(
    ICI / "annotees.npz",
    X=XL,
    y=lab["y"].to_numpy(int),
    site=lab["site"].to_numpy(str),
    point=lab["point"].astype(str).to_numpy(str),
    mic_id=lab["mic_id"].astype(str).to_numpy(str),
    label=lab["label"].astype(str).to_numpy(str),
    recording_id=lab["recording_id"].to_numpy(str),
    offset_s=lab["offset_s"].to_numpy(float),
    window_id=lab["window_id"].to_numpy(str),
)
R = recs.loc[rec_rows]
np.savez_compressed(
    ICI / "enregistrements.npz",
    X=np.stack(rec_embs),
    recording_id=np.array(rec_rows, str),
    site=R["site"].to_numpy(str),
    mic_id=R["mic_id"].to_numpy(str),
    start_utc=R["start_utc"].to_numpy(str),
)
print("écrit", XL.shape, len(rec_rows))
