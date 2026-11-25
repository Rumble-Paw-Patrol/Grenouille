# ruff: noqa: E501
"""Fenêtres BirdNET 2.4 (3 s, 1 024 nombres) pour la prez PNA, en une passe (lecture seule).

    uv run python documentation/prez/scripts/presentation_7/extraire_birdnet.py

Lit la base (mode=ro) et le stock birdnet-bacpipe1.3.5 (`paths.embeddings` de config/local.yaml)
partition par partition. Écrit dans `rapports/presentation_7/` :
- `birdnet_hasard.npz` : 600 fenêtres ouvertes tirées au hasard par site (au prorata des
  partitions), avec site, enregistrement, décalage ;
- `birdnet_annotees.npz` : les fenêtres étiquetées par `inputs.dataset.training_set` (labels
  importés et intervalles du poste d'annotation, douteuses écartées, sans négatifs présumés),
  avec y, site, point d'écoute, enregistrement, décalage, label.
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
ENC, WIN, PAR_SITE = "birdnet-bacpipe1.3.5", 3.0, 600

cfg = load_config(LOCAL_CONFIG)
con = sqlite3.connect(f"file:{Path(cfg['paths']['db']).as_posix()}?mode=ro", uri=True)
recs = pd.read_sql_query("SELECT recording_id, site FROM recordings", con).set_index("recording_id")
store = store_for(cfg, ENC)
paths = list(store.fragments())
print(len(paths), "partitions ; exemple :", paths[0], flush=True)
tailles = {}
for p in paths:
    m = store.read_meta(p)
    tailles[p] = (m["recording_id"].map(recs["site"]).mode().iloc[0], len(m))
par_site = pd.Series([v[1] for v in tailles.values()]).groupby([v[0] for v in tailles.values()]).sum()
print(par_site.to_string(), flush=True)

rng = np.random.default_rng(0)
h_meta, h_emb, l_parts, l_emb = [], [], [], []
for k, p in enumerate(paths):
    meta, emb = store.read(p)
    site, n = tailles[p]
    ouverte = ~meta["gated"].to_numpy(bool) if "gated" in meta else np.ones(len(meta), bool)
    q = int(round(PAR_SITE * n / par_site[site]))
    cand = np.flatnonzero(ouverte)
    if q and len(cand):
        rows = np.sort(rng.choice(cand, size=min(q, len(cand)), replace=False))
        h_meta.append(meta.iloc[rows].assign(site=meta["recording_id"].iloc[rows].map(recs["site"]).to_numpy()))
        h_emb.append(emb[rows].astype(np.float32))
    data = training_set(con, meta.assign(dur_s=WIN), per_positive=0)
    if len(data):
        data = data[ouverte[data["row"].to_numpy()]]
        l_parts.append(data)
        l_emb.append(emb[data["row"].to_numpy()].astype(np.float32))
    print(f"{k + 1}/{len(paths)} {p.parent.name}/{p.stem} : {len(data)} annotées", flush=True)

H = pd.concat(h_meta, ignore_index=True)
np.savez_compressed(
    ICI / "birdnet_hasard.npz", X=np.concatenate(h_emb), site=H["site"].to_numpy(str),
    recording_id=H["recording_id"].to_numpy(str), offset_s=H["offset_s"].to_numpy(float),
)
L = pd.concat(l_parts, ignore_index=True)
print(L.groupby(["site", "y"]).size().unstack(fill_value=0).to_string())
print("enregistrements positifs :", L.loc[L.y == 1, "recording_id"].nunique())
np.savez_compressed(
    ICI / "birdnet_annotees.npz", X=np.concatenate(l_emb), y=L["y"].to_numpy(int),
    site=L["site"].to_numpy(str), point=L["point"].astype(str).to_numpy(str),
    label=L["label"].astype(str).to_numpy(str), recording_id=L["recording_id"].to_numpy(str),
    offset_s=L["offset_s"].to_numpy(float), window_id=L["window_id"].to_numpy(str),
)
print("écrit", len(H), "au hasard,", len(L), "annotées")
