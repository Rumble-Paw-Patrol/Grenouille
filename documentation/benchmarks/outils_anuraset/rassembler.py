"""Rassemble les sorties de bench.py (une espèce par processus) en `donnees/` d'un rapport.
Usage : rassembler.py <sorties bench> <dossier du rapport>"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from blanci.anuraset import _holm_by_level
from blanci.evaluate import average_precision, recall_at_precision

src, dst = Path(sys.argv[1]), Path(sys.argv[2]) / "donnees"
dst.mkdir(parents=True, exist_ok=True)
ESPECES = ["DENMIN", "PITAZU", "PHYCUV", "LEPLAT", "BOAFAB"]


def cat(k):
    return pd.concat([pd.read_csv(src / f"{sp}_{k}.csv") for sp in ESPECES], ignore_index=True)


cat("table").to_csv(dst / "tetes.csv", index=False)
cat("sites").to_csv(dst / "sites.csv", index=False)
cat("selection").to_csv(dst / "selection.csv", index=False)
comp = cat("comparisons").drop(columns=["p_holm", "significant_holm"], errors="ignore")
_holm_by_level(comp).to_csv(dst / "comparaisons.csv", index=False)  # Holm sur toutes les espèces
pd.DataFrame([{sp: float((src / f"{sp}_duree_s.txt").read_text()) for sp in ESPECES}]).to_csv(
    dst / "durees_s.csv", index=False
)

# Amorçage d'un site : seuil (précision 0,5) choisi sur les scores hors-pli des autres sites,
# appliqué au site tenu à l'écart ; comparé au seuil choisi sur place (oracle).
rows = []
for sp in ESPECES:
    z = np.load(src / f"{sp}_scores.npz", allow_pickle=True)
    y, site = z["y"].astype(int), z["site"].astype(str)
    for h, s in zip(z["heads"], z["scores"], strict=False):
        for st in np.unique(site):
            m = site == st
            if y[m].sum() == 0:
                continue
            _, thr = recall_at_precision(y[~m], s[~m], 0.5)
            hit = s[m] >= thr
            k = int((hit & (y[m] == 1)).sum())
            oracle, _ = recall_at_precision(y[m], s[m], 0.5)
            rows.append(
                {
                    "species": sp,
                    "head": str(h),
                    "site": st,
                    "n_pos": int(y[m].sum()),
                    "n_neg": int((y[m] == 0).sum()),
                    "ap_site": average_precision(y[m], s[m]),
                    "recall_seuil_ailleurs": k / y[m].sum(),
                    "precision_seuil_ailleurs": k / hit.sum() if hit.sum() else np.nan,
                    "alertes": int(hit.sum()),
                    "recall_seuil_sur_place": oracle,
                }
            )
pd.DataFrame(rows).to_csv(dst / "amorcage.csv", index=False)
print("ok", dst)
