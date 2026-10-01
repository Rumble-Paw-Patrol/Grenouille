"""Rassemble les sorties de global_bench.py en `donnees/` du rapport 07.
Usage : rassembler_global.py <sorties> <dossier du rapport>

- `transfert.csv` : encodeur × tête × espèce × niveau (fenêtre native, minute) : AP poolée,
  AP moyenne par site (sites où l'espèce chante), toutes les fenêtres du site tenu à l'écart.
- `transfert_sites.csv` : la même AP, site par site.
- `comparaisons.csv` : AP moyenne par site contre la référence fixée d'avance (perch_v2,
  logistique), bootstrap apparié qui tire les enregistrements dans chaque site ; minute : tous
  les encodeurs ; fenêtre : seulement ceux à fenêtre de 5 s (mêmes fenêtres). Holm par niveau.
- `seuil.csv` : fenêtre, seuil de précision 0,5 choisi sur les autres sites, appliqué au site.
- `courbe.csv` : les runs de la courbe d'amorçage.
- `encodeurs.csv` : fenêtre, dimension, débit d'encodage mesuré (CPU 4 cœurs).
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from blanci.embedding.store import EmbeddingStore
from blanci.evaluation.evaluate import average_precision, holm, recall_at_precision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apparie import minutes, paired, site_mean_ap, windows  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2]) / "donnees"
dst.mkdir(parents=True, exist_ok=True)
ENCODEURS = ["perch_v2", "perch_bird", "birdnet", "birdmae_base", "birdmae_huge", "protoclr"]
ESPECES = ["DENMIN", "PITAZU", "PHYCUV", "LEPLAT", "BOAFAB"]
REFERENCE = ("perch_v2", "logistic")
WINDOW_S = {"birdnet": 3.0, "protoclr": 6.0}  # les autres : 5 s
WINDOW_S = {e: WINDOW_S.get(e, 5.0) for e in ENCODEURS}
N_BOOT, SEED = 1000, 0


_OFFSETS: dict[str, np.ndarray] = {}


def offsets(enc: str) -> np.ndarray:
    """Début de chaque fenêtre, dans l'ordre du stock (celui des scores de global_bench)."""
    if enc not in _OFFSETS:
        meta, _ = EmbeddingStore(Path("data/embeddings_anuraset"), f"{enc}-bacpipe1.3.5@o0").load()
        _OFFSETS[enc] = meta["offset_s"].to_numpy()
    return _OFFSETS[enc]


def load(enc: str, sp: str) -> dict:
    z = np.load(src / f"{enc}_{sp}_transfert.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


rows, per_site, comps, seuils = [], [], [], []
for sp in ESPECES:
    data = {enc: load(enc, sp) for enc in ENCODEURS}
    mins = {enc: minutes(z) for enc, z in data.items()}
    wins = {enc: windows(z) for enc, z in data.items()}
    ref_min = mins[REFERENCE[0]]
    for enc, z in data.items():
        for level, frame in (("fenetre", wins[enc]), ("minute", mins[enc].reset_index())):
            ok = frame[~frame["y"].isna()]
            for h in z["heads"]:
                h = str(h)
                y, s, site = ok["y"].to_numpy(), ok[h].to_numpy(), ok["site"].to_numpy()
                for g in np.unique(site):
                    m = site == g
                    per_site.append(
                        {
                            "encoder": enc,
                            "species": sp,
                            "head": h,
                            "level": level,
                            "site": g,
                            "n_pos": int(y[m].sum()),
                            "n_neg": int((y[m] == 0).sum()),
                            "ap": average_precision(y[m], s[m]) if y[m].sum() else np.nan,
                        }
                    )
                rows.append(
                    {
                        "encoder": enc,
                        "species": sp,
                        "head": h,
                        "level": level,
                        "ap": average_precision(y, s),
                        "ap_site": site_mean_ap(y, s, site),
                        "n_pos": int(y.sum()),
                        "n": len(y),
                    }
                )
                if level == "fenetre":  # seuil choisi sur les autres sites
                    for g in np.unique(site):
                        m = site == g
                        if not y[m].sum():
                            continue
                        _, thr = recall_at_precision(y[~m], s[~m], 0.5)
                        hit = s[m] >= thr
                        k = int((hit & (y[m] == 1)).sum())
                        seuils.append(
                            {
                                "encoder": enc,
                                "species": sp,
                                "head": h,
                                "site": g,
                                "n_pos": int(y[m].sum()),
                                "n": int(m.sum()),
                                "recall_ailleurs": k / y[m].sum(),
                                "precision_ailleurs": k / hit.sum() if hit.sum() else np.nan,
                                "recall_sur_place": recall_at_precision(y[m], s[m], 0.5)[0],
                            }
                        )
        # Comparaisons appariées à la référence (fixée d'avance).
        for h in z["heads"]:
            h = str(h)
            if (enc, h) == REFERENCE:
                continue
            f = mins[enc].reset_index()
            f = f[~f["y"].isna()]
            r = ref_min.loc[f["rec"], REFERENCE[1]].to_numpy()
            comps.append(
                {
                    "encoder": enc,
                    "head": h,
                    "species": sp,
                    "level": "minute",
                    **paired(f, f[h].to_numpy(), r, N_BOOT, SEED),
                }
            )
            if WINDOW_S[enc] == WINDOW_S[REFERENCE[0]]:  # mêmes fenêtres : fenêtre appariée
                w = wins[enc].assign(offset=offsets(enc))
                ref = wins[REFERENCE[0]].assign(offset=offsets(REFERENCE[0]))
                w = w.merge(
                    ref[["rec", "offset", REFERENCE[1]]].rename(columns={REFERENCE[1]: "_ref"}),
                    on=["rec", "offset"],
                )
                assert len(w) == len(ref), (enc, len(w), len(ref))
                w = w[~w["y"].isna()]
                comps.append(
                    {
                        "encoder": enc,
                        "head": h,
                        "species": sp,
                        "level": "fenetre",
                        **paired(w, w[h].to_numpy(), w["_ref"].to_numpy(), N_BOOT, SEED),
                    }
                )
    print("espèce", sp, flush=True)

comp = pd.DataFrame(comps)
parts = []
for _, frame in comp.groupby("level"):
    frame = frame.copy()
    frame["p_holm"] = holm(frame["p"].to_numpy())
    frame["significant_holm"] = frame["p_holm"] < 0.05
    parts.append(frame)
pd.concat(parts, ignore_index=True).to_csv(dst / "comparaisons.csv", index=False)
pd.DataFrame(rows).to_csv(dst / "transfert.csv", index=False)
pd.DataFrame(per_site).to_csv(dst / "transfert_sites.csv", index=False)
pd.DataFrame(seuils).to_csv(dst / "seuil.csv", index=False)
curves = sorted(src.glob("*_courbe.csv"))
pd.concat([pd.read_csv(p) for p in curves], ignore_index=True).to_csv(
    dst / "courbe.csv", index=False
)

# Débit d'encodage (fenêtres/s, CPU 4 cœurs dans le cloud, une machine par session) : mesures
# des n° 141 (perch_v2 : 19 166 fenêtres en 37 min), 147, 148, 149.
DEBIT = {
    "perch_v2": 19166 / (37 * 60),
    "perch_bird": 3.66,
    "birdnet": 24.25,
    "birdmae_base": 7.6,
    "birdmae_huge": 1.11,
    "protoclr": 29.84,
}
con = sqlite3.connect("data/db/anuraset.sqlite")
enc_rows = []
for enc in ENCODEURS:
    model_id = f"{enc}-bacpipe1.3.5@o0"
    p = json.loads(
        con.execute("select params_json from models where model_id = ?", (model_id,)).fetchone()[0]
    )
    enc_rows.append(
        {
            "encoder": enc,
            "window_s": p["window_s"],
            "sample_rate": p["sample_rate"],
            "dim": p["dim"],
            "fenetres_par_s": DEBIT[enc],
            "temps_reel": DEBIT[enc] * p["window_s"],
        }
    )
pd.DataFrame(enc_rows).to_csv(dst / "encodeurs.csv", index=False)
durees = {p.name.split("_duree")[0]: float(p.read_text()) for p in src.glob("*_duree_s.txt")}
pd.Series(durees, name="secondes").rename_axis("job").to_csv(dst / "durees_s.csv")
print("ok", dst)
