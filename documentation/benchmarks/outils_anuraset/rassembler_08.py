"""Rassemble les sorties de global_bench.py de **tous** les encodeurs en `donnees/` du rapport 08.
Usage : rassembler_08.py <sorties> <dossier du rapport>

Encodeurs : ceux dont les cinq espèces ont une sortie `<encodeur>_<ESPECE>_transfert.npz`.
- `transfert.csv` : encodeur × tête × espèce × niveau (fenêtre native, minute) : AP poolée et AP
  moyenne par site (sites où l'espèce chante), toutes les fenêtres du site tenu à l'écart.
- `transfert_sites.csv` : la même AP, site par site.
- `meilleures.csv` : par encodeur, la tête à la meilleure AP moyenne par site (minute, moyenne
  des cinq espèces), **choisie après coup** parmi celles essayées, et la logistique.
- `comparaisons.csv` : minute, AP moyenne par site contre la référence fixée d'avance (perch_v2
  + logistique) pour la logistique et la meilleure tête de chaque encodeur (et toutes les
  têtes de perch_v2) ; bootstrap apparié (enregistrements tirés dans chaque site) ; Holm.
- `courbe.csv` : les runs de la courbe d'amorçage.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apparie import minutes, paired, site_mean_ap, windows  # noqa: E402

from blanci.evaluate import average_precision, holm  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2]) / "donnees"
dst.mkdir(parents=True, exist_ok=True)
ESPECES = ["DENMIN", "PITAZU", "PHYCUV", "LEPLAT", "BOAFAB"]
REFERENCE = ("perch_v2", "logistic")
N_BOOT, SEED = 1000, 0
PATTERN = re.compile(r"^(.*)_(" + "|".join(ESPECES) + r")_transfert\.npz$")

found: dict[str, set] = {}
for p in src.glob("*_transfert.npz"):
    m = PATTERN.match(p.name)
    if m:
        found.setdefault(m.group(1), set()).add(m.group(2))
ENCODEURS = sorted(e for e, sp in found.items() if len(sp) == len(ESPECES))
print("encodeurs :", len(ENCODEURS), ENCODEURS, flush=True)


def load(enc: str, sp: str) -> dict:
    z = np.load(src / f"{enc}_{sp}_transfert.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


rows, per_site, minute_frames = [], [], {}
for sp in ESPECES:
    for enc in ENCODEURS:
        z = load(enc, sp)
        mins = minutes(z).reset_index()
        minute_frames[(enc, sp)] = mins
        for level, frame in (("fenetre", windows(z)), ("minute", mins)):
            ok = frame[~frame["y"].isna()]
            y, site = ok["y"].to_numpy(), ok["site"].to_numpy()
            for h in z["heads"]:
                h = str(h)
                s = ok[h].to_numpy()
                keep = ~np.isnan(s)  # classifieur d'origine : espèce hors de ses classes
                if not keep.any():
                    continue
                for g in np.unique(site):
                    m = (site == g) & keep
                    per_site.append(
                        {
                            "encoder": enc,
                            "species": sp,
                            "head": h,
                            "level": level,
                            "site": g,
                            "n_pos": int(y[m].sum()),
                            "ap": average_precision(y[m], s[m]) if y[m].sum() else np.nan,
                        }
                    )
                rows.append(
                    {
                        "encoder": enc,
                        "species": sp,
                        "head": h,
                        "level": level,
                        "ap": average_precision(y[keep], s[keep]),
                        "ap_site": site_mean_ap(y[keep], s[keep], site[keep]),
                        "n_pos": int(y[keep].sum()),
                        "n": int(keep.sum()),
                    }
                )
    print("espèce", sp, flush=True)

table = pd.DataFrame(rows)
table.to_csv(dst / "transfert.csv", index=False)
pd.DataFrame(per_site).to_csv(dst / "transfert_sites.csv", index=False)

# Meilleure tête de chaque encodeur (après coup), sur les têtes jugées pour les cinq espèces.
minute = table[table["level"] == "minute"]
complete = minute.groupby(["encoder", "head"])["species"].nunique() == len(ESPECES)
means = minute.groupby(["encoder", "head"])["ap_site"].mean()[complete]
best_rows = []
for enc in ENCODEURS:
    part = means.loc[enc]
    best_rows.append(
        {
            "encoder": enc,
            "best_head": part.idxmax(),
            "ap_site_best": part.max(),
            "ap_site_logistic": part.get("logistic", np.nan),
            "heads_tried": len(part),
        }
    )
best = pd.DataFrame(best_rows).sort_values("ap_site_best", ascending=False)
best.to_csv(dst / "meilleures.csv", index=False)

# Comparaisons appariées à la référence, niveau minute.
comps = []
for sp in ESPECES:
    ref = minute_frames[(REFERENCE[0], sp)].set_index("rec")
    for enc in ENCODEURS:
        frame = minute_frames[(enc, sp)]
        heads = {"logistic", best.set_index("encoder").loc[enc, "best_head"]}
        if enc == REFERENCE[0]:
            heads = set(load(enc, sp)["heads"].astype(str)) - {REFERENCE[1]}
        f = frame[~frame["y"].isna()]
        mismatch = (ref.loc[f["rec"], "y"].to_numpy() != f["y"].to_numpy()).sum()
        if mismatch:
            print(f"  {enc} {sp} : {mismatch} minutes au label différent de la référence")
        r = ref.loc[f["rec"], REFERENCE[1]].to_numpy()
        for h in sorted(heads):
            if h not in f or f[h].isna().all():
                continue
            comps.append(
                {
                    "encoder": enc,
                    "head": h,
                    "species": sp,
                    "level": "minute",
                    **paired(f, f[h].to_numpy(), r, N_BOOT, SEED),
                }
            )
    print("comparaisons", sp, flush=True)
comp = pd.DataFrame(comps)
comp["p_holm"] = holm(comp["p"].to_numpy())
comp["significant_holm"] = comp["p_holm"] < 0.05
comp.to_csv(dst / "comparaisons.csv", index=False)

curves = sorted(src.glob("*_courbe.csv"))
pd.concat([pd.read_csv(p) for p in curves], ignore_index=True).to_csv(
    dst / "courbe.csv", index=False
)
print("ok", dst, flush=True)
