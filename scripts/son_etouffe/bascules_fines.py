"""Repérage fin des micros qui s'étouffent ou se libèrent en plein enregistrement.

La file `bascule_plein_enregistrement` (`file_bascules.py`) ne regardait que les voisins d'une suite
écartée, sur la piste 0, par tranches de 10 s et avec un seuil absolu : à CDR, ses 11 candidats
étaient des micros dans un sac (pose, retrait, fermetures éclair). Ce script change quatre choses.

1. **Tri sur tout le corpus, sans audio** (`tri`) : les indices de chaque enregistrement sont déjà
   en base (`qc_flags.indices`, piste 0). Un fichier est « bas » si `hf_ratio` < `muffled_hf_ratio`
   ou s'il tombe de ×4 sous la médiane de son propre micro. Pour chaque suite de fichiers bas,
   même courte (le drapeau en exige 4 de suite), on retient le dernier fichier sain avant, le
   premier et le dernier de la suite, le premier sain après ; la suite entière si elle a 3
   fichiers ou moins.
2. **Mesure fine** (`mesure`) : `hf_ratio` par tranche de 2 s sur **les deux pistes**.
3. **Marche** (`file`) : cherchée sur la médiane glissante de 3 tranches (une pointe de fermeture
   éclair ne la fait pas bouger) ; il faut un facteur ≥ 6 entre les deux côtés, un côté
   franchement étouffé (médiane < 0,08), l'autre franchement sain (≥ 0,12), et un saut bien plus
   grand que la dispersion de chaque côté.
4. **Forme et contexte** : largeur de la transition (tranches passées entre 20 et 80 % du saut) :
   brusque (≤ 1 tranche) = un geste (sac, main) ; progressive (≥ 3 tranches) = une bonnette qui
   s'imbibe ou qui sèche. Piste qui bascule (0 seule, 1 seule, les deux), synchronie avec les
   autres micros du site (une pose ou un retrait fait tomber beaucoup de micros ensemble) et
   bord de campagne (10 premières et 6 dernières heures d'une période de pose).

Lecture seule de la base et de l'audio ; n'écrit que `paths.reports/son_etouffe/bascules_fines/`
et la file `bascules_fines` dans `annotations/files_ecoute/`.

    uv run python scripts/son_etouffe/bascules_fines.py tri      # base seule
    uv run python scripts/son_etouffe/bascules_fines.py mesure   # disques branchés, reprenable
    uv run python scripts/son_etouffe/bascules_fines.py file     # détection, file d'écoute, tableau
    uv run python scripts/son_etouffe/bascules_fines.py degradees  # pistes 0 encodées dégradées
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import soundfile as sf

from blanci.annotation.selection import write_queue
from blanci.core.config import config_path, default_user_config, load_config
from blanci.core.locate import locate
from blanci.inputs.qc import qc_indices

TRANCHE_S = 2.0
BAS = 0.05  # `qc.muffled_hf_ratio`
CHUTE_LOG = 0.6  # ×4 sous la médiane du micro
SAUT_LOG = float(np.log10(6))
ETOUFFE, SAIN = 0.08, 0.12
NOM = "bascules_fines"
# Deux pistes 0 encodées (aucun drapeau : hors d'une suite de 4) et pourtant très étouffées, la
# piste 1 au même moment très bruyante (-6 et -12 dBFS, des centaines de clics par seconde) :
# 2MA01489 le 24/12/2025 à 19 h 30 locales, 2LA03021 le 25/12/2025 à 19 h 30 locales (CDR).
DEGRADEES = ("d9d01ec1f00108fe", "a878e4aa18fd50be")


def dossier(cfg: dict) -> Path:
    out = config_path(cfg, "reports") / "son_etouffe" / NOM
    out.mkdir(parents=True, exist_ok=True)
    return out


# --- 1. tri sur la base -----------------------------------------------------------------------


def lire_indices(db: Path) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    rows = []
    for rid, path, ds, site, mic, start, dur, q in con.execute(
        "SELECT recording_id, path, dataset, site, mic_id, start_utc, duration_s, qc_flags "
        "FROM recordings"
    ):
        f = json.loads(q) if q else {}
        i = f.get("indices")
        rows.append(
            (
                rid,
                path,
                ds,
                site,
                mic,
                start,
                dur,
                i["hf_ratio"] if i else None,
                bool(f.get("muffled_0")),
                bool(f.get("silent")),
            )
        )
    con.close()
    d = pd.DataFrame(
        rows,
        columns=[
            "recording_id",
            "path",
            "dataset",
            "site",
            "mic_id",
            "start_utc",
            "duration_s",
            "hf0",
            "m0",
            "silent",
        ],
    )
    d["t"] = pd.to_datetime(d["start_utc"], utc=True)
    d = d[d["duration_s"].between(119, 121) & d["hf0"].notna() & ~d["silent"]].copy()
    d["slot"] = d["t"].dt.round("30min")
    keys = ["dataset", "site", "mic_id", "slot"]
    return d.sort_values(keys).drop_duplicates(keys).reset_index(drop=True)


def periodes(d: pd.DataFrame) -> pd.DataFrame:
    """Bord de campagne : 10 premières et 6 dernières heures de chaque période de pose d'un site
    (une interruption de plus de 36 h sépare deux périodes)."""
    d = d.assign(bord=False)
    for (_, _), g in d.groupby(["dataset", "site"]):
        slots = np.sort(g["slot"].unique())
        coupe = np.flatnonzero(np.diff(slots) > np.timedelta64(36, "h")) + 1
        for bloc in np.split(slots, coupe):
            debut, fin = bloc[0], bloc[-1]
            tot = (d["dataset"] == g["dataset"].iloc[0]) & (d["site"] == g["site"].iloc[0])
            dans = tot & d["slot"].between(debut, fin)
            d.loc[dans & (d["slot"] <= debut + np.timedelta64(10, "h")), "bord"] = True
            d.loc[dans & (d["slot"] >= fin - np.timedelta64(6, "h")), "bord"] = True
    return d


def tri(cfg: dict) -> pd.DataFrame:
    d = lire_indices(config_path(cfg, "db"))
    d["lhf"] = np.log10(d["hf0"] + 1e-4)
    d["base"] = d.groupby(["dataset", "site", "mic_id"])["lhf"].transform(
        lambda s: s.rolling(97, center=True, min_periods=10).median()
    )
    d["bas"] = (d["hf0"] < BAS) | (d["lhf"] - d["base"] <= -CHUTE_LOG)
    d = periodes(d)
    bas = d[d["bas"]][["dataset", "site", "mic_id", "slot"]]
    garde = []
    for (ds, site, mic), g in d.groupby(["dataset", "site", "mic_id"], sort=False):
        g = g.reset_index(drop=True)
        b = g["bas"].to_numpy()
        i, run = 0, 0
        while i < len(g):
            if not b[i]:
                i += 1
                continue
            j = i
            while (
                j + 1 < len(g)
                and b[j + 1]
                and g["slot"][j + 1] - g["slot"][j] <= pd.Timedelta("35min")
            ):
                j += 1
            run += 1
            idx = set(range(i, j + 1)) if j - i < 3 else {i, j}
            for k, role in ((i - 1, "avant"), (j + 1, "apres")):
                if 0 <= k < len(g) and abs(
                    g["slot"][k] - g["slot"][i if role == "avant" else j]
                ) <= pd.Timedelta("35min"):
                    idx.add(k)
            autres = bas[
                (bas["dataset"] == ds)
                & (bas["site"] == site)
                & (bas["mic_id"] != mic)
                & bas["slot"].between(
                    g["slot"][i] - pd.Timedelta("1h"), g["slot"][j] + pd.Timedelta("1h")
                )
            ]
            for k in sorted(idx):
                garde.append(
                    g.iloc[k].to_dict()
                    | {
                        "suite": f"{mic}@{g['slot'][i]:%Y-%m-%dT%H:%M}",
                        "suite_n": j - i + 1,
                        "sync": autres["mic_id"].nunique(),
                    }
                )
            i = j + 1
    out = pd.DataFrame(garde).drop_duplicates("recording_id")
    out = out.drop(columns=["t", "lhf", "base"]).reset_index(drop=True)
    out.to_csv(dossier(cfg) / "tri.csv", index=False)
    print(f"{len(out)} fichiers à lire ({out['suite'].nunique()} suites de fichiers bas) :")
    print(out.groupby("site").size().to_string())
    return out


# --- 2. mesure fine ---------------------------------------------------------------------------


def mesurer(path: Path | str) -> dict[str, object]:
    """`hf_ratio` et RMS par tranche de 2 s, pour chaque piste. Une chaîne est une erreur."""
    if isinstance(path, str):
        return {"erreur": path}
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        n = int(TRANCHE_S * sr)
        out: dict[str, object] = {"sample_rate": sr}
        for piste in range(min(2, wav.shape[1])):
            x = wav[:, piste]
            parts = [qc_indices(x[i : i + n], sr) for i in range(0, len(x) - n + 1, n)]
            out[f"hf_{piste}"] = json.dumps([round(p["hf_ratio"], 4) for p in parts])
            out[f"rms_{piste}"] = json.dumps([round(p["rms_dbfs"], 1) for p in parts])
        return out
    except Exception as exc:  # fichier illisible : noté, pas bloquant
        return {"erreur": f"{type(exc).__name__}: {exc}"}


def mesure(cfg: dict, fils: int, limite: int | None, prioritaire: bool = False) -> None:
    out = dossier(cfg)
    cible = out / "mesures.csv"
    todo = pd.read_csv(out / "tri.csv")
    if prioritaire:
        todo = todo[~todo["bord"] & (todo["sync"] <= 2)]
    deja = set(pd.read_csv(cible)["recording_id"]) if cible.is_file() else set()
    todo = todo[~todo["recording_id"].isin(deja)]
    if limite:
        todo = todo.head(limite)
    print(f"{len(todo)} à lire, {len(deja)} déjà lus")
    raw, renamed = Path(config_path(cfg, "raw")), cfg["audio"].get("renamed")
    colonnes = list(pd.read_csv(cible, nrows=0).columns) if cible.is_file() else None
    rows: list[dict] = []

    def flush() -> None:
        nonlocal colonnes, rows
        if not rows:
            return
        frame = pd.DataFrame(rows)
        colonnes = colonnes or list(frame.columns)
        frame.reindex(columns=colonnes).to_csv(
            cible, mode="a", header=not cible.is_file(), index=False
        )
        rows = []

    def lire(rec) -> Path | str:
        p = locate(rec.path, raw, renamed=renamed)
        return "introuvable" if p is None else p

    debut, n = perf_counter(), 0
    recs = list(todo.itertuples())
    # Lecture série (disque USB à plateaux), calcul dans des processus.
    with ProcessPoolExecutor(max_workers=fils) as pool:
        attente: deque = deque()
        for rec in [*recs, None]:
            if rec is not None:
                attente.append((rec, pool.submit(mesurer, lire(rec))))
            while attente and (rec is None or len(attente) > 2 * fils):
                r, fut = attente.popleft()
                n += 1
                rows.append({"recording_id": r.recording_id} | fut.result())
                if n % 50 == 0 or n == len(recs):
                    flush()
                    print(f"  {n}/{len(recs)}, {n / (perf_counter() - debut):.1f}/s", flush=True)
    flush()


# --- 3. marche ---------------------------------------------------------------------------------


def mediane3(x: np.ndarray) -> np.ndarray:
    p = np.pad(x, 1, mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def marche(hf: np.ndarray) -> dict[str, float] | None:
    """Meilleure marche de la série des tranches, ou None."""
    lh = mediane3(np.log10(hf + 1e-4))
    meilleure, score_max = None, 0.0
    for k in range(3, len(lh) - 2):
        g, d = lh[:k], lh[k:]
        mg, md = float(np.median(g)), float(np.median(d))
        saut = md - mg
        bas, haut = min(mg, md), max(mg, md)
        if abs(saut) < SAUT_LOG or bas >= np.log10(ETOUFFE) or haut < np.log10(SAIN):
            continue
        disp = float(np.median(np.abs(g - mg)) + np.median(np.abs(d - md)))
        score = abs(saut) / (disp + 0.15)
        if score > score_max:
            meilleure = {"k": k, "saut": saut, "score": score, "bas": 10**bas, "haut": 10**haut}
            score_max = score
    if meilleure is None or score_max < 4:
        return None
    # La coupure du score est approximative : on la recale sur la plus forte marche locale (3
    # tranches de chaque côté), puis on mesure la largeur de la transition autour d'elle.
    k0 = meilleure["k"]
    candidats = range(max(3, k0 - 10), min(len(lh) - 2, k0 + 11))
    k = max(candidats, key=lambda i: abs(np.median(lh[i : i + 3]) - np.median(lh[i - 3 : i])))
    bas, haut = np.log10(meilleure["bas"]), np.log10(meilleure["haut"])
    pas = abs(meilleure["saut"])
    fenetre = np.arange(max(0, k - 15), min(len(lh), k + 15))
    entre = fenetre[(lh[fenetre] > bas + 0.2 * pas) & (lh[fenetre] < bas + 0.8 * pas)]
    meilleure |= {"k": k, "t_s": k * TRANCHE_S, "largeur": len(entre)}
    if len(entre):
        meilleure["t_s"] = float(np.mean(entre)) * TRANCHE_S + TRANCHE_S / 2
    return meilleure


def detecter(m: pd.DataFrame) -> pd.DataFrame:
    lignes = []
    for r in m.itertuples():
        if isinstance(getattr(r, "erreur", None), str) or not isinstance(r.hf_0, str):
            continue
        res = {
            p: marche(np.array(json.loads(getattr(r, f"hf_{p}"))))
            if isinstance(getattr(r, f"hf_{p}", None), str)
            else None
            for p in (0, 1)
        }
        lignes.append({"recording_id": r.recording_id, "m_0": res[0], "m_1": res[1]})
    return pd.DataFrame(lignes)


def forme(largeur: int) -> str:
    return "brusque" if largeur <= 1 else "progressive" if largeur >= 3 else "intermédiaire"


def trouvailles(cfg: dict) -> pd.DataFrame:
    out = dossier(cfg)
    t = pd.read_csv(out / "tri.csv")
    m = pd.read_csv(out / "mesures.csv")
    det = detecter(m)
    rows = []
    for r in det.itertuples():
        a, b = r.m_0, r.m_1
        if a is None and b is None:
            continue
        if a and b:
            piste = (
                "deux pistes"
                if (abs(a["k"] - b["k"]) <= 2 and a["saut"] * b["saut"] > 0)
                else "deux pistes décalées"
            )
        else:
            piste = "piste 0 seule" if a else "piste 1 seule"
        ref = a or b
        rows.append(
            {
                "recording_id": r.recording_id,
                "piste": piste,
                "marche_s": ref["t_s"],
                "sens": "saine puis étouffée" if ref["saut"] < 0 else "étouffée puis saine",
                "facteur": round(10 ** abs(ref["saut"]), 1),
                "forme": forme(ref["largeur"]),
                "largeur_s": ref["largeur"] * TRANCHE_S,
                "score": round(ref["score"], 1),
                "etouffe_hf": round(ref["bas"], 4),
                "sain_hf": round(ref["haut"], 4),
                "piste_marche": 0 if a else 1,
            }
        )
    f = pd.DataFrame(rows).merge(t, on="recording_id", how="left")
    f.to_csv(out / "marches.csv", index=False)
    return f


def file_ecoute(cfg: dict, f: pd.DataFrame) -> None:
    f = f[(f["bord"] == False) & (f["sync"] <= 2)].copy()  # noqa: E712  (bord, mic isolé)
    f = f.sort_values(["forme", "score"], ascending=[False, False])
    # 60 s pour une transition progressive (le séchage se joue sur 20 à 60 s), 30 s sinon.
    dur = np.where(f["forme"] == "progressive", 60.0, 30.0)
    f["offset_s"] = np.clip(f["marche_s"] - dur / 2, 0, f["duration_s"] - dur).round(1)
    q = pd.DataFrame(
        {
            "recording_id": f["recording_id"],
            "path": f["path"],
            "site": f["site"],
            "mic_id": f["mic_id"],
            "start_utc": f["start_utc"],
            "offset_s": f["offset_s"],
            "dur_s": dur,
            "score": f["score"],
            "reason": f"{NOM}/" + f["site"] + "/" + f["forme"],
            "source": "flag",
            "dataset": f["dataset"],
            "piste_0_hf": f["hf0"].round(4),
            "piste_1_hf": np.nan,
            "cause": (
                "piste "
                + f["piste_marche"].astype(str)
                + " : "
                + f["sens"]
                + ", marche à "
                + f["marche_s"].astype(int).astype(str)
                + " s, "
                + f["forme"]
                + " ("
                + f["largeur_s"].astype(int).astype(str)
                + " s), "
                + f["piste"]
                + ", ×"
                + f["facteur"].astype(str)
            ),
        }
    )
    chemin = write_queue(
        cfg,
        q,
        NOM,
        about={
            "origine": "scripts/son_etouffe/bascules_fines.py : tri sur la base, mesure par "
            "tranches de 2 s sur les deux pistes",
            "contenu": "marche d'un facteur 6 ou plus de hf_ratio dans le fichier, hors bords de "
            "campagne et hors pose simultanée de plus de 2 autres micros ; extrait de 30 s",
        },
    )
    print(f"écrit : {chemin} ({len(q)} extraits)")


def degradees(cfg: dict) -> None:
    """File d'écoute des pistes 0 encodées et très dégradées : 30 s au milieu du fichier."""
    con = sqlite3.connect(f"file:{config_path(cfg, 'db').as_posix()}?mode=ro", uri=True)
    marques = ",".join("?" * len(DEGRADEES))
    base = pd.read_sql_query(
        "SELECT recording_id, path, dataset, site, mic_id, start_utc, qc_flags FROM recordings "
        f"WHERE recording_id IN ({marques})",
        con,
        params=list(DEGRADEES),
    )
    con.close()
    ind = base["qc_flags"].map(json.loads)
    q = pd.DataFrame(
        {
            "recording_id": base["recording_id"],
            "path": base["path"],
            "site": base["site"],
            "mic_id": base["mic_id"],
            "start_utc": base["start_utc"],
            "offset_s": 45.0,
            "dur_s": 30.0,
            "score": np.nan,
            "reason": "piste0_encodee_degradee/" + base["site"] + "/" + base["mic_id"],
            "source": "flag",
            "dataset": base["dataset"],
            "piste_0_hf": ind.map(lambda f: f["indices"]["hf_ratio"]).round(4),
            "piste_1_hf": ind.map(lambda f: f["indices_1"]["hf_ratio"]).round(4),
            "cause": "piste 0 encodée sans drapeau, hf_ratio constant sur tout le fichier",
        }
    )
    chemin = write_queue(
        cfg,
        q,
        "piste0_encodee_degradee",
        about={
            "origine": "scripts/son_etouffe/bascules_fines.py degradees",
            "contenu": "les deux pistes 0 les plus étouffées parmi celles encodées sans drapeau et "
            "dont la piste 1 est valable à l'œil des indices ; extrait de 30 s",
        },
    )
    print(f"écrit : {chemin}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("etape", choices=["tri", "mesure", "file", "degradees"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--fils", type=int, default=6)
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument(
        "--prioritaire",
        action="store_true",
        help="mesure : seulement hors bord de campagne et avec au plus 2 autres micros bas",
    )
    args = ap.parse_args()
    cfg = load_config(args.config or default_user_config())
    if args.etape == "tri":
        tri(cfg)
    elif args.etape == "mesure":
        mesure(cfg, args.fils, args.limite, args.prioritaire)
    elif args.etape == "degradees":
        degradees(cfg)
    else:
        f = trouvailles(cfg)
        print(f"{len(f)} fichiers avec une marche\n{f['forme'].value_counts().to_string()}")
        file_ecoute(cfg, f)


if __name__ == "__main__":
    main()
