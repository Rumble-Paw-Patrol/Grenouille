"""Son étouffé, piste par piste : `hf_ratio` et `rms_dbfs` sur les deux canaux des micros
signalés `muffled` (calculé, noté à l'écoute ou dit étouffé dans un commentaire), sur toute
leur campagne, puis la règle des suites (`muffled_runs`, n° 194) piste par piste.

`blanci qc` ne lit que `audio.channel` (piste 0, micro 1 à 6 dB) : un micro étouffé sur la
seule piste 1 n'est jamais vu, et un micro écarté pour sa piste 0 a peut-être une piste 1
valable. Lecture seule : la base est ouverte en `mode=ro`, rien n'est écrit dans
`recordings.qc_flags` ni sur les disques d'audio. Sorties : `paths.reports/son_etouffe/`
(mesures, enregistrements, suites, épisodes, micros) et la file `paths.queues/son_etouffe/`.

    uv run python scripts/son_etouffe/qc_deux_pistes.py                 # lit l'audio (reprenable), puis tout
    uv run python scripts/son_etouffe/qc_deux_pistes.py --sans-audio    # refait tableaux et file seulement
    uv run python scripts/son_etouffe/qc_deux_pistes.py --micros Kaw/SMA10283 --limite 20 --sortie <dossier>

Chaque fichier est lu une fois : `qc_indices` sur chaque colonne de `sf.read(always_2d=True)`
donne exactement `qc_indices(*load_audio(path, 0))` et `qc_indices(*load_audio(path, 1))`.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import soundfile as sf

from blanci.annotation.selection import write_queue
from blanci.core.ahead import ahead
from blanci.core.config import config_path, default_user_config, load_config
from blanci.core.locate import locate
from blanci.inputs.qc import muffled_candidate, muffled_runs, parse_flags, qc_indices

# Piste « valable » : ratio d'au moins deux fois le seuil d'`muffled`. Entre le seuil et ce
# plancher, la piste n'est pas candidate sans être franchement claire : douteux.
CLEAR_FACTOR = 2.0
# Part d'une suite où une piste doit être étouffée (ou valable) pour le dire de la suite.
MOSTLY = 0.8
# Deux suites du même micro et du même état forment un épisode si rien ne les sépare que la
# pause du programme (nuit : 19:30 → 05:00, heure locale) : pas d'enregistrement entre elles.
EPISODE_GAP = pd.Timedelta(hours=12)
# Commentaires d'écoute qui disent un son étouffé (le poste n'a pas de réponse pour ça).
MUFFLED_COMMENT = re.compile(r"touff|dans (?:le |un )?sac", re.IGNORECASE)
EXTRACT_S = 30.0
MEASURES = ["recording_id", "fichier", "hf_ratio_0", "rms_dbfs_0", "hf_ratio_1", "rms_dbfs_1"]


def read_only(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)


def noted_by_comment(con: sqlite3.Connection) -> pd.Series:
    """{enregistrement : commentaires d'écoute (fenêtre ou extrait) qui disent étouffé / sac}."""
    rows = con.execute(
        """SELECT w.recording_id, l.conditions FROM labels l JOIN windows w USING (window_id)
           UNION ALL SELECT recording_id, conditions FROM spans"""
    ).fetchall()
    found: dict[str, list[str]] = {}
    for rid, conditions in rows:
        c = json.loads(conditions) if conditions else {}
        comment = c.get("comment") or ""
        if MUFFLED_COMMENT.search(comment):
            channel = c.get("channel_listened")
            heard = "" if channel is None else f" (piste {channel} à l'écoute)"
            found.setdefault(rid, []).append(comment + heard)
    return pd.Series({rid: " | ".join(texts) for rid, texts in found.items()}, dtype=str)


def selection(con: sqlite3.Connection, micros: list[str] | None = None) -> pd.DataFrame:
    """Enregistrements à mesurer : campagne entière (jeu, site, micro) de tout micro qui a un
    enregistrement `muffled` calculé, noté `artefact_muffled` ou dit étouffé en commentaire
    (ou des micros `site/micro` donnés)."""
    rec = pd.read_sql_query(
        "SELECT recording_id, path, dataset, site, mic_id, start_utc, duration_s, sample_rate, "
        "channels, qc_flags FROM recordings",
        con,
    )
    flags = rec.pop("qc_flags").map(parse_flags)
    rec["in_bag_calcule"] = flags.map(lambda f: bool(f.get("muffled")))  # piste 0 (avant le n° 203)
    rec["in_bag_ecoute"] = flags.map(lambda f: "muffled" in f.get("annotated", ()))
    rec["ecarte_autre"] = flags.map(
        lambda f: ",".join(k for k in ("silent", "duration_off", "off_campaign") if f.get(k))
    )
    rec["hf_ratio_0_base"] = flags.map(lambda f: f.get("indices", {}).get("hf_ratio"))
    rec["commentaire_etouffe"] = rec["recording_id"].map(noted_by_comment(con)).fillna("")
    rec["note_ecoute"] = rec["in_bag_ecoute"] | (rec["commentaire_etouffe"] != "")
    if micros:
        seeds = rec[(rec["site"] + "/" + rec["mic_id"]).isin(micros)]
    else:
        seeds = rec[rec["in_bag_calcule"] | rec["note_ecoute"]]
    keys = seeds[["dataset", "site", "mic_id"]].drop_duplicates()
    return rec.merge(keys, on=["dataset", "site", "mic_id"]).sort_values(
        ["dataset", "site", "mic_id", "start_utc"], ignore_index=True
    )


def measure(cfg: dict, todo: pd.DataFrame, out: Path, workers: int) -> None:
    """Indices des deux pistes, ajoutés à `out` au fil de l'eau ; reprenable (ce qui y est
    déjà est sauté)."""
    done = set(pd.read_csv(out)["recording_id"]) if out.exists() else set()
    todo = todo[~todo["recording_id"].isin(done)]
    print(f"{len(done)} déjà mesurés, {len(todo)} à lire", flush=True)
    raw, renamed = config_path(cfg, "raw"), cfg["audio"].get("renamed")

    def prepare(rec) -> dict | None:
        full = locate(rec.path, raw, renamed=renamed)
        if full is None:
            return None
        try:
            wav, sr = sf.read(full, dtype="float32", always_2d=True)
        except Exception:  # fichier illisible : déjà signalé à l'inventaire
            return None
        row = {"recording_id": rec.recording_id, "fichier": str(full)}
        for c in range(min(2, wav.shape[1])):
            indices = qc_indices(wav[:, c], sr)
            row[f"hf_ratio_{c}"], row[f"rms_dbfs_{c}"] = indices["hf_ratio"], indices["rms_dbfs"]
        return row

    missing, start = 0, perf_counter()
    write_header = not out.exists()
    with out.open("a", encoding="utf-8", newline="") as f:
        for n, (_, row) in enumerate(ahead(todo.itertuples(), prepare, 2 * workers, workers), 1):
            if row is None:
                missing += 1
                continue
            pd.DataFrame([row]).reindex(columns=MEASURES).to_csv(
                f, header=write_header, index=False
            )
            write_header = False
            if n % 200 == 0:
                f.flush()
                rate = n / (perf_counter() - start)
                left = (len(todo) - n) / rate / 60
                print(
                    f"  {n}/{len(todo)}, {rate:.1f} par seconde, reste ≈ {left:.0f} min", flush=True
                )
    if missing:
        print(f"{missing} enregistrements introuvables ou illisibles (disque non branché ?)")


def track_flags(rec: pd.DataFrame, qc: dict) -> pd.DataFrame:
    """Piste par piste : candidat (`qc_flags`, seuil actuel) et étouffé (candidat retenu par
    la règle des suites, `muffled_runs`). Une piste absente (fichier mono) n'est pas candidate."""
    rec = rec.copy()
    for c in (0, 1):
        measured = rec[f"hf_ratio_{c}"].notna()
        rec[f"candidat_{c}"] = [
            bool(m) and muffled_candidate({"hf_ratio": h, "rms_dbfs": r}, qc)
            for m, h, r in zip(measured, rec[f"hf_ratio_{c}"], rec[f"rms_dbfs_{c}"], strict=True)
        ]
        kept = muffled_runs(
            rec[measured].assign(candidate=rec.loc[measured, f"candidat_{c}"]),
            int(qc.get("muffled_min_run", 4)),
            float(qc.get("muffled_max_gap_min", 60.0)),
        )
        rec[f"etouffe_{c}"] = rec["recording_id"].isin(kept)
    return rec


def number_runs(rec: pd.DataFrame, qc: dict) -> pd.DataFrame:
    """Suites (`suite`) : enregistrements consécutifs du même micro, sans trou de plus de
    `muffled_max_gap_min`, où une piste au moins est candidate."""
    gap = pd.Timedelta(minutes=float(qc.get("muffled_max_gap_min", 60.0)))
    rec = rec.assign(t=pd.to_datetime(rec["start_utc"], utc=True, errors="coerce"), suite="")
    for (_, site, mic), mic_rows in rec.sort_values("t").groupby(["dataset", "site", "mic_id"]):
        hit = mic_rows["candidat_0"] | mic_rows["candidat_1"]
        number = (~hit | (mic_rows["t"].diff() > gap)).cumsum()[hit]
        for k, (_, ids) in enumerate(number.groupby(number), 1):
            rec.loc[ids.index, "suite"] = f"{site}/{mic}/s{k:02d}"
    return rec


def clear(h: float, qc: dict) -> bool:
    return bool(pd.notna(h) and h >= CLEAR_FACTOR * float(qc["muffled_hf_ratio"]))


def state(part: pd.DataFrame, qc: dict) -> str:
    """État d'une suite : piste(s) étouffée(s) d'après la règle des suites, et l'autre
    franchement valable. Trop courte pour la règle (< `muffled_min_run`) : « courte »."""
    e0, e1 = part["etouffe_0"].mean(), part["etouffe_1"].mean()
    if e0 == 0 and e1 == 0:
        return "courte"
    if e0 >= MOSTLY and e1 >= MOSTLY:
        return "deux pistes"
    for muffled, other in ((e0, 1), (e1, 0)):
        if muffled >= MOSTLY and part[f"etouffe_{other}"].sum() == 0:
            ok = part[f"hf_ratio_{other}"].map(lambda h: clear(h, qc)).mean() >= MOSTLY
            return f"piste {1 - other} seule" if ok else "douteux"
    return "douteux"


def summarize(part: pd.DataFrame) -> dict:
    return {
        "dataset": part["dataset"].iloc[0],
        "site": part["site"].iloc[0],
        "mic_id": part["mic_id"].iloc[0],
        "debut_utc": part["start_utc"].min(),
        "fin_utc": part["start_utc"].max(),
        "n": len(part),
        "candidats_0": int(part["candidat_0"].sum()),
        "candidats_1": int(part["candidat_1"].sum()),
        "etouffes_0": int(part["etouffe_0"].sum()),
        "etouffes_1": int(part["etouffe_1"].sum()),
        "in_bag_calcule": int(part["in_bag_calcule"].sum()),
        "notes_ecoute": int(part["note_ecoute"].sum()),
        "hf_ratio_0_med": round(part["hf_ratio_0"].median(), 4),
        "hf_ratio_1_med": round(part["hf_ratio_1"].median(), 4),
        "rms_dbfs_0_med": round(part["rms_dbfs_0"].median(), 1),
        "rms_dbfs_1_med": round(part["rms_dbfs_1"].median(), 1),
    }


def runs_table(rec: pd.DataFrame, qc: dict) -> pd.DataFrame:
    rows = [
        {"suite": name, **summarize(part), "etat": state(part, qc)}
        for name, part in rec[rec["suite"] != ""].groupby("suite", sort=False)
    ]
    return pd.DataFrame(rows)


def number_episodes(rec: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Épisodes (`episode`) : suites voisines du même micro et du même état que seule la
    pause du programme sépare (aucun enregistrement entre elles, moins de `EPISODE_GAP`)."""
    rec = rec.assign(episode="")
    by_run = runs.set_index("suite")["etat"]
    for (_, site, mic), mic_rows in rec.sort_values("t").groupby(["dataset", "site", "mic_id"]):
        k, last, last_t, last_pos = 0, None, None, None
        positions = {rid: i for i, rid in enumerate(mic_rows["recording_id"])}
        for name, part in mic_rows[mic_rows["suite"] != ""].groupby("suite", sort=False):
            first_pos = positions[part["recording_id"].iloc[0]]
            joined = (
                last is not None
                and by_run[name] == by_run[last]
                and first_pos == last_pos + 1
                and part["t"].iloc[0] - last_t <= EPISODE_GAP
            )
            k += not joined
            rec.loc[part.index, "episode"] = f"{site}/{mic}/e{k:02d}"
            last, last_t = name, part["t"].iloc[-1]
            last_pos = positions[part["recording_id"].iloc[-1]]
    return rec


def episodes_table(rec: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    by_run = runs.set_index("suite")["etat"]
    rows = []
    for name, part in rec[rec["episode"] != ""].groupby("episode", sort=False):
        rows.append(
            {
                "episode": name,
                **summarize(part),
                "suites": part["suite"].nunique(),
                "etat": by_run[part["suite"].iloc[0]],
            }
        )
    return pd.DataFrame(rows)


CONCLUSIONS = {
    "deux pistes": "perdu",
    "piste 0 seule": "récupérable par la piste 1",
    "piste 1 seule": "récupérable par la piste 0",
    "douteux": "douteux",
}


def micros_table(rec: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, site, mic), part in rec.groupby(["dataset", "site", "mic_id"]):
        mine = episodes[(episodes["site"] == site) & (episodes["mic_id"] == mic)]
        real = mine[mine["etat"] != "courte"]
        n_by = real.groupby("etat")["n"].sum()
        kinds = {CONCLUSIONS[e] for e in real["etat"]}
        # Noté étouffé à l'oreille sans suite assez longue pour la règle : à réécouter.
        heard = part["note_ecoute"] & ~part["episode"].isin(real["episode"])
        kinds = sorted(kinds | ({"douteux"} if heard.any() else set()))
        outside = part[part["suite"] == ""]
        if not kinds:
            conclusion = "aucune suite"
        elif len(kinds) == 1:
            conclusion = kinds[0]
        else:
            conclusion = "mixte : " + ", ".join(kinds)
        rows.append(
            {
                "dataset": ds,
                "site": site,
                "mic_id": mic,
                "enregistrements": len(part),
                "mesures": int(part["hf_ratio_0"].notna().sum()),
                "in_bag_calcule": int(part["in_bag_calcule"].sum()),
                "notes_ecoute": int(part["note_ecoute"].sum()),
                "episodes": len(real),
                "suites_courtes": int((mine["etat"] == "courte").sum()),
                **{f"n_{e.replace(' ', '_')}": int(n_by.get(e, 0)) for e in CONCLUSIONS},
                "debut": real["debut_utc"].min() if len(real) else "",
                "fin": real["fin_utc"].max() if len(real) else "",
                "hf_ratio_0_hors_suites": round(outside["hf_ratio_0"].median(), 4),
                "hf_ratio_1_hors_suites": round(outside["hf_ratio_1"].median(), 4),
                "conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows)


PRIORITY = {"piste 1 seule": 0, "piste 0 seule": 1, "douteux": 2, "courte": 3, "deux pistes": 4}


def listening_queue(rec: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Un extrait par épisode (30 s au centre d'un enregistrement noté à l'écoute s'il y en a,
    sinon de celui du milieu), plus chaque enregistrement dit étouffé à l'écoute hors de tout
    épisode ; les cas « une seule piste étouffée » d'abord."""
    picks = []
    for e in episodes.itertuples():
        if e.etat == "courte" and not e.notes_ecoute:
            continue
        part = rec[rec["episode"] == e.episode].sort_values("t")
        pool = part[part["note_ecoute"]] if part["note_ecoute"].any() else part
        picks.append((pool.iloc[len(pool) // 2], e.etat, e.episode))
    lone = rec[(rec["episode"] == "") & rec["note_ecoute"]]
    picks += [(row, "noté à l'écoute", "") for _, row in lone.iterrows()]
    if not picks:
        return pd.DataFrame()
    rows = [r for r, _, _ in picks]
    out = pd.DataFrame(
        {
            "recording_id": [r.recording_id for r in rows],
            "path": [r.path for r in rows],
            "site": [r.site for r in rows],
            "mic_id": [r.mic_id for r in rows],
            "start_utc": [r.start_utc for r in rows],
            "offset_s": [max(0.0, (float(r.duration_s) - EXTRACT_S) / 2) for r in rows],
            "dur_s": EXTRACT_S,
            "score": np.nan,
            "reason": [f"son_etouffe/{etat.replace(' ', '_')}" for _, etat, _ in picks],
            "source": "flag",
            "episode": [e for _, _, e in picks],
            "hf_ratio_0": [round(r.hf_ratio_0, 4) for r in rows],
            "hf_ratio_1": [round(r.hf_ratio_1, 4) for r in rows],
        }
    )
    order = out["reason"].str.removeprefix("son_etouffe/").str.replace("_", " ").map(PRIORITY)
    out = out.assign(o=order.fillna(2.5)).sort_values(["o", "site", "mic_id", "start_utc"])
    return out.drop(columns="o").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sans-audio", action="store_true", help="ne relit pas l'audio")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--micros", nargs="*", help="site/micro à mesurer à la place")
    parser.add_argument("--limite", type=int, default=0, help="essai : N enregistrements")
    parser.add_argument("--sortie", type=Path, help="essai : tout écrire dans ce dossier")
    args = parser.parse_args()

    cfg = load_config(default_user_config())
    qc = cfg["qc"]
    folder = args.sortie or config_path(cfg, "reports") / "son_etouffe"
    folder.mkdir(parents=True, exist_ok=True)
    measured = folder / "mesures_brutes.csv"

    with read_only(config_path(cfg, "db")) as con:
        rec = selection(con, args.micros)
    if args.limite:
        rec = rec.head(args.limite)
    n_micros = rec.groupby(["site", "mic_id"]).ngroups
    print(f"{len(rec)} enregistrements, {n_micros} micros", flush=True)
    if not args.sans_audio:
        measure(cfg, rec, measured, args.workers)
    if not measured.exists():
        sys.exit("aucune mesure : brancher le disque de l'audio et relancer sans --sans-audio")

    rec = rec.merge(pd.read_csv(measured), on="recording_id", how="left")
    complete = rec["hf_ratio_0"].notna().all()
    print(f"{rec['hf_ratio_0'].notna().sum()} mesurés sur {len(rec)}", flush=True)
    both = rec.dropna(subset=["hf_ratio_0", "hf_ratio_0_base"])
    if len(both):
        gap = (both["hf_ratio_0"] - both["hf_ratio_0_base"]).abs().max()
        print(f"piste 0 contre les indices de `blanci qc` : écart max {gap:.1e} ({len(both)})")
    rec = number_runs(track_flags(rec, qc), qc)
    runs = runs_table(rec, qc)
    rec = number_episodes(rec, runs) if len(runs) else rec.assign(episode="")
    episodes = episodes_table(rec, runs) if len(runs) else pd.DataFrame(columns=["etat"])
    micros = micros_table(rec, episodes)

    keep = [
        "recording_id",
        "dataset",
        "site",
        "mic_id",
        "start_utc",
        "path",
        "channels",
        "sample_rate",
        "ecarte_autre",
        "in_bag_calcule",
        "in_bag_ecoute",
        "commentaire_etouffe",
        "hf_ratio_0_base",
        "hf_ratio_0",
        "rms_dbfs_0",
        "hf_ratio_1",
        "rms_dbfs_1",
        "candidat_0",
        "candidat_1",
        "etouffe_0",
        "etouffe_1",
        "suite",
        "episode",
    ]
    rec[keep].to_csv(folder / "enregistrements.csv", index=False)
    runs.to_csv(folder / "suites.csv", index=False)
    episodes.to_csv(folder / "episodes.csv", index=False)
    micros.to_csv(folder / "micros.csv", index=False)
    print(f"écrit : {folder}")

    if not complete:
        print("file d'écoute non écrite : des enregistrements ne sont pas mesurés")
    elif (queue := listening_queue(rec, episodes)).empty:
        print("file d'écoute non écrite : aucun épisode à écouter")
    else:
        target = cfg if not args.sortie else cfg | {"paths": cfg["paths"] | {"queues": str(folder)}}
        about = {
            "méthode": "suites `muffled` recalculées piste par piste (scripts/son_etouffe/qc_deux_pistes.py), "
            "un extrait de 30 s par épisode, cas à une seule piste étouffée d'abord",
            "mesures": str(folder),
        }
        path = write_queue(target, queue, "son_etouffe", about)
        with (path.parent / "LISEZMOI.md").open("a", encoding="utf-8") as f:
            f.write(HOW_TO_LISTEN)
        print(f"file : {path} ({len(queue)} extraits)")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(micros.to_string(index=False))


HOW_TO_LISTEN = """## Comment écouter

Chaque extrait représente un épisode : une suite d'enregistrements consécutifs du même micro
où une piste au moins a `hf_ratio` < 0,05 (énergie au-dessus de 2 kHz / énergie totale),
règle des suites appliquée piste par piste (`scripts/son_etouffe/qc_deux_pistes.py`). Colonnes
`hf_ratio_0` et `hf_ratio_1` : valeur de l'enregistrement choisi, piste 0 = micro 1 (6 dB),
piste 1 = micro 2 (18 dB) dans le poste.

Pour chaque extrait, écouter **les deux micros** (bascule micro 1 / micro 2 du poste) et
écrire dans le commentaire, mot pour mot, l'un de :

- `étouffé : micro 1` (piste 0 étouffée, micro 2 valable)
- `étouffé : micro 2` (piste 1 étouffée, micro 1 valable)
- `étouffé : les deux`
- `pas étouffé`

Le label et les classes se cochent comme d'habitude (ce qu'on entend sur le micro valable).
Le poste n'a pas de réponse « son étouffé » : ce commentaire est relu par le script, rien
n'est encore écrit dans les drapeaux de la base.

Ordre : `piste_1_seule` (jamais signalés : seul le micro 2 est étouffé), `piste_0_seule`
(écartés aujourd'hui, micro 2 peut-être valable), `douteux`, `noté_à_l'écoute` et `courte`
(notés à l'oreille sans suite retenue), puis `deux_pistes` (perdus, pour contrôle).
"""


if __name__ == "__main__":
    main()
