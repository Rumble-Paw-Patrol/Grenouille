"""Files d'écoute et de réencodage du son étouffé, d'après les mesures de `qc_deux_pistes.py`.

Règle (Léonard, 26/10) : un enregistrement n'est écarté que si **les deux pistes** sont
étouffées (suite d'au moins `muffled_min_run` enregistrements du même micro, piste par piste).
Si une piste est valable il est gardé, avec son drapeau ; douteux, il va à l'écoute.

Écrit, dans `paths.queues` :
- `muffled_perdus`   : deux pistes étouffées, un extrait de 5 s par enregistrement ;
- `muffled_douteux`  : une piste étouffée et l'autre pas franchement claire, ou noté étouffé à
                       l'écoute sans suite assez longue ; un extrait de 5 s par enregistrement ;
- `autres_drapeaux`  : durée anormale / hors relevé (ex-`ecartes`, motifs 3 à 5), enregistrements
                       entiers, durée intacte ;
et `paths.reports/reencodage_piste1/` : les enregistrements gardés grâce à la seule piste 1
(piste 0 étouffée, jamais encodés), à encoder avec `audio.channel` = 1.

Lecture seule de la base. Avec `--remplacer`, les files `ecartes` et `son_etouffe` sont
supprimées une fois les nouvelles écrites.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

import pandas as pd

from blanci.annotation.selection import queue_dir, write_queue
from blanci.core.config import config_path, default_user_config, load_config
from blanci.inputs.qc import flag_raised, parse_flags, positive_recordings

EXTRACT_S = 5.0
# Ratio d'une piste « franchement valable » : le double du seuil d'étouffement (0,05).
CLEAR_HF = 0.10
OTHER_FLAGS = ("silent", "duration_off", "off_campaign")


def classify(rec: pd.DataFrame) -> pd.DataFrame:
    """État de chaque enregistrement mesuré : perdu, douteux, garde_piste_1 (piste 0
    étouffée, piste 1 valable : à encoder), garde_piste_0 (piste 1 étouffée, piste 0 valable),
    ou vide."""
    e0, e1 = rec["etouffe_0"], rec["etouffe_1"]
    clear0, clear1 = rec["hf_ratio_0"] >= CLEAR_HF, rec["hf_ratio_1"] >= CLEAR_HF
    state = pd.Series("", index=rec.index)
    state[rec["note_ecoute"]] = "douteux"  # noté étouffé à l'oreille, sans suite assez longue
    state[e0 & ~e1] = "douteux"
    state[e1 & ~e0] = "douteux"
    state[e0 & ~e1 & clear1] = "garde_piste_1"
    state[e1 & ~e0 & clear0] = "garde_piste_0"
    state[e0 & e1] = "perdu"
    cause = pd.Series("", index=rec.index)
    cause[state == "douteux"] = "noté à l'écoute sans suite"
    cause[(state == "douteux") & e0 & ~e1] = "piste 0 étouffée, piste 1 limite"
    cause[(state == "douteux") & e1 & ~e0] = "piste 1 étouffée, piste 0 limite"
    return rec.assign(etat=state, cause=cause)


def extracts(rec: pd.DataFrame, durations: pd.Series, name: str) -> pd.DataFrame:
    """Un extrait de 5 s au centre de chaque enregistrement, rangé par jeu, site, micro, heure."""
    dur = rec["recording_id"].map(durations).astype(float)
    out = pd.DataFrame(
        {
            "recording_id": rec["recording_id"],
            "path": rec["path"],
            "site": rec["site"],
            "mic_id": rec["mic_id"],
            "start_utc": rec["start_utc"],
            "offset_s": ((dur - EXTRACT_S) / 2).clip(lower=0).round(2),
            "dur_s": dur.clip(upper=EXTRACT_S),
            "score": float("nan"),
            "reason": name + "/" + rec["site"] + "/" + rec["mic_id"],
            "source": "flag",
            "dataset": rec["dataset"],
            "piste_0_hf": rec["hf_ratio_0"].round(4),
            "piste_1_hf": rec["hf_ratio_1"].round(4),
            "cause": rec["cause"],
        }
    )
    return out.sort_values(["dataset", "site", "mic_id", "start_utc"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--remplacer", action="store_true", help="supprime ecartes et son_etouffe")
    args = parser.parse_args()

    cfg = load_config(default_user_config())
    reports = config_path(cfg, "reports")
    measured = pd.read_csv(reports / "son_etouffe" / "enregistrements.csv").fillna(
        {"commentaire_etouffe": ""}
    )
    measured["note_ecoute"] = measured["in_bag_ecoute"] | (measured["commentaire_etouffe"] != "")

    db = Path(config_path(cfg, "db")).as_posix()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    protected = positive_recordings(con)
    rows = con.execute("SELECT recording_id, duration_s, qc_flags, path FROM recordings").fetchall()
    durations = pd.Series({r[0]: r[1] for r in rows})
    flags = {r[0]: parse_flags(r[2]) for r in rows}
    other = {
        rid: ",".join(k for k in OTHER_FLAGS if flag_raised(f, k))
        for rid, f in flags.items()
        if any(flag_raised(f, k) for k in OTHER_FLAGS)
    }

    measured = classify(measured)
    measured["autres"] = measured["recording_id"].map(other).fillna("")
    # Les autres drapeaux passent en premier : l'enregistrement est écarté de toute façon.
    kept = measured[(measured["autres"] == "") & ~measured["recording_id"].isin(protected)]
    lost = kept[kept["etat"] == "perdu"]
    doubtful = kept[kept["etat"] == "douteux"]
    spared_1 = kept[kept["etat"] == "garde_piste_1"]
    spared_0 = kept[kept["etat"] == "garde_piste_0"]

    queues = queue_dir(cfg)
    old = {name: queues / name / "candidats.csv" for name in ("ecartes", "son_etouffe")}
    others = pd.read_csv(old["ecartes"])
    others = others[~others["reason"].str.contains("micro_dans_sac")].copy()
    others["reason"] = others["reason"].str.replace("ecartes/", "autres_drapeaux/", regex=False)

    about = {
        "origine": "scripts/son_etouffe/files_son_etouffe.py, d'après F:/data/reports/son_etouffe/"
    }
    write_queue(
        cfg,
        extracts(lost, durations, "muffled_perdus"),
        "muffled_perdus",
        about
        | {
            "contenu": "deux pistes étouffées (règle des suites sur chaque piste) ; extrait de 5 s "
            "au centre de chaque enregistrement ; à confirmer en basculant micro 1 / micro 2"
        },
    )
    write_queue(
        cfg,
        extracts(doubtful, durations, "muffled_douteux"),
        "muffled_douteux",
        about
        | {
            "contenu": "une piste étouffée et l'autre limite (hf 0,05-0,10) ou non retenue "
            "par la règle des suites, ou noté étouffé à l'écoute sans suite ; extrait de 5 s ; "
            "la colonne `cause` dit pourquoi"
        },
    )
    write_queue(
        cfg,
        others,
        "autres_drapeaux",
        about
        | {
            "contenu": "durée anormale et/ou hors relevé (ex-motifs 3 à 5 de `ecartes`) ; "
            "enregistrements entiers, durée inchangée"
        },
    )

    reencode = spared_1[["recording_id", "path", "dataset", "site", "mic_id", "start_utc"]].assign(
        channel=1,
        hf_ratio_0=spared_1["hf_ratio_0"].round(4),
        hf_ratio_1=spared_1["hf_ratio_1"].round(4),
    )
    folder = reports / "reencodage_piste1"
    folder.mkdir(parents=True, exist_ok=True)
    reencode.sort_values(["dataset", "site", "mic_id", "start_utc"]).to_csv(
        folder / "enregistrements.csv", index=False
    )
    print(f"perdus {len(lost)}, douteux {len(doubtful)}, autres drapeaux {len(others)}")
    print(f"gardés par la piste 1 (à encoder) {len(spared_1)}, par la piste 0 {len(spared_0)}")
    print(f"protégés (A. blanci entendu) : {measured['recording_id'].isin(protected).sum()}")
    print(doubtful["cause"].value_counts().to_string())
    if args.remplacer:
        for name in old:
            shutil.rmtree(queues / name)
        print("supprimées : ecartes, son_etouffe")


if __name__ == "__main__":
    main()
