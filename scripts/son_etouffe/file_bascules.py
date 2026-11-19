"""File d'écoute des bascules en plein enregistrement (notebook 06, § 12).

Un micro qui tombe ou se fait recouvrir bascule le plus souvent entre deux enregistrements. Les
enregistrements qui encadrent une bascule sont découpés en tranches de 10 s (`hf_seg_0`, mesuré
par `scripts/controle_qualite/echantillon_pistes.py`) : une marche d'un facteur 10 de `hf_ratio`
à travers le seuil, avec au moins 2 tranches de chaque côté, est une bascule en direct. Même
règle que le notebook. Chaque candidat est un extrait de 20 s centré sur la marche.

Lecture seule de la base et du CSV de mesures ; n'écrit que la file `bascule_plein_enregistrement`
dans `annotations/files_ecoute/`.

    uv run python scripts/controle_qualite/echantillon_pistes.py   # une fois, disques branchés
    uv run python scripts/son_etouffe/file_bascules.py
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from blanci.annotation.selection import write_queue
from blanci.core.config import config_path, default_user_config, load_config

SEUIL = 0.05  # `qc.muffled_hf_ratio`
NOM = "bascule_plein_enregistrement"


def marche(seg: np.ndarray) -> tuple[float, int | None]:
    """Meilleure marche de la série des tranches : (écart des médianes en log10, rang)."""
    meilleure: tuple[float, int | None] = (0.0, None)
    la = np.log10(seg + 1e-6)
    for k in range(2, len(seg) - 1):
        bas, haut = np.median(la[:k]), np.median(la[k:])
        entre = min(bas, haut) < np.log10(SEUIL) <= max(bas, haut)
        if abs(bas - haut) > abs(meilleure[0]) and entre:
            meilleure = (float(haut - bas), k)
    return meilleure


def main() -> None:
    cfg = load_config(default_user_config())
    mesures = pd.read_csv(config_path(cfg, "reports") / "pistes" / "echantillon_pistes.csv")
    bords = mesures[mesures["motif"].astype(str).str.startswith("bascule")].copy()
    segments = bords["hf_seg_0"].map(lambda s: np.array(json.loads(s)) if isinstance(s, str)
                                     else np.array([]))
    trouve = segments.map(lambda s: marche(s) if len(s) > 4 else (0.0, None))
    bords["saut"] = trouve.map(lambda b: b[0])
    bords["rang"] = trouve.map(lambda b: b[1])
    vivantes = bords[bords["saut"].abs() >= 1].copy()
    print(f"{len(bords)} enregistrements aux bascules, "
          f"{len(vivantes)} avec une marche dans le fichier")

    con = sqlite3.connect(f"file:{config_path(cfg, 'db').as_posix()}?mode=ro", uri=True)
    base = pd.read_sql_query("SELECT recording_id, path, duration_s FROM recordings", con)
    con.close()
    vivantes = vivantes.merge(base, on="recording_id", how="left")
    centre = vivantes["rang"].astype(float) * 10
    vivantes["offset_s"] = np.clip(centre - 10, 0, vivantes["duration_s"] - 20).round(1)
    sens = np.where(vivantes["saut"] < 0, "saine puis étouffée", "étouffée puis saine")
    file_ecoute = pd.DataFrame({
        "recording_id": vivantes["recording_id"], "path": vivantes["path"],
        "site": vivantes["site"],
        "mic_id": vivantes["mic_id"], "start_utc": vivantes["start_utc"],
        "offset_s": vivantes["offset_s"], "dur_s": 20.0, "score": np.nan,
        "reason": f"{NOM}/" + vivantes["site"] + "/" + vivantes["mic_id"], "source": "flag",
        "dataset": vivantes["dataset"], "piste_0_hf": vivantes["hf_ratio_0"].round(4),
        "piste_1_hf": vivantes["hf_ratio_1"].round(4),
        "cause": "",
    })
    file_ecoute["cause"] = [f"piste 0 {s}, marche à {int(r) * 10} s"
                            for s, r in zip(sens, vivantes["rang"], strict=True)]
    file_ecoute = file_ecoute.sort_values(["site", "mic_id", "start_utc"]).reset_index(drop=True)
    chemin = write_queue(cfg, file_ecoute, NOM, about={
        "origine": "scripts/son_etouffe/file_bascules.py, d'après pistes/echantillon_pistes.csv",
        "contenu": "enregistrements où hf_ratio (piste 0) fait une marche d'un facteur 10 à "
                   "travers le seuil pendant le fichier ; extrait de 20 s centré sur la marche",
    })
    print(f"écrit : {chemin} ({len(file_ecoute)} extraits)")


if __name__ == "__main__":
    main()
