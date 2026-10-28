"""Tirage au hasard d'extraits à faire vérifier par un encadrant.

On tire parmi les extraits que l'annotateur a écoutés dans le poste (méthode par intervalles), à
leur dernière version : une annotation remplacée par une séance d'annotation plus récente
(`dataset.add_versions`, `remplace_par`) n'est jamais tirée, sa remplaçante peut l'être. Les
extraits déjà envoyés dans un tirage précédent (`exclude`) sont écartés aussi, pour ne pas faire
vérifier deux fois le même.

La file écrite par `scripts/verification/tirage_verification.py` est une file d'écoute ordinaire
(`candidats.csv`) : `blanci paquet <nom> --file <file>` en fait un paquet à envoyer.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blanci.annotation.workbench import CANDIDATE_COLUMNS
from blanci.inputs.dataset import add_versions, annotated_spans

# Extraits de la méthode par intervalles, écoutés à fond ; « flag » (contrôle rapide d'un
# enregistrement écarté) n'est pas un jugement à faire vérifier.
DEFAULT_SOURCES = ("plan",)
REASON = "verification/hasard"


def _key(recording_id: pd.Series, start_s: pd.Series, end_s: pd.Series) -> pd.Series:
    return (
        recording_id.astype(str)
        + ":"
        + start_s.round(2).astype(str)
        + ":"
        + end_s.round(2).astype(str)
    )


def verification_sample(
    con: sqlite3.Connection,
    annotators: Sequence[str],
    n: int,
    seed: int = 0,
    sources: Sequence[str] = DEFAULT_SOURCES,
    exclude: Sequence[pd.DataFrame] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """`n` extraits au hasard (moins s'il n'en reste pas assez), au format d'une file d'écoute
    (`CANDIDATE_COLUMNS`, `source` = « audit »), avec le décompte de ce qui a été écarté.

    - `annotators` : noms de l'annotateur (casse ignorée) ;
    - `sources` : `source` des extraits tirables (`DEFAULT_SOURCES`) ;
    - `exclude` : files d'écoute déjà envoyées (recording_id, offset_s, dur_s)."""
    mine = {a.casefold() for a in annotators}
    extracts, _ = annotated_spans(con)
    extracts = extracts[extracts["annotator"].fillna("").str.casefold().isin(mine)]
    extracts = add_versions(extracts.assign(annotator=sorted(mine)[0] if mine else ""))
    extracts = extracts[extracts["source"].isin(sources)]
    replaced = extracts["remplace_par"].notna()
    keys = _key(extracts["recording_id"], extracts["start_s"], extracts["end_s"])
    sent = set()
    for queue in exclude:
        sent |= set(
            _key(
                queue["recording_id"],
                queue["offset_s"].astype(float),
                (queue["offset_s"] + queue["dur_s"]).astype(float),
            )
        )
    already_sent = keys.isin(sent)
    pool = extracts[~replaced & ~already_sent]
    rng = np.random.default_rng(seed)
    picked = pool.iloc[np.sort(rng.permutation(len(pool))[:n])] if len(pool) else pool
    sample = pd.DataFrame(
        {
            "recording_id": picked["recording_id"],
            "path": picked["path"],
            "site": picked["site"],
            "mic_id": picked["mic_id"],
            "start_utc": picked["start_utc"],
            "offset_s": picked["start_s"],
            "dur_s": picked["end_s"] - picked["start_s"],
            "score": np.nan,
            "reason": REASON,
            "source": "audit",
        }
    )[CANDIDATE_COLUMNS]
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    info = {
        "extraits_ecoutes": int((~replaced).sum()),
        "versions_remplacees_exclues": int(replaced.sum()),
        "deja_envoyes_exclus": int((already_sent & ~replaced).sum()),
        "tirables": len(pool),
        "tires": len(sample),
    }
    return sample, info


def read_sent(paths: Sequence[Path]) -> list[pd.DataFrame]:
    """Files d'écoute (`candidats.csv`) déjà envoyées, pour `verification_sample(exclude=…)`."""
    return [pd.read_csv(p) for p in paths]
