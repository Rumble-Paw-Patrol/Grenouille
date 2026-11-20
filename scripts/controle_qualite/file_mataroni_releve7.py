"""File d'écoute `mataroni_releve7` : les 18 fichiers du 19/05/2026 face au même micro en janvier.

Relevé 7 de Mataroni (`audio/R7_2026-05_Mataroni/2LA03595_GI07`) : 18 fichiers de 120 s de
2LA03595, le 19/05/2026 de 9 h 30 à 18 h (heure locale), un par demi-heure, quatre mois après la
relève du relevé 3 (13/01/2026). Gardés par `keep` (relevé déclaré dans `qc.surveys`) ; soupçon :
un autre programme ou un autre projet, resté sur la carte SD. Deux indices de réglage : les gains
sont inversés (piste 0 à 18 dB, piste 1 à 6 dB, contre 6 / 18 dB en décembre et janvier) et la
position GPS de l'en-tête est un autre lieu de réglage que celle du relevé 3.

Chaque créneau du 19/05 est suivi du même créneau du même micro (point GI07) au relevé 3, le
12/01/2026 (dernier jour plein de la pose) : comparer le paysage sonore à l'oreille. Pour
comparer à gain égal, écouter la piste à 6 dB : piste 1 au relevé 7, piste 0 au relevé 3
(colonne `cause`). Enregistrements entiers. Les 18 du relevé 7 ont déjà été écoutés le 10/11
(file `hors_programme_et_releve7`) : leurs labels restent, la file ne les écrase pas.

Lecture seule de la base ; écrit la file dans `paths.queues`.

    uv run python scripts/controle_qualite/file_mataroni_releve7.py
"""

from __future__ import annotations

import pandas as pd
from files_apres_prez4 import entiers

from blanci.annotation.selection import write_queue
from blanci.core.config import default_user_config, load_config
from blanci.exploration import inventaire as inv

MICRO = "2LA03595"
RELEVE7 = "audio/R7_2026-05_Mataroni"
RELEVE3 = "Projet blanci 2025/RELEVE 3 Mataroni - 06-13 janv 2026"
REPERE = pd.Timestamp("2026-01-12")  # jour plein du relevé 3, créneaux 06:00-18:30 complets
# Piste à 6 dB de chaque relevé (en-têtes GUANO lus le 20/11/2026 sur les 200 fichiers du micro).
PISTE_6_DB = {RELEVE7: 1, RELEVE3: 0}


def indices(fl: dict, piste: int) -> dict:
    return fl.get("indices" if piste == 0 else "indices_1") or {}


def cause(r: pd.Series) -> str:
    piste = PISTE_6_DB[r["survey"]]
    i = indices(r["fl"], piste)
    gains = "18 / 6 dB" if r["survey"] == RELEVE7 else "6 / 18 dB"
    return (f"gains piste 0 / 1 : {gains} ; écouter la piste {piste} (6 dB) : "
            f"{i.get('rms_dbfs', float('nan')):.1f} dBFS, hf {i.get('hf_ratio', float('nan')):.2f}")


def main() -> None:
    cfg = load_config(default_user_config())
    rec = inv.load_recordings(inv.open_database(cfg), cfg)
    micro = rec[rec["mic_id"] == MICRO]
    r7 = micro[micro["survey"] == RELEVE7].sort_values("start_utc")
    assert len(r7) == 18, f"{len(r7)} fichiers au relevé 7"
    r3 = micro[(micro["survey"] == RELEVE3) & (micro["day"] == REPERE)]
    r3 = r3[r3["slot"].isin(r7["slot"])]
    assert len(r3) == 18, f"{len(r3)} créneaux repères le {REPERE:%d/%m/%Y}"
    rows = pd.concat([r7.assign(role="relevé 7 (19/05/2026)", rang=0),
                      r3.assign(role="repère relevé 3 (12/01/2026)", rang=1)])
    rows = rows.sort_values(["slot", "rang"], ignore_index=True)
    file = entiers(rows, "mataroni_releve7/" + rows["slot"].astype(str))
    file.insert(file.columns.get_loc("source"), "role", rows["role"].to_numpy())
    file["cause"] = rows.apply(cause, axis=1).to_numpy()
    write_queue(cfg, file, "mataroni_releve7", {
        "origine": "scripts/controle_qualite/file_mataroni_releve7.py (20/11/2026)",
        "contenu": f"les 18 fichiers du relevé 7 de Mataroni ({MICRO}, point GI07, 19/05/2026, "
                   "9 h 30 - 18 h locale), chacun suivi du même créneau du même micro au "
                   "relevé 3 le 12/01/2026 ; enregistrements entiers",
        "écoute": "piste à 6 dB : piste 1 au relevé 7 (gains inversés), piste 0 au relevé 3",
        "question": "même lieu, même habitat qu'en janvier, ou autre programme / autre projet "
                    "resté sur la carte SD ? S'il n'est pas du projet : retirer `keep` et "
                    "sortir `audio/R7_2026-05_Mataroni` de `qc.surveys`"})
    print(file[["recording_id", "start_utc", "role", "cause"]].to_string(index=False))


if __name__ == "__main__":
    main()
