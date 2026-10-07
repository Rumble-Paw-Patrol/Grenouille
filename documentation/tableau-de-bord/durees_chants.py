"""Durée des chants des cinq espèces d'AnuraSet retenues : quantiles pour la « signature des
chants » du tableau de bord.

    uv run python documentation/tableau-de-bord/durees_chants.py [strong_labels.zip]

Lit les chants datés d'AnuraSet (`strong_labels.zip`, Zenodo 8342596 ; par défaut
`data/external/anuraset/strong_labels.zip`) et écrit `durees_chants.csv` (versionné) : 10e, 25e,
50e, 75e et 90e centiles de la durée par espèce. Mêmes chants que le profil des benchmarks
(`anuraset.species_profile`) : les annotations de plus de 5 s, des chœurs continus, sont exclues ;
médiane et 90e centile y sont donc identiques à `especes.csv` du benchmark 01.
"""

from __future__ import annotations

import sys
from pathlib import Path

from blanci.evaluation.anuraset import read_strong_labels

ICI = Path(__file__).resolve().parent
ESPECES = ["DENMIN", "PITAZU", "PHYCUV", "LEPLAT", "BOAFAB"]
MAX_CHANT_S = 5.0


def main() -> None:
    source = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ICI.parents[1] / "data/external/anuraset/strong_labels.zip"
    )
    chants = read_strong_labels(source)
    chants["duree_s"] = chants["end_s"] - chants["start_s"]
    brefs = chants[(chants["duree_s"] <= MAX_CHANT_S) & chants["species"].isin(ESPECES)]
    q = brefs.groupby("species")["duree_s"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    q.columns = ["p10_s", "q1_s", "mediane_s", "q3_s", "p90_s"]
    q.insert(0, "n_chants_brefs", brefs.groupby("species").size())
    q.reindex(ESPECES).round(4).to_csv(ICI / "durees_chants.csv", index_label="espece")
    print(q.round(3))


if __name__ == "__main__":
    main()
