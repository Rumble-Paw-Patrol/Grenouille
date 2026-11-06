"""Vérifie les doublons écartés à l'inventaire : même nom suffit-il ? Compare le contenu.

Jusqu'au 05/11/2026, l'inventaire écartait un fichier dès que son nom (à la casse près) était
déjà inventorié sous un autre chemin encore présent : les cartes SD rapportées d'un relevé
gardent les fichiers du précédent. Le nom seul ne prouve rien : deux enregistrements
différents peuvent porter le même nom (horloge remise à zéro, numéro de série recopié). Ce
script relit chaque paire de `paths.reports/ingest_duplicates_*.csv` (écarté, conservé) et
compare ce qui identifie un enregistrement (`ingest.same_recording`) : nombre d'échantillons,
fréquence, pistes, numéro de série et horodatage GUANO, empreinte de la première seconde
décodée. Avec `--disque <racine>`, cherche aussi les noms répétés sous un dossier.

Lecture seule. Les fichiers écartés ne sont que sur le disque d'origine (Biophonia) : le
brancher. Une paire dont une copie est introuvable est notée « introuvable ».

    uv run python scripts/controle_qualite/verifier_doublons.py
    uv run python scripts/controle_qualite/verifier_doublons.py --disque F:/audio

Sortie : `paths.reports/doublons/verification_doublons[_<disque>].csv` (une ligne par paire, verdict
`identiques`, `conflit` avec la raison, ou `introuvable`).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from blanci.core.config import config_path, default_user_config, load_config
from blanci.core.locate import locate
from blanci.inputs.ingest import iter_audio_files, same_recording


def recorded_pairs(reports: Path) -> pd.DataFrame:
    files = sorted(reports.glob("ingest_duplicates_*.csv"))
    frames = [pd.read_csv(f).assign(rapport=f.name) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def repeated_names(root: Path) -> pd.DataFrame:
    """Paires (premier vu, suivant) de fichiers du même nom sous `root`, dans l'ordre naturel."""
    seen: dict[str, list[Path]] = defaultdict(list)
    for path in iter_audio_files(root):
        seen[path.stem.lower()].append(path)
    rows = [
        {"ecarte": str(p), "conserve": str(paths[0]), "rapport": f"disque {root}"}
        for paths in seen.values()
        if len(paths) > 1
        for p in paths[1:]
    ]
    return pd.DataFrame(rows, columns=["ecarte", "conserve", "rapport"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--disque", type=Path, default=None, help="dossier des noms répétés")
    args = ap.parse_args()
    cfg = load_config(args.config or default_user_config())
    reports = config_path(cfg, "reports")
    raw, renamed = config_path(cfg, "raw"), cfg["audio"].get("renamed")
    pairs = repeated_names(args.disque) if args.disque else recorded_pairs(reports)
    print(f"{len(pairs)} paires à vérifier")

    def find(p: str) -> Path | None:
        path = Path(p)
        return path if path.is_absolute() and path.is_file() else locate(p, raw, renamed=renamed)

    verdicts, reasons = [], []
    for n, row in enumerate(pairs.itertuples(), 1):
        a, b = find(row.ecarte), find(row.conserve)
        if a is None or b is None:
            verdicts.append("introuvable")
            reasons.append("écarté absent" if a is None else "conservé absent")
        else:
            try:
                same, why = same_recording(a, b)
            except Exception as exc:  # fichier illisible
                same, why = False, f"illisible : {type(exc).__name__}: {exc}"
            verdicts.append("identiques" if same else "conflit")
            reasons.append(why)
        if n % 500 == 0:
            print(f"  {n}/{len(pairs)}", flush=True)
    out = pairs.assign(verdict=verdicts, raison=reasons)
    suffix = f"_{args.disque.drive.rstrip(':') or args.disque.name}" if args.disque else ""
    target = reports / "doublons" / f"verification_doublons{suffix}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    print(out["verdict"].value_counts().to_string())
    print(f"écrit : {target}")


if __name__ == "__main__":
    main()
