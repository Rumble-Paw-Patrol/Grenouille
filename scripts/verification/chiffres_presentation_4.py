"""Rejoue les chiffres de la présentation n° 4 sur la base d'aujourd'hui, et dit d'où ils viennent.

Chaque chiffre est recalculé par une fonction de `blanci/exploration/inventaire.py`
(`presentation_4_numbers`) ; ce script ne calcule rien lui-même. Il lit, en lecture seule :

- la base (`paths.db`) ;
- s'ils sont lisibles, deux rapports de `paths.reports` : `pistes/echantillon_pistes.csv`
  (gains, diapo 6) et `ingest_duplicates_*.csv` (doublons, diapo 8) ;
- `documentation/tableau-de-bord/structure.yaml` (les relevés empilés sur leur site).

Il écrit un tableau Markdown daté, avec l'empreinte de la base et le commit du code : deux
personnes qui obtiennent la même empreinte ont la même base, et doivent trouver les mêmes chiffres.

    uv run python scripts/verification/chiffres_presentation_4.py
    options : --config <fichier>, --sortie <fichier .md>
"""

from __future__ import annotations

import argparse
import subprocess
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from blanci.core.config import config_path, default_user_config, load_config
from blanci.exploration import inventaire as inv

ROOT = Path(__file__).resolve().parents[2]
SORTIE = ROOT / "documentation" / "prez" / "chiffres_presentation_4.md"


def sans_accents(nom: str) -> str:
    """« Patawa Est » → « PatawaEst », « Trésor » → « Tresor » : les noms de site de la base."""
    nom = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode()
    return nom.replace(" ", "")


def commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=False).stdout.strip()

    sale = git("status", "--porcelain", "--", "blanci/exploration/inventaire.py")
    return git("rev-parse", "--short", "HEAD") + (" (inventaire.py modifié)" if sale else "")


def valeur(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    v = float(v)
    texte = f"{v:,.0f}" if v == round(v) else f"{v:,.3f}".rstrip("0")
    return texte.replace(",", " ").replace(".", ",")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--sortie", type=Path, default=SORTIE)
    args = ap.parse_args()
    cfg = load_config(args.config or default_user_config())
    con = inv.open_database(cfg)
    reports = config_path(cfg, "reports")
    lus, absents = [], []
    pistes = duplicates = None
    csv = reports / "pistes" / "echantillon_pistes.csv"
    if csv.is_file():
        pistes = pd.read_csv(csv)
        lus.append("`pistes/echantillon_pistes.csv`")
    else:
        absents.append("`pistes/echantillon_pistes.csv`")
    rapports = sorted(reports.glob("ingest_duplicates_*.csv")) if reports.is_dir() else []
    if rapports:
        duplicates = pd.concat([pd.read_csv(f) for f in rapports], ignore_index=True)
        lus.append(", ".join(f"`{f.name}`" for f in rapports))
    else:
        absents.append("`ingest_duplicates_*.csv`")
    structure = ROOT / "documentation" / "tableau-de-bord" / "structure.yaml"
    lieux = {sans_accents(k): sans_accents(v) for k, v in
             yaml.safe_load(structure.read_text(encoding="utf-8"))["lieux"].items()}

    table = inv.presentation_4_numbers(con, cfg, lieux=lieux, pistes=pistes,
                                       duplicates=duplicates)
    fp = inv.database_fingerprint(con)
    con.close()

    lignes = [
        "# Chiffres de la présentation n° 4, rejoués",
        "",
        f"Rejoué le {datetime.now():%d/%m/%Y à %H h %M} par "
        "`uv run python scripts/verification/chiffres_presentation_4.py`, code au commit "
        f"`{commit()}`. Chaque chiffre vient de la fonction citée, dans "
        "`blanci/exploration/inventaire.py` (fiches : `blanci/exploration/inventaire.md`).",
        "",
        f"Base : `{config_path(cfg, 'db').name}`, schéma v{fp['user_version']}, "
        f"{valeur(fp['rows']['recordings'])} lignes dans `recordings`, empreinte de l'inventaire "
        f"`{fp['inventory_sha256'][:16]}`, des drapeaux `{fp['flags_sha256'][:16]}`, dernier "
        f"extrait écouté le {fp['last_span'] or '—'}. Même empreinte = même base : les chiffres "
        "doivent être les mêmes.",
        "",
        "Rapports lus : " + (", ".join(lus) or "aucun") + ". Absents : "
        + (", ".join(absents) or "aucun") + ".",
        "",
        "| Diapo | Chiffre | Annoncé | Recalculé | Statut | Fonction |",
        "|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        lignes.append(f"| {r.diapo} | {r.chiffre} | {valeur(r.annonce)} | "
                      f"{valeur(r.recalcule)} | {r.statut} | `{r.fonction}` |")
    lignes.append("")
    args.sortie.write_text("\n".join(lignes), encoding="utf-8")
    print(table["statut"].value_counts().to_string())
    print(f"écrit : {args.sortie}")


if __name__ == "__main__":
    main()
