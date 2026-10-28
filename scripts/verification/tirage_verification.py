"""Tire au hasard des extraits déjà annotés, à faire vérifier par un encadrant.

Parmi les extraits écoutés dans le poste (à leur dernière version), tire `--n` extraits sans
remise. Sont exclus : les extraits réécoutés (voir `reannotations.csv` dans `annotations/`),
les extraits des tirages déjà envoyés (`--exclure`) et, par défaut, les extraits de contrôle
rapide (`flag`). Écrit une file d'écoute dans `paths.queues/<nom>/` ; elle devient un paquet
avec `blanci paquet <nom> --file <nom>`.

Lecture seule de la base.

    uv run python scripts/verification/tirage_verification.py verif_1 --n 50 --seed 1
    uv run python scripts/verification/tirage_verification.py verif_2 --n 50 --exclure verif_1
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from blanci.annotation.selection import queue_dir, write_queue
from blanci.annotation.verification import DEFAULT_SOURCES, read_sent, verification_sample
from blanci.core.config import config_path, default_user_config, load_config

MY_NAMES = ("Léonard", "L")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("nom", help="nom de la file écrite dans paths.queues")
    parser.add_argument("--n", type=int, required=True, help="nombre d'extraits à tirer")
    parser.add_argument("--seed", type=int, default=0, help="graine du tirage (défaut 0)")
    parser.add_argument(
        "--annotateur", nargs="+", default=list(MY_NAMES), help="défaut : Léonard, L"
    )
    parser.add_argument("--source", nargs="+", default=list(DEFAULT_SOURCES), help="défaut : plan")
    parser.add_argument("--exclure", nargs="*", default=[], help="files déjà envoyées (noms)")
    args = parser.parse_args()

    cfg = load_config(default_user_config())
    queues = queue_dir(cfg)
    if (queues / args.nom).exists():
        raise SystemExit(f"la file existe déjà : {queues / args.nom} (autre nom ou la supprimer)")
    previous = read_sent([queues / name / "candidats.csv" for name in args.exclure])

    db = Path(config_path(cfg, "db")).as_posix()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    sample, info = verification_sample(
        con, args.annotateur, args.n, args.seed, args.source, previous
    )
    if sample.empty:
        raise SystemExit(f"aucun extrait tirable ({info})")
    about = {
        "origine": "scripts/verification/tirage_verification.py",
        "annotateur": ", ".join(args.annotateur),
        "graine": args.seed,
        "sources": ", ".join(args.source),
        "déjà envoyés (exclus)": ", ".join(args.exclure),
        **{k.replace("_", " "): v for k, v in info.items()},
    }
    path = write_queue(cfg, sample, args.nom, about)
    print(f"{info['tires']} extraits tirés sur {info['tirables']} tirables : {path}")
    n_re, n_sent = info["reannotes_exclus"], info["deja_envoyes_exclus"]
    print(f"  écartés : {n_re} réannotés, {n_sent} déjà envoyés")
    print(f"Paquet : uv run blanci paquet {args.nom} --file {args.nom}")
    if info["tires"] < args.n:
        print(f"  Attention : {info['tires']} extraits seulement sur {args.n} demandés.")


if __name__ == "__main__":
    main()
