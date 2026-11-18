"""Affiche les blocs bruts d'un fichier WAV de Song Meter, tels qu'ils sont écrits : `fmt `
(pistes, fréquence), `data` (le son, donc la durée) et `guan` (en-tête GUANO, texte brut).
Sert à la diapo 5 de la présentation n° 4 (le fichier avant l'inventaire). Lecture seule.

    uv run python documentation/prez/scripts/entete_wav.py FICHIER.wav
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path


def blocs(path: Path) -> None:
    with path.open("rb") as f:
        riff, _, wave = struct.unpack("<4sI4s", f.read(12))
        if riff != b"RIFF" or wave != b"WAVE":
            raise SystemExit(f"{path.name} : pas un WAV")
        pistes = taux = alignement = None
        while tete := f.read(8):
            if len(tete) < 8:
                break
            nom, taille = struct.unpack("<4sI", tete)
            nom = nom.decode("latin-1")
            if nom == "fmt ":
                corps = f.read(taille)
                _, pistes, taux, _, alignement = struct.unpack("<HHIIH", corps[:14])
                print(f"[fmt ] {pistes} pistes · {taux} Hz")
            elif nom == "data":
                n = taille // alignement if alignement else 0
                duree = f" = {n / taux:.1f} s" if taux else ""
                print(f"[data] {n} échantillons par piste{duree}")
                f.seek(taille, 1)
            elif nom == "guan":
                texte = f.read(taille).rstrip(b"\0").decode("utf-8", "replace")
                print("[guan] " + texte.rstrip("\n"))
            else:
                print(f"[{nom}] {taille} octets")
                f.seek(taille, 1)
            if taille % 2:
                f.seek(1, 1)


if __name__ == "__main__":
    for chemin in sys.argv[1:]:
        blocs(Path(chemin))
