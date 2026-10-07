"""Retrouver un enregistrement sur les disques branchés, sans configuration (DECISIONS n° 195).

La base garde le chemin de chaque enregistrement relatif à la racine du disque utilisée à
l'inventaire (`paths.raw`). Ce module essaie ce chemin sous chaque racine possible, dans
l'ordre : racines qui ont déjà marché (la dernière d'abord), `paths.raw`, racines retenues
(dossier collé à la main sur le poste d'annotation), puis les disques branchés (`/Volumes/*`
sur Mac, `D:` à `Z:` sur Windows, `/media/<moi>/*` et `/mnt/*` sous Linux), et un niveau de
dossier sous chacun d'eux (disque inventorié depuis un sous-dossier).

Coût : un test d'existence de fichier par racine essayée, quelques dizaines au plus pour le
premier enregistrement ; ensuite la racine trouvée passe devant et un seul test suffit.
Plusieurs disques : chaque enregistrement est cherché sur tous, le premier qui l'a gagne.
"""

from __future__ import annotations

import os
import string
import sys
from pathlib import Path

# Racines qui ont déjà donné un enregistrement, la plus récente d'abord (le temps du processus).
_FOUND: list[Path] = []


def remembered_file(db: Path) -> Path:
    """Fichier des racines collées à la main, à côté de la base (hors git, propre au poste)."""
    return Path(db).parent / "racines_audio.txt"


def remembered_roots(db: Path) -> list[Path]:
    path = remembered_file(db)
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()) for line in lines if line.strip()]


def remember_root(db: Path, root: Path) -> None:
    """Ajoute `root` en tête des racines retenues (sans doublon)."""
    roots = [Path(root), *(r for r in remembered_roots(db) if r != Path(root))]
    path = remembered_file(db)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{r}\n" for r in roots), encoding="utf-8")


def mounted_disks() -> list[Path]:
    """Disques branchés, selon le système."""
    if sys.platform == "win32":
        return [Path(f"{d}:/") for d in string.ascii_uppercase[3:] if os.path.exists(f"{d}:/")]
    bases = [Path("/Volumes")] if sys.platform == "darwin" else []
    user = os.environ.get("USER", "")
    bases += [Path("/media") / user, Path("/run/media") / user, Path("/mnt")]
    disks = []
    for base in bases:
        try:
            disks += sorted(p for p in base.iterdir() if p.is_dir())
        except OSError:
            continue
    return disks


def _children(root: Path) -> list[Path]:
    try:
        return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        return []


def candidate_roots(raw: Path | None, extra: list[Path] = ()):
    """Racines à essayer, dans l'ordre, sans doublon ; les disques ne sont parcourus que si
    les racines connues n'ont pas suffi."""
    seen: set[Path] = set()

    def fresh(roots):
        for root in roots:
            if root not in seen:
                seen.add(root)
                yield root

    yield from fresh([*_FOUND, *([Path(raw)] if raw else []), *map(Path, extra)])
    disks = mounted_disks()
    yield from fresh(disks)
    yield from fresh(child for disk in disks for child in _children(disk))


def locate(rel: str, raw: Path | None, extra: list[Path] = ()) -> Path | None:
    """Chemin complet de l'enregistrement `rel` (relatif, séparateurs POSIX), ou None."""
    rel = str(rel).replace("\\", "/")
    for root in candidate_roots(raw, extra):
        full = root / rel
        if full.is_file():
            if root in _FOUND:
                _FOUND.remove(root)
            _FOUND.insert(0, root)
            return full
    return None
