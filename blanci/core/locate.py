"""Retrouver un enregistrement sur les disques branchés, sans configuration (DECISIONS n° 195).

La base garde le chemin de chaque enregistrement relatif à la racine du disque utilisée à
l'inventaire (`paths.raw`). Ce module essaie ce chemin sous chaque racine possible, dans
l'ordre : racines qui ont déjà marché (la dernière d'abord), `paths.raw`, racines retenues
(dossier collé à la main sur le poste d'annotation), puis les disques branchés (`/Volumes/*`
sur Mac, les disques locaux `D:` à `Z:` sur Windows, sans les lecteurs réseau, `/media/<moi>/*`
et `/mnt/*` sous Linux), et un niveau de dossier sous chacun d'eux (disque inventorié depuis
un sous-dossier). Un partage réseau sert s'il est collé à la main (racine retenue).

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


# Types de lecteur Windows (GetDriveTypeW) gardés : amovible, fixe, disque virtuel en mémoire.
# Les lecteurs réseau (partages de l'ONF) et les CD sont écartés : tester un fichier sur un
# partage coûte une demi-seconde, et sonder les lettres ainsi branchées prenait 15 s.
_LOCAL_DRIVE_TYPES = {2, 3, 6}


def _local_drive_letters() -> list[str]:
    """Lettres D à Z attribuées à un disque local (USB compris), sans toucher aux autres."""
    import ctypes

    kernel32 = ctypes.windll.kernel32
    attached = kernel32.GetLogicalDrives()
    return [
        d
        for i, d in enumerate(string.ascii_uppercase)
        if i >= 3 and attached >> i & 1 and kernel32.GetDriveTypeW(f"{d}:\\") in _LOCAL_DRIVE_TYPES
    ]


def mounted_disks() -> list[Path]:
    """Disques branchés, selon le système."""
    if sys.platform == "win32":
        return [Path(f"{d}:/") for d in _local_drive_letters()]
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


def variants(rel: str, renamed: dict[str, str] | None = None) -> list[str]:
    """`rel`, puis le même chemin sous chaque nom de dossier qui a remplacé le sien à la copie
    sur un disque du stage (`audio.renamed`)."""
    rel = str(rel).replace("\\", "/")
    out = [rel]
    for old, new in (renamed or {}).items():
        old = old.strip("/")
        if rel.startswith(old + "/"):
            out.append(new.strip("/") + rel[len(old) :])
    return out


def locate(
    rel: str, raw: Path | None, extra: list[Path] = (), renamed: dict[str, str] | None = None
) -> Path | None:
    """Chemin complet de l'enregistrement `rel` (relatif, séparateurs POSIX), ou None."""
    names = variants(rel, renamed)
    for root in candidate_roots(raw, extra):
        for name in names:
            full = root / name
            if full.is_file():
                if root in _FOUND:
                    _FOUND.remove(root)
                _FOUND.insert(0, root)
                return full
    return None
