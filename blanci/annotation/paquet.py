"""Paquet d'écoute à envoyer à un encadrant qui n'a ni GitHub ni les disques (DECISIONS n° 205).

Un dossier (et son .zip) qui se suffit à lui-même :

    <nom>/
      Ouvrir le poste d'annotation.bat      double-clic sous Windows (.command sur Mac)
      LISEZ-MOI.txt                         la marche à suivre, pour l'encadrant
      A_RENVOYER/                           ses annotations, réécrites après chaque envoi
      donnees/
        base.sqlite                         les enregistrements de la file et nos extraits
        files/<nom>/candidats.csv           la file d'écoute, seule proposée par le poste
        audio/<nom du fichier>.flac         l'audio des extraits seulement (silence ailleurs)
      programme/                            le code du poste et config/local.yaml

L'audio garde le nom et la durée de l'enregistrement d'origine : les temps et les
identifiants sont les mêmes que chez nous, les CSV renvoyés (qui portent `recording_id`) se
relisent tels quels (`blanci import-labels`). Hors des extraits, le fichier est silencieux : le FLAC
ne coûte presque rien. Le poste n'a besoin que du groupe `app` (pas torch ni TensorFlow).
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from blanci.core.config import PROJECT_ROOT
from blanci.core.db import connect
from blanci.core.locate import locate

# Du dépôt, seul ce qui fait tourner le poste ; les annotations du dossier `annotations/`
# restent chez nous, sauf les deux listes que le poste propose.
PROGRAM_FILES = (
    "pyproject.toml",
    "README.md",  # le pyproject le cite : sans lui, uv ne construit pas le paquet
    "uv.lock",
    ".python-version",
    "config/default.yaml",
    "annotations/annotateurs.txt",
    "annotations/etiquettes.json",
)
PROGRAM_DIRS = ("blanci",)
SEND_BACK = "A_RENVOYER"

LAUNCH_BAT = r"""@echo off
chcp 65001 >nul
title Poste d'annotation
cd /d "%~dp0programme"
rem Environnement Python hors du paquet, sous un chemin court : les fichiers de scikit-learn
rem depasseraient sinon les 260 caracteres d'un chemin Windows ; partage par tous les paquets.
set "UV_PROJECT_ENVIRONMENT=%LOCALAPPDATA%\poste-blanci"
where uv >nul 2>nul
if errorlevel 1 (
  echo Premiere ouverture : installation de uv, qui installe Python et le poste.
  powershell -NoProfile -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
echo Ouverture du poste dans le navigateur (la premiere fois, quelques minutes)...
echo Laisser cette fenetre ouverte pendant l'ecoute ; la fermer pour arreter le poste.
uv sync --frozen --no-default-groups --group app
uv run --no-sync blanci annotate
pause
"""

LAUNCH_COMMAND = """#!/bin/sh
cd "$(dirname "$0")/programme" || exit 1
export UV_PROJECT_ENVIRONMENT="$HOME/.poste-blanci"  # partagé par tous les paquets
if ! command -v uv >/dev/null 2>&1; then
  echo "Première ouverture : installation de uv, qui installe Python et le poste."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"
fi
echo "Ouverture du poste dans le navigateur (la première fois, quelques minutes)..."
echo "Laisser cette fenêtre ouverte pendant l'écoute ; la fermer pour arrêter le poste."
uv sync --frozen --no-default-groups --group app
uv run --no-sync blanci annotate
"""

README = """Paquet d'écoute « {name} », envoyé par {author}
{line}

{message}

1. Ouvrir le poste : double-cliquer sur « Ouvrir le poste d'annotation.bat » (Windows) ou
   « Ouvrir le poste d'annotation.command » (Mac). La première fois, il installe ce qu'il
   faut (connexion internet, quelques minutes) ; le poste s'ouvre ensuite dans le navigateur.
   Laisser la fenêtre noire ouverte pendant l'écoute.
2. Dans le panneau de gauche, « Annotateur » : choisir son nom (ou « Ajouter quelqu'un… »).
3. La file d'écoute de ce paquet est déjà ouverte : {n} extraits. Les intervalles de
   {author} sont en tirets, à son nom, sous le spectrogramme. Pour chaque extrait :
   - d'accord : « Reprendre les intervalles de {author} », puis « Envoyer l'extrait » ;
   - sinon, tracer ses propres intervalles (clic gauche + glisser), changer l'étiquette,
     en ajouter une (« Étiquettes ») ;
   - un avis, un doute : le champ « Commentaire » ;
   - « Envoyer l'extrait » enregistre et passe au suivant.
   Le mode d'emploi (souris, clavier) est en haut du panneau de gauche.
4. À la fin, le poste dit que la file est terminée. Les réponses sont dans le dossier
   « {send_back} » de ce paquet (deux fichiers à son nom, en .csv) : les envoyer par mail à
   {author}. On peut s'arrêter en route et reprendre plus tard : tout est gardé, et le dossier
   est à jour après chaque extrait envoyé.
"""


def extracts_with_label(
    con: sqlite3.Connection, queue: pd.DataFrame, label: str, annotator: str | None = None
) -> pd.DataFrame:
    """Candidats de `queue` dont le dernier envoi (de `annotator`, ou de n'importe qui) sur
    l'extrait [offset_s, offset_s + dur_s] a au moins un intervalle `label`."""
    from blanci.annotation.workbench import extract_intervals

    keep = []
    for row in queue.itertuples(index=False):
        start, end = float(row.offset_s), float(row.offset_s) + float(row.dur_s)
        saved = extract_intervals(con, row.recording_id, start, end, annotator)
        keep.append(any(i[2] == label for ivs in saved.values() for i in ivs))
    return queue[keep].reset_index(drop=True)


def _program_files(root: Path) -> list[Path]:
    """Fichiers du programme, relatifs au dépôt : ceux que git suit (le reste du dossier,
    caches et essais, ne part pas), sinon tout `blanci/` hors caches."""
    try:
        tracked = subprocess.run(
            ["git", "ls-files", *PROGRAM_DIRS],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        ).stdout.splitlines()
        files = [Path(f) for f in tracked]
    except (OSError, subprocess.CalledProcessError):
        files = [
            p.relative_to(root)
            for d in PROGRAM_DIRS
            for p in (root / d).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        ]
    files += [Path(f) for f in PROGRAM_FILES]
    return [f for f in files if (root / f).is_file()]


def _write_audio(source: Path, target: Path, spans: list[tuple[float, float]]) -> None:
    """`source` en FLAC, même durée et mêmes canaux, silencieux hors de `spans` (s)."""
    with sf.SoundFile(source) as f:
        sr, frames, channels = f.samplerate, f.frames, f.channels
        audio = np.zeros((frames, channels), dtype="int16")
        for start, end in spans:
            a, b = max(0, round(start * sr)), min(frames, round(end * sr))
            f.seek(a)
            audio[a:b] = f.read(b - a, dtype="int16", always_2d=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, audio, sr, format="FLAC", subtype="PCM_16")


def build_package(
    con: sqlite3.Connection,
    cfg: dict,
    queue: pd.DataFrame,
    name: str,
    author: str,
    out_root: Path,
    message: str = "",
    root: Path = PROJECT_ROOT,
) -> Path:
    """Écrit le paquet `out_root/<name>/` (voir le module) et `out_root/<name>.zip` ; renvoie
    le chemin du .zip. `queue` : la file d'écoute (colonnes d'une file du poste). Refuse un
    paquet qui existe déjà et un enregistrement introuvable sur les disques branchés."""
    from blanci.core.config import config_path

    folder = out_root / name
    if folder.exists():
        raise FileExistsError(f"le paquet existe déjà : {folder}")
    ids = list(dict.fromkeys(queue["recording_id"]))
    recordings = pd.read_sql_query(
        f"SELECT * FROM recordings WHERE recording_id IN ({', '.join('?' * len(ids))})",
        con,
        params=ids,
    ).set_index("recording_id")
    renamed = cfg["audio"].get("renamed")
    sources = {
        rid: locate(str(recordings.loc[rid, "path"]), config_path(cfg, "raw"), [], renamed)
        for rid in ids
    }
    missing = [str(recordings.loc[r, "path"]) for r, s in sources.items() if s is None]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} enregistrements introuvables (disque débranché ?), dont {missing[0]}"
        )

    # Audio à plat, sous le seul nom du fichier : les dossiers d'origine (« Projet Phénologie
    # blanci/Troisième relevé… ») dépasseraient les 260 caractères d'un chemin Windows une
    # fois le paquet dézippé dans Téléchargements. Les CSV renvoyés portent recording_id.
    short = {rid: Path(str(recordings.loc[rid, "path"]).replace("\\", "/")).name for rid in ids}
    twice = {n for n in short.values() if list(short.values()).count(n) > 1}
    short = {rid: f"{rid}_{n}" if n in twice else n for rid, n in short.items()}
    data = folder / "donnees"
    for rid, group in queue.groupby("recording_id", sort=False):
        spans = [(o, o + d) for o, d in zip(group["offset_s"], group["dur_s"], strict=True)]
        _write_audio(sources[rid], data / "audio" / Path(short[rid]).with_suffix(".flac"), spans)

    queue_dir = data / "files" / name
    queue_dir.mkdir(parents=True)
    queue.assign(path=queue["recording_id"].map(short)).to_csv(
        queue_dir / "candidats.csv", index=False
    )
    (queue_dir / "LISEZMOI.md").write_text(
        f"Paquet envoyé par {author} : {len(queue)} extraits.\n\n{message}\n", encoding="utf-8"
    )

    base = connect(data / "base.sqlite")
    columns = ", ".join(c for (_, c, *_) in base.execute("PRAGMA table_info(recordings)"))
    base.executemany(
        f"INSERT INTO recordings ({columns}) VALUES ({', '.join('?' * (columns.count(',') + 1))})",
        con.execute(
            f"SELECT {columns} FROM recordings WHERE recording_id IN ({', '.join('?' * len(ids))})",
            ids,
        ).fetchall(),
    )
    base.executemany(
        "UPDATE recordings SET path = ? WHERE recording_id = ?",
        [(n, rid) for rid, n in short.items()],
    )
    _copy_spans(con, base, queue)
    base.commit()
    base.close()

    program = folder / "programme"
    for rel in _program_files(root):
        (program / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, program / rel)
    local = {
        "paths": {"raw": "../donnees/audio", "db": "../donnees/base.sqlite",
                  "queues": "../donnees/files"},
        "paquet": {"nom": name, "de": author, "renvoi": f"../{SEND_BACK}"},
    }  # fmt: skip
    (program / "config" / "local.yaml").write_text(
        yaml.safe_dump(local, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    (folder / SEND_BACK).mkdir()
    (folder / "LISEZ-MOI.txt").write_text(
        README.format(
            name=name,
            author=author,
            line="=" * (len(name) + len(author) + 30),
            message=message or "Merci pour votre écoute !",
            n=len(queue),
            send_back=SEND_BACK,
        ),
        encoding="utf-8",
    )
    (folder / "Ouvrir le poste d'annotation.bat").write_text(
        LAUNCH_BAT.replace("\n", "\r\n"), encoding="utf-8", newline=""
    )
    command = folder / "Ouvrir le poste d'annotation.command"
    command.write_text(LAUNCH_COMMAND, encoding="utf-8", newline="\n")
    command.chmod(0o755)
    return Path(shutil.make_archive(str(folder), "zip", out_root, name))


def _copy_spans(src: sqlite3.Connection, dst: sqlite3.Connection, queue: pd.DataFrame) -> None:
    """Nos extraits annotés sur les extraits de la file (tous annotateurs, toutes les
    versions), avec leurs intervalles : le poste de l'encadrant les montre à leur nom."""
    span_cols = [c for (_, c, *_) in dst.execute("PRAGMA table_info(spans)") if c != "span_id"]
    iv_cols = [
        c
        for (_, c, *_) in dst.execute("PRAGMA table_info(intervals)")
        if c not in ("interval_id", "span_id")
    ]
    for row in queue.itertuples(index=False):
        start, end = float(row.offset_s), float(row.offset_s) + float(row.dur_s)
        spans = src.execute(
            f"SELECT span_id, {', '.join(span_cols)} FROM spans WHERE recording_id = ? "
            "AND ABS(start_s - ?) < 0.006 AND ABS(end_s - ?) < 0.006 ORDER BY span_id",
            (row.recording_id, start, end),
        ).fetchall()
        for old_id, *values in spans:
            new_id = dst.execute(
                f"INSERT INTO spans ({', '.join(span_cols)}) "
                f"VALUES ({', '.join('?' * len(span_cols))})",
                values,
            ).lastrowid
            dst.executemany(
                f"INSERT INTO intervals (span_id, {', '.join(iv_cols)}) "
                f"VALUES (?, {', '.join('?' * len(iv_cols))})",
                [
                    (new_id, *iv)
                    for iv in src.execute(
                        f"SELECT {', '.join(iv_cols)} FROM intervals WHERE span_id = ?",
                        (old_id,),
                    )
                ],
            )
