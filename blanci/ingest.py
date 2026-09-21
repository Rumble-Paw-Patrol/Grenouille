"""Inventaire des enregistrements : scan récursif, en-têtes WAV, nommage Song Meter, QC.

Arborescence attendue : <racine>/<jeu>/<site>/<micro>/**/*.wav. Si le dossier micro manque,
le micro est pris dans l'en-tête GUANO (numéro de série) ou le préfixe du nom de fichier.
La racine audio n'est jamais modifiée.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO

import soundfile as sf

from blanci.audio import load_audio
from blanci.db import recording_id_for
from blanci.qc import qc_flags, qc_indices

# <préfixe>_<AAAAMMJJ>_<HHMMSS>[_suffixe].wav — convention Wildlife Acoustics.
SONGMETER_NAME = re.compile(r"^(?P<prefix>.+?)_(?P<date>\d{8})_(?P<time>\d{6})(?:_.*)?$")


@dataclass
class IngestReport:
    added: int = 0
    skipped: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def iter_wav_files(directory: Path) -> Iterator[Path]:
    """Fichiers WAV triés.

    Ignore les fichiers cachés, dont les `._*` que macOS sème sur un disque externe.
    """
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".wav" and not path.name.startswith("."):
            yield path


def read_guano(source: Path | BinaryIO) -> dict[str, str]:
    """Métadonnées GUANO (bloc RIFF `guan`) écrites par les Song Meter ; {} si absentes."""
    if isinstance(source, Path):
        with open(source, "rb") as f:
            return read_guano(f)
    source.seek(0)
    header = source.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        return {}
    while True:
        chunk = source.read(8)
        if len(chunk) < 8:
            return {}
        size = int.from_bytes(chunk[4:], "little")
        if chunk[:4] == b"guan":
            text = source.read(size).decode("utf-8", errors="replace")
            break
        source.seek(size + (size & 1), 1)
    meta = {}
    for line in text.splitlines()[1:]:
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta


def parse_songmeter_name(stem: str) -> tuple[str, datetime] | None:
    """(préfixe, horodatage naïf) depuis le nom de fichier, ou None."""
    match = SONGMETER_NAME.match(stem)
    if not match:
        return None
    try:
        stamp = datetime.strptime(match["date"] + match["time"], "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return match["prefix"], stamp


def start_utc(guano: dict[str, str], stem: str, utc_offset_h: float) -> str | None:
    """Début en UTC : horodatage GUANO en priorité, sinon nom de fichier + décalage configuré."""
    local_tz = timezone(timedelta(hours=utc_offset_h))
    stamp = None
    if "Timestamp" in guano:
        try:
            stamp = datetime.fromisoformat(guano["Timestamp"])
        except ValueError:
            stamp = None
    if stamp is None:
        parsed = parse_songmeter_name(stem)
        if parsed is None:
            return None
        stamp = parsed[1]
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=local_tz)
    return stamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def describe_recording(
    path: Path, root: Path, dataset: str, cfg: dict[str, Any], run_qc: bool, hash_file: bool
) -> dict[str, Any]:
    """Ligne de la table recordings pour un fichier ; lit le fichier une seule fois."""
    rel = path.relative_to(root).as_posix()
    parts = path.relative_to(root / dataset).parts
    data = path.read_bytes()
    buffer = io.BytesIO(data)
    info = sf.info(buffer)
    guano = read_guano(buffer)
    parsed = parse_songmeter_name(path.stem)

    mic_id = parts[1] if len(parts) >= 3 else None
    if mic_id is None:
        mic_id = guano.get("Serial") or (parsed[0] if parsed else None)

    qc = None
    if run_qc:
        buffer.seek(0)
        wav, sr = load_audio(buffer)
        qc = json.dumps(qc_flags(qc_indices(wav, sr), cfg["qc"]))

    return {
        "recording_id": recording_id_for(rel),
        "path": rel,
        "dataset": dataset,
        "site": parts[0] if len(parts) >= 2 else None,
        "mic_id": mic_id,
        "start_utc": start_utc(guano, path.stem, cfg["recorder"]["filename_utc_offset_h"]),
        "duration_s": info.frames / info.samplerate,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "sha256": hashlib.sha256(data).hexdigest() if hash_file else None,
        "qc_flags": qc,
    }


def ingest(
    con: sqlite3.Connection,
    root: Path,
    dataset: str,
    cfg: dict[str, Any],
    run_qc: bool = True,
    hash_file: bool = True,
    force: bool = False,
    progress_every: int = 500,
) -> IngestReport:
    """Inventorie <root>/<dataset> ; reprenable (les fichiers déjà inventoriés sont sautés)."""
    directory = root / dataset
    if not directory.is_dir():
        raise FileNotFoundError(f"dossier introuvable : {directory}")
    known = {row[0] for row in con.execute("SELECT path FROM recordings")}
    report = IngestReport()
    columns = (
        "recording_id, path, dataset, site, mic_id, start_utc, duration_s, "
        "sample_rate, channels, sha256, qc_flags"
    )
    updates = ", ".join(f"{c} = excluded.{c}" for c in columns.split(", ")[1:])
    sql = (
        f"INSERT INTO recordings ({columns}) VALUES ({', '.join('?' * 11)}) "
        f"ON CONFLICT(recording_id) DO UPDATE SET {updates}"
    )

    for n, path in enumerate(iter_wav_files(directory), start=1):
        rel = path.relative_to(root).as_posix()
        if rel in known and not force:
            report.skipped += 1
            continue
        try:
            row = describe_recording(path, root, dataset, cfg, run_qc, hash_file)
        except Exception as exc:  # fichier tronqué ou illisible : signalé, pas bloquant
            report.errors.append((rel, f"{type(exc).__name__}: {exc}"))
            continue
        con.execute(sql, tuple(row.values()))
        report.added += 1
        if n % progress_every == 0:
            con.commit()
            print(f"  {n} fichiers parcourus ({report.added} ajoutés)", flush=True)
    con.commit()
    return report
