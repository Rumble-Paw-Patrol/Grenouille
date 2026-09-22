"""Inventaire des enregistrements : scan récursif, en-têtes, nommage Song Meter, QC.

Le micro est le numéro de série de l'enregistreur (GUANO, sinon préfixe du nom de fichier :
`2LA04530_20260106_103000.wav` → `2LA04530`). Le site est donné à l'inventaire (`--site`),
ou lu dans l'arborescence <racine>/<jeu>/<site>/<micro>/ si elle est respectée.
La racine audio n'est jamais modifiée.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO

import soundfile as sf

from blanci.audio import load_audio
from blanci.db import recording_id_for
from blanci.qc import apply_metadata_flags, qc_flags, qc_indices

# <préfixe>_<AAAAMMJJ>_<HHMMSS>[_suffixe] — convention Wildlife Acoustics.
SONGMETER_NAME = re.compile(r"^(?P<prefix>.+?)_(?P<date>\d{8})_(?P<time>\d{6})(?:_.*)?$")

# Formats lus par libsndfile et utilisés sur le terrain. Les Song Meter Mini 2 écrivent en
# WAV ou en FLAC selon leur réglage de compression ; les deux se lisent de la même façon.
AUDIO_SUFFIXES = (".wav", ".flac")


@dataclass
class IngestReport:
    added: int = 0
    skipped: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)  # (écarté, conservé)
    relocated: int = 0  # déjà connus sous un chemin qui n'existe plus : chemin mis à jour
    flagged: dict[str, int] = field(default_factory=dict)  # drapeaux d'inventaire, total


def iter_audio_files(directory: Path, suffixes: Sequence[str] = AUDIO_SUFFIXES) -> Iterator[Path]:
    """Fichiers audio triés.

    Ignore les fichiers cachés, dont les `._*` que macOS sème sur un disque externe.
    """
    wanted = {s.lower() for s in suffixes}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in wanted and not path.name.startswith("."):
            yield path


def read_guano(source: Path | BinaryIO) -> dict[str, str]:
    """Métadonnées GUANO (bloc RIFF `guan`) écrites par les Song Meter ; {} si absentes.

    Propre au WAV : un FLAC renvoie {} et l'horodatage vient alors du nom de fichier.
    """
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


# Décalage horaire GUANO écrit par les Song Meter sans zéro : « -3:00 », « +0:00 ».
_GUANO_OFFSET = re.compile(r"([+-])(\d):(\d{2})$")


def parse_guano_timestamp(value: str) -> datetime | None:
    """Horodatage GUANO, fuseau compris ; None s'il est illisible.

    Les Song Meter écrivent « 2025-12-27 15:30:00-3:00 », que `fromisoformat` refuse faute
    de zéro devant l'heure du décalage. Le fuseau varie d'un enregistreur à l'autre (heure
    locale ou UTC selon le réglage) : le perdre fausse l'heure de 3 h sans le dire.
    """
    try:
        return datetime.fromisoformat(_GUANO_OFFSET.sub(r"\g<1>0\2:\3", value.strip()))
    except ValueError:
        return None


def start_utc(guano: dict[str, str], stem: str, utc_offset_h: float) -> str | None:
    """Début en UTC : horodatage GUANO en priorité, sinon nom de fichier + décalage configuré."""
    local_tz = timezone(timedelta(hours=utc_offset_h))
    stamp = parse_guano_timestamp(guano["Timestamp"]) if "Timestamp" in guano else None
    if stamp is None:
        parsed = parse_songmeter_name(stem)
        if parsed is None:
            return None
        stamp = parsed[1]
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=local_tz)
    return stamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def describe_recording(
    path: Path,
    root: Path,
    dataset: str,
    cfg: dict[str, Any],
    run_qc: bool,
    hash_file: bool,
    site: str | None = None,
) -> dict[str, Any]:
    """Ligne de la table recordings pour un fichier ; lit le fichier une seule fois.

    Micro : numéro de série GUANO, sinon préfixe du nom de fichier, sinon dossier. Sur le
    terrain, le même enregistreur s'appelle « SM4 A », « SM A_SMA13417 » ou « SMA13417 » selon
    le relevé : seul le numéro de série est stable.
    Site : `site` s'il est donné, sinon le dossier sous <racine>/<jeu> (<jeu>/<site>/<micro>/).
    """
    rel = path.relative_to(root).as_posix()
    try:
        parts = path.relative_to(root / dataset).parts
    except ValueError:  # dossier scanné hors de <racine>/<jeu> : pas d'inférence par dossier
        parts = ()
    # Contrôle qualité et empreinte demandent tout le fichier (23 Mo en 48 kHz stéréo) ;
    # sans eux, l'en-tête suffit et l'inventaire d'un disque entier prend des minutes.
    data = path.read_bytes() if run_qc or hash_file else None
    source: Path | BinaryIO = io.BytesIO(data) if data is not None else path
    info = sf.info(source)
    guano = read_guano(source)
    parsed = parse_songmeter_name(path.stem)

    mic_id = guano.get("Serial") or (parsed[0] if parsed else None)
    if mic_id is None and len(parts) >= 3:
        mic_id = parts[1]
    if site is None and len(parts) >= 2:
        site = parts[0]

    qc = None
    if run_qc:
        wav, sr = load_audio(io.BytesIO(data), cfg.get("audio", {}).get("channel", "mean"))
        qc = json.dumps(qc_flags(qc_indices(wav, sr), cfg["qc"]))

    return {
        "recording_id": recording_id_for(rel),
        "path": rel,
        "dataset": dataset,
        "site": site,
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
    scan: Path | None = None,
    site: str | None = None,
) -> IngestReport:
    """Inventorie `scan` (défaut : <root>/<jeu>) ; chemins stockés relatifs à `root`.

    Reprenable : un chemin déjà inventorié est sauté. Un **doublon** (même nom de fichier,
    à la casse près, déjà inventorié sous un autre chemin) est écarté et signalé : les cartes
    SD rapportées d'un relevé contiennent encore les fichiers du relevé précédent, copies
    exactes qui compteraient deux fois, sous deux sites. Le premier inventorié l'emporte :
    inventorier les relevés dans l'ordre chronologique, pour que chaque fichier reste
    rattaché au relevé où il a été enregistré.

    Si l'autre chemin n'existe plus sous la racine (fichiers copiés sur un autre disque,
    dossiers réorganisés), ce n'est pas un doublon mais un déplacement : l'identifiant ne
    dépend que du nom de fichier, la ligne est mise à jour et labels, fenêtres et
    embeddings restent attachés.
    """
    root = Path(root)
    directory = Path(scan) if scan is not None else root / dataset
    if not directory.is_dir():
        raise FileNotFoundError(f"dossier introuvable : {directory}")
    try:
        directory.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{directory} n'est pas sous la racine audio {root} (paths.raw) : "
            "les chemins stockés doivent lui être relatifs"
        ) from exc

    known = {row[0] for row in con.execute("SELECT path FROM recordings")}
    by_stem = {Path(p).stem.lower(): p for p in known}
    report = IngestReport()
    columns = (
        "recording_id, path, dataset, site, mic_id, start_utc, duration_s, "
        "sample_rate, channels, sha256, qc_flags"
    )
    # Un réinventaire sans contrôle qualité ni empreinte n'efface pas ceux déjà calculés.
    keep = ("sha256", "qc_flags")
    updates = ", ".join(
        f"{c} = COALESCE(excluded.{c}, recordings.{c})" if c in keep else f"{c} = excluded.{c}"
        for c in columns.split(", ")[1:]
    )
    sql = (
        f"INSERT INTO recordings ({columns}) VALUES ({', '.join('?' * 11)}) "
        f"ON CONFLICT(recording_id) DO UPDATE SET {updates}"
    )

    suffixes = cfg.get("audio", {}).get("suffixes", AUDIO_SUFFIXES)
    for n, path in enumerate(iter_audio_files(directory, suffixes), start=1):
        rel = path.relative_to(root).as_posix()
        if rel in known and not force:
            report.skipped += 1
            continue
        stem = path.stem.lower()
        moved = False
        if stem in by_stem and by_stem[stem] != rel:
            if (root / by_stem[stem]).exists():
                report.duplicates.append((rel, by_stem[stem]))  # l'autre copie est toujours là
                continue
            # L'ancien chemin n'existe plus sous cette racine : le fichier a été copié ou
            # déplacé (nouveau disque, dossiers réorganisés). Même identifiant, donc labels et
            # embeddings conservés ; seule la ligne de l'inventaire est mise à jour.
            moved = True
        try:
            row = describe_recording(path, root, dataset, cfg, run_qc, hash_file, site=site)
        except Exception as exc:  # fichier tronqué ou illisible : signalé, pas bloquant
            report.errors.append((rel, f"{type(exc).__name__}: {exc}"))
            continue
        con.execute(sql, tuple(row.values()))
        by_stem[stem] = rel
        if moved:
            report.relocated += 1
        else:
            report.added += 1
        if n % progress_every == 0:
            con.commit()
            print(f"  {n} fichiers parcourus ({report.added} ajoutés)", flush=True)
    con.commit()
    # Les drapeaux d'inventaire dépendent de toute la série d'un micro : recalculés ici.
    report.flagged = apply_metadata_flags(con, cfg["qc"])
    return report
