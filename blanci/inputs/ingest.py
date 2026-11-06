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

import numpy as np
import soundfile as sf

from blanci.core.db import recording_id_for
from blanci.inputs.qc import (
    apply_annotation_flags,
    apply_audio_flags,
    apply_metadata_flags,
    audio_qc,
    parse_flags,
)

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
    # Même nom, contenu différent : (écarté faute d'identifiant libre, conservé, raison).
    conflicts: list[tuple[str, str, str]] = field(default_factory=list)
    relocated: int = 0  # déjà connus sous un chemin qui n'existe plus : chemin mis à jour
    flagged: dict[str, int] = field(default_factory=dict)  # drapeaux d'inventaire, total


def iter_audio_files(directory: Path, suffixes: Sequence[str] = AUDIO_SUFFIXES) -> Iterator[Path]:
    """Fichiers audio triés dans l'ordre naturel : « RELEVE 2 » avant « RELEVE 10 », pour que
    la règle « le premier inventorié l'emporte » suive l'ordre des relevés.

    Ignore les fichiers cachés, dont les `._*` que macOS sème sur un disque externe.
    """
    wanted = {s.lower() for s in suffixes}
    for path in sorted(directory.rglob("*"), key=natural_key):
        if path.is_file() and path.suffix.lower() in wanted and not path.name.startswith("."):
            yield path


def natural_key(path: Path) -> list[str | int]:
    """Clé de tri naturel : les nombres du chemin comparés comme des nombres."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.as_posix())]


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


SIGNATURE_S = 1.0  # secondes d'audio décodé comparées entre deux copies
SIGNATURE_TOLERANCE = 4 / 32768  # écart d'échantillon toléré : quelques pas de quantification


def audio_signature(path: Path) -> dict[str, Any]:
    """Ce qui identifie un enregistrement au-delà de son nom : nombre d'échantillons, fréquence,
    pistes, numéro de série et horodatage GUANO (WAV seulement), et la première seconde
    décodée (`head`). Décodée, pas lue en octets : un WAV et le FLAC tiré du même
    enregistrement (compression sans perte) donnent le même son, à un pas d'arrondi près."""
    info = sf.info(path)
    guano = read_guano(path) if path.suffix.lower() == ".wav" else {}
    frames = min(info.frames, int(SIGNATURE_S * info.samplerate))
    head, _ = sf.read(path, frames=frames, dtype="float32", always_2d=True)
    return {
        "frames": info.frames,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "serial": guano.get("Serial"),
        "timestamp": guano.get("Timestamp"),
        "head": head,
    }


def same_recording(a: Path, b: Path) -> tuple[bool, str]:
    """(même enregistrement, raison du désaccord). Les champs GUANO ne se comparent que s'ils
    sont présents dans les deux fichiers (un FLAC n'en a pas) ; l'audio, échantillon par
    échantillon sur la première seconde, à `SIGNATURE_TOLERANCE` près."""
    sa, sb = audio_signature(a), audio_signature(b)
    for key in ("frames", "sample_rate", "channels", "serial", "timestamp"):
        if sa[key] is not None and sb[key] is not None and sa[key] != sb[key]:
            return False, f"{key} : {sa[key]} ≠ {sb[key]}"
    gap = float(np.abs(sa["head"] - sb["head"]).max()) if sa["head"].size else 0.0
    if gap > SIGNATURE_TOLERANCE:
        return False, f"audio : écart maximal {gap:.4f} sur la première seconde"
    return True, ""


def clock_corrected(stem: str, mic_id: str | None, corrections: Any) -> bool:
    """L'horloge de ce fichier est dans `recorder.clock_corrections` : micro, et horodatage du
    nom de fichier dans [from, to[ (« AAAA-MM-JJ » ou « AAAA-MM-JJTHH:MM », comparés au format
    ISO). Son heure de début se lit alors dans le nom, pas dans le GUANO."""
    parsed = parse_songmeter_name(stem)
    if parsed is None or mic_id is None:
        return False
    stamp = parsed[1].isoformat()
    return any(
        str(c["mic_id"]) == mic_id and str(c["from"]) <= stamp < str(c["to"])
        for c in corrections or ()
    )


def apply_clock_corrections(con: sqlite3.Connection, recorder: dict[str, Any]) -> int:
    """Remet l'heure de début des horloges corrigées (`recorder.clock_corrections`) sur le nom
    de fichier et y pose `clock_corrected` (une remarque). Idempotent : l'heure se recalcule
    depuis le nom, jamais depuis l'heure déjà rangée. Un enregistrement sorti de la période
    perd la remarque, pas l'heure corrigée (seul un réinventaire relit le GUANO). Renvoie le
    nombre d'enregistrements concernés."""
    corrections = recorder.get("clock_corrections") or []
    mics = sorted({str(c["mic_id"]) for c in corrections})
    if not mics:
        return 0
    rows = con.execute(
        "SELECT recording_id, path, mic_id, start_utc, qc_flags FROM recordings "
        f"WHERE mic_id IN ({', '.join('?' * len(mics))})",
        mics,
    ).fetchall()
    offset = float(recorder["filename_utc_offset_h"])
    concerned, updates = 0, []
    for rid, path, mic, start, qc in rows:
        stem = Path(path).stem
        if not clock_corrected(stem, mic, corrections):
            # Hors période (plage resserrée) : la remarque part, l'heure rangée reste.
            flags = parse_flags(qc)
            if flags.pop("clock_corrected", None) is not None:
                updates.append((start, json.dumps(flags), rid))
            continue
        concerned += 1
        new_start = start_utc({}, stem, offset)
        flags = parse_flags(qc)
        new_flags = flags | {"clock_corrected": True}
        if new_start != start or new_flags != flags:
            updates.append((new_start, json.dumps(new_flags), rid))
    if updates:
        with con:
            con.executemany(
                "UPDATE recordings SET start_utc = ?, qc_flags = ? WHERE recording_id = ?", updates
            )
    return concerned


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
        qc = json.dumps(audio_qc(io.BytesIO(data), cfg["qc"]))
    # Horloge corrigée : l'heure du nom de fichier fait foi (`apply_clock_corrections`).
    if clock_corrected(path.stem, mic_id, cfg["recorder"].get("clock_corrections")):
        guano = {k: v for k, v in guano.items() if k != "Timestamp"}

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
    à la casse près, déjà inventorié sous un autre chemin, **et même contenu** : nombre
    d'échantillons, fréquence, pistes, GUANO, première seconde décodée, `same_recording`) est
    écarté et signalé : les cartes SD rapportées d'un relevé contiennent encore les fichiers du
    relevé précédent, copies exactes qui compteraient deux fois, sous deux sites. Même nom et
    contenu différent : un **conflit** (`report.conflicts`), signalé, jamais confondu avec un
    doublon ; le second fichier n'est pas inventorié (même identifiant), il faut trancher à la
    main. Le premier inventorié l'emporte :
    inventorier les relevés dans l'ordre chronologique, pour que chaque fichier reste
    rattaché au relevé où il a été enregistré.

    Si l'autre chemin n'existe plus sous la racine (fichiers copiés sur un autre disque,
    dossiers réorganisés) et que le site est le même, ce n'est pas un doublon mais un
    déplacement : l'identifiant ne dépend que du nom de fichier, la ligne est mise à jour et
    labels, fenêtres et embeddings restent attachés. Sous un autre site, c'est la copie
    laissée sur une carte SD d'un relevé suivant, vue depuis un disque qui ne porte pas
    l'original : un doublon, écarté, jamais rattaché au nouveau site.
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

    inventory = con.execute("SELECT path, site FROM recordings").fetchall()
    known = {row[0] for row in inventory}
    by_stem = {Path(p).stem.lower(): p for p in known}
    site_of = {Path(p).stem.lower(): str(s or "").lower() for p, s in inventory}
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
                # L'autre copie est toujours là : doublon si le contenu concorde (taille,
                # GUANO, audio décodé), sinon conflit de noms, signalé (même identifiant).
                try:
                    same, why = same_recording(path, root / by_stem[stem])
                except Exception as exc:  # fichier illisible : signalé, pas bloquant
                    same, why = False, f"illisible : {type(exc).__name__}: {exc}"
                if same:
                    report.duplicates.append((rel, by_stem[stem]))
                else:
                    report.conflicts.append((rel, by_stem[stem], why))
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
        if moved and str(row["site"] or "").lower() != site_of.get(stem, ""):
            # Autre site : la copie d'un relevé précédent, pas un déplacement (DECISIONS n° 144).
            report.duplicates.append((rel, by_stem[stem]))
            continue
        con.execute(sql, tuple(row.values()))
        by_stem[stem] = rel
        site_of[stem] = str(row["site"] or "").lower()
        if moved:
            report.relocated += 1
        else:
            report.added += 1
        if n % progress_every == 0:
            con.commit()
            print(f"  {n} fichiers parcourus ({report.added} ajoutés)", flush=True)
    con.commit()
    apply_clock_corrections(con, cfg["recorder"])
    # Les drapeaux d'inventaire dépendent de toute la série d'un micro : recalculés ici.
    report.flagged = apply_metadata_flags(con, cfg["qc"], cfg["recorder"]["filename_utc_offset_h"])
    if run_qc:  # `muffled_*` rangés ne sont que des candidats : règle des suites (n° 194, 203)
        apply_audio_flags(con, cfg["qc"])
    # Un contrôle audio refait réécrit qc_flags : les drapeaux posés à l'écoute sont remis.
    apply_annotation_flags(con)
    return report
