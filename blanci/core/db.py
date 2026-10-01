"""SQLite : schéma (§13.3), migrations par `PRAGMA user_version`, identifiants."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

# Une entrée par version ; ne jamais modifier une entrée publiée, en ajouter une.
MIGRATIONS = [
    """
    CREATE TABLE recordings (
        recording_id TEXT PRIMARY KEY,
        path         TEXT UNIQUE NOT NULL,   -- relatif à paths.raw, séparateurs POSIX
        dataset      TEXT NOT NULL,
        site         TEXT,
        mic_id       TEXT,
        start_utc    TEXT,
        duration_s   REAL,
        sample_rate  INTEGER,
        channels     INTEGER,
        sha256       TEXT,
        qc_flags     TEXT                    -- JSON : rain, saturation, in_bag, silent, indices
    );
    CREATE INDEX recordings_site_mic ON recordings(dataset, site, mic_id);

    CREATE TABLE windows (
        window_id    TEXT PRIMARY KEY,
        recording_id TEXT NOT NULL REFERENCES recordings(recording_id),
        offset_s     REAL NOT NULL,
        dur_s        REAL NOT NULL
    );
    CREATE INDEX windows_recording ON windows(recording_id);

    CREATE TABLE labels (
        label_id   INTEGER PRIMARY KEY,
        window_id  TEXT NOT NULL REFERENCES windows(window_id),
        label      TEXT NOT NULL,
        quality    TEXT,
        species    TEXT,
        conditions TEXT,                     -- JSON : tags, co_occurring, comment, ...
        annotator  TEXT,
        source     TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX labels_window ON labels(window_id);

    -- Labels en ajout seul (§13.7) : une correction est une nouvelle ligne.
    CREATE TRIGGER labels_no_update BEFORE UPDATE ON labels
    BEGIN SELECT RAISE(ABORT, 'labels are append-only'); END;
    CREATE TRIGGER labels_no_delete BEFORE DELETE ON labels
    BEGIN SELECT RAISE(ABORT, 'labels are append-only'); END;

    CREATE TABLE models (
        model_id    TEXT PRIMARY KEY,
        kind        TEXT NOT NULL,           -- encoder, head, fusion, threshold
        name        TEXT NOT NULL,
        version     TEXT NOT NULL,
        sha256      TEXT,
        params_json TEXT,
        created_at  TEXT NOT NULL
    );

    CREATE TABLE scores (
        window_id TEXT NOT NULL,
        model_id  TEXT NOT NULL,
        score     REAL NOT NULL,
        PRIMARY KEY (window_id, model_id)
    );

    CREATE TABLE decisions (
        recording_id TEXT NOT NULL,
        encoder_id   TEXT NOT NULL,
        head_version TEXT NOT NULL,
        threshold_id TEXT NOT NULL,
        fraction     REAL,
        status       TEXT NOT NULL,
        created_at   TEXT NOT NULL
    );

    -- Fichiers d'annotation déjà importés (idempotence de import-labels).
    CREATE TABLE imports (
        file_sha256 TEXT PRIMARY KEY,
        path        TEXT NOT NULL,
        n_labels    INTEGER NOT NULL,
        imported_at TEXT NOT NULL
    );
    """,
    # 2 — débuts de notes par enregistrement (module séquentiel, §3), calculés une fois.
    """
    CREATE TABLE onsets (
        recording_id TEXT PRIMARY KEY REFERENCES recordings(recording_id),
        channel      TEXT NOT NULL,          -- micro lu (0, 1 ou mean)
        onsets_json  TEXT NOT NULL,          -- secondes depuis le début, liste JSON
        computed_at  TEXT NOT NULL
    );
    """,
]


def connect(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    migrate(con)
    return con


def migrate(con: sqlite3.Connection) -> None:
    version = con.execute("PRAGMA user_version").fetchone()[0]
    for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
        con.executescript(script)
        con.execute(f"PRAGMA user_version = {number}")
    con.commit()


def model_params(con: sqlite3.Connection, model_id: str, kind: str | None = None) -> dict:
    """Paramètres JSON d'un modèle enregistré."""
    sql = "SELECT params_json FROM models WHERE model_id = ?"
    args: tuple = (model_id,)
    if kind is not None:
        sql += " AND kind = ?"
        args += (kind,)
    row = con.execute(sql, args).fetchone()
    if row is None:
        label = f"{kind} " if kind else ""
        raise ValueError(f"{label}inconnu dans la table models : {model_id}")
    return json.loads(row["params_json"])


def encoder_params(con: sqlite3.Connection, encoder_id: str) -> dict:
    """Paramètres enregistrés par `embed` : f_e, fenêtre, pas, dimension, débit mesuré."""
    try:
        return model_params(con, encoder_id, kind="encoder")
    except ValueError as exc:
        raise ValueError(f"encodeur inconnu : {encoder_id} (lancer `blanci embed`)") from exc


def register_model(
    con: sqlite3.Connection,
    model_id: str,
    kind: str,
    name: str,
    version: str,
    params: dict,
    sha256: str | None = None,
) -> None:
    """Ajoute ou met à jour une ligne du registre (§13.3)."""
    con.execute(
        "INSERT INTO models (model_id, kind, name, version, sha256, params_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(model_id) DO UPDATE SET "
        "params_json = excluded.params_json, sha256 = excluded.sha256",
        (model_id, kind, name, version, sha256, json.dumps(params), utc_now()),
    )
    con.commit()


def next_version(con: sqlite3.Connection, kind: str, name: str) -> str:
    """Version suivante d'un modèle : v1, v2, … Les versions précédentes restent en base."""
    versions = [
        int(row["version"][1:])
        for row in con.execute(
            "SELECT version FROM models WHERE kind = ? AND name = ?", (kind, name)
        )
        if str(row["version"]).startswith("v") and str(row["version"])[1:].isdigit()
    ]
    return f"v{max(versions, default=0) + 1}"


def recording_key(path: str) -> str:
    """Identité d'un enregistrement : son nom sans dossier ni extension, en minuscules.

    `<série>_<AAAAMMJJ>_<HHMMSS>` est unique pour un enregistreur donné et survit à la copie
    sur un autre disque, à une réorganisation des dossiers et au passage WAV ↔ FLAC.
    """
    return Path(path.replace("\\", "/")).stem.lower()


def recording_id_for(path: str) -> str:
    """Identifiant stable, indépendant du dossier et de la machine (voir `recording_key`)."""
    return hashlib.sha256(recording_key(path).encode("utf-8")).hexdigest()[:16]


# Durée des fenêtres annotées (Biophonia) et de la grille w3 : leur identifiant garde la forme
# historique « <enregistrement>:<décalage> ». Toute autre durée est écrite dans l'identifiant.
LEGACY_WINDOW_S = 3.0


def window_id_for(recording_id: str, offset_s: float, dur_s: float = LEGACY_WINDOW_S) -> str:
    """Identifiant d'une fenêtre : enregistrement, décalage et durée (DECISIONS n° 65).

    Sans la durée, une fenêtre de 5 s (grille de beats) et une annotation de 3 s au même
    décalage, ou un label d'enregistrement entier (0 s, 120 s) et la fenêtre de 3 s à 0 s,
    partageraient la même ligne de `windows`, et la seconde hériterait de la durée de la
    première.
    """
    base = f"{recording_id}:{offset_s:.2f}"
    return base if abs(dur_s - LEGACY_WINDOW_S) < 1e-6 else f"{base}/{dur_s:.2f}"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
