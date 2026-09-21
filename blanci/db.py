"""SQLite : schéma (§13.3), migrations par `PRAGMA user_version`, identifiants."""

from __future__ import annotations

import hashlib
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


def recording_id_for(rel_path: str) -> str:
    """Identifiant stable : chemin relatif à la racine audio, indépendant de la machine."""
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:16]


def window_id_for(recording_id: str, offset_s: float) -> str:
    return f"{recording_id}:{offset_s:.2f}"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
