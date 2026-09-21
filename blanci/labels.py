"""Import des annotations reçues, schéma de labels (§5), analyse des commentaires libres.

Un fichier reçu (CSV ou Excel) est importé en une transaction : chaque ligne devient une
fenêtre (enregistrement, décalage, durée annotée) et un label en ajout seul. Le commentaire
brut est toujours conservé dans `conditions.comment`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from blanci.db import utc_now, window_id_for

POSITIVE_LABELS = ("blanci", "blanci_solo", "blanci_chorus")
LABELS = POSITIVE_LABELS + (
    "blanci_uncertain",
    "bird",
    "amphibian",
    "orthoptera",
    "amphibian_contact_call",
    "rain",
    "artefact_in_bag",
    "background",
    "other",
    "uncertain",
)
QUALITIES = ("A", "B", "C")
SOURCES = ("import", "similarity", "active", "random", "audit")
TARGET_SPECIES = "Anomaloglossus blanci"

Kind = Literal["positive", "negative"]


def normalize(text: str) -> str:
    """Minuscules, sans accents ni espaces multiples."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower()).strip()


@dataclass(frozen=True)
class SpeciesRule:
    pattern: str  # sur texte normalisé
    family: str  # label
    name: str
    genus: str | None = None
    generic: bool = False  # « Genre sp. », écarté si une espèce du même genre est citée


# Faux amis relevés lors de l'annotation (H14). Noms d'oiseaux vernaculaires [À VÉRIFIER].
SPECIES_RULES = (
    SpeciesRule(r"fourmilier tachete", "bird", "Fourmilier tacheté"),
    SpeciesRule(r"moucherolle manakin", "bird", "Moucherolle manakin"),
    SpeciesRule(r"tangara mordore", "bird", "Tangara mordoré"),
    SpeciesRule(r"pigeon plombe", "bird", "Pigeon plombé"),
    SpeciesRule(r"eveque de rothschild", "bird", "Évêque de Rothschild"),
    SpeciesRule(r"scleru?r?e a bec court", "bird", "Sclérure à bec court"),
    SpeciesRule(r"scleru?r?e obscure", "bird", "Sclérure obscure"),
    SpeciesRule(r"myrmidon", "bird", "Myrmidon"),
    SpeciesRule(r"psittacide", "bird", "Psittacidae"),
    SpeciesRule(r"pic a cou rouge", "bird", "Pic à cou rouge"),
    SpeciesRule(r"\bmartinet\b", "bird", "Martinet"),
    SpeciesRule(r"\bpiau\b", "bird", "Piauhau hurleur"),  # « piau » : Lipaugus vociferans ? [À VÉRIFIER]
    SpeciesRule(r"(?:adenomera |a\. ?)?andreae", "amphibian", "Adenomera andreae"),
    SpeciesRule(
        r"(?:hyalinobatrachium |h\. ?)?cappellei",
        "amphibian",
        "Hyalinobatrachium cappellei",
        "Hyalinobatrachium",
    ),
    SpeciesRule(
        r"(?:hyalinobatrachium |h\. ?)?mondolfii",
        "amphibian",
        "Hyalinobatrachium mondolfii",
        "Hyalinobatrachium",
    ),
    SpeciesRule(
        r"(?:hyalinobatrachium |h\. ?)?iaspidiense",
        "amphibian",
        "Hyalinobatrachium iaspidiense",
        "Hyalinobatrachium",
    ),
    SpeciesRule(
        r"hyalinobatrachium", "amphibian", "Hyalinobatrachium sp.", "Hyalinobatrachium", True
    ),
    # « A. hahneli » : Ameerega hahneli (Dendrobatidae) ; la feuille de route écrit Allobates. [À VÉRIFIER]
    SpeciesRule(r"(?:ameerega |a\. ?)?hahneli", "amphibian", "Ameerega hahneli"),
    SpeciesRule(r"(?:allobates |a\.? ?)?femoralis", "amphibian", "Allobates femoralis"),
    SpeciesRule(r"(?:amazophrynella )?\bteko\b", "amphibian", "Amazophrynella teko"),
    SpeciesRule(r"otophryne", "amphibian", "Otophryne sp."),
)

CONTACT_CALL = re.compile(r"amphibiens? de contact|cris? d.?interaction|cris? de contact")
IN_BAG = re.compile(r"\bdans (?:le |un )?sac\b")
ORTHOPTERA = re.compile(r"orthoptere|grillon|sauterelle")
BIRD = re.compile(r"\boiseaux?\b")
AMPHIBIAN = re.compile(r"\bamphibiens?\b")
UNCERTAIN = re.compile(r"suspect|probablement pas blanci")
RAIN = re.compile(r"\bpluie\b")
BACKGROUND = re.compile(r"aucun chant|rien d.?audible")
CHORUS = re.compile(r"\bchoeur\b|\bchorus\b|plusieurs (?:individus|males|chanteurs)")
SOLO = re.compile(r"\bsolo\b|individu (?:isole|seul)|male seul")

CONDITION_TAGS = {
    "second_plan": re.compile(r"second plan|arriere[- ]plan"),
    "distant": re.compile(r"lointaine?s?\b"),
    "rain": RAIN,
    "noise": re.compile(r"bruits? parasites?|bruite"),
    "uncertain_mention": re.compile(r"\?"),
}


@dataclass
class ParsedComment:
    label: str
    species: str | None
    quality: str | None
    conditions: dict[str, Any] = field(default_factory=dict)


def find_species(norm: str) -> list[SpeciesRule]:
    """Espèces citées, dans l'ordre d'apparition dans le texte."""
    found: list[tuple[int, SpeciesRule]] = []
    taken: list[tuple[int, int]] = []
    for rule in SPECIES_RULES:
        for m in re.finditer(rule.pattern, norm):
            if any(m.start() < end and start < m.end() for start, end in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), rule))
    found.sort(key=lambda item: item[0])
    rules = [rule for _, rule in found]
    specific_genera = {r.genus for r in rules if r.genus and not r.generic}
    rules = [r for r in rules if not (r.generic and r.genus in specific_genera)]
    return list(dict.fromkeys(rules))


def infer_quality(tags: list[str], co_occurring: list[str]) -> str:
    """Qualité A/B/C (Courtois et al. 2025) déduite du commentaire, faute de colonne.

    C : lointain ou second plan ; B : chevauchement (pluie, bruit, autre espèce) ; A sinon.
    """
    if {"distant", "second_plan"} & set(tags):
        return "C"
    if {"rain", "noise"} & set(tags) or co_occurring:
        return "B"
    return "A"


def parse_comment(comment: str | None, kind: Kind, quality: str | None = None) -> ParsedComment:
    raw = "" if comment is None or pd.isna(comment) else str(comment).strip()
    norm = normalize(raw)
    tags = [tag for tag, pattern in CONDITION_TAGS.items() if pattern.search(norm)]
    rules = find_species(norm)
    species = [rule.name for rule in rules]
    conditions: dict[str, Any] = {"tags": tags, "comment": raw}

    if kind == "positive":
        if species:
            conditions["co_occurring"] = species
        if quality is None:
            quality = infer_quality(tags, species)
            conditions["quality_inferred"] = True
        label = "blanci_chorus" if CHORUS.search(norm) else "blanci_solo" if SOLO.search(norm) else "blanci"
        return ParsedComment(label, TARGET_SPECIES, quality, conditions)

    if IN_BAG.search(norm):
        label = "artefact_in_bag"
    elif CONTACT_CALL.search(norm):
        label = "amphibian_contact_call"
    elif rules:
        label = rules[0].family
    elif ORTHOPTERA.search(norm):
        label = "orthoptera"
    elif BIRD.search(norm):
        label = "bird"
    elif AMPHIBIAN.search(norm):
        label = "amphibian"
    elif UNCERTAIN.search(norm):
        label = "uncertain"
    elif RAIN.search(norm):
        label = "rain"
    elif BACKGROUND.search(norm):
        label = "background"
    else:
        label = "other"
    return ParsedComment(label, "; ".join(species) or None, None, conditions)


# --- Lecture des fichiers reçus -------------------------------------------------------------


def read_annotation_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"encodage non reconnu : {path}")


def column_key(name: str) -> str:
    """« Début (s) » → « debut » : sans casse, accents, unités entre parenthèses ni ponctuation."""
    text = re.sub(r"\(.*?\)", "", normalize(name))
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def detect_columns(df: pd.DataFrame, candidates: dict[str, list[str]]) -> dict[str, str]:
    """Champ → nom de colonne réel, premier candidat trouvé."""
    by_key = {column_key(c): c for c in df.columns}
    found = {}
    for key, names in candidates.items():
        for name in names:
            if column_key(name) in by_key:
                found[key] = by_key[column_key(name)]
                break
    return found


def parse_offset(value: Any, unit: str, window_s: float) -> float:
    """Secondes depuis un nombre, « mm:ss » ou « hh:mm:ss » ; ou indice de fenêtre."""
    if isinstance(value, str) and ":" in value:
        seconds = 0.0
        for part in value.strip().split(":"):
            seconds = seconds * 60 + float(part.replace(",", "."))
    else:
        seconds = float(str(value).replace(",", "."))
    return seconds * window_s if unit == "window_index" else seconds


def file_key(name: str) -> str:
    """Nom de fichier sans dossier ni extension, en minuscules."""
    return Path(str(name).replace("\\", "/")).stem.lower()


def _normalize_quality(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    letter = str(value).strip().upper()[:1]
    return letter if letter in QUALITIES else None


@dataclass
class LabelImportReport:
    path: Path
    columns: dict[str, str]
    rows: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[tuple[int, str]] = field(default_factory=list)
    already_imported: bool = False
    inserted: int = 0

    def summary(self) -> str:
        labels = Counter(r["label"] for r in self.rows)
        qualities = Counter(r["quality"] or "-" for r in self.rows)
        species = Counter(r["species"] or "-" for r in self.rows)
        other = [r["comment"] for r in self.rows if r["label"] == "other"]
        lines = [
            f"{self.path.name} : {len(self.rows)} lignes lues, colonnes {self.columns}",
            "  labels : " + ", ".join(f"{k}={v}" for k, v in labels.most_common()),
            "  qualité : " + ", ".join(f"{k}={v}" for k, v in sorted(qualities.items())),
            "  espèces : " + ", ".join(f"{k}={v}" for k, v in species.most_common()),
        ]
        if other:
            lines.append(f"  {len(other)} commentaires non reconnus (label other) :")
            lines += [f"    - {c!r}" for c in sorted(set(other))]
        if self.unresolved:
            lines.append(f"  {len(self.unresolved)} lignes sans enregistrement correspondant :")
            lines += [f"    - ligne {i} : {msg}" for i, msg in self.unresolved[:20]]
        if self.already_imported:
            lines.append("  déjà importé (même SHA-256) : rien n'est ajouté")
        return "\n".join(lines)


def _resolve_recordings(con: sqlite3.Connection) -> dict[str, list[sqlite3.Row]]:
    index: dict[str, list[sqlite3.Row]] = {}
    for row in con.execute("SELECT recording_id, path, site, mic_id, duration_s FROM recordings"):
        index.setdefault(file_key(row["path"]), []).append(row)
    return index


def import_label_file(
    con: sqlite3.Connection,
    path: Path,
    cfg: dict[str, Any],
    kind: Kind | None,
    annotator: str | None = None,
    dry_run: bool = False,
    allow_partial: bool = False,
) -> LabelImportReport:
    """Importe un fichier d'annotation, tout ou rien (sauf `allow_partial`)."""
    icfg = cfg["labels"]["import"]
    df = read_annotation_table(path)
    columns = detect_columns(df, icfg["columns"])
    report = LabelImportReport(path=path, columns=columns)
    missing = [key for key in ("file", "offset_s") if key not in columns]
    if missing:
        raise ValueError(
            f"{path.name} : colonnes {missing} introuvables parmi {list(df.columns)} ; "
            "compléter labels.import.columns dans la config"
        )
    if kind is None and "label" not in columns:
        raise ValueError(f"{path.name} : pas de colonne label, préciser --kind positive|negative")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report.already_imported = (
        con.execute("SELECT 1 FROM imports WHERE file_sha256 = ?", (digest,)).fetchone() is not None
    )
    recordings = _resolve_recordings(con)
    window_s = float(icfg["window_s"])

    for i, record in enumerate(df.to_dict("records"), start=2):  # ligne 1 = en-tête
        row_kind = kind
        if row_kind is None:
            value = normalize(record[columns["label"]])
            positive = "blanci" in value and not re.search(r"\bpas\b", value)
            row_kind = "positive" if positive else "negative"
        quality = _normalize_quality(record.get(columns.get("quality", ""), None))
        parsed = parse_comment(record.get(columns.get("comment", ""), None), row_kind, quality)

        matches = recordings.get(file_key(record[columns["file"]]), [])
        for key in ("site", "mic_id"):
            if key in columns and len(matches) > 1:
                wanted = normalize(record[columns[key]])
                matches = [m for m in matches if normalize(m[key] or "") == wanted]
        try:
            offset = parse_offset(record[columns["offset_s"]], icfg["offset_unit"], window_s)
        except ValueError:
            report.unresolved.append((i, f"décalage illisible : {record[columns['offset_s']]!r}"))
            continue
        if len(matches) != 1:
            status = "introuvable" if not matches else f"ambigu ({len(matches)} candidats)"
            report.unresolved.append((i, f"{record[columns['file']]} : {status}"))
            continue
        recording = matches[0]
        if offset < 0 or offset + window_s > recording["duration_s"] + 0.05:
            report.unresolved.append((i, f"décalage {offset} s hors de l'enregistrement"))
            continue
        report.rows.append(
            {
                "recording_id": recording["recording_id"],
                "offset_s": offset,
                "label": parsed.label,
                "quality": parsed.quality,
                "species": parsed.species,
                "conditions": parsed.conditions,
                "comment": parsed.conditions["comment"],
            }
        )

    if dry_run or report.already_imported or (report.unresolved and not allow_partial):
        return report

    created_at = utc_now()
    annotator = annotator or icfg["annotator"]
    with con:
        for row in report.rows:
            wid = window_id_for(row["recording_id"], row["offset_s"])
            con.execute(
                "INSERT OR IGNORE INTO windows (window_id, recording_id, offset_s, dur_s) "
                "VALUES (?, ?, ?, ?)",
                (wid, row["recording_id"], round(row["offset_s"], 2), window_s),
            )
            con.execute(
                "INSERT INTO labels (window_id, label, quality, species, conditions, annotator, "
                "source, created_at) VALUES (?, ?, ?, ?, ?, ?, 'import', ?)",
                (
                    wid,
                    row["label"],
                    row["quality"],
                    row["species"],
                    json.dumps(row["conditions"], ensure_ascii=False),
                    annotator,
                    created_at,
                ),
            )
        con.execute(
            "INSERT INTO imports (file_sha256, path, n_labels, imported_at) VALUES (?, ?, ?, ?)",
            (digest, str(path), len(report.rows), created_at),
        )
    report.inserted = len(report.rows)
    return report
