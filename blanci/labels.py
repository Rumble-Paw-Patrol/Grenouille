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
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
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
    SpeciesRule(
        r"\bpiau\b", "bird", "Piauhau hurleur"
    ),  # « piau » : Lipaugus vociferans ? [À VÉRIFIER]
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
    # « A. hahneli » : Ameerega hahneli (Dendrobatidae) ; la feuille de route
    # écrit Allobates. [À VÉRIFIER]
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


def comment_fields(comment: str | None) -> dict[str, Any]:
    """Champs tirés d'un commentaire libre, sans rien décider du label : conditions (pluie,
    lointain, second plan, bruit…) et espèces citées. Sert au poste d'annotation, où le label
    est choisi par l'annotateur et le commentaire écrit à côté."""
    if not comment or not str(comment).strip():
        return {}
    norm = normalize(comment)
    fields: dict[str, Any] = {
        "tags": [tag for tag, pattern in CONDITION_TAGS.items() if pattern.search(norm)]
    }
    species = [rule.name for rule in find_species(norm)]
    if species:
        fields["co_occurring"] = species
    return fields


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
        # Sans commentaire, la qualité reste inconnue : un « vrai » sec ne dit pas que le
        # chant était clair, et un A par défaut viderait de sens le rappel par qualité (§6).
        if quality is None and raw:
            quality = infer_quality(tags, species)
            conditions["quality_inferred"] = True
        label = (
            "blanci_chorus"
            if CHORUS.search(norm)
            else "blanci_solo"
            if SOLO.search(norm)
            else "blanci"
        )
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


def _contains_words(column: str, candidate: str) -> bool:
    """« nom_de_l_enregistrement » contient « nom » et « enregistrement », mais pas « time »."""
    words, wanted = column.split("_"), candidate.split("_")
    return any(words[i : i + len(wanted)] == wanted for i in range(len(words) - len(wanted) + 1))


def detect_columns(df: pd.DataFrame, candidates: dict[str, list[str]]) -> dict[str, str]:
    """Champ → nom de colonne réel.

    Deux passes : d'abord les égalités exactes, ensuite les libellés composés
    (« Nom de l'enregistrement » pour `fichier`). Une colonne ne sert qu'à un seul champ,
    le premier de la config — d'où l'ordre file, offset, comment, verdict, label, score.
    """
    by_key = {column_key(c): c for c in df.columns}
    found: dict[str, str] = {}
    taken: set[str] = set()

    for match_exactly in (True, False):
        for field_name, names in candidates.items():
            if field_name in found:
                continue
            for name in names:
                wanted = column_key(name)
                for key, column in by_key.items():
                    if key in taken:
                        continue
                    hit = key == wanted if match_exactly else _contains_words(key, wanted)
                    if hit:
                        found[field_name] = column
                        taken.add(key)
                        break
                if field_name in found:
                    break
    return found


def parse_offset(value: Any, unit: str, window_s: float) -> float:
    """Secondes depuis un nombre, « mm:ss », « hh:mm:ss », une durée Excel, ou un indice.

    Excel rend une cellule au format horaire en `datetime.time` ou en `Timedelta` selon son
    format : les deux comptent depuis le début de l'enregistrement, pas depuis une date.
    """
    if isinstance(value, timedelta):
        seconds = value.total_seconds()
    elif isinstance(value, time):
        seconds = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    elif isinstance(value, datetime):  # cellule horaire lue comme date du 1900-01-00
        seconds = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    elif isinstance(value, str) and ":" in value:
        seconds = 0.0
        for part in value.strip().split(":"):
            seconds = seconds * 60 + float(part.replace(",", "."))
    else:
        seconds = float(str(value).replace(",", "."))
    return seconds * window_s if unit == "window_index" else seconds


# Verdicts d'une colonne « vérif manuelle ». Tout ce qui n'est reconnu ni ici ni comme
# commentaire est signalé ligne par ligne : un « oui » pris pour un négatif passerait inaperçu.
VERDICT_YES = re.compile(
    r"^(o|oui|y|yes|v|vrai|true|ok|x|1|1\.0|confirme[e]?|valide[e]?|certain[e]?|"
    r"present[e]?|avere[e]?|blanci)$"
)
VERDICT_NO = re.compile(
    r"^(n|non|no|f|faux|false|ko|0|0\.0|rejete[e]?|invalide|absent[e]?|"
    r"pas blanci|non blanci|erreur|faux positif)$"
)


# Labels qui ne reconnaissent rien de précis : ils ne valent pas verdict.
VAGUE_LABELS = ("other", "uncertain")

# « à vérif », « à conf », « à revoir » : l'expert n'a pas encore tranché. Ni label ni
# blocage : la ligne est mise de côté et listée, elle reviendra dans une file de vérification.
VERDICT_PENDING = re.compile(r"^a (verif|verifier|conf|confirmer|revoir|reecouter)\b")

# Colonnes sans nom où l'on range des commentaires (« Colonne1 » d'un tableau Excel,
# « Unnamed: 12 » d'une cellule sans en-tête) : leur texte rejoint le commentaire.
ANONYMOUS_COLUMN = re.compile(r"^(colonne|column|unnamed)_?\d*$")


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value))


def parse_verdict(value: Any) -> bool | None:
    """True / False depuis une colonne de vérification, None si la valeur n'est pas concluante."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | float | np.number) and float(value) in (0.0, 1.0):
        return bool(value)
    norm = normalize(value)
    if VERDICT_YES.match(norm):
        return True
    if VERDICT_NO.match(norm):
        return False
    if re.search(r"\bblanci\b", norm):  # « blanci lointain », « blanci malgré la pluie »
        return not re.search(r"\b(pas|non|aucun|sans)\b", norm)
    return None


# Clé S3 de l'ancien prestataire : « 2353462-2la03550_20260108_143000.flac ». Le préfixe
# numérique n'est retiré que devant un nom Song Meter, pour ne jamais tronquer un vrai nom.
S3_PREFIX = re.compile(r"^\d+-(?=.+_\d{8}_\d{6}$)")


def file_key(name: str) -> str:
    """Nom de fichier sans dossier, extension ni préfixe S3, en minuscules."""
    stem = Path(str(name).replace("\\", "/")).stem.lower()
    return S3_PREFIX.sub("", stem)


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
    unverified: int = 0  # verdict vide : détection jamais écoutée, pas un label
    pending: list[tuple[int, str]] = field(default_factory=list)  # « à vérif », « à conf »
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
        scores = [
            r["conditions"]["previous_model_score"]
            for r in self.rows
            if "previous_model_score" in r["conditions"]
        ]
        if scores:
            lines.append(
                f"  score de l'ancien modèle : {len(scores)} lignes, "
                f"min {min(scores):.3f}, médiane {sorted(scores)[len(scores) // 2]:.3f}, "
                f"max {max(scores):.3f}"
            )
        silent = sum(1 for c in other if not c)
        if silent:
            lines.append(f"  {silent} négatifs sans commentaire (label other : espèce inconnue)")
        if len(other) > silent:
            lines.append(f"  {len(other) - silent} commentaires non reconnus (label other) :")
            lines += [f"    - {c!r}" for c in sorted({c for c in other if c})]
        if self.unverified:
            lines.append(
                f"  {self.unverified} lignes sans vérification (détections jamais écoutées) : "
                "ignorées, ce ne sont pas des labels"
            )
        if self.pending:
            lines.append(f"  {len(self.pending)} lignes en attente de vérification, de côté :")
            lines += [f"    - ligne {i} : {msg}" for i, msg in self.pending]
        if self.unresolved:
            lines.append(f"  {len(self.unresolved)} lignes non résolues (bloquent l'import) :")
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
    if kind is None and not ({"verdict", "label"} & set(columns)):
        raise ValueError(
            f"{path.name} : ni colonne de vérification ni colonne label parmi {list(df.columns)} ; "
            "préciser --kind positive|negative, ou compléter labels.import.columns"
        )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report.already_imported = (
        con.execute("SELECT 1 FROM imports WHERE file_sha256 = ?", (digest,)).fetchone() is not None
    )
    recordings = _resolve_recordings(con)
    window_s = float(icfg["window_s"])

    # Colonne qui porte le verdict : « vérif manuelle » de préférence, sinon « label ».
    verdict_column = columns.get("verdict") or columns.get("label")
    comment_columns = ([columns["comment"]] if "comment" in columns else []) + [
        c
        for c in df.columns
        if ANONYMOUS_COLUMN.match(column_key(str(c))) and c not in columns.values()
    ]
    headers = {str(c).strip() for c in df.columns}

    for i, record in enumerate(df.to_dict("records"), start=2):  # ligne 1 = en-tête
        row_kind = kind
        verdict_text = record.get(verdict_column) if verdict_column else None
        if row_kind is None:
            if _is_blank(verdict_text):
                # Détection de l'ancien modèle que personne n'a écoutée : pas un label.
                report.unverified += 1
                continue
            if isinstance(verdict_text, str) and VERDICT_PENDING.match(normalize(verdict_text)):
                report.pending.append((i, f"{record[columns['file']]} : {verdict_text!r}"))
                continue
            verdict = parse_verdict(verdict_text)
            if verdict is None:
                # Pas un oui/non. La cellule nomme-t-elle un faux ami reconnu ? Si oui, c'est
                # un négatif : l'expert a écrit ce qu'il a entendu à la place d'A. blanci.
                # Sinon (« à revoir », « ? »), on ne devine pas : la ligne est signalée.
                probe = parse_comment(verdict_text, "negative")
                if probe.label in VAGUE_LABELS:
                    report.unresolved.append(
                        (i, f"vérification illisible : {verdict_text!r} (utiliser --kind)")
                    )
                    continue
                verdict = False
            row_kind = "positive" if verdict else "negative"

        quality = _normalize_quality(record.get(columns.get("quality", ""), None))
        # Commentaire = texte de la vérification (il nomme souvent le faux ami ou la qualité)
        # + colonne commentaire + colonnes sans nom. Un texte égal à un en-tête (« Colonne1 »
        # recopié dans une cellule) est un reste de mise en forme, pas un commentaire.
        texts: list[str] = []
        for value in [verdict_text, *(record.get(c) for c in comment_columns)]:
            if isinstance(value, str) and value.strip() and value.strip() not in headers:
                if value.strip() not in texts:
                    texts.append(value.strip())
        parsed = parse_comment(" | ".join(texts) or None, row_kind, quality)
        if "score" in columns:
            score = record[columns["score"]]
            if not pd.isna(score):
                parsed.conditions["previous_model_score"] = float(score)

        referenced = str(record[columns["file"]])
        matches = recordings.get(file_key(referenced), [])
        if len(matches) > 1:
            # Même nom en .wav et en .flac : on préfère l'extension citée, si elle existe.
            suffix = Path(referenced.replace("\\", "/")).suffix.lower()
            same_suffix = [m for m in matches if Path(m["path"]).suffix.lower() == suffix]
            if same_suffix:
                matches = same_suffix
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
            wid = window_id_for(row["recording_id"], row["offset_s"], window_s)
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


# --- Détections d'un détecteur indépendant (Blancinet) ----------------------------------------


@dataclass
class DetectionImportReport:
    model_id: str
    rows: int = 0
    stored: int = 0
    unverified: int = 0  # détections que personne n'a écoutées
    not_found: int = 0  # fichier absent de l'inventaire (autre relevé, doublon écarté)
    ambiguous: int = 0
    unreadable: int = 0


def import_detections(
    con: sqlite3.Connection, path: Path, cfg: dict[str, Any], model_id: str = "blancinet"
) -> DetectionImportReport:
    """Range toutes les détections d'un export (vérifiées ou non) comme scores de `model_id`.

    Ce ne sont pas des labels : un score dit où le détecteur a entendu A. blanci, pas ce qu'un
    humain a entendu. Ils servent à ne pas tirer de négatif présumé là où le détecteur
    entend A. blanci (`dataset.detected_blanci`). Réimporter remplace les scores, sans rien
    dupliquer. Le fichier reçu n'est jamais modifié.
    """
    icfg = cfg["labels"]["import"]
    df = read_annotation_table(Path(path))
    columns = detect_columns(df, icfg["columns"])
    for needed in ("file", "offset_s", "score"):
        if needed not in columns:
            raise ValueError(f"colonne {needed!r} introuvable dans {Path(path).name}")
    recordings = _resolve_recordings(con)
    window_s = float(icfg["window_s"])
    report = DetectionImportReport(model_id, rows=len(df))
    windows, scores = [], []
    for record in df.to_dict("records"):
        matches = recordings.get(file_key(str(record[columns["file"]])), [])
        if len(matches) != 1:
            report.not_found += not matches
            report.ambiguous += len(matches) > 1
            continue
        score = record[columns["score"]]
        try:
            offset = parse_offset(record[columns["offset_s"]], icfg["offset_unit"], window_s)
        except ValueError:
            report.unreadable += 1
            continue
        if pd.isna(score):
            report.unreadable += 1
            continue
        rid = matches[0]["recording_id"]
        wid = window_id_for(rid, offset, window_s)
        windows.append((wid, rid, round(offset, 2), window_s))
        scores.append((wid, model_id, float(score)))
        if "verdict" in columns and _is_blank(record[columns["verdict"]]):
            report.unverified += 1
    with con:
        con.executemany(
            "INSERT OR IGNORE INTO windows (window_id, recording_id, offset_s, dur_s) "
            "VALUES (?, ?, ?, ?)",
            windows,
        )
        con.executemany(
            "INSERT INTO scores (window_id, model_id, score) VALUES (?, ?, ?) "
            "ON CONFLICT(window_id, model_id) DO UPDATE SET score = excluded.score",
            scores,
        )
        con.execute(
            "INSERT INTO models (model_id, kind, name, version, params_json, created_at) "
            "VALUES (?, 'detector', ?, ?, ?, ?) ON CONFLICT(model_id) DO UPDATE SET "
            "params_json = excluded.params_json",
            (
                model_id,
                model_id,
                Path(path).stem,
                json.dumps({"source": Path(path).name, "windows": len(scores)}),
                utc_now(),
            ),
        )
    report.stored = len(scores)
    return report
