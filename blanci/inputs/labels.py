"""Schéma de labels (§5), analyse des commentaires libres, lecture des tableaux d'un outil
externe (réponses sur des extraits exportés, `selection.import_clip_labels`)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

POSITIVE_LABELS = ("blanci", "blanci_solo", "blanci_chorus")
LABELS = POSITIVE_LABELS + (
    "blanci_uncertain",
    "bird",
    "amphibian",
    "orthoptera",
    "amphibian_contact_call",
    "false_friend",
    "rain",
    "artefact_in_bag",
    "background",
    "other",
    "uncertain",
)
QUALITIES = ("A", "B", "C")
# Origine d'un label = méthode de sélection qui a proposé la fenêtre (DECISIONS n° 99) : les
# tours d'annotation se comparent par source (lesquels trouvent des positifs, lesquels des
# négatifs durs). « bulk » : label propagé sans écoute à tout un groupe homogène (étiquetage en
# bloc, `selection.label_cluster`) ; les fenêtres écoutées de ce groupe restent « cluster ».
# « gap » : faux négatif suspect, fenêtre négative encadrée de positives (n° 102).
# « flag » : enregistrement écarté par un drapeau (horloge douteuse…), écouté pour juger
# ce qu'il vaut (n° 154) ; jamais mêlé à l'audit aléatoire, qui mesure le rappel.
# « external » : réponse d'un outil externe (Raven, Audacity, YAPAT…) sur des extraits exportés
# (`selection.import_clip_labels`).
# « plan » : lot 1 et jeu de test tirés par plan, sans détecteur (`annotation.plan`, n° 196).
SOURCES = (
    "similarity",
    "active",
    "random",
    "audit",
    "mining",
    "phenology",
    "suspect",
    "coverage",
    "cluster",
    "bulk",
    "gap",
    "external",
    "flag",
    "plan",
)


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

CONDITION_TAGS = {
    "second_plan": re.compile(r"second plan|arriere[- ]plan"),
    "distant": re.compile(r"lointaine?s?\b"),
    "rain": re.compile(r"\bpluie\b"),
    "noise": re.compile(r"bruits? parasites?|bruite"),
    "uncertain_mention": re.compile(r"\?"),
}


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


# --- Tableaux d'un outil externe -----------------------------------------------------------


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
    le premier de `candidates`.
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
