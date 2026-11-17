"""Étiquettes d'intervalle et qualités proposées par le poste d'annotation, propres au projet :
`annotations/etiquettes.json`.

- **Étiquettes** : celles du schéma (A. blanci, plusieurs, ?, faux ami) et celles ajoutées par
  les annotateurs (nom et couleur, code `custom_<nom simplifié>`, stocké tel quel dans
  `intervals.label` ; l'entraînement les ignore). Le « − » du poste en retire une de la liste
  proposée sans l'oublier : elle reste dans `retirees`, et le « + » la propose de nouveau, avec
  celles déjà présentes dans les annotations. Réajouter un nom connu (même code, ou même nom à
  la casse près) reprend l'étiquette existante : jamais deux codes pour une même espèce.
- **Qualités** : A, B, C au départ ; le poste en ajoute (D…) ou en retire de la même façon.

Ancien format (avant le 17/11/2026) : {code : {name, color}} des seules étiquettes ajoutées ;
relu tel quel (les quatre du schéma devant), réécrit au nouveau format à la première
modification.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from blanci.core.config import project_path
from blanci.service import CUSTOM_PREFIX as PREFIX

LABELS_FILE = project_path("annotations/etiquettes.json")
DEFAULT_COLOR = "#B08CF2"
QUALITY_PATTERN = r"[A-Za-z0-9+\-]{1,8}"


def is_custom(code: str) -> bool:
    return str(code).startswith(PREFIX)


def slug(name: str) -> str:
    """« Crapaud ? » → « custom_crapaud » (sans accents, minuscules, tirets bas)."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return PREFIX + re.sub(r"[^a-z0-9]+", "_", plain.lower()).strip("_")


def name_of_code(code: str) -> str:
    """Nom lisible d'un code sans nom connu : « custom_crapaud_buffle » → « Crapaud buffle »."""
    return str(code).removeprefix(PREFIX).replace("_", " ").strip().capitalize() or str(code)


def _builtin() -> list[dict[str, str]]:
    """Les quatre étiquettes du schéma, avec leur couleur du poste."""
    from blanci.annotation.style import LABEL_COLORS
    from blanci.annotation.workbench import INTERVAL_ANSWERS

    return [
        {"code": c, "name": n, "color": LABEL_COLORS.get(c, DEFAULT_COLOR)}
        for c, n in INTERVAL_ANSWERS
    ]


def _entry(code: str, item: dict) -> dict[str, str] | None:
    name = item.get("name") or item.get("nom")
    if not name:
        return None
    color = str(item.get("color") or item.get("couleur") or DEFAULT_COLOR)
    return {"code": str(code), "name": str(name), "color": color}


def read_set(path: Path | None = None) -> dict:
    """{"labels", "removed"} (listes de {code, name, color}, dans l'ordre proposé) et
    {"qualities", "removed_qualities"} (listes de codes). Sans fichier : le schéma et A, B, C."""
    from blanci.inputs.labels import QUALITIES

    path = path or LABELS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    if "etiquettes" not in data:  # ancien format : étiquettes ajoutées seulement
        custom = [_entry(c, v) for c, v in data.items() if isinstance(v, dict) and is_custom(c)]
        return {
            "labels": _builtin() + [e for e in custom if e],
            "removed": [],
            "qualities": list(QUALITIES),
            "removed_qualities": [],
        }

    def entries(key: str) -> list[dict[str, str]]:
        out = [_entry(i.get("code", ""), i) for i in data.get(key, []) if isinstance(i, dict)]
        return [e for e in out if e and e["code"]]

    labels = entries("etiquettes")
    removed = [e for e in entries("retirees") if e["code"] not in {x["code"] for x in labels}]
    qualities = [str(q) for q in data.get("qualites", QUALITIES)]
    removed_q = [str(q) for q in data.get("qualites_retirees", []) if str(q) not in qualities]
    return {
        "labels": labels,
        "removed": removed,
        "qualities": qualities,
        "removed_qualities": removed_q,
    }


def _write(state: dict, path: Path | None) -> None:
    path = path or LABELS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    def entries(items: list[dict]) -> list[dict]:
        return [{"code": e["code"], "nom": e["name"], "couleur": e["color"]} for e in items]

    data = {
        "etiquettes": entries(state["labels"]),
        "retirees": entries(state["removed"]),
        "qualites": state["qualities"],
        "qualites_retirees": state["removed_qualities"],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def known_labels(path: Path | None = None) -> dict[str, dict[str, str]]:
    """{code : {"name", "color"}} de toutes les étiquettes connues, proposées ou retirées (pour
    nommer les intervalles déjà posés)."""
    state = read_set(path)
    return {e["code"]: {"name": e["name"], "color": e["color"]} for e in _all(state)}


def _all(state: dict) -> list[dict[str, str]]:
    seen, out = set(), []
    for e in [*state["labels"], *state["removed"], *_builtin()]:
        if e["code"] not in seen:
            seen.add(e["code"])
            out.append(e)
    return out


def _find(state: dict, name: str, code: str | None) -> dict[str, str] | None:
    """L'étiquette connue de ce code, de ce nom (à la casse près) ou de ce nom simplifié."""
    folded, simple = name.casefold(), slug(name)
    for e in _all(state):
        if e["code"] == code or e["name"].casefold() == folded or e["code"] == simple:
            return e
    return None


def add_label(
    name: str, color: str | None = None, code: str | None = None, path: Path | None = None
) -> str:
    """Propose une étiquette (à la fin de la liste) ; renvoie son code. Un nom ou un code déjà
    connus (étiquette retirée, du schéma, déjà proposée) reprennent l'étiquette existante, qui
    ne change de couleur que si `color` est donnée. `code` : celui d'une étiquette vue dans les
    annotations (« + » qui la propose), sinon déduit du nom."""
    name = name.strip()
    if not name:
        raise ValueError("nom d'étiquette vide")
    if color is not None and not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError(f"couleur inconnue : {color!r} (attendue : #RRGGBB)")
    state = read_set(path)
    found = _find(state, name, code)
    if found is None:
        new = code or slug(name)
        if new == PREFIX:
            raise ValueError(f"nom inutilisable : {name!r} (lettres ou chiffres attendus)")
        found = {"code": new, "name": name, "color": DEFAULT_COLOR}
    entry = dict(found, color=(color or found["color"]).upper())
    if any(e["code"] == entry["code"] for e in state["labels"]):  # déjà proposée : sa place
        state["labels"] = [entry if e["code"] == entry["code"] else e for e in state["labels"]]
    else:
        state["labels"].append(entry)
    state["removed"] = [e for e in state["removed"] if e["code"] != entry["code"]]
    _write(state, path)
    return entry["code"]


def remove_label(code: str, path: Path | None = None) -> None:
    """Retire l'étiquette de la liste proposée, sans l'oublier (le « + » la repropose) ; les
    intervalles déjà posés la gardent."""
    state = read_set(path)
    gone = [e for e in state["labels"] if e["code"] == code]
    if not gone:
        return
    state["labels"] = [e for e in state["labels"] if e["code"] != code]
    state["removed"] = [*gone, *(e for e in state["removed"] if e["code"] != code)]
    _write(state, path)


def add_quality(quality: str, path: Path | None = None) -> str:
    """Propose une qualité (D, A+…), à la fin de la liste ; renvoie son code."""
    quality = quality.strip()
    if not re.fullmatch(QUALITY_PATTERN, quality):
        raise ValueError(f"qualité inutilisable : {quality!r} (1 à 8 lettres ou chiffres)")
    state = read_set(path)
    if quality not in state["qualities"]:
        state["qualities"].append(quality)
    state["removed_qualities"] = [q for q in state["removed_qualities"] if q != quality]
    _write(state, path)
    return quality


def remove_quality(quality: str, path: Path | None = None) -> None:
    """Retire une qualité de la liste proposée (les intervalles déjà notés la gardent)."""
    state = read_set(path)
    if quality not in state["qualities"]:
        return
    state["qualities"] = [q for q in state["qualities"] if q != quality]
    state["removed_qualities"] = [quality, *(q for q in state["removed_qualities"] if q != quality)]
    _write(state, path)


def suggestions(used: list[str], path: Path | None = None) -> list[dict[str, str]]:
    """Étiquettes que le « + » propose : les retirées, et celles des annotations (`used` : codes
    vus dans la base) qui ne sont plus proposées, sans doublon."""
    state = read_set(path)
    active = {e["code"] for e in state["labels"]}
    known = {e["code"]: e for e in _all(state)}
    out, seen = [], set(active)
    for e in state["removed"]:
        if e["code"] not in seen:
            seen.add(e["code"])
            out.append(e)
    for code in used:
        if code and code not in seen:
            seen.add(code)
            out.append(
                known.get(code)
                or {"code": code, "name": name_of_code(code), "color": DEFAULT_COLOR}
            )
    return out


# --- Noms d'avant le 17/11/2026, gardés pour `blanci import-labels` et le paquet d'écoute ---


def read_custom(path: Path | None = None) -> dict[str, dict[str, str]]:
    """{code : {"name", "color"}} des étiquettes ajoutées (hors schéma) encore proposées."""
    return {
        e["code"]: {"name": e["name"], "color": e["color"]}
        for e in read_set(path)["labels"]
        if is_custom(e["code"])
    }


def add_custom(name: str, color: str, path: Path | None = None) -> str:
    """`add_label` avec une couleur (recolore une étiquette déjà connue)."""
    return add_label(name, color, path=path)


def remove_custom(code: str, path: Path | None = None) -> None:
    remove_label(code, path)
