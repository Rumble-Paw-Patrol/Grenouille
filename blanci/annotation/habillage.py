"""Habillages du poste d'annotation : palettes (mode jour / nuit × style) et préférences.

Deux étages : `BRAISE_DARK` et `BRAISE_LIGHT` donnent toutes les variables CSS de la page (un
jeu par mode, repris du tableau de bord) ; les autres styles n'écrasent que ce qui change
(teintes, fonds, lueurs). « Braise » est l'habillage d'origine.

Le choix de l'utilisateur (mode, style) est gardé dans un petit fichier JSON par utilisateur,
d'où il passe d'une séance à l'autre ; `style.py` en tire la feuille de style de la page, et le
visualiseur et le bandeau le reçoivent en argument (`look`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

BRAISE_DARK = {
    "bg": "#050B0A",
    "bg-2": "#08120F",
    "surface": "rgba(16, 29, 25, .72)",
    "surface-solid": "#0F1C18",
    "surface-2": "rgba(255, 255, 255, .055)",
    "ink": "#E9F2EC",
    "muted": "#93A89D",
    "line": "rgba(160, 210, 185, .14)",
    "line-fort": "rgba(160, 210, 185, .28)",
    "accent": "#F5A53A",
    "accent-2": "#E4573F",
    "accent-3": "#9E2A6B",
    "accent-ink": "#1A0B03",
    "gold": "#FFE08A",
    "gold-soft": "rgba(255, 210, 110, .12)",
    "teal": "#4FD8C0",
    "ok": "#5FD394",
    "ok-soft": "rgba(95, 211, 148, .13)",
    "warn": "#F5B547",
    "warn-soft": "rgba(245, 181, 71, .13)",
    "bad": "#F07A63",
    "bad-soft": "rgba(240, 122, 99, .14)",
    "nuit-ligne": "rgba(160, 210, 185, .16)",
    "ombre": "0 20px 50px -24px rgba(0, 0, 0, .8), 0 1px 0 rgba(255, 255, 255, .04) inset",
    "aura-1": "rgba(228, 87, 63, .20)",
    "aura-2": "rgba(79, 216, 192, .13)",
    "aura-3": "rgba(158, 42, 107, .18)",
    "voile": "5, 11, 10",
    "fond-side": "linear-gradient(180deg, rgba(6, 14, 12, .94), rgba(4, 9, 8, .97))",
    "fond-bande": "radial-gradient(120% 140% at 0% 0%, #1A0E1C 0%, #070C0B 46%, #040807 100%)",
    "halo-1": "rgba(228, 87, 63, .22)",
    "halo-2": "rgba(79, 216, 192, .10)",
    "titre": "linear-gradient(100deg, #FFFFFF 0%, #FFF3D6 40%, #FFD27A 70%, #F5A53A 100%)",
    "titre-ombre": "drop-shadow(0 2px 10px rgba(4, 8, 7, .9))",
    "fond-fenetre": "rgba(10, 20, 17, .97)",
    "ombre-fenetre": "0 30px 80px -24px rgba(0, 0, 0, .95)",
    "ombre-menu": "0 20px 50px -20px rgba(0, 0, 0, .9)",
}

BRAISE_LIGHT = {
    "bg": "#EDF3EF",
    "bg-2": "#E3ECE6",
    "surface": "rgba(255, 255, 255, .78)",
    "surface-solid": "#FFFFFF",
    "surface-2": "rgba(20, 60, 45, .06)",
    "ink": "#11201A",
    "muted": "#55675E",
    "line": "rgba(20, 70, 50, .14)",
    "line-fort": "rgba(20, 70, 50, .26)",
    "accent": "#B9530D",
    "accent-2": "#C2352A",
    "accent-3": "#8A1F5C",
    "accent-ink": "#FFFFFF",
    "gold": "#A86F00",
    "gold-soft": "rgba(215, 150, 20, .14)",
    "teal": "#137F70",
    "ok": "#237A4A",
    "ok-soft": "rgba(35, 122, 74, .12)",
    "warn": "#9A5E00",
    "warn-soft": "rgba(200, 130, 0, .13)",
    "bad": "#B33A28",
    "bad-soft": "rgba(179, 58, 40, .12)",
    "nuit-ligne": "rgba(20, 70, 50, .14)",
    "ombre": "0 18px 40px -26px rgba(20, 50, 40, .45)",
    "aura-1": "rgba(235, 120, 60, .16)",
    "aura-2": "rgba(30, 160, 140, .13)",
    "aura-3": "rgba(170, 60, 120, .09)",
    "voile": "237, 243, 239",
    "fond-side": "linear-gradient(180deg, rgba(255, 255, 255, .82), rgba(237, 243, 239, .92))",
    "fond-bande": "radial-gradient(120% 140% at 0% 0%, #F8E7DC 0%, #EEF3EF 46%, #E5EDE8 100%)",
    "halo-1": "rgba(235, 120, 60, .16)",
    "halo-2": "rgba(30, 160, 140, .10)",
    "titre": "linear-gradient(100deg, #11201A 0%, #3A2414 45%, #8A3A0C 75%, #B9530D 100%)",
    "titre-ombre": "none",
    "fond-fenetre": "rgba(255, 255, 255, .97)",
    "ombre-fenetre": "0 30px 80px -24px rgba(20, 50, 40, .45)",
    "ombre-menu": "0 20px 50px -20px rgba(20, 50, 40, .35)",
}

# Identifiant (celui que gardent les préférences et que lisent les composants) → nom affiché.
STYLES = {"braise": "Braise", "lagune": "Lagune", "ardoise": "Ardoise"}
# Pastille de chaque style (boutons de l'en-tête).
STYLE_ICONS = {"braise": "🔥", "lagune": "🌊", "ardoise": "🪨"}
# Projet de départ : celui dont les fichiers sont ceux de `annotations/` (voir `projets.py`).
DEFAULT_PROJECT = "defaut"

_NO_AURA = {
    "aura-1": "rgba(0, 0, 0, 0)",
    "aura-2": "rgba(0, 0, 0, 0)",
    "aura-3": "rgba(0, 0, 0, 0)",
}

# Écarts de chaque style par rapport à Braise : (variables en sombre, variables en clair).
# Lagune : bleu-vert d'eau claire ; Ardoise : sobre, gris-bleu, sans lueurs.
_STYLE_VARS = {
    "lagune": (
        {
            "accent": "#4FD8C0",
            "accent-2": "#2A9DB5",
            "accent-3": "#3B5BA8",
            "accent-ink": "#03130F",
            "gold": "#BDF3E8",
            "gold-soft": "rgba(79, 216, 192, .13)",
            "teal": "#F5A53A",
            "bg": "#04090D",
            "bg-2": "#07111A",
            "surface-solid": "#0D1A24",
            "surface": "rgba(13, 26, 36, .72)",
            "voile": "4, 9, 13",
            "aura-1": "rgba(42, 157, 181, .20)",
            "aura-2": "rgba(79, 216, 192, .12)",
            "aura-3": "rgba(59, 91, 168, .20)",
            "halo-1": "rgba(42, 157, 181, .22)",
            "halo-2": "rgba(79, 216, 192, .10)",
            "fond-side": "linear-gradient(180deg, rgba(5, 12, 18, .94), rgba(3, 8, 12, .97))",
            "fond-bande": "radial-gradient(120% 140% at 0% 0%, #0B1B2A 0%, #050B10 46%, "
            "#03070A 100%)",
            "titre": "linear-gradient(100deg, #FFFFFF 0%, #E2FAF4 40%, #9BEBDC 70%, #4FD8C0 100%)",
            "fond-fenetre": "rgba(8, 17, 24, .97)",
        },
        {
            "accent": "#0F7C70",
            "accent-2": "#1F6FA3",
            "accent-3": "#3B4F9A",
            "gold": "#0F6B90",
            "gold-soft": "rgba(20, 130, 150, .13)",
            "teal": "#B9530D",
            "bg": "#EAF3F5",
            "bg-2": "#DEEBEE",
            "voile": "234, 243, 245",
            "aura-1": "rgba(30, 130, 170, .15)",
            "aura-2": "rgba(30, 160, 140, .13)",
            "aura-3": "rgba(60, 90, 170, .09)",
            "halo-1": "rgba(30, 130, 170, .15)",
            "fond-side": "linear-gradient(180deg, rgba(255, 255, 255, .82), "
            "rgba(234, 243, 245, .92))",
            "fond-bande": "radial-gradient(120% 140% at 0% 0%, #D9ECF2 0%, #EAF3F5 46%, "
            "#E0EDEF 100%)",
            "titre": "linear-gradient(100deg, #11201A 0%, #12383A 45%, #0E5E66 75%, #0F7C70 100%)",
        },
    ),
    "ardoise": (
        {
            "accent": "#8FB0FF",
            "accent-2": "#6A8CE0",
            "accent-3": "#4B5FA8",
            "accent-ink": "#08101F",
            "gold": "#C9D8FF",
            "gold-soft": "rgba(143, 176, 255, .12)",
            "teal": "#8FB0FF",
            "bg": "#0C0E12",
            "bg-2": "#11141A",
            "surface-solid": "#161A21",
            "surface": "rgba(22, 26, 33, .78)",
            "voile": "12, 14, 18",
            "ink": "#E6E9EE",
            "muted": "#9AA3B0",
            "line": "rgba(190, 200, 220, .13)",
            "line-fort": "rgba(190, 200, 220, .26)",
            "nuit-ligne": "rgba(190, 200, 220, .15)",
            **_NO_AURA,
            "halo-1": "rgba(143, 176, 255, .10)",
            "halo-2": "rgba(0, 0, 0, 0)",
            "fond-side": "linear-gradient(180deg, rgba(14, 16, 21, .96), rgba(10, 12, 16, .98))",
            "fond-bande": "linear-gradient(180deg, #14171E 0%, #0C0E12 100%)",
            "titre": "linear-gradient(100deg, #FFFFFF 0%, #DCE6FF 60%, #8FB0FF 100%)",
            "fond-fenetre": "rgba(16, 19, 25, .97)",
        },
        {
            "accent": "#2F5BC4",
            "accent-2": "#3A6FD6",
            "accent-3": "#4B5FA8",
            "gold": "#2F5BC4",
            "gold-soft": "rgba(47, 91, 196, .12)",
            "teal": "#2F5BC4",
            "bg": "#F3F4F7",
            "bg-2": "#E8EAEF",
            "voile": "243, 244, 247",
            "ink": "#161A22",
            "muted": "#5B6472",
            "line": "rgba(40, 50, 80, .14)",
            "line-fort": "rgba(40, 50, 80, .26)",
            "nuit-ligne": "rgba(40, 50, 80, .14)",
            "surface-2": "rgba(40, 50, 80, .06)",
            **_NO_AURA,
            "halo-1": "rgba(47, 91, 196, .08)",
            "halo-2": "rgba(0, 0, 0, 0)",
            "fond-side": "linear-gradient(180deg, rgba(255, 255, 255, .85), "
            "rgba(243, 244, 247, .94))",
            "fond-bande": "linear-gradient(180deg, #E6E9F0 0%, #F3F4F7 100%)",
            "titre": "linear-gradient(100deg, #161A22 0%, #26335C 60%, #2F5BC4 100%)",
        },
    ),
}

# Ce que reprennent le visualiseur et le bandeau (leurs variables ont d'autres noms).
COMPONENT_KEYS = ("accent", "accent-2", "accent-ink", "gold", "titre", "bg-2", "surface-solid")


def palette(mode: str, style: str) -> dict[str, str]:
    """Variables CSS de la page pour `mode` ("dark" ou "light") et `style` (identifiant)."""
    base = dict(BRAISE_LIGHT if mode == "light" else BRAISE_DARK)
    if style in _STYLE_VARS:
        base.update(_STYLE_VARS[style][mode == "light"])
    return base


def declarations(values: dict[str, str], indent: str = "  ") -> str:
    return "\n".join(f"{indent}--{k}: {v};" for k, v in values.items())


def prefs_path() -> Path:
    """Fichier des préférences ; `BLANCI_POSTE_PREFS` le déplace (tests)."""
    env = os.environ.get("BLANCI_POSTE_PREFS")
    return Path(env) if env else Path.home() / ".blanci" / "poste.json"


def _read_raw() -> dict:
    try:
        raw = json.loads(prefs_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def read_prefs() -> dict[str, str | None]:
    """{"theme": "light" | "dark" | None (celui du système), "projet": nom du projet actif,
    "style": style du projet actif}. Chaque projet a son style (un projet = un préréglage)."""
    raw = _read_raw()
    theme = raw.get("theme")
    project = raw.get("projet")
    project = project if isinstance(project, str) and project else DEFAULT_PROJECT
    styles = raw.get("styles") if isinstance(raw.get("styles"), dict) else {}
    if raw.get("style") in STYLES:  # ancien fichier : un seul style, celui du projet de départ
        styles = {DEFAULT_PROJECT: raw["style"], **styles}
    style = styles.get(project)
    return {
        "theme": theme if theme in ("light", "dark") else None,
        "projet": project,
        "style": style if style in STYLES else "braise",
    }


def write_prefs(
    theme: str | None = None, style: str | None = None, projet: str | None = None
) -> None:
    """Retient `theme`, `projet` (le projet actif) et/ou `style` (celui du projet actif, après
    changement de projet éventuel) ; ce qui n'est pas donné reste comme avant. Un dossier en
    lecture seule n'empêche pas d'annoter : le choix vaut alors pour la séance seulement."""
    raw = _read_raw()
    prefs = read_prefs()
    styles = raw.get("styles") if isinstance(raw.get("styles"), dict) else {}
    if raw.get("style") in STYLES:
        styles = {DEFAULT_PROJECT: raw["style"], **styles}
    out = {k: v for k, v in raw.items() if k != "style"}  # le reste (`confirm_off`) est gardé
    out |= {"theme": raw.get("theme"), "projet": prefs["projet"], "styles": styles}
    if theme in ("light", "dark"):
        out["theme"] = theme
    if projet:
        out["projet"] = projet
    if style in STYLES:
        out["styles"][out["projet"]] = style
    try:
        path = prefs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out), encoding="utf-8")
    except OSError:
        pass


# Fenêtres « Êtes-vous sûr ? » du poste : « valider » (envoi qui diffère de ceux déjà faits),
# « quitter » (extrait quitté sans valider ce qui y est tracé ou écrit).
CONFIRMS = ("valider", "quitter")


def confirm_on(name: str) -> bool:
    """Vrai tant que « Ne plus afficher » n'a pas été coché dans la fenêtre `name`."""
    off = _read_raw().get("sans_confirmation")
    return not (isinstance(off, list) and name in off)


def confirm_off(name: str) -> None:
    """« Ne plus afficher » : la fenêtre `name` ne s'ouvre plus (retenu dans les préférences)."""
    raw = _read_raw()
    off = raw.get("sans_confirmation") if isinstance(raw.get("sans_confirmation"), list) else []
    raw["sans_confirmation"] = sorted({*off, name})
    try:
        path = prefs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw), encoding="utf-8")
    except OSError:
        pass


def rename_style(old: str, new: str) -> None:
    """Un projet renommé garde son style, et reste le projet actif s'il l'était."""
    raw = _read_raw()
    styles = raw.get("styles") if isinstance(raw.get("styles"), dict) else {}
    if old in styles:
        styles[new] = styles.pop(old)
    raw["styles"] = styles
    if raw.get("projet") == old:
        raw["projet"] = new
    try:
        prefs_path().write_text(json.dumps(raw), encoding="utf-8")
    except OSError:
        pass


def next_style(style: str) -> str:
    ids = list(STYLES)
    return ids[(ids.index(style) + 1) % len(ids)] if style in ids else ids[0]


def look() -> dict:
    """Ce que reçoivent les composants : mode choisi (ou None), style actif, styles proposés
    (`id`, `name`, `icon`) et variables de couleur du style dans les deux modes (`vars`)."""
    prefs = read_prefs()
    style = str(prefs["style"])
    return {
        "theme": prefs["theme"],
        "style": style,
        "styles": [{"id": k, "name": v, "icon": STYLE_ICONS[k]} for k, v in STYLES.items()],
        "vars": {m: {k: palette(m, style)[k] for k in COMPONENT_KEYS} for m in ("dark", "light")},
    }
