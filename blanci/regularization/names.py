"""Noms des têtes régularisées : lecture (`parse_head`), nom canonique, validation, réglages
principaux (DECISIONS n° 108)."""

from __future__ import annotations

import re

IMPLEMENTED = (13, 15, 17, 18, 19, 20, 21, 27, 28, 36, 37)  # fenêtres, poids, pénalités
IMPLEMENTED += (40, 41, 42, 43, 45, 46, 47, 59, 64)  # entraînement des têtes torch
IMPLEMENTED += (76, 79, 81)  # sélection des réglages, bagging, pseudo-étiquetage
DESCRIPTIONS = {
    13: "chaque micro pèse autant dans sa classe",
    15: "négatifs annotés surpondérés face aux présumés",
    17: "normalisation L2 des embeddings",
    18: "ACP avant la tête",
    19: "centrage par micro",
    20: "AdaBN : centrage et réduction par micro",
    21: "retrait des directions du micro (INLP)",
    27: "pénalité L1",
    28: "pénalité Elastic Net",
    36: "poids des classes",
    37: "biais par micro",
    40: "weight decay par validation groupée",
    41: "AdamW",
    42: "arrêt précoce",
    43: "entropie de l'attention",
    45: "dropout des jetons",
    46: "dropout des dimensions",
    47: "rétrécissement vers la logistique",
    59: "warm-up et écrêtage du gradient",
    64: "moyenne des poids (EMA, SWA)",
    76: "grille de C plus fine",
    79: "bagging par micros",
    81: "pseudo-étiquetage",
}
# Réglage principal de chaque R, celui que « =v » remplace.
MAIN_PARAMETER = {
    15: "hard_weight",
    18: "components",
    21: "iterations",
    28: "l1_ratio",
    36: "power",
    37: "scale",
    41: "weight_decay",
    42: "monitor",
    43: "strength",
    45: "p",
    46: "p",
    47: "strength",
    59: "warmup",
    64: "method",
    76: "points",
    79: "bags",
    81: "weight",
}
WEIGHTED_HEADS = ("logistic", "cascade", "logistic_to_prototype", "loss")
PENALIZED_HEADS = ("logistic", "cascade")
GROUP_BIAS_HEADS = ("logistic", "cascade", "loss")
C_HEADS = ("logistic", "cascade", "logistic_to_prototype", "loss", "multiclass")  # R76
CONTEXT_HEADS = ("multiclass", "dann")  # têtes qui lisent le contexte même sans suffixe
PSEUDO_HEADS = ("logistic", "logistic_to_prototype", "loss")  # R81
BAGGED_HEADS = ("logistic", "logistic_to_prototype", "loss")  # entraînements légers (R79)
# Régularisations des têtes entraînées avec torch, et celles que chacune accepte.
TORCH_REGULARIZATIONS = {
    "attentive": (40, 41, 42, 43, 45, 46, 47, 59, 64),
    "gated": (40, 42, 46, 59, 64),
    "dann": (40, 42, 46, 59, 64),
}
# Réglages principaux qui prennent un mot plutôt qu'un nombre : R42=ap, R42=loss.
WORD_VALUES = {37: ("glmm",), 42: ("ap", "loss"), 64: ("ema", "swa")}
_SUFFIX = re.compile(r"^R(\d+)(?:=([0-9.eE+-]+|[a-z_]+))?$")


# --- Noms des têtes -------------------------------------------------------------------------


def parse_head(spec: str) -> tuple[str, dict[int, float | str | None]]:
    """« logistic:max+R18=16+R13 » → ("logistic:max", {13: None, 18: 16.0}) ;
    « attentive+R42=loss » → ("attentive", {42: "loss"}) (`WORD_VALUES`)."""
    base, *suffixes = spec.split("+")
    regs: dict[int, float | str | None] = {}
    for suffix in suffixes:
        match = _SUFFIX.match(suffix.strip())
        number = int(match.group(1)) if match else None
        value = match.group(2) if match else None
        if match and value is not None and number in WORD_VALUES:
            if value not in WORD_VALUES[number]:
                raise ValueError(
                    f"R{number}={value} : valeurs possibles {', '.join(WORD_VALUES[number])}"
                )
            regs[number] = value
            continue
        try:
            regs[number] = float(value) if value is not None else None
        except (TypeError, ValueError):
            match = None
        if not match:
            raise ValueError(f"régularisation illisible dans {spec!r} : {suffix!r} (ex. R18=16)")
    return base.strip(), regs


def head_name(base: str, regs: dict[int, float | str | None]) -> str:
    """Nom canonique : régularisations triées par numéro."""
    parts = [base]
    for number in sorted(regs):
        value = regs[number]
        if value is None:
            parts.append(f"R{number}")
        elif isinstance(value, str):
            parts.append(f"R{number}={value}")
        else:
            parts.append(f"R{number}={int(value) if float(value).is_integer() else value}")
    return "+".join(parts)


def canonical(spec: str) -> str:
    return head_name(*parse_head(spec))


def validate(base: str, regs: dict[int, float | str | None]) -> None:
    """Refuse une régularisation inconnue ou sans objet pour cette tête."""
    unknown = sorted(set(regs) - set(IMPLEMENTED))
    if unknown:
        raise ValueError(
            f"R{unknown[0]} n'est pas programmée comme suffixe (programmées : "
            f"{', '.join(f'R{n}' for n in IMPLEMENTED)} ; R22 = pooling gem, R30 = tête "
            "logistic_to_prototype, R31 = tête lda_shrunk)"
        )
    if not regs:
        return
    family = "logistic" if base.startswith("logistic:") else base.split(":")[0]
    torch_only = {n for allowed in TORCH_REGULARIZATIONS.values() for n in allowed}
    if base == "multiclass":
        refused = sorted(set(regs) - {17, 18, 19, 20, 21, 76})
        if refused:
            raise ValueError(f"multiclass : R{refused[0]} sans objet (R17–R21, R76 seulement)")
        return
    if base in TORCH_REGULARIZATIONS:
        refused = sorted(set(regs) - set(TORCH_REGULARIZATIONS[base]))
        if refused:
            accepted = ", ".join(f"R{n}" for n in TORCH_REGULARIZATIONS[base])
            reason = "l'attentive lit les jetons bruts" if base == "attentive" else base
            raise ValueError(f"{reason} : R{refused[0]} sans objet ({accepted} seulement)")
        return
    if torch_only & set(regs):
        number = min(torch_only & set(regs))
        raise ValueError(f"R{number} : têtes attentive et gated seulement (entraînées avec torch)")
    if {19, 20} <= set(regs):
        raise ValueError("R20 contient déjà le centrage de R19 : l'une ou l'autre")
    if {27, 28} <= set(regs):
        raise ValueError("R27 (L1) et R28 (Elastic Net) : l'une ou l'autre")
    if base.startswith("logistic:") and {19, 20} & set(regs):
        raise ValueError(
            f"{base} : R19/R20 n'existent que pour l'embedding par défaut (statistiques du stock)"
        )
    if {13, 15, 36} & set(regs) and family not in WEIGHTED_HEADS:
        raise ValueError(f"R13/R15/R36 (poids) : têtes {', '.join(WEIGHTED_HEADS)} seulement")
    if 76 in regs and family not in C_HEADS:
        raise ValueError(f"R76 (grille de C) : têtes {', '.join(C_HEADS)} seulement")
    if 79 in regs and family not in BAGGED_HEADS:
        raise ValueError(f"R79 (bagging) : têtes {', '.join(BAGGED_HEADS)} seulement")
    if 81 in regs and family not in PSEUDO_HEADS:
        raise ValueError(f"R81 (pseudo-étiquetage) : têtes {', '.join(PSEUDO_HEADS)} seulement")
    if {79, 81} <= set(regs):
        raise ValueError("R79 + R81 : pas programmé ensemble")
    if 37 in regs and family not in GROUP_BIAS_HEADS:
        raise ValueError(f"R37 (biais par micro) : têtes {', '.join(GROUP_BIAS_HEADS)} seulement")
    if regs.get(37) == "glmm" and family == "loss":
        raise ValueError("R37=glmm (σ estimé) : têtes logistic et cascade seulement")
    if {27, 28} & set(regs) and family not in PENALIZED_HEADS:
        raise ValueError(f"R27/R28 (pénalité) : têtes {', '.join(PENALIZED_HEADS)} seulement")


def needs_domain(specs: list[str]) -> bool:
    return any({19, 20} & set(parse_head(s)[1]) for s in specs)


def needs_pool(specs: list[str]) -> bool:
    return any(81 in parse_head(s)[1] for s in specs)
