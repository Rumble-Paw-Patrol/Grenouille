"""Régularisations des têtes (DECISIONS n° 108), numérotées comme la liste du 25/09/2026
(R1–R84, `documentation/regularisation.md`).

Toutes coupées par défaut. Une tête du benchmark les active dans son nom :
`logistic+R18=16+R19` = régression logistique, ACP à 16 composantes (R18), centrage par micro
(R19). « =v » remplace le réglage principal de la section `regularization` de la config. Le nom
canonique (régularisations triées par numéro) est celui des rapports et des scores hors-pli :
on sait toujours lesquelles ont servi.

Programmées ici, appliquées dans cet ordre :

| R | où | quoi |
|---|---|---|
| R19 | fenêtre | centrage par micro : − moyenne de tout le stock du micro (sans labels) |
| R20 | fenêtre | AdaBN : (x − moyenne du micro) / écart-type du micro |
| R17 | fenêtre | embedding ramené à la norme 1 |
| R18 | pli | ACP ajustée sur l'entraînement du pli (`components`) |
| R21 | pli | retrait des directions qui trahissent le micro (négatifs seulement) |
| R37 | fenêtre | indicatrices du micro ajoutées à l'entrée : un biais par micro, pénalisé |
| R13 | poids | chaque micro pèse autant dans sa classe |
| R15 | poids | négatifs annotés (faux amis, espèces) × `hard_weight` face aux présumés |
| R36 | poids | poids des classes (n / 2·n_classe)^`power` : 1 équilibré (défaut), 0 aucun |
| R27 | pénalité | L1 (lasso) au lieu de L2 |
| R28 | pénalité | Elastic Net (`l1_ratio`) |

Têtes entraînées avec torch (DECISIONS n° 117, 121) : toute la mécanique est ici, les têtes
(`attentive.py`, `gated.py`) l'appellent dans leur boucle d'entraînement.

| R | têtes | quoi | fonction |
|---|---|---|---|
| R40 | attentive, gated | weight decay par validation groupée (`grid`) | `fit_with_options` |
| R41 | attentive | AdamW (weight decay découplé, `weight_decay`) | `optimise` |
| R42 | attentive, gated | époques par validation groupée (`R42=ap`, `=loss`) | `fit_with_options` |
| R45 | attentive | dropout des jetons (`p`) | `keep_mask` |
| R46 | attentive, gated | dropout des dimensions du vecteur agrégé (`p`) | `dropout` |
| R47 | attentive | départ et rétrécissement vers la logistique (`strength`) | `logistic_start` |
| R59 | attentive, gated | warm-up du pas (`warmup` époques), écrêtage du gradient | `optimise` |

Prêtes pour les réseaux à venir (fine-tuning, distillation, modèle maison), sans appel encore :
R62 `l2_sp_penalty` (déjà utilisée par R47), R63 `distillation_loss`.

Réglage choisi par validation groupée : `grouped_search`, commun au C des têtes (R26,
`head.select_C`), au weight decay (R40) et au C de la fusion (R50, `choose_fusion_C`).

Ce module est l'index de toutes les régularisations programmées. Celles qui sont une tête ou
un réglage d'un autre étage vivent là où elles s'appliquent, et appellent ce module quand elles
ont une mécanique propre :

| R | où | comment l'activer |
|---|---|---|
| R22 | `pooling.gem` | tête `logistic:gem` (gem2, gem5…) |
| R26 | `head.fit_logistic` | la L2, toujours là ; C par `grouped_search` |
| R30, R31 | `head.py` | têtes `logistic_to_prototype`, `lda_shrunk` |
| R34, R35 | `losses.py` | têtes `loss:<nom>`, `--methods losses` |
| R37 | `head.standardize` | échelle des colonnes : `group_bias_scale` (ici) |
| R39 | `head.py` | têtes `knn:k=…`, `exemplar:k=…` (`:w`) ; calcul : `nearest_similarity` (ici) |
| R50 | `fusion.fit_fusion_model` | méthode `logistic+R50` ; C : `choose_fusion_C` (ici) |
| R57 | `stacking.py` | toujours là : la fusion n'apprend que sur des scores hors-pli |
| R85 | `gated.py` | tête `gated` |

R19 et R20 lisent le stock d'embeddings entier du micro (`store_domain_statistics`) : aucune
étiquette, ce que la chaîne aura aussi sur un nouveau site. Elles ne valent que pour
l'embedding par défaut (pas pour les jetons résumés `logistic:<pooling>`). R18 et R21 sont
ajustées dans chaque pli sur les seules fenêtres d'entraînement ; le C des têtes logistiques est
ensuite choisi sur ces fenêtres transformées.

R37 (DECISIONS n° 116) : une colonne par micro de l'entraînement, non standardisée ; son poids
est le biais du micro, d'a priori N(0, σ²) (σ = `scale`, en logit, indépendant de C : voir
`head.standardize`). Un micro absent de l'entraînement (le micro du pli de test, un nouveau
site) a toutes ses colonnes à 0 : biais commun. Le biais absorbe le niveau de chaque micro
pendant l'apprentissage, w n'a plus à le coder. Sur données simulées, un σ petit (biais « très
pénalisés ») laissait le raccourci dans w : le mécanisme dépend de σ, à choisir sur la base
complète (`logistic+R37=0.3`, `=1`, `=3`, `=10`) ; σ = 3 est un défaut provisoire (n° 119).
Écartée au tri du 26/09 : R33 (norme maximale, équivalente à la L2 pour une tête linéaire).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

IMPLEMENTED = (13, 15, 17, 18, 19, 20, 21, 27, 28, 36, 37)  # fenêtres, poids, pénalités
IMPLEMENTED += (40, 41, 42, 45, 46, 47, 59, 64)  # entraînement des têtes torch
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
    "attentive": (40, 41, 42, 45, 46, 47, 59, 64),
    "gated": (40, 42, 46, 59, 64),
    "dann": (40, 42, 46, 59, 64),
}
# Réglages principaux qui prennent un mot plutôt qu'un nombre : R42=ap, R42=loss.
WORD_VALUES = {42: ("ap", "loss"), 64: ("ema", "swa")}
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
    if {27, 28} & set(regs) and family not in PENALIZED_HEADS:
        raise ValueError(f"R27/R28 (pénalité) : têtes {', '.join(PENALIZED_HEADS)} seulement")


def needs_domain(specs: list[str]) -> bool:
    return any({19, 20} & set(parse_head(s)[1]) for s in specs)


def needs_pool(specs: list[str]) -> bool:
    return any(81 in parse_head(s)[1] for s in specs)


def sample_pool(
    con, store, filters: dict | None, by: str, exclude: set, size: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """R81 : (embeddings, groupe) d'au plus `size` fenêtres non annotées du stock, tirées au
    hasard partition par partition, hors fenêtres du benchmark (`exclude`) et fenêtres
    arrêtées par une porte."""
    from blanci.dataset import recordings_table
    from blanci.store import gated_mask

    group_of = recordings_table(con).set_index("recording_id")[by].astype(str)
    rng = np.random.default_rng(seed)
    fragments = list(store.fragments(filters))
    per_fragment = max(1, int(np.ceil(size / max(len(fragments), 1))))
    embs, groups = [], []
    for path in fragments:
        meta, emb = store.read(path)
        keep = np.flatnonzero(~gated_mask(meta) & ~meta["window_id"].isin(exclude).to_numpy())
        if not len(keep):
            continue
        keep = rng.choice(keep, size=min(per_fragment, len(keep)), replace=False)
        embs.append(np.asarray(emb[keep], dtype=np.float32))
        groups.append(group_of.loc[meta["recording_id"].to_numpy()[keep]].to_numpy())
    if not embs:
        raise ValueError(f"stock vide pour {store.encoder_id} : pas de réservoir (R81)")
    return np.vstack(embs)[:size], np.concatenate(groups)[:size]


# --- R19, R20 : statistiques par micro sur le stock entier ------------------------------------


@dataclass(frozen=True)
class DomainStats:
    """Moyenne et écart-type des embeddings de chaque groupe (micro ou site), sans labels."""

    by: str
    names: np.ndarray  # (G,) noms des groupes
    mean: np.ndarray  # (G, d)
    std: np.ndarray  # (G, d)
    count: np.ndarray  # (G,) fenêtres

    def rows(self, groups: np.ndarray) -> np.ndarray:
        """Indice du groupe de chaque fenêtre."""
        index = {str(name): i for i, name in enumerate(self.names)}
        missing = sorted({str(g) for g in groups} - set(index))
        if missing:
            raise ValueError(f"pas de statistiques de stock pour {missing[:3]} (R19/R20)")
        return np.array([index[str(g)] for g in groups], dtype=int)


class _Accumulator:
    """Sommes par groupe, partition par partition (un stock entier ne tient pas en mémoire)."""

    def __init__(self) -> None:
        self.sums: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}

    def add(self, emb: np.ndarray, groups: np.ndarray) -> None:
        for name in np.unique(groups):
            block = emb[groups == name].astype(np.float64)
            s, sq, n = self.sums.get(str(name), (0.0, 0.0, 0))
            self.sums[str(name)] = (
                s + block.sum(axis=0),
                sq + (block**2).sum(axis=0),
                n + len(block),
            )

    def stats(self, by: str) -> DomainStats:
        names = sorted(self.sums)
        count = np.array([self.sums[n][2] for n in names])
        mean = np.stack([self.sums[n][0] / self.sums[n][2] for n in names])
        var = np.stack([self.sums[n][1] / self.sums[n][2] for n in names]) - mean**2
        return DomainStats(
            by,
            np.array(names, dtype=object),
            mean.astype(np.float32),
            np.sqrt(np.clip(var, 0.0, None)).astype(np.float32),
            count,
        )


def domain_statistics(emb: np.ndarray, groups: np.ndarray, by: str = "point") -> DomainStats:
    acc = _Accumulator()
    acc.add(np.asarray(emb), np.asarray(groups).astype(str))
    return acc.stats(by)


def store_domain_statistics(con, store, filters: dict | None, by: str = "point") -> DomainStats:
    """Statistiques de chaque micro (ou site) sur tout son stock, fenêtres arrêtées exclues."""
    from blanci.dataset import recordings_table
    from blanci.store import gated_mask

    group_of = recordings_table(con).set_index("recording_id")[by].astype(str)
    acc = _Accumulator()
    for path in store.fragments(filters):
        meta, emb = store.read(path)
        keep = ~gated_mask(meta)
        acc.add(emb[keep], group_of.loc[meta.loc[keep, "recording_id"]].to_numpy())
    if not acc.sums:
        raise ValueError(f"stock vide pour {store.encoder_id} : pas de statistiques (R19/R20)")
    return acc.stats(by)


# --- R21 : directions qui prédisent le micro ----------------------------------------------------


def mean_directions(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Base orthonormée (d, r ≤ G − 1) des écarts entre moyennes des groupes.

    Une fois ce sous-espace retiré, tous les micros ont la même moyenne : aucun classifieur
    linéaire ne les distingue plus par leur fond moyen (principe de LEACE, Belrose et al.
    2023, ici en projection orthogonale). Solution fermée, au plus un micro − 1 directions.
    """
    X = np.asarray(X, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    names = np.unique(groups)
    if len(names) < 2:
        return np.zeros((X.shape[1], 0))
    means = np.stack([X[groups == g].mean(axis=0) for g in names])
    U, S, _ = np.linalg.svd((means - X.mean(axis=0)).T, full_matrices=False)
    return U[:, S > max(S[0], 1e-12) * 1e-6] if S[0] > 1e-12 else np.zeros((X.shape[1], 0))


def nuisance_directions(
    X: np.ndarray,
    groups: np.ndarray,
    iterations: int = 3,
    max_directions: int = 128,
    C: float = 1.0,
    seed: int = 0,
    tolerance: float = 0.05,
) -> np.ndarray:
    """INLP (Ravfogel et al. 2020) : base orthonormée (d, r) des directions qui prédisent le
    groupe. À chaque tour, une régression logistique apprend à reconnaître le micro ; ses
    directions sont retirées et l'on recommence sur ce qui reste.

    Arrêt dès que le micro n'est plus prédit mieux que le hasard (exactitude en validation
    croisée ≤ part du groupe majoritaire + `tolerance`). Limite constatée sur données simulées :
    quand les micros sont très séparés, chaque tour en retire les normales mais il reste
    toujours une direction qui les sépare, et l'INLP finit par retirer le chant ; d'où la
    méthode `means` par défaut (`mean_directions`).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = np.asarray(X, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    Q = np.zeros((X.shape[1], 0))
    names, counts = np.unique(groups, return_counts=True)
    if len(names) < 2:
        return Q
    chance = counts.max() / counts.sum()
    n_folds = int(min(3, counts.min()))
    Xc = X - X.mean(axis=0)
    for _ in range(int(iterations)):
        if Q.shape[1] >= max_directions:
            break
        R = Xc - (Xc @ Q) @ Q.T
        scale = R.std(axis=0)
        scale[scale < 1e-9] = 1.0
        clf = LogisticRegression(C=C, max_iter=1000, random_state=seed)
        if n_folds >= 2:
            folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
            if cross_val_score(clf, R / scale, groups, cv=folds).mean() <= chance + tolerance:
                break
        clf.fit(R / scale, groups)
        W = clf.coef_ / scale  # normales des frontières dans l'espace des embeddings
        W = W - (W @ Q) @ Q.T
        U, S, _ = np.linalg.svd(W.T, full_matrices=False)
        if not len(S) or S[0] < 1e-12:
            break
        B = U[:, S > S[0] * 1e-6][:, : max_directions - Q.shape[1]]
        Q = np.hstack([Q, B])
    return Q


# --- R13, R15 : poids des exemples ------------------------------------------------------------


def sample_weights(
    y: np.ndarray,
    groups: np.ndarray,
    hard: np.ndarray | None,
    regs: dict[int, float | str | None],
    hard_weight: float = 3.0,
    class_power: float = 1.0,
) -> np.ndarray | None:
    """Poids par fenêtre, de moyenne 1 dans chaque classe (l'équilibre des classes reste celui
    de `class_weight="balanced"`), sauf R36. None si ni R13, ni R15, ni R36.

    R36 : les têtes donnent à chaque classe le poids n / (2·n_classe) (« balanced ») ; R36 le
    remplace par (n / (2·n_classe))^`class_power` : 1 = équilibré, 0 = aucun rééquilibrage
    (chaque fenêtre compte 1), 0,5 = entre les deux."""
    if not {13, 15, 36} & set(regs):
        return None
    y = np.asarray(y).astype(int)
    w = np.ones(len(y))
    if 13 in regs:
        for c in (0, 1):
            idx = np.flatnonzero(y == c)
            _, inverse, counts = np.unique(
                np.asarray(groups)[idx].astype(str), return_inverse=True, return_counts=True
            )
            w[idx] = 1.0 / counts[inverse]
    if 15 in regs and hard is not None:
        w[np.asarray(hard, dtype=bool) & (y == 0)] *= hard_weight
    for c in (0, 1):
        idx = y == c
        if idx.any():
            w[idx] *= idx.sum() / w[idx].sum()
    if 36 in regs:
        balanced = len(y) / (2.0 * np.maximum(np.bincount(y, minlength=2), 1))
        w *= (balanced ** (float(class_power) - 1.0))[y]
    return w


def group_indicators(groups: np.ndarray, train: np.ndarray) -> np.ndarray:
    """R37 : (n, micros de l'entraînement) — 1 si la fenêtre est de ce micro, sinon 0.

    Les micros absents de `train` n'ont pas de colonne : leurs fenêtres ont le biais commun."""
    groups = np.asarray(groups).astype(str)
    names = np.unique(groups[train])
    return (groups[:, None] == names[None, :]).astype(np.float32)


# --- Assemblage -------------------------------------------------------------------------------


@dataclass
class Context:
    """Ce que certaines régularisations savent de chaque fenêtre, hors embedding."""

    groups: np.ndarray  # (n,) groupe `regularization.by` (R13, R21)
    hard: np.ndarray | None = None  # (n,) négatif annoté (R15)
    domain: DomainStats | None = None  # R19, R20
    domain_rows: np.ndarray | None = None  # (n,) indice dans `domain`
    classes: np.ndarray | None = None  # (n,) classe du son (R67, `window_classes`)
    # R81 : fenêtres non annotées du stock (embeddings, groupe, indice dans `domain`)
    pool: np.ndarray | None = None
    pool_groups: np.ndarray | None = None
    pool_domain_rows: np.ndarray | None = None

    def subset(self, rows: np.ndarray) -> Context:
        return Context(
            self.groups[rows],
            None if self.hard is None else self.hard[rows],
            self.domain,
            None if self.domain_rows is None else self.domain_rows[rows],
            None if self.classes is None else self.classes[rows],
            self.pool,
            self.pool_groups,
            self.pool_domain_rows,
        )


@dataclass
class Regularizer:
    """Les régularisations d'une tête, avec leurs réglages et le contexte des fenêtres."""

    regs: dict[int, float | str | None]
    params: dict[str, Any]
    context: Context

    def param(self, number: int, key: str, default: Any) -> Any:
        value = self.regs.get(number)
        if value is not None and MAIN_PARAMETER.get(number) == key:
            return value
        return (self.params.get(f"R{number}") or {}).get(key, default)

    def window_transform(self, X: np.ndarray, domain_rows: np.ndarray | None = None) -> np.ndarray:
        """R19 / R20 puis R17 : fenêtre par fenêtre, sans apprentissage. `domain_rows` : pour
        d'autres fenêtres que celles du contexte (le réservoir de R81)."""
        X = np.asarray(X, dtype=np.float32)
        if {19, 20} & set(self.regs):
            ctx = self.context
            rows = ctx.domain_rows if domain_rows is None else domain_rows
            if ctx.domain is None or rows is None:
                raise ValueError("R19/R20 demandent les statistiques du stock (Context.domain)")
            X = X - ctx.domain.mean[rows]
            if 20 in self.regs:
                eps = float(self.param(20, "eps", 1e-6))
                X = X / (ctx.domain.std[rows] + eps)
        if 17 in self.regs:
            from blanci.index import l2_normalize

            X = l2_normalize(X)
        return X.astype(np.float32)

    def fit_projection(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0):
        """R18 puis R21, ajustées sur les fenêtres d'entraînement : renvoie x ↦ projection."""
        steps = []
        Z = np.asarray(X, dtype=np.float64)
        if 18 in self.regs:
            from sklearn.decomposition import PCA

            k = int(self.param(18, "components", 32))
            pca = PCA(n_components=max(1, min(k, *Z.shape)), random_state=seed).fit(Z)
            steps.append(pca.transform)
            Z = pca.transform(Z)
        if 21 in self.regs:
            negatives = np.asarray(y) == 0
            how = self.param(21, "method", "means")
            if how == "means":
                Q = mean_directions(Z[negatives], np.asarray(groups)[negatives])
            elif how == "inlp":
                Q = nuisance_directions(
                    Z[negatives],
                    np.asarray(groups)[negatives],
                    iterations=int(self.param(21, "iterations", 3)),
                    max_directions=int(self.param(21, "max_directions", 128)),
                    C=float(self.param(21, "C", 1.0)),
                    seed=seed,
                    tolerance=float(self.param(21, "tolerance", 0.05)),
                )
            else:
                raise ValueError(f"R21 : méthode inconnue {how!r} (means ou inlp)")
            steps.append(lambda A, Q=Q: A - (A @ Q) @ Q.T)
        if not steps:
            return None

        def project(A: np.ndarray) -> np.ndarray:
            A = np.asarray(A, dtype=np.float64)
            for step in steps:
                A = step(A)
            return A.astype(np.float32)

        return project

    pool_prepared: np.ndarray | None = None  # R81 : réservoir transformé par `prepare`
    pool_prepared_groups: np.ndarray | None = None

    def pool_windows(self, project, train: np.ndarray) -> np.ndarray:
        """R81 : le réservoir de fenêtres non annotées, passé par les mêmes transformations que
        les fenêtres annotées. `pool_from: train` (défaut) : seulement les micros
        d'entraînement ; `all` : aussi ceux du pli jugé (adaptation sans labels à un micro
        nouveau)."""
        ctx = self.context
        if ctx.pool is None or ctx.pool_groups is None:
            raise ValueError("R81 demande un réservoir de fenêtres non annotées (Context.pool)")
        keep = np.ones(len(ctx.pool), dtype=bool)
        if self.param(81, "pool_from", "train") == "train":
            keep = np.isin(
                np.asarray(ctx.pool_groups).astype(str),
                np.unique(np.asarray(ctx.groups)[train].astype(str)),
            )
        rows = None if ctx.pool_domain_rows is None else ctx.pool_domain_rows[keep]
        pool = self.window_transform(ctx.pool[keep], rows)
        if project is not None:
            pool = project(pool)
        self.pool_prepared_groups = np.asarray(ctx.pool_groups)[keep]
        return np.asarray(pool, dtype=np.float32)

    def grid(self, C_grid):
        """Grille de C de la tête : R76 la remplace par une grille plus fine."""
        if 76 in self.regs and C_grid:
            return fine_grid(C_grid, int(self.param(76, "points", 13)))
        return C_grid

    def pseudo_options(self) -> dict[str, Any]:
        """Réglages de R81 (`with_pseudo_labels`), section `regularization.R81`."""
        return {
            "min_score": float(self.param(81, "min_score", 3.0)),
            "max_fraction": float(self.param(81, "max_fraction", 0.01)),
            "weight": float(self.param(81, "weight", 0.3)),
            "rounds": int(self.param(81, "rounds", 1)),
        }

    def dann_options(self) -> dict[str, Any]:
        """Réglages de la tête `dann` (R66), section `regularization.R66` de la config."""
        defaults = {
            "hidden": 32,
            "strength": 1.0,
            "domain_on": "negatives",
            "weight_decay": 1e-2,
            "epochs": 300,
            "lr": 1e-2,
        }
        return {k: type(v)(self.param(66, k, v)) for k, v in defaults.items()}

    @property
    def bags(self) -> int:
        """R79 : nombre de tirages bootstrap (0 : pas de bagging)."""
        return int(self.param(79, "bags", 20)) if 79 in self.regs else 0

    def torch_options(self) -> dict[str, Any]:
        """Options des têtes torch (`fit_with_options`) : R40–R42, R45–R47, R59."""
        options: dict[str, Any] = {}
        if 40 in self.regs:
            options["weight_decays"] = [
                float(v) for v in self.param(40, "grid", [1e-4, 1e-3, 1e-2, 1e-1])
            ]
            options["n_splits"] = int(self.param(40, "n_splits", 3))
        if 41 in self.regs:
            options["optimizer"] = "adamw"
            options["weight_decay"] = float(self.param(41, "weight_decay", 1e-2))
        if 42 in self.regs:
            options["early_stopping"] = True
            options["monitor"] = str(self.param(42, "monitor", "ap"))  # provisoire (n° 119)
            options["epochs"] = int(self.param(42, "max_epochs", 300))
            options["n_splits"] = int(self.param(42, "n_splits", options.get("n_splits", 3)))
        if 45 in self.regs:
            options["token_dropout"] = float(self.param(45, "p", 0.2))
        if 46 in self.regs:
            options["dim_dropout"] = float(self.param(46, "p", 0.2))
        if 47 in self.regs:
            options["shrink"] = float(self.param(47, "strength", 1e-2))
        if 59 in self.regs:
            options["warmup"] = int(self.param(59, "warmup", 20))
            options["clip_norm"] = float(self.param(59, "clip_norm", 1.0))
        if 64 in self.regs:
            options["average"] = str(self.param(64, "method", "ema"))
            options["ema_decay"] = float(self.param(64, "ema_decay", 0.99))
            options["swa_start"] = float(self.param(64, "swa_start", 0.75))
        return options

    def fit_options(self, bias_columns: int = 0) -> dict[str, float]:
        """Options de l'ajustement : pénalité (R27, R28 : l1_ratio, 0 = L2, R26) et biais par
        micro (R37 : nombre de colonnes d'indicatrices, échelle)."""
        options: dict[str, float] = {}
        if 27 in self.regs:
            options["l1_ratio"] = 1.0
        elif 28 in self.regs:
            options["l1_ratio"] = float(self.param(28, "l1_ratio", 0.5))
        if bias_columns:
            options |= {
                "bias_columns": bias_columns,
                "bias_scale": float(self.param(37, "scale", 3.0)),
            }
        return options

    def prepare(
        self, X: np.ndarray, y: np.ndarray, train: np.ndarray, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, float]]:
        """(X transformé, toutes fenêtres ; poids des fenêtres `train` ou None ; options de la
        logistique)."""
        X = self.window_transform(X)
        project = self.fit_projection(X[train], y[train], self.context.groups[train], seed)
        if project is not None:
            X = project(X)
        pool = self.pool_windows(project, train) if 81 in self.regs else None
        bias_columns = 0
        if 37 in self.regs:
            indicators = group_indicators(self.context.groups, train)
            bias_columns = indicators.shape[1]
            X = np.hstack([np.asarray(X, dtype=np.float32), indicators])
            if pool is not None:
                names = np.unique(np.asarray(self.context.groups)[train].astype(str))
                pool_groups = np.asarray(self.pool_prepared_groups).astype(str)
                pool = np.hstack(
                    [pool, (pool_groups[:, None] == names[None, :]).astype(np.float32)]
                )
        self.pool_prepared = pool
        weights = sample_weights(
            y[train],
            self.context.groups[train],
            None if self.context.hard is None else self.context.hard[train],
            self.regs,
            float(self.param(15, "hard_weight", 3.0)),
            float(self.param(36, "power", 0.0)),
        )
        return X, weights, self.fit_options(bias_columns)


# --- R26, R40, R50 : un réglage choisi par validation groupée -----------------------------------


def usable_folds(
    y: np.ndarray, groups: np.ndarray, n_splits: int | str, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plis groupés internes (micros entiers) dont l'entraînement et le test ont les deux
    classes ; [] s'il n'y a qu'un micro."""
    from blanci.evaluate import grouped_folds

    y, groups = np.asarray(y).astype(int), np.asarray(groups)
    if len(np.unique(groups)) < 2:
        return []
    return [
        (train, test)
        for train, test in grouped_folds(y, groups, n_splits, seed)
        if len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2
    ]


# R75 : règle de choix. "one_se" (défaut, `head.selection_rule`) : parmi les valeurs dont l'AP
# moyenne est à moins d'une erreur type de la meilleure, la plus régularisante ; "best" : la
# meilleure AP moyenne. `configure` la règle depuis la config au démarrage de la CLI.
SELECTION_RULES = ("one_se", "best")
_SELECTION = {"rule": "one_se"}


def configure(cfg: dict) -> None:
    """Réglages globaux de ce module tirés de la config (R75 : `head.selection_rule`)."""
    rule = (cfg.get("head", {}) or {}).get("selection_rule", "one_se")
    if rule not in SELECTION_RULES:
        raise ValueError(f"head.selection_rule : {rule!r} (connues : {SELECTION_RULES})")
    _SELECTION["rule"] = rule


def pick(
    means: dict,
    ses: dict,
    rule: str | None = None,
    more_regularized: str = "low",
) -> Any:
    """Valeur retenue parmi `means` (AP moyenne par valeur de la grille). R75, règle du
    « 1 écart-type » : toute valeur dont la moyenne est à moins d'une erreur type (`ses`) de la
    meilleure est indiscernable d'elle ; on prend alors la plus régularisante — la plus petite
    (`more_regularized="low"` : C) ou la plus grande ("high" : weight decay). None si aucune
    valeur n'a d'AP."""
    rule = rule or _SELECTION["rule"]
    valid = {k: v for k, v in means.items() if np.isfinite(v)}
    if not valid:
        return None
    best = max(valid, key=valid.get)
    if rule == "best":
        return best
    se = ses.get(best, 0.0)
    floor = valid[best] - (se if np.isfinite(se) else 0.0)
    close = [k for k, v in valid.items() if v >= floor]
    return min(close) if more_regularized == "low" else max(close)


def grouped_search(
    score,
    grid,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int | str = 5,
    seed: int = 0,
    rule: str | None = None,
    more_regularized: str = "low",
) -> tuple[Any, dict, dict]:
    """Force d'une régularisation choisie par validation groupée interne : C des têtes
    linéaires (R26, `head.select_C`), weight decay des têtes torch (R40), C de la fusion
    (R50). `score(valeur, train, test)` rend l'AP sur `test` d'un modèle appris sur `train`.

    Renvoie (valeur retenue par `pick` (R75), ou None ; {valeur: AP moyenne} ; {valeur :
    erreur type de cette moyenne, écart-type des AP des plis / √plis}). Les deux derniers
    forment le chemin de régularisation (R76)."""
    folds = usable_folds(y, groups, n_splits, seed)
    means, ses = {}, {}
    for value in grid:
        aps = [score(value, train, test) for train, test in folds]
        finite = np.array([ap for ap in aps if np.isfinite(ap)], dtype=float)
        means[value] = float(finite.mean()) if len(finite) else float("nan")
        ses[value] = float(finite.std(ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0
    return pick(means, ses, rule, more_regularized), means, ses


def fine_grid(grid, points: int = 13) -> list[float]:
    """R76 : grille plus fine, `points` valeurs régulièrement espacées en échelle log entre la
    plus petite et la plus grande de `grid` (13 entre 0,001 et 10 : un facteur ~2,2 entre deux
    voisines, au lieu de 10)."""
    low, high = float(min(grid)), float(max(grid))
    return [float(v) for v in np.logspace(np.log10(low), np.log10(high), int(points))]


def choose_fusion_C(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C_grid: list[float] | tuple[float, ...],
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[float | None, dict[float, float]]:
    """R50 : C de la fusion logistique (`fusion.fit_fusion_model`, méthode `logistic+R50`).
    (None, {}) faute de deux valeurs, de deux micros ou d'un pli à deux classes."""
    from blanci.evaluate import average_precision
    from blanci.fusion import fit_fusion_model

    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
    if len(C_grid) < 2 or not usable_folds(y, groups, n_splits, seed):
        return None, {}
    columns = [f"x{j}" for j in range(X.shape[1])]

    def score(C, train, test):
        model = fit_fusion_model("logistic", X[train], y[train], columns, C=C)
        return average_precision(y[test], model.decision(X[test]))

    best, results, _ = grouped_search(score, [float(c) for c in C_grid], y, groups, n_splits, seed)
    return best, results


# --- R74 : ce que le choix n'a pas vu -----------------------------------------------------------


def fold_ids(n: int, folds) -> np.ndarray:
    """Numéro du pli de test de chaque fenêtre, d'après les plis (train, test) d'un `OOFScores`."""
    out = np.full(n, -1)
    for f, (_, test) in enumerate(folds):
        out[test] = f
    return out


def cross_fitted_threshold(
    y: np.ndarray, scores: np.ndarray, folds: np.ndarray, min_precision: float
) -> dict[str, Any]:
    """R74 : seuil de précision plancher choisi pour chaque pli sur les scores hors-pli des
    *autres* plis, puis appliqué à ce pli. Précision et rappel obtenus : ce que le seuil fera
    sur des micros qu'il n'a pas vus (le seuil choisi et jugé sur les mêmes scores est
    optimiste)."""
    from blanci.evaluate import recall_at_precision

    y, scores, folds = np.asarray(y).astype(int), np.asarray(scores, float), np.asarray(folds)
    decided = np.zeros(len(y), dtype=bool)
    thresholds = {}
    for f in np.unique(folds[folds >= 0]):
        other, this = (folds != f) & (folds >= 0), folds == f
        _, t = recall_at_precision(y[other], scores[other], min_precision)
        thresholds[int(f)] = float(t)
        decided[this] = scores[this] >= t
    tp = int((decided & (y == 1)).sum())
    return {
        "precision": tp / int(decided.sum()) if decided.any() else float("nan"),
        "recall": tp / max(int((y == 1).sum()), 1),
        "thresholds": thresholds,
    }


def selection_estimate(
    scores: dict[str, np.ndarray],
    y: np.ndarray,
    folds: np.ndarray,
    recordings: np.ndarray | None = None,
) -> dict[str, Any]:
    """R74 et R80 : ce que vaut la procédure « garder la variante à la meilleure AP ».

    Pour chaque pli, la variante est choisie sur les scores hors-pli des autres plis, puis
    jugée sur ce pli : le choix ne voit jamais les labels qu'on mesure. Comparée à l'AP, sur
    les mêmes plis, de la variante qui gagne le tableau (choisie en voyant tout), elle dit
    combien le gagnant doit à la chance. AP au niveau enregistrement si `recordings` est
    donné. Renvoie les variantes choisies, et les deux AP moyennes par pli."""
    from blanci.evaluate import average_precision, to_recordings

    y, folds = np.asarray(y).astype(int), np.asarray(folds)

    def ap(values: np.ndarray, mask: np.ndarray) -> float:
        if recordings is None:
            return average_precision(y[mask], values[mask])
        rec = to_recordings(values[mask], y[mask], np.asarray(recordings)[mask])
        return average_precision(rec["y"].to_numpy(), rec["score"].to_numpy())

    everything = folds >= 0
    overall = {name: ap(np.asarray(v, float), everything) for name, v in scores.items()}
    finite = {k: v for k, v in overall.items() if np.isfinite(v)}
    if not finite:
        return {}
    winner = max(finite, key=finite.get)
    chosen, selected, naive = {}, [], []
    for f in np.unique(folds[everything]):
        other, this = everything & (folds != f), folds == f
        perf = {name: ap(np.asarray(v, float), other) for name, v in scores.items()}
        perf = {k: v for k, v in perf.items() if np.isfinite(v)}
        if not perf:
            continue
        pick_f = max(perf, key=perf.get)
        chosen[int(f)] = pick_f
        selected.append(ap(np.asarray(scores[pick_f], float), this))
        naive.append(ap(np.asarray(scores[winner], float), this))
    return {
        "winner": winner,
        "winner_ap": finite[winner],
        "chosen": chosen,
        "fold_ap_winner": float(np.nanmean(naive)) if naive else float("nan"),
        "fold_ap_selection": float(np.nanmean(selected)) if selected else float("nan"),
    }


# --- R81 : pseudo-étiquetage ---------------------------------------------------------------------


def pseudo_positives(
    scores: np.ndarray, min_score: float = 3.0, max_fraction: float = 0.01
) -> np.ndarray:
    """R81 : indices des fenêtres non annotées retenues comme pseudo-positifs — score (logit de
    la tête) ≥ `min_score`, et au plus la fraction `max_fraction` du réservoir (les mieux
    notées) : un garde-fou si la tête note haut trop de fenêtres."""
    scores = np.asarray(scores, dtype=float)
    above = np.flatnonzero(scores >= min_score)
    cap = int(np.floor(max_fraction * len(scores)))
    if len(above) > cap:
        above = above[np.argsort(-scores[above])[:cap]]
    return np.sort(above)


def with_pseudo_labels(
    fit_rows,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    pool: np.ndarray,
    min_score: float = 3.0,
    max_fraction: float = 0.01,
    weight: float = 0.3,
    rounds: int = 1,
):
    """R81 (auto-apprentissage, Lee 2013) : une tête apprise sur les labels note le réservoir ;
    les fenêtres sûres (`pseudo_positives`) rejoignent l'entraînement comme positifs, au poids
    `weight` (un avis du modèle ne vaut pas un label d'expert), et la tête est réapprise,
    `rounds` fois. `fit_rows(X, y, poids)` apprend une tête. Pas de pseudo-négatifs : les
    fenêtres non annotées notées bas sont aussi incertaines que les négatifs présumés (n° 106).
    Risque : le biais de confirmation (la tête étiquette ce qu'elle reconnaît déjà, erreurs
    comprises)."""
    y = np.asarray(y).astype(int)
    base_w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
    model = fit_rows(X, y, sample_weight)
    chosen = np.array([], dtype=int)
    for _ in range(int(rounds)):
        chosen = pseudo_positives(model.decision(pool), min_score, max_fraction)
        if not len(chosen):
            break
        model = fit_rows(
            np.vstack([X, pool[chosen]]),
            np.r_[y, np.ones(len(chosen), dtype=int)],
            np.r_[base_w, np.full(len(chosen), float(weight))],
        )
    model.meta = getattr(model, "meta", {}) | {"pseudo_positives": int(len(chosen))}
    return model


# --- R66 : inversion du gradient (DANN) -------------------------------------------------------


def grad_reverse(torch, x, strength: float):
    """R66 : couche d'inversion du gradient (Ganin et al. 2016). À l'aller, l'identité ; au
    retour, le gradient multiplié par −`strength`. Placée entre la représentation et le
    classifieur de micro, elle fait apprendre au classifieur à reconnaître le micro, et à la
    représentation à l'en empêcher."""

    class _Reverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            return inputs.view_as(inputs)

        @staticmethod
        def backward(ctx, grad):
            return -strength * grad

    return _Reverse.apply(x)


def dann_strength(progress: float, strength: float = 1.0) -> float:
    """R66 : force de l'inversion selon l'avancement p ∈ [0, 1] de l'entraînement,
    λ(p) = strength · (2 / (1 + e^(−10 p)) − 1) : nulle au départ (la tête apprend d'abord le
    chant), pleine ensuite (Ganin et al. 2016)."""
    return float(strength) * (2.0 / (1.0 + np.exp(-10.0 * float(progress))) - 1.0)


# --- R67 : classes des sons ----------------------------------------------------------------------

# Labels d'annotation → classe de la tête multi-classes. Les négatifs présumés (sans label)
# rejoignent le fond : par définition, aucun événement n'y a été noté (n° 124).
CLASS_OF_LABEL = {
    "bird": "oiseau",
    "amphibian": "amphibien",
    "amphibian_contact_call": "amphibien",
    "orthoptera": "orthoptère",
    "rain": "pluie",
    "background": "fond",
    "artefact_in_bag": "artefact",
    "other": "autre",
}
POSITIVE_CLASS = "blanci"


def window_classes(
    labels: np.ndarray, y: np.ndarray, presumed: np.ndarray | None = None
) -> np.ndarray:
    """R67 : classe de chaque fenêtre — `blanci` pour les positifs, la classe de son label
    d'annotation sinon (`CLASS_OF_LABEL`), `fond` pour les négatifs présumés et les labels
    absents, `autre` pour un label inconnu."""
    y = np.asarray(y).astype(int)
    out = []
    for i, label in enumerate(np.asarray(labels, dtype=object)):
        if y[i] == 1:
            out.append(POSITIVE_CLASS)
        elif (presumed is not None and bool(presumed[i])) or label is None or label != label:
            out.append("fond")
        else:
            out.append(CLASS_OF_LABEL.get(str(label), "autre"))
    return np.array(out, dtype=object)


def merge_rare_classes(classes: np.ndarray, min_count: int = 10) -> np.ndarray:
    """R67 : une classe de moins de `min_count` fenêtres (hors `blanci`) rejoint `autre` ;
    si `autre` reste sous le seuil, elle rejoint `fond`. Une classe de 3 exemples ne s'apprend
    pas, elle ajoute du bruit."""
    classes = np.asarray(classes, dtype=object).copy()
    names, counts = np.unique(classes, return_counts=True)
    for name, count in zip(names, counts, strict=True):
        if name not in (POSITIVE_CLASS, "fond", "autre") and count < min_count:
            classes[classes == name] = "autre"
    if 0 < (classes == "autre").sum() < min_count:
        classes[classes == "autre"] = "fond"
    return classes


# --- R79 : bagging -------------------------------------------------------------------------------


def bootstrap_weights(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """R79 : tirage bootstrap des **micros** (avec remise, autant que de micros) ; poids de
    chaque fenêtre = nombre de fois où son micro est tiré (0 : absent de ce tirage). Tirer
    des fenêtres ferait croire à des exemples indépendants."""
    names, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    counts = np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names))
    return counts[inverse].astype(float)


@dataclass
class Bagged:
    """Têtes apprises sur des tirages bootstrap ; score = moyenne de leurs scores."""

    models: list

    def decision(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.decision(X) for m in self.models], axis=0)


def bagged(
    fit,
    y: np.ndarray,
    groups: np.ndarray,
    bags: int,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
) -> Bagged:
    """R79 (Breiman 1996) : `bags` têtes, chacune apprise sur un tirage bootstrap des micros
    (`bootstrap_weights`), leurs scores moyennés. Chaque tête a ses lubies (les exemples
    qu'elle a vus) ; la moyenne les efface : la variance baisse, le biais ne bouge pas.
    `fit(lignes, poids)` apprend une tête sur ces lignes ; les tirages sans l'une des deux
    classes sont refaits."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    models, attempts = [], 0
    while len(models) < bags and attempts < 10 * bags:
        attempts += 1
        w = bootstrap_weights(groups, rng)
        rows = np.flatnonzero(w > 0)
        if len(np.unique(y[rows])) < 2:
            continue
        if sample_weight is not None:
            w = w * np.asarray(sample_weight, dtype=float)
        models.append(fit(rows, w[rows]))
    if not models:
        raise ValueError("R79 : aucun tirage bootstrap n'a les deux classes")
    return Bagged(models)


# --- R37 : échelle des colonnes de biais ---------------------------------------------------------


def group_bias_scale(sigma: float, C: float) -> float:
    """Échelle des indicatrices de micro (`head.standardize`) : σ / √C. Sous la pénalité ½‖w‖²
    et le terme C · Σ perte, le biais b d'un micro coûte alors b² / 2σ² rapporté aux données,
    l'a priori N(0, σ²), quel que soit C."""
    return float(sigma) / float(np.sqrt(C))


# --- R39 : k plus proches voisins ----------------------------------------------------------------


def nearest_similarity(sims: np.ndarray, k: int = 1, weighted: bool = False) -> np.ndarray:
    """Similarité de chaque ligne à ses k références les plus proches (R39, DECISIONS n° 115),
    appelée par les têtes `knn:k=…` et `exemplar:k=…` (`head.py`).

    k = 1 : le plus proche seul. k > 1 : moyenne des k plus proches, plus lisse, moins sensible
    à une référence bizarre. `weighted` : moyenne pondérée par 1 / distance (distance
    euclidienne entre vecteurs de norme 1, √(2 − 2·cos), comme `weights="distance"` de
    scikit-learn) : les voisins très proches comptent davantage, entre k = 1 et la moyenne."""
    k = max(1, min(int(k), sims.shape[1]))
    top = -np.partition(-sims, k - 1, axis=1)[:, :k] if k > 1 else sims.max(axis=1)[:, None]
    if not weighted:
        return top.mean(axis=1)
    w = 1.0 / np.maximum(np.sqrt(np.clip(2.0 - 2.0 * top, 0.0, None)), 1e-6)
    return (w * top).sum(axis=1) / w.sum(axis=1)


# --- Têtes et réseaux entraînés avec torch : R40–R42, R45–R47, R59, R62, R63 ----------------------


def optimise(
    parameters: list,
    decayed: list[bool],
    loss_of,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    optimizer: str = "adam",
    validation_loss=None,
    patience: int | None = 20,
    curve: list[float] | None = None,
    warmup: int = 0,
    clip_norm: float | None = None,
    average: str | None = None,
    ema_decay: float = 0.99,
    swa_start: float = 0.75,
) -> int:
    """Descente en lot entier, partagée par l'attentive et la sonde à portes (R85), et par les
    réseaux à venir.

    `loss_of()` : perte d'entraînement (dropout compris) ; weight decay sur les seuls paramètres
    `decayed`. `optimizer` : "adam" (L2 couplée, défaut historique) ou "adamw" (R41).
    `validation_loss()` : critère à minimiser sur des micros tenus à l'écart (R42), relevé à
    chaque époque dans `curve` (liste remplie en place) ; avec `patience`, l'entraînement
    s'arrête `patience` époques après la meilleure, dont les paramètres sont restaurés.
    R59 : `warmup` époques de montée linéaire du pas d'apprentissage, `clip_norm` : norme
    maximale du gradient (écrêtage). R64, `average` : les poids gardés sont une moyenne des
    poids successifs — "ema" : moyenne glissante (poids × `ema_decay` à chaque pas), "swa" :
    moyenne simple à partir de la fraction `swa_start` de l'entraînement ; la validation (R42)
    juge alors ces poids moyens. Renvoie l'époque retenue (la dernière sans validation)."""
    import torch

    kinds = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if optimizer not in kinds:
        raise ValueError(f"optimiseur inconnu : {optimizer!r} (adam, adamw)")
    groups = [
        {
            "params": [p for p, d in zip(parameters, decayed, strict=True) if d],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for p, d in zip(parameters, decayed, strict=True) if not d],
            "weight_decay": 0.0,
        },
    ]
    if average not in (None, "ema", "swa"):
        raise ValueError(f"R64 : moyenne inconnue {average!r} (ema, swa)")
    opt = kinds[optimizer]([g for g in groups if g["params"]], lr=lr)
    averager = WeightAverage(parameters, average, ema_decay, swa_start, int(epochs))
    best, best_epoch, best_state = float("inf"), int(epochs), None
    for epoch in range(1, int(epochs) + 1):
        if warmup:
            for group in opt.param_groups:
                group["lr"] = lr * min(1.0, epoch / warmup)
        opt.zero_grad()
        loss = loss_of()
        loss.backward()
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
        opt.step()
        averager.update(epoch)
        if validation_loss is None:
            continue
        with torch.no_grad(), averager.swapped():
            current = float(validation_loss())
            if current < best - 1e-7:
                best_state = [p.detach().clone() for p in parameters]
        if curve is not None:
            curve.append(current)
        if current < best - 1e-7:
            best, best_epoch = current, epoch
        elif patience is not None and epoch - best_epoch >= patience:
            break
    if best_state is None:
        averager.apply()
    else:
        with torch.no_grad():
            for p, value in zip(parameters, best_state, strict=True):
                p.copy_(value)
    return best_epoch


class WeightAverage:
    """R64 : moyenne des poids pendant l'entraînement (Izmailov et al. 2018 pour SWA ; EMA).

    À la fin d'un entraînement, les poids oscillent autour d'un minimum (petits lots, pas
    d'apprentissage grand) ; la dernière époque est un point pris au hasard dans ces
    oscillations, la moyenne tombe plus près du centre du creux. Sans `method`, rien ne
    change."""

    def __init__(self, parameters, method, decay: float, start: float, epochs: int):
        self.parameters, self.method, self.decay = parameters, method, float(decay)
        self.start = max(1, int(np.ceil(float(start) * epochs)))
        self.mean = None
        self.count = 0

    def update(self, epoch: int) -> None:
        if self.method is None:
            return
        current = [p.detach().clone() for p in self.parameters]
        if self.method == "ema":
            self.mean = (
                current
                if self.mean is None
                else [
                    self.decay * m + (1.0 - self.decay) * c
                    for m, c in zip(self.mean, current, strict=True)
                ]
            )
        elif epoch >= self.start:  # swa
            self.count += 1
            if self.mean is None:
                self.mean = current
            else:
                self.mean = [
                    m + (c - m) / self.count for m, c in zip(self.mean, current, strict=True)
                ]

    def apply(self) -> None:
        """Remplace les poids par leur moyenne."""
        if self.mean is None:
            return
        for p, m in zip(self.parameters, self.mean, strict=True):
            p.data.copy_(m)

    def swapped(self):
        """Contexte : les poids moyens le temps d'une évaluation, puis les poids courants."""
        from contextlib import contextmanager

        @contextmanager
        def swap():
            if self.mean is None:
                yield
                return
            saved = [p.detach().clone() for p in self.parameters]
            self.apply()
            try:
                yield
            finally:
                for p, v in zip(self.parameters, saved, strict=True):
                    p.data.copy_(v)

        return swap()


# --- R61, R62 : fine-tuning d'un réseau pré-entraîné (`finetune.py`, à écrire) ------------------


def layerwise_lr_groups(
    layers: list, lr: float, decay: float = 0.8, weight_decay: float = 0.0
) -> list[dict]:
    """R61, pas d'apprentissage par couche (LLRD) : la dernière couche de `layers` (de bas en
    haut) reçoit `lr`, celle du dessous lr × decay, puis lr × decay²… Les couches basses,
    génériques, bougent peu ; les hautes, spécialisées, s'adaptent. Groupes pour un optimiseur
    torch."""
    n = len(layers)
    return [
        {
            "params": list(layer.parameters()),
            "lr": lr * decay ** (n - 1 - i),
            "weight_decay": weight_decay,
        }
        for i, layer in enumerate(layers)
    ]


def unfreezing_schedule(n_layers: int, epochs: int, every: int) -> list[int]:
    """R61, dégel progressif (ULMFiT, Howard et Ruder 2018) : nombre de couches du haut qui
    apprennent à chaque époque — 1, puis une de plus toutes les `every` époques."""
    return [min(n_layers, 1 + epoch // max(1, every)) for epoch in range(epochs)]


def unfreeze_top(layers: list, n_top: int) -> None:
    """R61 : seules les `n_top` dernières couches de `layers` apprennent, les autres sont
    gelées."""
    for i, layer in enumerate(layers):
        for p in layer.parameters():
            p.requires_grad_(i >= len(layers) - n_top)


def snapshot(model) -> dict:
    """R62 : copie des poids pré-entraînés d'un réseau torch, avant l'adaptation."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def l2_sp_model_penalty(torch, model, reference: dict, alpha: float, beta: float = 0.0):
    """R62, L2-SP complet (Li, Grandvalet et Davoine 2018, « SP » = starting point) :
    α/2 Σ ‖θ − θ⁰‖² sur les poids qui existaient avant l'adaptation (`reference`, de
    `snapshot`, clés = noms des poids dans `model`), β/2 Σ ‖θ‖² sur les poids nouveaux (tête,
    adaptateurs LoRA) ; les poids gelés ne comptent pas."""
    total = 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in reference:
            total = total + 0.5 * alpha * ((p - reference[name]) ** 2).sum()
        elif beta:
            total = total + 0.5 * beta * (p**2).sum()
    return total


def validation_criterion(torch, loss_fn, scores_of, y_val: np.ndarray, monitor: str = "loss"):
    """Critère de R42, à minimiser : la perte d'entraînement sur les fenêtres de validation
    ("loss"), ou 1 − AP ("ap"). La perte monte dès que la tête devient trop sûre d'elle, même
    quand son classement s'améliore encore (vu sur données simulées, DECISIONS n° 117 ; le
    choix entre les deux se fera sur la base complète, n° 119)."""
    from blanci.evaluate import average_precision

    y_np = np.asarray(y_val).astype(int)
    target = torch.from_numpy(y_np.astype(np.float32))
    if monitor == "loss":
        return lambda: loss_fn(scores_of(), target)
    if monitor == "ap":
        return lambda: 1.0 - average_precision(y_np, scores_of().numpy())
    raise ValueError(f"R42 : critère inconnu {monitor!r} (loss ou ap)")


def keep_mask(torch, n: int, t: int, p: float):
    """R45 : jetons gardés (n, t), chacun masqué avec la probabilité p, au moins un par
    fenêtre."""
    keep = torch.rand(n, t) >= p
    keep[torch.arange(n), torch.randint(t, (n,))] = True
    return keep


def dropout(torch, x, p: float, train: bool):
    """R46 : dropout (sorties éteintes avec la probabilité p, les autres × 1/(1 − p)), à
    l'entraînement seulement."""
    if train and p > 0:
        return torch.nn.functional.dropout(x, p, training=True)
    return x


def logistic_start(
    Z: np.ndarray, y: np.ndarray, C: float, seed: int = 0
) -> tuple[np.ndarray, float]:
    """R47 : (w₀, b₀) d'une logistique apprise sur Z et réécrite dans l'espace de Z (sans sa
    standardisation) : le point de départ, et la cible du rétrécissement, de l'attentive."""
    from blanci.head import fit_logistic

    head = fit_logistic(Z, np.asarray(y).astype(int), C, seed)
    w0 = head.coef / head.scale
    return w0, float(head.intercept - float(head.coef @ (head.mean / head.scale)))


def l2_sp_penalty(torch, parameters: list, references: list, strength: float):
    """R62 (L2-SP) : ½λ Σ ‖θ − θ_réf‖², l'écart aux poids de référence plutôt qu'à zéro. Les
    poids pré-entraînés pour un réseau adapté (fine-tuning, `finetune.py`) ; ceux de la
    logistique pour l'attentive (R47)."""
    total = 0.0
    for p, ref in zip(parameters, references, strict=True):
        total = total + ((p - ref) ** 2).sum()
    return 0.5 * strength * total


def distillation_loss(torch, student_logits, teacher_logits, temperature: float = 2.0):
    """R63 : perte de la distillation (détecteur distillé, `detectors/distilled.py`). L'élève
    apprend les probabilités de l'enseignant (la chaîne gelée), adoucies par la température T :
    entropie croisée binaire entre σ(élève / T) et σ(enseignant / T), × T² pour garder
    l'échelle des gradients (Hinton et al. 2015). Les labels souples de l'enseignant disent
    aussi « à peu près » et « pas sûr », ce qui régularise l'élève."""
    t = float(temperature)
    target = torch.sigmoid(teacher_logits / t)
    return torch.nn.functional.binary_cross_entropy_with_logits(student_logits / t, target) * t**2


def fit_with_options(
    fit,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 0,
    *,
    weight_decays: list[float] | None = None,
    early_stopping: bool = False,
    monitor: str = "ap",
    n_splits: int = 3,
    **options,
):
    """Tête torch `fit` (fit_attentive, fit_gated) entourée de R40 et R42, sur plis groupés.

    R40 (`weight_decays`) : chaque valeur est jugée par l'AP moyenne en validation groupée
    interne (`n_splits` plis, micros entiers, `grouped_search`), la meilleure est retenue. R42
    (`early_stopping`) : dans chaque pli interne, la courbe du critère de validation
    (`monitor` : "loss" ou "ap") est relevée à chaque époque ; l'époque retenue minimise la
    courbe moyenne des plis, et la tête est réentraînée sur tout `X` pour ce nombre d'époques
    (en lot entier, une époque = un pas, quel que soit l'effectif). Un pli unique de 1 à 3
    micros donnait une époque au hasard (données simulées, DECISIONS n° 117). Faute de deux
    micros, ou d'une classe dans un pli, R40/R42 sont sautées.
    """
    from blanci.evaluate import average_precision

    X, y, groups = np.asarray(X), np.asarray(y).astype(int), np.asarray(groups)
    extra: dict[str, Any] = {}
    if weight_decays and len(weight_decays) > 1:

        def score(wd, train, test):
            head = fit_with_options(
                fit,
                X[train],
                y[train],
                groups[train],
                seed,
                early_stopping=early_stopping,
                monitor=monitor,
                n_splits=n_splits,
                **options | {"weight_decay": wd},
            )
            return average_precision(y[test], head.decision(X[test]))

        best, results, _ = grouped_search(
            score, weight_decays, y, groups, n_splits, seed, more_regularized="high"
        )
        if best is not None:
            options["weight_decay"] = best
            extra["weight_decay_cv"] = {str(k): v for k, v in results.items()}

    def with_groups(rows) -> dict:
        """Les têtes qui apprennent aussi le micro (R66, `needs_groups`) reçoivent ses groupes."""
        return {"groups": groups[rows]} if getattr(fit, "needs_groups", False) else {}

    everything = np.arange(len(y))
    folds = usable_folds(y, groups, n_splits, seed) if early_stopping else []
    if folds:
        curves = [
            fit(
                X[train],
                y[train],
                seed=seed,
                validation=(X[test], y[test]),
                patience=None,
                monitor=monitor,
                **options,
                **with_groups(train),
            ).meta["validation_curve"]
            for train, test in folds
        ]
        mean_curve = np.nanmean(np.vstack(curves), axis=0)
        best_epoch = int(np.nanargmin(mean_curve)) + 1
        head = fit(X, y, seed=seed, **(options | {"epochs": best_epoch}), **with_groups(everything))
        head.meta |= extra | {
            "early_stopping": {"best_epoch": best_epoch, "monitor": monitor, "folds": len(curves)}
        }
        return head
    head = fit(X, y, seed=seed, **options, **with_groups(everything))
    head.meta |= extra
    return head


def regularizer_for(
    spec: str, cfg: dict, context: Context | None
) -> tuple[str, str, Regularizer | None]:
    """(nom canonique, tête de base, régularisation ou None) d'une tête du benchmark."""
    base, regs = parse_head(spec)
    validate(base, regs)
    name = head_name(base, regs)
    if not regs and base not in CONTEXT_HEADS:
        return name, base, None
    if context is None:
        raise ValueError(f"{name} : contexte des fenêtres manquant")
    return name, base, Regularizer(regs, cfg.get("regularization", {}) or {}, context)
