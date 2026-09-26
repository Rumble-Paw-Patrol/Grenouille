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

Têtes entraînées avec torch (DECISIONS n° 117, `blanci/attentive.py`) :

| R | têtes | quoi |
|---|---|---|
| R40 | attentive, gated | weight decay choisi par validation groupée sur `grid` |
| R41 | attentive | AdamW (weight decay découplé, `weight_decay`) |
| R42 | attentive, gated | nombre d'époques par validation groupée (`R42=ap`, `R42=loss`) |
| R45 | attentive | dropout des jetons (`p`) |
| R46 | attentive, gated | dropout des dimensions du vecteur agrégé (`p`) |
| R47 | attentive | départ et rétrécissement vers la logistique (`strength` λ) |

Ailleurs : R22 = pooling `gem` (`blanci/pooling.py`), R30 = tête `logistic_to_prototype`,
R31 = tête `lda_shrunk` (`blanci/head.py`), R26 = la L2, déjà là, R39 = têtes `knn:k=…`,
`exemplar:k=…`.

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

IMPLEMENTED = (13, 15, 17, 18, 19, 20, 21, 27, 28, 36, 37, 40, 41, 42, 45, 46, 47)
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
}
WEIGHTED_HEADS = ("logistic", "cascade", "logistic_to_prototype", "loss")
PENALIZED_HEADS = ("logistic", "cascade")
GROUP_BIAS_HEADS = ("logistic", "cascade", "loss")
# Régularisations des têtes entraînées avec torch, et celles que chacune accepte.
TORCH_REGULARIZATIONS = {"attentive": (40, 41, 42, 45, 46, 47), "gated": (40, 42, 46)}
# Réglages principaux qui prennent un mot plutôt qu'un nombre : R42=ap, R42=loss.
WORD_VALUES = {42: ("ap", "loss")}
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
    if 37 in regs and family not in GROUP_BIAS_HEADS:
        raise ValueError(f"R37 (biais par micro) : têtes {', '.join(GROUP_BIAS_HEADS)} seulement")
    if {27, 28} & set(regs) and family not in PENALIZED_HEADS:
        raise ValueError(f"R27/R28 (pénalité) : têtes {', '.join(PENALIZED_HEADS)} seulement")


def needs_domain(specs: list[str]) -> bool:
    return any({19, 20} & set(parse_head(s)[1]) for s in specs)


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

    def subset(self, rows: np.ndarray) -> Context:
        return Context(
            self.groups[rows],
            None if self.hard is None else self.hard[rows],
            self.domain,
            None if self.domain_rows is None else self.domain_rows[rows],
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

    def window_transform(self, X: np.ndarray) -> np.ndarray:
        """R19 / R20 puis R17 : fenêtre par fenêtre, sans apprentissage."""
        X = np.asarray(X, dtype=np.float32)
        if {19, 20} & set(self.regs):
            ctx = self.context
            if ctx.domain is None or ctx.domain_rows is None:
                raise ValueError("R19/R20 demandent les statistiques du stock (Context.domain)")
            X = X - ctx.domain.mean[ctx.domain_rows]
            if 20 in self.regs:
                eps = float(self.param(20, "eps", 1e-6))
                X = X / (ctx.domain.std[ctx.domain_rows] + eps)
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

    def torch_options(self) -> dict[str, Any]:
        """Options des têtes torch (`attentive.fit_with_options`) : R40–R42, R45–R47."""
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
        bias_columns = 0
        if 37 in self.regs:
            indicators = group_indicators(self.context.groups, train)
            bias_columns = indicators.shape[1]
            X = np.hstack([np.asarray(X, dtype=np.float32), indicators])
        weights = sample_weights(
            y[train],
            self.context.groups[train],
            None if self.context.hard is None else self.context.hard[train],
            self.regs,
            float(self.param(15, "hard_weight", 3.0)),
            float(self.param(36, "power", 0.0)),
        )
        return X, weights, self.fit_options(bias_columns)


def regularizer_for(
    spec: str, cfg: dict, context: Context | None
) -> tuple[str, str, Regularizer | None]:
    """(nom canonique, tête de base, régularisation ou None) d'une tête du benchmark."""
    base, regs = parse_head(spec)
    validate(base, regs)
    name = head_name(base, regs)
    if not regs:
        return name, base, None
    if context is None:
        raise ValueError(f"{name} : contexte des fenêtres manquant")
    return name, base, Regularizer(regs, cfg.get("regularization", {}) or {}, context)
