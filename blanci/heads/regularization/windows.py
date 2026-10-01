"""Régularisations fenêtre par fenêtre et poids des exemples : statistiques du stock par micro
(R19, R20), directions du micro (R21), poids (R13, R15, R36), indicatrices de biais (R37)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    from blanci.embedding.store import gated_mask
    from blanci.inputs.dataset import recordings_table

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


def site_of(groups: np.ndarray) -> np.ndarray:
    """Site de chaque point « site/micro » (ce qui précède le premier « / »)."""
    return np.array([g.split("/", 1)[0] for g in np.asarray(groups).astype(str)])


def bias_indicators(
    groups: np.ndarray, train: np.ndarray, sites: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """R37 : (indicatrices, niveau de chaque colonne). Niveau 0 : une colonne par micro de
    l'entraînement ; avec `sites`, niveau 1 : une colonne par site de l'entraînement (biais de
    site et biais de micro emboîtés, comme les effets aléatoires d'un GLMM site/micro). Un
    micro nouveau d'un site connu reçoit alors le biais de son site ; un site nouveau, le biais
    commun."""
    micro = group_indicators(groups, train)
    levels = np.zeros(micro.shape[1], dtype=int)
    if not sites:
        return micro, levels
    site = group_indicators(site_of(groups), train)
    return np.hstack([micro, site]), np.concatenate([levels, np.ones(site.shape[1], dtype=int)])
