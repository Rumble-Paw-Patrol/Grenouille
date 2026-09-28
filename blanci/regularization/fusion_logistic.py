"""Fusion logistique régularisée : C par validation groupée (R50), contraintes (R52, R53, R56),
descripteurs courbés (R54, R55)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from blanci.regularization.selection import grouped_search, usable_folds


def choose_fusion_C(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C_grid: list[float] | tuple[float, ...],
    n_splits: int = 5,
    seed: int = 0,
    method: str = "logistic",
    columns: list[str] | None = None,
    **options,
) -> tuple[float | None, dict[float, float]]:
    """R50 : C de la fusion logistique `method` (`logistic`, ou ses variantes `+R52`, `+R53`,
    `+R56`), choisi par validation groupée. (None, {}) faute de deux valeurs, de deux micros
    ou d'un pli à deux classes."""
    from blanci.evaluate import average_precision
    from blanci.fusion import fit_fusion_model

    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
    if len(C_grid) < 2 or not usable_folds(y, groups, n_splits, seed):
        return None, {}
    columns = columns or [f"x{j}" for j in range(X.shape[1])]

    def score(C, train, test):
        model = fit_fusion_model(method, X[train], y[train], columns, C=C, **options)
        return average_precision(y[test], model.decision(X[test]))

    best, results, _ = grouped_search(score, [float(c) for c in C_grid], y, groups, n_splits, seed)
    return best, results


# --- R52, R53, R56 : fusion logistique contrainte -----------------------------------------------

FUSION_SUFFIXES = {
    50: "C par validation groupée",
    52: "poids ≥ 0 (combinaison convexe à l'échelle près)",
    53: "sélection L1 des entrées",
    54: "descripteurs en paliers monotones (classes)",
    55: "descripteurs en courbes lisses (GAM, P-splines)",
    56: "pas de veto : contribution des descripteurs plafonnée",
}
# « R54=v », « R55=v », « R56=v » : le réglage principal, à la place de celui de la config.
FUSION_PARAMETER = {54: "bins", 55: "smoothness", 56: "cap"}


def fusion_settings(method: str) -> dict[int, float | None] | None:
    """« logistic+R50+R55=10 » → {50: None, 55: 10.0} ; « logistic » → {} ; None si ce n'est
    pas la fusion logistique. Suffixes possibles : `FUSION_SUFFIXES` ; valeur (« =v ») pour
    ceux de `FUSION_PARAMETER`. R54 et R55 s'excluent (deux façons de courber un
    descripteur), R53 et R55 aussi (la L1 ne lisse pas une courbe)."""
    base, *parts = method.split("+")
    if base != "logistic":
        return None
    settings: dict[int, float | None] = {}
    for part in parts:
        match = re.fullmatch(r"R(\d+)(?:=([0-9.eE+-]+))?", part.strip())
        if not match or int(match.group(1)) not in FUSION_SUFFIXES:
            raise ValueError(
                f"fusion {method!r} : suffixe {part!r} inconnu "
                f"({', '.join(f'R{n}' for n in FUSION_SUFFIXES)})"
            )
        number, value = int(match.group(1)), match.group(2)
        if value is not None and number not in FUSION_PARAMETER:
            raise ValueError(f"fusion {method!r} : R{number} ne prend pas de valeur")
        settings[number] = float(value) if value is not None else None
    if {54, 55} <= settings.keys():
        raise ValueError(f"fusion {method!r} : R54 ou R55, pas les deux")
    if {53, 55} <= settings.keys():
        raise ValueError(f"fusion {method!r} : R53 (L1) ne se combine pas avec R55 (lissage)")
    return settings


def fusion_variant(method: str) -> set[int] | None:
    """« logistic+R50+R52 » → {50, 52} ; « logistic » → set() ; None si ce n'est pas la
    fusion logistique. Suffixes possibles : `FUSION_SUFFIXES`."""
    settings = fusion_settings(method)
    return None if settings is None else set(settings)


def fit_constrained_logistic(
    z: np.ndarray,
    y: np.ndarray,
    C: float,
    nonneg: bool | np.ndarray = False,
    l1: bool = False,
    cap: float | None = None,
    head_index: int | None = None,
    penalty: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """(coefficients, biais) d'une logistique de fusion à classes équilibrées, sous contraintes.

    Perte : C · Σ poids · log(1 + e^(−t·s)) + pénalité, avec
    - R52 (`nonneg`) : coefficients ≥ 0 sur des entrées déjà orientées (« plus haut = plus
      A. blanci ») — pour classer, c'est une combinaison convexe à l'échelle près. Un booléen
      pour toutes les colonnes, ou un masque par colonne (R54 : les paliers seulement) ;
    - R53 (`l1`) : pénalité ‖w‖₁ au lieu de ½‖w‖² : les entrées inutiles tombent à 0 (exactement
      avec R52, à ~0 sinon) ;
    - R55 (`penalty`) : pénalité ½ wᵀ P w au lieu de ½‖w‖² (P : `DescriptorBasis.penalty`) ;
    - R56 (`cap`) : s = w_tête·z_tête + cap · tanh(Σ_autres w_j z_j / cap) + b. Les
      descripteurs déplacent le score d'au plus `cap` (en logit) : une tête assez sûre d'elle
      passe toujours, aucun descripteur n'a de droit de veto.
    """
    from scipy.optimize import minimize
    from scipy.special import expit

    z, y = np.asarray(z, dtype=float), np.asarray(y).astype(int)
    n, k = z.shape
    counts = np.bincount(y, minlength=2)
    sw = (n / (2.0 * np.maximum(counts, 1)))[y]
    t = 2.0 * y - 1.0
    positive = np.broadcast_to(np.asarray(nonneg, dtype=bool), (k,)).copy()
    if l1 and penalty is not None:
        raise ValueError("pénalité L1 ou quadratique, pas les deux")
    others = np.ones(k, dtype=bool)
    if cap is not None:
        if head_index is None:
            raise ValueError("R56 : la fusion n'a pas d'entrée « head » à protéger")
        others[head_index] = False
    eps = 1e-6

    def score(w, b):
        if cap is None:
            return z @ w + b, None
        u = z[:, others] @ w[others]
        return z[:, ~others] @ w[~others] + cap * np.tanh(u / cap) + b, u

    def objective(theta):
        w, b = theta[:-1], theta[-1]
        s, u = score(w, b)
        m = t * s
        loss = C * float(sw @ np.logaddexp(0.0, -m))
        g_s = -C * sw * t * expit(-m)
        if cap is None:
            grad_w = z.T @ g_s
        else:
            grad_w = np.empty(k)
            grad_w[~others] = z[:, ~others].T @ g_s
            grad_w[others] = z[:, others].T @ (g_s * (1.0 - np.tanh(u / cap) ** 2))
        if l1:  # |w| = w sur les colonnes ≥ 0 ; ailleurs, une valeur absolue lissée
            smooth = np.sqrt(w[~positive] ** 2 + eps**2)
            loss += float(w[positive].sum() + smooth.sum())
            grad_w = grad_w + np.where(positive, 1.0, 0.0)
            grad_w[~positive] += w[~positive] / smooth
        elif penalty is not None:
            pw = penalty @ w
            loss += 0.5 * float(w @ pw)
            grad_w = grad_w + pw
        else:
            loss += 0.5 * float(w @ w)
            grad_w = grad_w + w
        return loss, np.append(grad_w, g_s.sum())

    start = np.append(np.full(k, 0.01), 0.0)
    bounds = [(0.0, None) if p else (None, None) for p in positive] + [(None, None)]
    # Petits produits matrice-vecteur : BLAS à plusieurs fils y perd jusqu'à 100× son temps.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api="blas"):
        result = minimize(
            objective, start, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000}
        )
    return result.x[:-1], float(result.x[-1])


# --- R54, R55 : descripteurs courbés dans la fusion ----------------------------------------------


def is_descriptor(column: str) -> bool:
    """R54, R55 : les entrées courbées sont les descripteurs du module séquentiel. Le score de
    la tête (« head ») et les autres sources (« head:<encodeur> », « congeners:… »), des
    scores déjà faits pour classer, restent linéaires."""
    return column != "head" and ":" not in column


@dataclass
class DescriptorBasis:
    """R54, R55 : chaque descripteur (entrée standardisée z_j de la fusion) devient un bloc de
    colonnes, les autres entrées restent telles quelles. La fusion reste une logistique : sa
    contribution f_j(z_j) = bloc · coefficients n'est simplement plus une droite.

    - `kind` = "steps" (R54) : colonnes 1[s_j·z_j > t] aux seuils t, quantiles de
      l'entraînement (`bins` classes), s_j l'orientation de l'entrée (+1 si elle monte avec le
      label). Coefficients ≥ 0 : f_j est une marche qui ne fait que monter dans le sens « plus
      A. blanci », chaque coefficient est la hauteur d'une marche.
    - `kind` = "spline" (R55) : B-splines cubiques sur `segments` intervalles égaux entre les
      quantiles 1 % et 99 % de l'entraînement (P-splines, Eilers et Marx 1996). Au-delà, la
      courbe reste plate : pas d'extrapolation sur un site aux valeurs jamais vues.

    Colonnes centrées sur l'entraînement : f_j vaut 0 en moyenne, le biais de la fusion garde
    le niveau (et R56 plafonne une contribution centrée). `expanded` : entrée → réglages ; une
    entrée constante à l'entraînement n'est pas développée."""

    kind: str
    n_inputs: int
    expanded: dict[int, dict[str, Any]]

    def _raw(self, spec: dict[str, Any], x: np.ndarray) -> np.ndarray:
        if self.kind == "steps":
            cuts = np.asarray(spec["cuts"], dtype=float)
            return (spec["sign"] * x[:, None] > cuts[None, :]).astype(float)
        from scipy.interpolate import BSpline

        knots, degree = np.asarray(spec["knots"], dtype=float), int(spec["degree"])
        x = np.clip(x, knots[degree], knots[-degree - 1])
        return BSpline.design_matrix(x, knots, degree).toarray()

    def blocks(self) -> list[tuple[int, int]]:
        """Colonnes (début, fin) de chaque entrée dans la matrice développée."""
        out, start = [], 0
        for j in range(self.n_inputs):
            spec = self.expanded.get(j)
            width = len(spec["center"]) if spec else 1
            out.append((start, start + width))
            start += width
        return out

    def transform(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        parts = []
        for j in range(self.n_inputs):
            spec = self.expanded.get(j)
            if spec is None:
                parts.append(z[:, j : j + 1])
            else:
                parts.append(self._raw(spec, z[:, j]) - np.asarray(spec["center"]))
        return np.hstack(parts)

    def penalty(self, smoothness: float = 1.0, order: int = 2) -> np.ndarray:
        """R55 : matrice P de la pénalité ½ wᵀ P w. Entrée linéaire : 1 (la L2 ordinaire).
        Bloc de spline : G + `smoothness` · DᵀD, avec G = BᵀB / n la taille de la courbe (pour
        une droite w·z, fᵀf / n = w² : la même L2 qu'une entrée linéaire) et D les différences
        d'ordre `order` des coefficients voisins (la courbure). Quand C baisse, la courbe
        rétrécit comme une entrée ordinaire ; `smoothness` grand : la courbe tend vers une
        droite (le noyau de D d'ordre 2)."""
        blocks = self.blocks()
        size = blocks[-1][1]
        out = np.zeros((size, size))
        for j, (a, b) in enumerate(blocks):
            spec = self.expanded.get(j)
            if spec is None:
                out[a, a] = 1.0
                continue
            diff = np.diff(np.eye(b - a), n=order, axis=0)
            gram = np.asarray(spec["gram"], dtype=float)
            out[a:b, a:b] = gram + float(smoothness) * diff.T @ diff + 1e-6 * np.eye(b - a)
        return out

    def amplitudes(self, coef: np.ndarray) -> np.ndarray:
        """Amplitude de la contribution de chaque entrée quand elle va de −2 à +2 écarts-types
        (4·|w| pour une entrée linéaire) : la « part » de chaque entrée dans la fusion."""
        grid = np.linspace(-2.0, 2.0, 401)
        out = []
        for j, (a, b) in enumerate(self.blocks()):
            spec = self.expanded.get(j)
            if spec is None:
                out.append(4.0 * abs(float(coef[a])))
                continue
            x = grid
            if self.kind == "steps":
                cuts = np.asarray(spec["cuts"], dtype=float) * spec["sign"]
                x = np.concatenate([grid, cuts[np.abs(cuts) <= 2.0] + 1e-9])
            f = self._raw(spec, x) @ coef[a:b]
            out.append(float(f.max() - f.min()))
        return np.asarray(out)

    def curve(self, j: int, coef: np.ndarray, x: np.ndarray) -> np.ndarray:
        """f_j(x) : la contribution de l'entrée j (z standardisé) au logit de la fusion."""
        a, b = self.blocks()[j]
        spec = self.expanded.get(j)
        if spec is None:
            return np.asarray(x, dtype=float) * float(coef[a])
        return (self._raw(spec, np.asarray(x, dtype=float)) - np.asarray(spec["center"])) @ coef[
            a:b
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_inputs": self.n_inputs,
            "expanded": [[j, spec] for j, spec in sorted(self.expanded.items())],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DescriptorBasis:
        return cls(d["kind"], int(d["n_inputs"]), {int(j): spec for j, spec in d["expanded"]})


def fit_descriptor_basis(
    kind: str,
    z: np.ndarray,
    sign: np.ndarray,
    expand: list[int],
    bins: int = 5,
    segments: int = 8,
    degree: int = 3,
) -> DescriptorBasis:
    """R54 (`kind` = "steps", `bins` classes de même effectif) ou R55 ("spline", `segments`
    intervalles, degré `degree`) : la base de chaque entrée de `expand`, apprise sur z
    (entraînement de la fusion, entrées standardisées)."""
    if kind not in ("steps", "spline"):
        raise ValueError(f"R54, R55 : kind {kind!r} (steps ou spline)")
    z = np.asarray(z, dtype=float)
    basis = DescriptorBasis(kind, z.shape[1], {})
    for j in expand:
        x = z[:, j]
        if kind == "steps":
            oriented = sign[j] * x
            cuts = np.unique(np.quantile(oriented, np.linspace(0, 1, int(bins) + 1)[1:-1]))
            cuts = cuts[(cuts >= oriented.min()) & (cuts < oriented.max())]
            if not len(cuts):
                continue
            spec: dict[str, Any] = {"sign": float(sign[j]), "cuts": cuts.tolist()}
        else:
            lo, hi = np.quantile(x, [0.01, 0.99])
            if hi - lo < 1e-9:
                continue
            step = (hi - lo) / int(segments)
            knots = lo + step * np.arange(-int(degree), int(segments) + int(degree) + 1)
            spec = {"knots": knots.tolist(), "degree": int(degree)}
        raw = basis._raw(spec, x)
        spec["center"] = raw.mean(axis=0).tolist()
        if kind == "spline":
            centred = raw - raw.mean(axis=0)
            spec["gram"] = (centred.T @ centred / len(x)).tolist()
        basis.expanded[j] = spec
    return basis


def fit_fusion_logistic(
    z: np.ndarray,
    y: np.ndarray,
    sign: np.ndarray,
    columns: list[str],
    variant: set[int],
    C: float,
    cap: float = 2.0,
    bins: int = 5,
    segments: int = 8,
    smoothness: float = 1.0,
) -> tuple[np.ndarray, float, DescriptorBasis | None, int | None]:
    """Fusion logistique contrainte (R52, R53, R54, R55, R56) sur les entrées standardisées z :
    (coefficients sur la matrice développée, biais, base des descripteurs ou None, colonne de
    la tête dans la matrice développée ou None). Les coefficients intègrent déjà
    l'orientation : décision = base(z) · coefficients + biais (R56 : voir
    `fit_constrained_logistic`)."""
    kind = "steps" if 54 in variant else "spline" if 55 in variant else None
    basis = None
    if kind is not None:
        expand = [j for j, c in enumerate(columns) if is_descriptor(c)]
        basis = fit_descriptor_basis(kind, z, sign, expand, bins, segments)
    design = basis.transform(z) if basis is not None else np.asarray(z, dtype=float)
    blocks = basis.blocks() if basis is not None else [(j, j + 1) for j in range(z.shape[1])]
    linear = np.zeros(design.shape[1], dtype=bool)
    flip = np.ones(design.shape[1])
    for j, (a, _) in enumerate(blocks):
        if basis is None or j not in basis.expanded:
            linear[a] = True
            if 52 in variant:
                flip[a] = sign[j]
    nonneg = (linear & (52 in variant)) | (~linear & (kind == "steps"))
    head = blocks[list(columns).index("head")][0] if "head" in columns else None
    coef, intercept = fit_constrained_logistic(
        design * flip,
        y,
        C,
        nonneg=nonneg,
        l1=53 in variant,
        cap=float(cap) if 56 in variant else None,
        head_index=head,
        penalty=basis.penalty(smoothness) if kind == "spline" else None,
    )
    return coef * flip, intercept, basis, head
