"""Briques communes des rassemblements du benchmark global (07, 08) : scores par minute, AP
moyenne par site, bootstrap apparié qui tire les enregistrements dans chaque site (l'AP d'un
tirage se calcule sur des scores triés une fois, pondérés par le nombre de tirages)."""

import numpy as np
import pandas as pd

from blanci.evaluation.evaluate import average_precision, bootstrap_p


def minutes(z: dict) -> pd.DataFrame:
    """Une ligne par enregistrement : site, label (NaN si douteux), score max de chaque tête."""
    df = pd.DataFrame({"rec": z["recording_id"], "site": z["site"], "y": z["y"]})
    for h, s in zip(z["heads"], z["scores"], strict=True):
        df[str(h)] = s
    agg = {"site": "first", "y": lambda v: np.nan if v.isna().all() else float(v.max())}
    agg.update({str(h): "max" for h in z["heads"]})
    return df.groupby("rec").agg(agg)


def windows(z: dict) -> pd.DataFrame:
    df = pd.DataFrame({"rec": z["recording_id"], "site": z["site"], "y": z["y"]})
    for h, s in zip(z["heads"], z["scores"], strict=True):
        df[str(h)] = s
    return df


def site_mean_ap(y: np.ndarray, s: np.ndarray, site: np.ndarray) -> float:
    aps = [average_precision(y[site == g], s[site == g]) for g in np.unique(site)]
    aps = [a for a in aps if not np.isnan(a)]
    return float(np.mean(aps)) if aps else np.nan


class _Sorted:
    """Scores d'un site triés une fois ; l'AP d'un tirage bootstrap ne change que les poids
    (nombre de tirages de chaque enregistrement) : même valeur que l'AP sur les fenêtres
    dupliquées (ex æquo groupés aux seuils distincts, comme `average_precision`)."""

    def __init__(self, y: np.ndarray, s: np.ndarray, rec_index: np.ndarray):
        order = np.argsort(-s, kind="mergesort")
        self.y, self.rec = y[order], rec_index[order]
        self.last = np.r_[np.flatnonzero(np.diff(s[order])), len(s) - 1]  # fin de chaque ex æquo

    def ap(self, counts: np.ndarray) -> float:
        w = counts[self.rec]
        tp = np.cumsum(w * self.y)[self.last]
        fp = np.cumsum(w * (1 - self.y))[self.last]
        if tp[-1] == 0:
            return np.nan
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        return float(np.sum(np.diff(np.r_[0.0, tp]) / tp[-1] * precision))


def paired(
    frame: pd.DataFrame, a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict:
    """AP moyenne par site, A − B ; enregistrements tirés avec remise dans chaque site. `p` :
    2 × (min(k(Δ ≤ 0), k(Δ ≥ 0)) + 1) / (n + 1), jamais nulle ; la plus petite atteignable est
    2/(n + 1), donc avec Holm sur M comparaisons il faut n_boot ≳ 2M/α pour qu'une différence
    puisse rester significative."""
    y, site, rec = frame["y"].to_numpy(), frame["site"].to_numpy(), frame["rec"].to_numpy()
    parts = []
    for g in np.unique(site):
        m = site == g
        if not y[m].sum():
            continue
        codes, index = np.unique(rec[m], return_inverse=True)
        parts.append((len(codes), _Sorted(y[m], a[m], index), _Sorted(y[m], b[m], index)))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        da, db = [], []
        for n, sa, sb in parts:
            counts = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            da.append(sa.ap(counts))
            db.append(sb.ap(counts))
        diffs.append(np.nanmean(da) - np.nanmean(db))
    d = np.asarray(diffs)
    d = d[~np.isnan(d)]
    return {
        "diff": site_mean_ap(y, a, site) - site_mean_ap(y, b, site),
        "lo": float(np.quantile(d, 0.025)),
        "hi": float(np.quantile(d, 0.975)),
        "p": bootstrap_p(d),
    }
