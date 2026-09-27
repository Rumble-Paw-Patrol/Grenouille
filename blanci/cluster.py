"""Voie non supervisée (§5 bis) : HDBSCAN sur ACP des embeddings, tests C0 et C1.

- C0 (exploration) : échantillon global ; les groupes suivent-ils les micros (AMI groupe/micro
  élevé = l'embedding encode surtout le paysage sonore du point) ? Groupes dominés par un
  enregistrement signalé (saturation, micro dans sac) = anomalies à écouter.
- C1 (le clustering peut-il détecter ?) : positifs connus + fenêtres des mêmes micros aux mêmes
  heures. Réussi si le meilleur groupe rassemble ≥ 50 % des positifs, avec un enrichissement
  ≥ 20 (part de positifs du groupe / part globale), et si l'AMI groupe/micro reste faible.
  Seuils du §5 bis (jugement), dans `cluster` de la config.

La note occupe 2–3 % d'une fenêtre : les groupes ressemblent d'abord à des paysages sonores.
C'est ce que C1 mesure ; un échec n'empêche pas les autres usages (négatifs en volume, C2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score

from blanci.index import l2_normalize

NOISE = -1


def cluster_embeddings(
    X: np.ndarray,
    n_components: int = 50,
    min_cluster_size: int = 15,
    min_samples: int | None = 5,
    seed: int = 0,
) -> np.ndarray:
    """Groupe de chaque fenêtre (−1 = bruit HDBSCAN) : normalisation L2, ACP, HDBSCAN."""
    X = l2_normalize(np.asarray(X, dtype=np.float32))
    k = max(1, min(n_components, X.shape[1], len(X) - 1))
    Z = PCA(n_components=k, random_state=seed).fit_transform(X)
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True)
    return model.fit_predict(Z)


def ami(labels: np.ndarray, groups: np.ndarray) -> float:
    """AMI entre groupes et une partition connue (micros), bruit HDBSCAN exclu."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    kept = labels != NOISE
    if kept.sum() < 2 or len(np.unique(labels[kept])) < 2:
        return float("nan")
    return float(adjusted_mutual_info_score(groups[kept], labels[kept]))


def cluster_table(
    labels: np.ndarray,
    mics: np.ndarray,
    y: np.ndarray | None = None,
    flagged: np.ndarray | None = None,
) -> pd.DataFrame:
    """Une ligne par groupe : taille, micro dominant et sa part, positifs, rappel, enrichissement,
    part de fenêtres d'enregistrements signalés."""
    df = pd.DataFrame({"cluster": labels, "mic": mics})
    if y is not None:
        df["y"] = np.asarray(y).astype(int)
    if flagged is not None:
        df["flagged"] = np.asarray(flagged).astype(bool)
    rows = []
    total_pos = int(df["y"].sum()) if "y" in df else 0
    base_rate = total_pos / len(df) if len(df) else float("nan")
    for cluster, part in df.groupby("cluster"):
        counts = part["mic"].value_counts()
        row: dict[str, Any] = {
            "cluster": int(cluster),
            "n": len(part),
            "top_mic": counts.index[0],
            "top_mic_share": float(counts.iloc[0] / len(part)),
            "n_mics": int(part["mic"].nunique()),
        }
        if "y" in part:
            n_pos = int(part["y"].sum())
            precision = n_pos / len(part)
            row |= {
                "n_pos": n_pos,
                "recall": n_pos / total_pos if total_pos else float("nan"),
                "precision": precision,
                "enrichment": precision / base_rate if base_rate else float("nan"),
            }
        if "flagged" in part:
            row["flagged_share"] = float(part["flagged"].mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    order = "n_pos" if "n_pos" in table else "n"
    return table.sort_values(order, ascending=False, kind="stable").reset_index(drop=True)


def c0_summary(
    labels: np.ndarray, mics: np.ndarray, sites: np.ndarray | None = None
) -> dict[str, float]:
    kept = labels != NOISE
    out = {
        "n_windows": int(len(labels)),
        "n_clusters": int(len(np.unique(labels[kept]))),
        "noise_share": float(1 - kept.mean()) if len(labels) else float("nan"),
        "ami_mic": ami(labels, mics),
    }
    if sites is not None and len(np.unique(sites)) > 1:
        out["ami_site"] = ami(labels, sites)
    return out


def variance_partition(X: np.ndarray, sites: np.ndarray, mics: np.ndarray) -> dict[str, float]:
    """Où est la variance des embeddings (normés) : part due au **site** (écart des moyennes de
    site à la moyenne générale), au **micro dans son site** (écart des moyennes de micro à celle
    de leur site), et **dans un micro** (heure, météo, chant, bruit). Décomposition exacte des
    sommes de carrés (ANOVA emboîtée), les trois parts font 1. Question de Léonard (27/09) :
    les micros d'un même site diffèrent-ils autant que deux sites ? Si la part « micro » domine
    la part « site », généraliser d'un micro à l'autre est déjà aussi dur que d'un site à
    l'autre. `mics` : le point (site/micro), pour qu'un micro déplacé compte une fois par site."""
    X = l2_normalize(np.asarray(X, dtype=np.float64))
    sites, mics = np.asarray(sites).astype(str), np.asarray(mics).astype(str)
    grand = X.mean(axis=0)
    total = float(((X - grand) ** 2).sum())
    between_sites = between_mics = 0.0
    for site in np.unique(sites):
        in_site = sites == site
        site_mean = X[in_site].mean(axis=0)
        between_sites += in_site.sum() * float(((site_mean - grand) ** 2).sum())
        for mic in np.unique(mics[in_site]):
            in_mic = in_site & (mics == mic)
            between_mics += in_mic.sum() * float(((X[in_mic].mean(axis=0) - site_mean) ** 2).sum())
    if total <= 0:
        return {"share_site": float("nan"), "share_mic": float("nan"), "share_within": float("nan")}
    return {
        "share_site": between_sites / total,
        "share_mic": between_mics / total,
        "share_within": 1.0 - (between_sites + between_mics) / total,
        "n_sites": int(len(np.unique(sites))),
        "n_mics": int(len(np.unique(mics))),
    }


def pca_information_curve(X: np.ndarray, mics: np.ndarray | None = None) -> pd.DataFrame:
    """Part de la variance des embeddings **perdue** quand on ne garde que les k premières
    composantes de l'ACP, pour k = 1 … d (question de Léonard, 27/09) : l'ACP de R18 et du
    clustering (`cluster.pca_components`) garde les k premières, ce tableau dit ce qu'elle jette.

    Colonnes : `components` (k), `lost_raw` (en %, embeddings tels quels, comme R18 seule) et,
    avec `mics`, `lost_centred` (après centrage par micro, comme R19 puis R18 : l'ACP ne voit
    plus ce qui distingue les micros). La variance n'est pas l'information utile : une direction
    de faible variance peut porter le chant (la note occupe 2–3 % d'une fenêtre). La courbe dit
    combien de dimensions portent l'essentiel du **paysage sonore** ; ce qu'elles valent pour
    A. blanci se lit dans le benchmark (`logistic+R18=16`, `=32`, `=64`).

    Chaque courbe est en % de sa propre variance totale ; `attrs["mic_share"]` : la part de la
    variance brute que le centrage par micro retire (les différences entre micros)."""
    X = np.asarray(X, dtype=np.float64)
    totals = {}

    def lost(A: np.ndarray, name: str) -> np.ndarray:
        A = A - A.mean(axis=0)
        eig = np.clip(np.linalg.eigvalsh(A.T @ A)[::-1], 0.0, None)
        totals[name] = total = eig.sum()
        return 100.0 * (1.0 - np.cumsum(eig) / total) if total > 0 else np.zeros(len(eig))

    out = pd.DataFrame({"components": np.arange(1, X.shape[1] + 1), "lost_raw": lost(X, "raw")})
    if mics is not None:
        mics = np.asarray(mics).astype(str)
        centred = X.copy()
        for mic in np.unique(mics):
            rows = mics == mic
            centred[rows] -= centred[rows].mean(axis=0)
        out["lost_centred"] = lost(centred, "centred")
        if totals["raw"] > 0:
            out.attrs["mic_share"] = float(1.0 - totals["centred"] / totals["raw"])
    out.iloc[:, 1:] = out.iloc[:, 1:].clip(lower=0.0)
    return out


def components_for(curve: pd.DataFrame, column: str, kept: float) -> int:
    """Plus petit nombre de composantes qui garde au moins `kept` % de la variance."""
    enough = curve.loc[curve[column] <= 100.0 - kept + 1e-9, "components"]
    return int(enough.iloc[0]) if len(enough) else int(curve["components"].iloc[-1])


def c1_verdict(
    labels: np.ndarray,
    y: np.ndarray,
    mics: np.ndarray,
    min_recall: float = 0.5,
    min_enrichment: float = 20.0,
    max_ami_mic: float = 0.3,
) -> dict[str, Any]:
    """Critères de C1 (§5 bis) sur le meilleur groupe (hors bruit), choisi par rappel."""
    table = cluster_table(labels, mics, y)
    real = table[table["cluster"] != NOISE]
    summary = c0_summary(labels, mics)
    if real.empty or real["n_pos"].sum() == 0:
        best = {"cluster": None, "recall": 0.0, "enrichment": float("nan")}
    else:
        best = real.sort_values(["recall", "enrichment"], ascending=False).iloc[0].to_dict()
    ami_mic = summary["ami_mic"]
    passed = (
        best["recall"] >= min_recall
        and best["enrichment"] >= min_enrichment
        and (np.isnan(ami_mic) or ami_mic <= max_ami_mic)
    )
    return summary | {
        "best_cluster": best["cluster"],
        "best_recall": float(best["recall"]),
        "best_enrichment": float(best["enrichment"]),
        "passed": bool(passed),
    }
