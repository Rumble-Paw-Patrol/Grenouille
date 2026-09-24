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


def c0_summary(labels: np.ndarray, mics: np.ndarray) -> dict[str, float]:
    kept = labels != NOISE
    return {
        "n_windows": int(len(labels)),
        "n_clusters": int(len(np.unique(labels[kept]))),
        "noise_share": float(1 - kept.mean()) if len(labels) else float("nan"),
        "ami_mic": ami(labels, mics),
    }


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
