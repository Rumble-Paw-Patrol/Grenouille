"""Outil de sélection des candidats à écouter (§5, DECISIONS n° 99) : toutes les méthodes, une
sortie.

Chaque méthode rend une file au format du poste d'annotation (`workbench.CANDIDATE_COLUMNS` :
recording_id, path, site, mic_id, start_utc, offset_s, dur_s, score, reason, source), écrite
en `paths.reports/candidats_<nom>.csv` et ouverte telle quelle par `blanci annotate`. La
colonne `source` suit la fenêtre jusqu'au label : on saura quelle méthode a trouvé quoi.

- `active` : file 60-20-20 (incertains, meilleurs scores, aléatoire stratifié micro × heure),
  proportions réglables ; baseline, plus d'aléatoire en début d'entraînement ;
- `similarity` : fenêtres proches des positifs, moins les négatifs appariés ; positifs
  faciles, amorcer un nouveau site ;
- `coverage` : les fenêtres les plus éloignées de tout ce qui est déjà annoté (k-centres) ;
  YAPAT fait maison, version automatique : écouter ce qu'on n'a jamais entendu ;
- `cluster` : une dizaine de fenêtres par groupe HDBSCAN ; `label_cluster` étiquette ensuite
  le groupe entier s'il est homogène (négatifs en volume, positifs si un groupe est pur) ;
- `audit` : enregistrements entiers tirés au hasard (micro × heure) ; seule mesure du rappel
  indépendante de Blancinet ;
- `random` : fenêtres au hasard (site × micro, heures de pic) ; faux négatifs confiants ;
- `negative_mining` : scores élevés là où A. blanci est improbable, ou fenêtres proches des
  faux amis annotés ; apprendre les confusions ;
- `phenology` : gabarit — heures de pic et saison haute d'abord, une part hors strates ;
- `suspects` : détections isolées (statut « suspect ») ; faux positifs probables ou chant
  ponctuel ;
- `gaps` : le miroir — fenêtres négatives encadrées de positives (DECISIONS n° 102) ; mode
  `scores` : trous du modèle dans un chant (faux négatifs du modèle : positifs difficiles) ;
  mode `labels` : négatifs annotés entre deux positifs annotés, à réécouter ;
- `congeners` : fenêtres où Perch entend un *Anomaloglossus* congénère ;
- `blancinet` : détections Blancinet jamais écoutées.

La carte des embeddings (projection 2-D, `embedding_map`) sert au poste d'annotation : on y
choisit à la main une zone à écouter (YAPAT fait maison, sélection interactive). YAPAT lui-même
(Docker, PostgreSQL, embeddings BirdNET, licence non commerciale) reste un outil externe :
`export_clips` lui prépare les extraits, `import_clip_labels` relit ses réponses.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blanci.config import config_path
from blanci.dataset import current_labels, embedded_training_set, recordings_table
from blanci.db import encoder_params
from blanci.index import l2_normalize, search
from blanci.store import EmbeddingStore, gated_mask
from blanci.workbench import CANDIDATE_COLUMNS, _drop_labelled, _round_robin

SELECTION_METHODS = (
    "active",
    "similarity",
    "coverage",
    "cluster",
    "audit",
    "random",
    "negative_mining",
    "phenology",
    "suspects",
    "gaps",
    "congeners",
    "blancinet",
)
NEEDS_ENCODER = (
    "active",
    "similarity",
    "coverage",
    "cluster",
    "negative_mining",
    "suspects",
    "congeners",
)


def _store(cfg: dict, encoder_id: str) -> EmbeddingStore:
    return EmbeddingStore(config_path(cfg, "embeddings"), encoder_id)


def _finish(
    con: sqlite3.Connection,
    frame: pd.DataFrame,
    reason,
    source: str,
    drop_labelled: bool = True,
) -> pd.DataFrame:
    """Colonnes du poste d'annotation (chemin, site, micro, heure lus en base) + colonnes en
    plus (groupe, coordonnées de carte) ; fenêtres déjà étiquetées retirées. `reason` : texte,
    ou série alignée sur `frame`."""
    if frame.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    frame = frame.assign(reason=reason, source=source)
    if "score" not in frame:
        frame["score"] = np.nan
    info = recordings_table(con)[["recording_id", "path", "site", "mic_id", "start_utc"]]
    frame = frame.drop(columns=[c for c in ("path", "site", "mic_id", "start_utc") if c in frame])
    out = frame.merge(info, on="recording_id", how="left")
    out = out.assign(offset_s=out["offset_s"].astype(float).round(2))
    if drop_labelled:  # une réécoute (mode labels des trous) garde les fenêtres étiquetées
        out = _drop_labelled(con, out)
    extra = [c for c in ("cluster", "encoder_id", "x", "y") if c in out.columns]
    return out[CANDIDATE_COLUMNS + extra].reset_index(drop=True)


def _head_model(con: sqlite3.Connection, encoder_id: str, version: str) -> tuple[str, dict]:
    from blanci.service import head_id, load_head

    _, params = load_head(con, encoder_id, version)
    return head_id(encoder_id, params["version"]), params


def _best_windows(con: sqlite3.Connection, model_id: str, recording_ids) -> pd.DataFrame:
    """Fenêtre de score maximal de chaque enregistrement (celle qu'on écoute d'abord)."""
    from blanci.service import _in_chunks

    rows = _in_chunks(
        con,
        "SELECT w.recording_id, w.offset_s, w.dur_s, s.score FROM scores s "
        "JOIN windows w USING (window_id) WHERE s.model_id = '"
        + model_id.replace("'", "''")
        + "' AND w.recording_id IN ({})",
        list(recording_ids),
    )
    scores = pd.DataFrame(rows, columns=["recording_id", "offset_s", "dur_s", "score"])
    if scores.empty:
        return scores
    return scores.loc[scores.groupby("recording_id")["score"].idxmax()].reset_index(drop=True)


# --- Méthodes ----------------------------------------------------------------------------------


def active_candidates(
    con, cfg, encoder_id, n=None, mix=None, version="default", **_
) -> pd.DataFrame:
    """File 60-20-20 (proportions `mix`, défaut `active.mix`), unité l'enregistrement ; on
    écoute sa fenêtre de score maximal."""
    from blanci.service import make_queue

    model_id, params = _head_model(con, encoder_id, version)
    queue = make_queue(con, encoder_id, cfg, n, params["version"], tuple(mix) if mix else None)
    best = _best_windows(con, model_id, queue["recording_id"])
    out = queue[["recording_id", "reason"]].merge(best, on="recording_id", how="inner")
    return _finish(con, out, "active_" + out["reason"], "active")


def similarity_candidates(
    con, cfg, encoder_id, n=300, site=None, paired_negatives=True, **_
) -> pd.DataFrame:
    from blanci.service import similarity_search

    found = similarity_search(
        con,
        encoder_id,
        cfg,
        k=n,
        filters={"site": site} if site else None,
        use_paired_negatives=paired_negatives,
    )
    found["dur_s"] = encoder_params(con, encoder_id)["window_s"]
    return _finish(con, found, "similarity", "similarity")


def coverage_candidates(
    con, cfg, encoder_id, n=50, pool=20000, site=None, seed=0, **_
) -> pd.DataFrame:
    """YAPAT fait maison, version automatique : k-centres gloutons. Part d'un échantillon du
    stock et ajoute à chaque tour la fenêtre la plus éloignée (cosinus) de tout ce qui est déjà
    annoté ou choisi : on écoute ce qu'on n'a jamais entendu."""
    store = _store(cfg, encoder_id)
    meta, X = store.sample(pool, filters={"site": site} if site else None, seed=seed)
    open_rows = ~gated_mask(meta)
    meta, X = meta[open_rows].reset_index(drop=True), l2_normalize(X[open_rows])
    if not len(meta):
        raise ValueError(f"aucun embedding pour {encoder_id}")
    distance = np.full(len(meta), 2.0)
    try:  # tout ce qui est déjà annoté dans le stock, pas seulement dans l'échantillon
        _, annotated = embedded_training_set(
            con, store, encoder_params(con, encoder_id)["window_s"], per_positive=0
        )
    except ValueError:  # rien d'annoté encore : la couverture part de zéro
        annotated = np.zeros((0, X.shape[1]), dtype=np.float32)
    annotated = l2_normalize(annotated) if len(annotated) else annotated
    for start in range(0, len(annotated), 2048):
        anchors = annotated[start : start + 2048]
        distance = np.minimum(distance, 1.0 - (X @ anchors.T).max(axis=1))
    chosen = []
    for _ in range(min(n, len(meta))):
        i = int(np.argmax(distance))
        chosen.append(i)
        distance = np.minimum(distance, 1.0 - X @ X[i])
        distance[i] = -1.0
    out = meta.iloc[chosen].assign(dur_s=encoder_params(con, encoder_id)["window_s"], score=np.nan)
    return _finish(con, out, "coverage", "coverage")


def cluster_candidates(
    con, cfg, encoder_id, n=10, pool=None, site=None, seed=None, **_
) -> pd.DataFrame:
    """Une dizaine de fenêtres (`n`) par groupe HDBSCAN d'un échantillon du stock. Les groupes
    sont gardés (`paths.reports/clusters/<encodeur>.parquet`) pour `label_cluster`."""
    from blanci.cluster import NOISE, cluster_embeddings

    ccfg = cfg["cluster"]
    seed = ccfg["seed"] if seed is None else seed
    meta, X = _store(cfg, encoder_id).sample(
        pool or ccfg["c0_sample"], filters={"site": site} if site else None, seed=seed
    )
    open_rows = ~gated_mask(meta)
    meta, X = meta[open_rows].reset_index(drop=True), X[open_rows]
    groups = cluster_embeddings(
        X, ccfg["pca_components"], ccfg["min_cluster_size"], ccfg["min_samples"], seed
    )
    window_s = encoder_params(con, encoder_id)["window_s"]
    assignments = meta.assign(cluster=groups, dur_s=window_s)
    path = cluster_path(cfg, encoder_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(path, index=False)
    rng = np.random.default_rng(seed)
    picked = []
    for _, part in assignments[assignments["cluster"] != NOISE].groupby("cluster"):
        take = rng.choice(len(part), size=min(n, len(part)), replace=False)
        picked.append(part.iloc[np.sort(take)])
    out = pd.concat(picked) if picked else assignments.iloc[:0]
    out = out.assign(encoder_id=encoder_id)  # le poste d'annotation en a besoin (bloc)
    return _finish(con, out, "cluster_" + out["cluster"].astype(str), "cluster")


def audit_candidates(con, cfg, n=30, sites=None, peak_hours=False, reason="audit", seed=0, **_):
    from blanci.workbench import recording_candidates

    return recording_candidates(con, cfg, n, sites, peak_hours, reason, seed)


def random_window_candidates(con, cfg, n=20, sites=None, peak_hours=True, seed=0, **_):
    from blanci.workbench import random_candidates

    return random_candidates(con, cfg, n, sites, peak_hours, seed)


def negative_mining_candidates(
    con, cfg, encoder_id, n=50, mode="unlikely", version="default", **_
) -> pd.DataFrame:
    """Négatifs durs probables, à écouter puis réinjecter.

    - `unlikely` : meilleurs scores de la tête dans les enregistrements sans positif, là où
      A. blanci est improbable (mois creux, `activity.reference.low_months`, ou hors des heures
      de pic) : ce qui score haut là est d'abord un faux ami ;
    - `false_friends` : fenêtres proches des faux amis déjà annotés (oiseaux, amphibiens,
      insectes…) et loin des positifs (recherche par similarité, requêtes négatives).
    """
    if mode == "unlikely":
        model_id, _ = _head_model(con, encoder_id, version)
        scores = pd.read_sql_query(
            "SELECT w.recording_id, w.offset_s, w.dur_s, s.score FROM scores s "
            "JOIN windows w USING (window_id) WHERE s.model_id = ?",
            con,
            params=(model_id,),
        )
        if scores.empty:
            raise ValueError(f"aucun score pour {model_id} (lancer `blanci score`)")
        from blanci.labels import POSITIVE_LABELS

        labels = current_labels(con)
        positives = set(labels.loc[labels["label"].isin(POSITIVE_LABELS), "recording_id"])
        rec = recordings_table(con).set_index("recording_id")
        offset = cfg["recorder"]["filename_utc_offset_h"]
        local = pd.to_datetime(rec["start_utc"], utc=True) + pd.Timedelta(hours=offset)
        low = set(cfg["activity"]["reference"]["low_months"])
        peak = cfg["peak_hours_local"]
        off_peak = ~np.logical_or.reduce(
            [(local.dt.hour >= a) & (local.dt.hour < b) for a, b in peak]
        )
        unlikely = set(rec.index[local.dt.month.isin(low) | off_peak])
        pool = scores[~scores["recording_id"].isin(positives)]
        narrowed = pool[pool["recording_id"].isin(unlikely)]
        pool = narrowed if len(narrowed) else pool
        best = pool.sort_values("score", ascending=False).drop_duplicates("recording_id")
        return _finish(con, best.head(n), "mining_unlikely", "mining")
    if mode == "false_friends":
        from blanci.dataset import pairing_options
        from blanci.frozen import frozen_recordings

        params = encoder_params(con, encoder_id)
        store = _store(cfg, encoder_id)
        data, X = embedded_training_set(
            con,
            store,
            params["window_s"],
            per_positive=0,
            **pairing_options(cfg),
            exclude_recordings=frozen_recordings(cfg),
        )
        friends = (data["y"] == 0) & data["label"].isin(
            ["bird", "amphibian", "orthoptera", "amphibian_contact_call", "other"]
        )
        if not friends.any():
            raise ValueError("aucun faux ami annoté (oiseau, amphibien, insecte…) : pas de requête")
        positives = X[(data["y"] == 1).to_numpy()]
        found = search(
            X[friends.to_numpy()], store, k=n * 3, negatives=positives if len(positives) else None
        )
        found = found.drop_duplicates("recording_id").head(n).assign(dur_s=params["window_s"])
        return _finish(con, found, "mining_false_friends", "mining")
    raise ValueError(f"mode inconnu : {mode!r} (unlikely ou false_friends)")


def phenology_candidates(con, cfg, n=40, whole=False, sites=None, seed=0, **_) -> pd.DataFrame:
    """Gabarit d'échantillonnage phénologique (`selection.phenology`) : strates « priorité »
    (heures de pic, mois forts), « secondaire » (heures de pic, mois secondaires) et « hors
    strates », tirées dans les proportions `shares`, à parts égales entre micros. À affiner avec
    l'étude de phénologie (Courtois et al. 2025) quand l'inventaire 2023 sera fini."""
    from blanci.embed import select_recordings
    from blanci.grid import window_grid

    pcfg = cfg.get("selection", {}).get("phenology", {})
    rng = np.random.default_rng(seed)
    recordings = select_recordings(con)
    if sites:
        recordings = recordings[recordings["site"].str.lower().isin({s.lower() for s in sites})]
    if recordings.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    offset = cfg["recorder"]["filename_utc_offset_h"]
    local = pd.to_datetime(recordings["start_utc"], utc=True) + pd.Timedelta(hours=offset)
    hour, month = local.dt.hour, local.dt.month
    site_hours = {k.lower(): v for k, v in (pcfg.get("site_hours") or {}).items()}
    default_hours = pcfg.get("priority_hours", cfg["peak_hours_local"])
    in_hours = np.array(
        [
            any(a <= h < b for a, b in site_hours.get(str(s).lower(), default_hours))
            for s, h in zip(recordings["site"], hour, strict=True)
        ]
    )
    priority = in_hours & month.isin(pcfg.get("priority_months", [1, 2, 3, 4])).to_numpy()
    secondary = in_hours & month.isin(pcfg.get("secondary_months", [11, 12])).to_numpy() & ~priority
    stratum = np.where(priority, "priority", np.where(secondary, "secondary", "outside"))
    recordings = recordings.assign(stratum=stratum)
    shares = pcfg.get("shares", {"priority": 0.6, "secondary": 0.2, "outside": 0.2})
    picked = []
    for name, share in shares.items():
        part = recordings[recordings["stratum"] == name]
        picked.append(_round_robin(part, "mic_id", round(n * share), rng))
    chosen = pd.concat(picked) if picked else recordings.iloc[:0]
    w3 = cfg["grids"]["w3"]
    if whole:
        chosen = chosen.assign(offset_s=0.0, dur_s=chosen["duration_s"].astype(float).round(2))
    else:
        offsets = []
        for r in chosen.itertuples():
            grid = window_grid(r.duration_s or w3["window_s"], w3["window_s"], w3["hop_s"])
            offsets.append(grid[int(rng.integers(len(grid)))][0])
        chosen = chosen.assign(offset_s=offsets, dur_s=w3["window_s"])
    return _finish(con, chosen, "phenology_" + chosen["stratum"], "phenology")


def suspect_candidates(con, cfg, encoder_id, n=50, version="default", **_) -> pd.DataFrame:
    """Enregistrements « suspect » (détection isolée, §1) des décisions de la tête : la fenêtre
    qui a déclenché. Faux positif probable, ou chant ponctuel (à demander au tuteur)."""
    model_id, params = _head_model(con, encoder_id, version)
    rows = pd.read_sql_query(
        "SELECT DISTINCT recording_id FROM decisions WHERE encoder_id = ? AND head_version = ? "
        "AND status = 'suspect'",
        con,
        params=(encoder_id, params["version"]),
    )
    if rows.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    best = _best_windows(con, model_id, rows["recording_id"]).sort_values("score", ascending=False)
    return _finish(con, best.head(n), "suspect", "suspect")


def gap_candidates(
    con, cfg, encoder_id=None, n=50, mode="scores", version="default", **_
) -> pd.DataFrame:
    """Faux négatifs suspects : fenêtres négatives encadrées de positives à moins de
    `sequential.gap_radius_s` (le miroir des détections isolées, DECISIONS n° 102).

    - `scores` : fenêtres sous le seuil de la tête, encadrées de fenêtres au-dessus dans le même
      enregistrement ; les plus hautes d'abord, une par enregistrement. Si A. blanci y chante,
      ce sont les positifs que le modèle rate : les plus utiles à annoter ;
    - `labels` : négatifs annotés entre deux positifs annotés, à réécouter (la réponse s'ajoute
      comme une correction, jamais par-dessus).
    """
    from blanci.sequential import GAP_RADIUS_S, surrounded_by_positives

    radius = float(cfg.get("sequential", {}).get("gap_radius_s", GAP_RADIUS_S))
    if mode == "labels":
        from blanci.dataset import positive_annotations
        from blanci.labels import POSITIVE_LABELS

        labels = current_labels(con)
        positives = positive_annotations(labels)
        negatives = labels[
            ~labels["label"].isin(POSITIVE_LABELS + ("blanci_uncertain", "uncertain"))
        ]
        rows = []
        for rid, part in negatives.groupby("recording_id"):
            pos = positives[positives["recording_id"] == rid]
            if pos.empty:
                continue
            centers = part["offset_s"] + part["dur_s"] / 2
            mask = surrounded_by_positives(centers, pos["offset_s"] + pos["dur_s"] / 2, radius)
            rows.append(part[mask])
        found = pd.concat(rows) if rows else negatives.iloc[:0]
        found = found[["recording_id", "offset_s", "dur_s"]].head(n)
        return _finish(con, found, "gap_labels", "gap", drop_labelled=False)
    if mode != "scores":
        raise ValueError(f"mode inconnu : {mode!r} (scores ou labels)")
    if not encoder_id:
        raise ValueError("gaps, mode scores : un encodeur (stock scoré par sa tête) est nécessaire")
    model_id, params = _head_model(con, encoder_id, version)
    scores = pd.read_sql_query(
        "SELECT w.recording_id, w.offset_s, w.dur_s, s.score FROM scores s "
        "JOIN windows w USING (window_id) WHERE s.model_id = ?",
        con,
        params=(model_id,),
    )
    if scores.empty:
        raise ValueError(f"aucun score pour {model_id} (lancer `blanci score`)")
    threshold = float(params["threshold"])
    holes = []
    for _, part in scores.groupby("recording_id"):
        above = part["score"].to_numpy() >= threshold
        if above.sum() < 2:
            continue
        centers = (part["offset_s"] + part["dur_s"] / 2).to_numpy()
        mask = ~above & surrounded_by_positives(centers, centers[above], radius)
        holes.append(part[mask])
    if not holes:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    found = pd.concat(holes).sort_values("score", ascending=False)
    found = found.drop_duplicates("recording_id").head(n)
    return _finish(con, found, "gap_scores", "gap")


def congener_window_candidates(con, cfg, encoder_id, n=30, sites=None, **_):
    from blanci.workbench import congener_candidates

    return congener_candidates(con, encoder_id, n, sites)


def blancinet_window_candidates(con, cfg, table=None, n=30, sites=None, seed=0, **_):
    from blanci.workbench import blancinet_candidates

    if table is None:
        raise ValueError("blancinet : l'export des détections est à donner (table=…)")
    return blancinet_candidates(con, Path(table), cfg, n, sites, seed)


_DISPATCH = {
    "active": active_candidates,
    "similarity": similarity_candidates,
    "coverage": coverage_candidates,
    "cluster": cluster_candidates,
    "audit": audit_candidates,
    "random": random_window_candidates,
    "negative_mining": negative_mining_candidates,
    "phenology": phenology_candidates,
    "suspects": suspect_candidates,
    "gaps": gap_candidates,
    "congeners": congener_window_candidates,
    "blancinet": blancinet_window_candidates,
}


def select_candidates(
    con: sqlite3.Connection,
    cfg: dict,
    method: str,
    encoder_id: str | None = None,
    **options: Any,
) -> pd.DataFrame:
    """File de candidats de la méthode `method` (voir le tableau du module)."""
    if method not in _DISPATCH:
        raise ValueError(f"méthode inconnue : {method!r} (connues : {SELECTION_METHODS})")
    if method in NEEDS_ENCODER:
        if not encoder_id:
            raise ValueError(f"{method} demande un encodeur (stock d'embeddings)")
        return _DISPATCH[method](con, cfg, encoder_id, **options)
    return _DISPATCH[method](con, cfg, encoder_id=encoder_id, **options)


def write_queue(cfg: dict, candidates: pd.DataFrame, name: str) -> Path:
    """`paths.reports/candidats_<nom>.csv`, ouvert par le poste d'annotation."""
    path = config_path(cfg, "reports") / f"candidats_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(path, index=False)
    return path


# --- Étiquetage en bloc par groupe (§5 bis, C2) ---------------------------------------------------


def cluster_path(cfg: dict, encoder_id: str) -> Path:
    return config_path(cfg, "reports") / "clusters" / f"{encoder_id.replace(':', '_')}.parquet"


def cluster_status(
    con: sqlite3.Connection, cfg: dict, encoder_id: str, min_checked: int = 10
) -> pd.DataFrame:
    """Par groupe : taille, fenêtres écoutées, labels entendus, homogène (assez d'écoutes, un
    seul label) — c'est ce qui autorise l'étiquetage en bloc."""
    path = cluster_path(cfg, encoder_id)
    if not path.exists():
        raise ValueError(f"aucun groupe pour {encoder_id} (blanci select --method cluster)")
    assignments = pd.read_parquet(path)
    labels = current_labels(con)[["window_id", "label", "source"]]
    heard = assignments.merge(labels[labels["source"] != "bulk"], on="window_id")
    rows = []
    for cluster, part in assignments.groupby("cluster"):
        mine = heard[heard["cluster"] == cluster]
        counts = mine["label"].value_counts()
        rows.append(
            {
                "cluster": int(cluster),
                "n_windows": len(part),
                "n_listened": len(mine),
                "labels": json.dumps(counts.to_dict(), ensure_ascii=False),
                "homogeneous": bool(len(mine) >= min_checked and len(counts) == 1),
                "label": counts.index[0] if len(counts) == 1 else None,
            }
        )
    return pd.DataFrame(rows)


def label_cluster(
    con: sqlite3.Connection,
    cfg: dict,
    encoder_id: str,
    cluster: int,
    label: str | None = None,
    annotator: str | None = None,
    min_checked: int = 10,
    force: bool = False,
) -> int:
    """Étiquette en bloc toutes les fenêtres non écoutées d'un groupe homogène (§5 bis, C2).

    Refusé si le groupe n'a pas `min_checked` fenêtres écoutées toutes du même label (sauf
    `force`), ou si `label` contredit ce qu'on a entendu. Source « bulk » : ces labels se
    distinguent toujours des labels écoutés un par un."""
    from blanci.cluster import NOISE
    from blanci.service import append_labels_bulk

    if cluster == NOISE:
        raise ValueError("le bruit HDBSCAN (−1) n'est pas un groupe")
    status = cluster_status(con, cfg, encoder_id, min_checked).set_index("cluster")
    if cluster not in status.index:
        raise ValueError(f"groupe inconnu : {cluster}")
    row = status.loc[cluster]
    heard = row["label"]
    if not row["homogeneous"] and not force:
        raise ValueError(
            f"groupe {cluster} non homogène ou trop peu écouté ({row['n_listened']} écoutes, "
            f"labels {row['labels']}) : écouter encore, ou --force"
        )
    label = label or heard
    if label is None:
        raise ValueError("label à donner : le groupe n'a pas de label unique")
    if heard is not None and label != heard and not force:
        raise ValueError(f"le groupe {cluster} a été entendu « {heard} », pas « {label} »")
    assignments = pd.read_parquet(cluster_path(cfg, encoder_id))
    members = assignments.loc[assignments["cluster"] == cluster]
    done = set(current_labels(con)["window_id"])
    todo = [w for w in members["window_id"] if w not in done]
    return append_labels_bulk(
        con,
        todo,
        label,
        "bulk",
        {
            "propagated_from_cluster": int(cluster),
            "encoder_id": encoder_id,
            "checked": int(row["n_listened"]),
            "forced": bool(force and not row["homogeneous"]),
        },
        annotator,
    )


# --- Carte des embeddings (YAPAT fait maison, sélection à la main) --------------------------------


def embedding_map(
    con: sqlite3.Connection,
    cfg: dict,
    encoder_id: str,
    n: int = 5000,
    method: str = "auto",
    site: str | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Échantillon du stock projeté en 2-D (x, y) avec son label s'il en a un et son score de
    tête s'il existe : la carte du poste d'annotation, où l'on entoure une zone à écouter.

    `method` : umap (si installé), tsne, pca ; auto = umap sinon t-SNE (PCA au-delà de 20 000
    fenêtres, t-SNE devenant lent)."""
    from sklearn.decomposition import PCA

    meta, X = _store(cfg, encoder_id).sample(n, filters={"site": site} if site else None, seed=seed)
    open_rows = ~gated_mask(meta)
    meta, X = meta[open_rows].reset_index(drop=True), l2_normalize(X[open_rows])
    if len(meta) < 3:
        raise ValueError(f"trop peu d'embeddings pour {encoder_id}")
    Z = PCA(n_components=min(50, X.shape[1], len(X) - 1), random_state=seed).fit_transform(X)
    if method == "auto":
        try:
            import umap  # noqa: F401

            method = "umap"
        except ImportError:
            method = "tsne" if len(meta) <= 20000 else "pca"
    if method == "umap":
        import umap

        xy = umap.UMAP(n_components=2, random_state=seed).fit_transform(Z)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = max(2.0, min(30.0, (len(Z) - 1) / 3))
        xy = TSNE(2, perplexity=perplexity, random_state=seed, init="pca").fit_transform(Z)
    elif method == "pca":
        xy = Z[:, :2]
    else:
        raise ValueError(f"projection inconnue : {method!r} (umap, tsne, pca)")
    labels = current_labels(con)[["window_id", "label"]]
    out = meta.assign(x=xy[:, 0], y=xy[:, 1], dur_s=encoder_params(con, encoder_id)["window_s"])
    out = out.merge(labels, on="window_id", how="left")
    return out.assign(label=out["label"].fillna("non écouté"))


def map_selection(con: sqlite3.Connection, selected: pd.DataFrame) -> pd.DataFrame:
    """File de candidats à partir des points entourés sur la carte."""
    return _finish(con, selected.assign(score=np.nan), "map", "coverage")


# --- Passerelle YAPAT -----------------------------------------------------------------------------


def export_clips(
    cfg: dict, candidates: pd.DataFrame, name: str, context_s: float = 2.0, channel: int = 0
) -> Path:
    """Extraits WAV des candidats (fenêtre ± `context_s`) dans `paths.exports/<nom>/`, avec
    `manifest.csv` (extrait → enregistrement, décalage, durée) : de quoi les charger dans YAPAT
    ou tout autre outil. L'audio d'origine n'est que lu."""
    import soundfile as sf

    from blanci.workbench import read_clip

    directory = config_path(cfg, "exports") / name
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(candidates.itertuples()):
        wav, sr, start = read_clip(
            config_path(cfg, "raw"), c.path, c.offset_s, c.dur_s, context_s, channel
        )
        clip = f"{i:05d}_{Path(str(c.path)).stem}_{c.offset_s:.2f}.wav"
        sf.write(directory / clip, wav, sr, subtype="PCM_16")
        rows.append(
            {
                "clip": clip,
                "recording_id": c.recording_id,
                "offset_s": c.offset_s,
                "dur_s": c.dur_s,
                "clip_start_s": start,
                "site": c.site,
                "mic_id": c.mic_id,
                "reason": c.reason,
            }
        )
    manifest = directory / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def import_clip_labels(
    con: sqlite3.Connection,
    manifest: Path,
    answers: Path,
    label_map: dict[str, str] | None = None,
    annotator: str | None = None,
) -> int:
    """Relit les réponses d'un outil externe sur des extraits exportés : une colonne nommant
    l'extrait (fichier), une colonne de label (valeurs du schéma, ou traduites par
    `label_map`), un commentaire éventuel. Chaque réponse devient un label (source « yapat »).

    Le format d'export de YAPAT n'est pas documenté : colonnes reconnues comme à l'import des
    annotations, à vérifier au premier fichier réel."""
    from blanci.labels import LABELS, detect_columns, read_annotation_table
    from blanci.workbench import save_answer

    clips = pd.read_csv(manifest).set_index("clip")
    table = read_annotation_table(Path(answers))
    columns = detect_columns(
        table,
        {
            "file": ["clip", "file", "fichier", "filename", "audio", "nom"],
            "label": ["label", "etiquette", "classe", "class", "annotation"],
            "comment": ["commentaire", "comment", "notes", "remarque"],
        },
    )
    for needed in ("file", "label"):
        if needed not in columns:
            raise ValueError(f"colonne {needed!r} introuvable dans {Path(answers).name}")
    label_map = label_map or {}
    written, unknown = 0, set()
    for record in table.to_dict("records"):
        clip = Path(str(record[columns["file"]])).name
        raw = str(record[columns["label"]]).strip()
        label = label_map.get(raw, raw)
        if clip not in clips.index:
            continue
        if label not in LABELS:
            unknown.add(raw)
            continue
        c = clips.loc[clip]
        comment = record.get(columns.get("comment", ""), None)
        save_answer(
            con,
            {
                "recording_id": c["recording_id"],
                "offset_s": float(c["offset_s"]),
                "dur_s": float(c["dur_s"]),
                "reason": c.get("reason"),
                "source": "yapat",
            },
            label,
            annotator or "yapat",
            comment=None if comment is None or pd.isna(comment) else str(comment),
        )
        written += 1
    if unknown:
        raise ValueError(
            f"{written} labels écrits ; valeurs sans correspondance (label_map) : {sorted(unknown)}"
        )
    return written
