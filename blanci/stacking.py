"""Stacking : entrées de niveau 1, emplacement du module séquentiel, benchmark de la fusion
(DECISIONS n° 94–95).

Entrées de niveau 1 d'une fusion, toutes hors-pli, sur les plis communs (`dataset.folds_for`) :

- `head` : score de la tête logistique de l'encodeur principal ;
- descripteurs du module séquentiel (`fusion.columns`) : rythme dans la fenêtre (débuts de
  notes, calculés sur l'audio) et persistance dans l'enregistrement (calculée sur les scores de
  la tête, pli par pli) ;
- autres sources (`fusion.sources`, ensemble de modèles) :
  - `head:<encodeur>` : tête logistique d'un autre encodeur, apprise pli par pli sur ses propres
    fenêtres, appliquée à toutes les fenêtres des enregistrements testés, ramenée sur la grille
    de l'encodeur principal (`fusion.project_scores`) ;
  - `congeners:<encodeur perch_v2>` : plus grand logit des *Anomaloglossus* congénères.

Emplacement du module séquentiel (`sequential.position`, une liste, vide = pas du tout) :

- `upstream` (amont) : le rythme sert de porte — une fenêtre sans assez de débuts de notes, ou
  d'intervalles d'A. blanci, prend le score le plus bas (portes `notes`, `rhythm` de
  `sequential.upstream`, les mêmes qu'à l'encodage ; module et seuillage ne font qu'un,
  DECISIONS n° 103). Les portes spectrales agissent à l'encodage (stock `+g-…`) ;
- `parallel` : les descripteurs de rythme (audio, indépendants de l'encodeur) entrent dans la
  fusion à côté du score de la tête ;
- `downstream` (aval) : les descripteurs de persistance, qui n'existent qu'après la tête (ils
  résument ses scores sur l'enregistrement), entrent dans la fusion.

`fusion_benchmark` compare chaque emplacement × chaque méthode de fusion à la tête seule, sur
les mêmes plis (bootstrap apparié par enregistrement), et donne la part de chaque entrée.
"""

from __future__ import annotations

import sqlite3
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from blanci.config import config_path
from blanci.dataset import embedded_training_set, folds_for, pairing_options
from blanci.db import encoder_params
from blanci.evaluate import evaluate, grouped_folds, paired_bootstrap, to_recordings
from blanci.frozen import frozen_recordings
from blanci.fusion import (
    FUSION_METHODS,
    FusionModel,
    fit_fusion_model,
    fusion_model_oof,
    project_scores,
)
from blanci.head import OOFScores, _choose_C, fit_logistic
from blanci.oof import labels_fingerprint, oof_frame, save_oof
from blanci.sequential import (
    GATED_SCORE,
    PERSISTENCE_COLUMNS,
    RHYTHM_COLUMNS,
    apply_gate,
    gate_mask,
    load_onsets,
    onset_counts,
    onset_gates,
    recording_persistence,
    window_rhythm,
)
from blanci.store import EmbeddingStore, gated_mask

SEQ_POSITIONS = ("upstream", "parallel", "downstream")
PERSISTENCE_THRESHOLD = 0.0  # frontière de décision de la tête logistique
DEFAULT_POSITIONS = (
    [],
    ["upstream"],
    ["parallel"],
    ["downstream"],
    ["parallel", "downstream"],
    ["upstream", "parallel", "downstream"],
)


def positions_from(value: Any) -> list[str]:
    """Emplacement : liste, « parallel,downstream », « none » ou vide (pas du tout)."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [] if value.strip() in ("", "none", "aucun") else value.split(",")
    out = [str(v).strip() for v in value if str(v).strip()]
    unknown = sorted(set(out) - set(SEQ_POSITIONS))
    if unknown:
        raise ValueError(f"emplacements inconnus : {unknown} (connus : {SEQ_POSITIONS})")
    return [p for p in SEQ_POSITIONS if p in out]  # ordre canonique


def position_tag(position: list[str]) -> str:
    return "+".join(position) if position else "aucun"


def store_rows(store: EmbeddingStore, recording_ids: set[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Toutes les fenêtres encodées de ces enregistrements (persistance = tout l'enregistrement)."""
    metas, embs = [], []
    for path in store.fragments():
        meta, emb = store.read(path)
        keep = meta["recording_id"].isin(recording_ids).to_numpy()
        if keep.any():
            metas.append(meta[keep])
            embs.append(emb[keep])
    if not metas:
        return pd.DataFrame(columns=["window_id", "recording_id", "offset_s"]), np.zeros((0, 0))
    return pd.concat(metas, ignore_index=True), np.concatenate(embs).astype(np.float32)


def upstream_pass(counts: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Fenêtres qui passent les portes de rythme du module en amont (`sequential.upstream`,
    `onset_gates`) ; valeur manquante : passe."""
    gates, combine = onset_gates(cfg)
    return gate_mask(counts, gates, combine)


def sequential_columns(cfg: dict) -> list[str]:
    """Descripteurs du module : `sequential.columns` (ou l'ancien `fusion.columns`)."""
    seq = cfg.get("sequential", {}) or {}
    return list(seq.get("columns") or cfg.get("fusion", {}).get("columns") or [])


def sequential_features(
    windows: pd.DataFrame, persistence: pd.DataFrame, onsets: dict, cfg: dict
) -> pd.DataFrame:
    """Rythme dans chaque fenêtre + persistance de son enregistrement."""
    rhythm = window_rhythm(windows, onsets, tuple(cfg["signal"]["ioi_range_s"]))
    persist = persistence.reindex(windows["recording_id"]).set_index(windows.index)
    return pd.concat([rhythm, persist], axis=1)


def seq_columns(position: list[str], cfg: dict) -> list[str]:
    """Descripteurs de `fusion.columns` que l'emplacement fait entrer dans la fusion."""
    columns = sequential_columns(cfg)
    out = []
    if "parallel" in position:
        out += [c for c in columns if c in RHYTHM_COLUMNS]
    if "downstream" in position:
        out += [c for c in columns if c in PERSISTENCE_COLUMNS]
    return out


def _source_kind(source: str) -> tuple[str, str]:
    """« head:<id> » ou « congeners:<id> » ; un identifiant nu est une tête."""
    kind, _, eid = source.partition(":")
    if not eid:
        return "head", kind
    if kind not in ("head", "congeners"):
        raise ValueError(f"source inconnue : {source!r} (head:<encodeur> ou congeners:<encodeur>)")
    return kind, eid


# --- Entrées de niveau 1 -----------------------------------------------------------------------


@dataclass
class Level1:
    encoder_id: str
    data: pd.DataFrame  # fenêtres évaluées (avec dur_s)
    y: np.ndarray
    groups: np.ndarray
    recordings: np.ndarray
    assignment: dict[str, int]
    head: OOFScores
    features: pd.DataFrame  # rythme + persistance
    counts: pd.DataFrame  # notes, rhythm (porte amont)
    sources: pd.DataFrame  # une colonne par source supplémentaire
    n_with_onsets: int


def build_level1(
    con: sqlite3.Connection,
    cfg: dict,
    encoder_id: str,
    sources: list[str] | tuple[str, ...] = (),
    require_onsets: bool = False,
) -> Level1:
    """Entrées de niveau 1 de l'encodeur principal (voir le module)."""
    head_cfg, bench = cfg["head"], cfg["benchmark"]
    enc = encoder_params(con, encoder_id)
    data, X = embedded_training_set(
        con,
        EmbeddingStore(config_path(cfg, "embeddings"), encoder_id),
        enc["window_s"],
        per_positive=bench["negatives_per_positive"],
        **pairing_options(cfg),
        seed=head_cfg["seed"],
        exclude_recordings=frozen_recordings(cfg),
    )
    data = data.assign(dur_s=enc["window_s"])
    y, groups = data["y"].to_numpy(), data["point"].to_numpy()
    recordings = data["recording_id"].to_numpy()
    onsets = load_onsets(con, set(recordings))
    if require_onsets and not onsets:
        raise ValueError(
            "aucun début de note calculé pour ces enregistrements : ils se calculent pendant "
            "`blanci embed`, ou avec `blanci onsets`"
        )
    meta_all, emb_all = store_rows(
        EmbeddingStore(config_path(cfg, "embeddings"), encoder_id), set(recordings)
    )
    assignment = folds_for(con, cfg)
    folds = grouped_folds(y, groups, head_cfg["n_splits"], head_cfg["seed"], assignment)
    gated = data["gated"].to_numpy()
    head_oof = np.full(len(y), np.nan)
    persistence_parts = []
    for train, test in folds:
        train = train[~gated[train]]  # fenêtres arrêtées : jamais apprises
        C = _choose_C(
            X[train],
            y[train],
            groups[train],
            head_cfg["C_grid"],
            head_cfg["n_splits"],
            head_cfg["seed"],
        )
        fold_head = fit_logistic(X[train], y[train], C, head_cfg["seed"])
        head_oof[test] = apply_gate(fold_head.decision(X[test]), ~gated[test])
        mask = meta_all["recording_id"].isin(set(recordings[test])).to_numpy()
        scored = meta_all[mask].assign(
            score=apply_gate(fold_head.decision(emb_all[mask]), ~gated_mask(meta_all[mask]))
        )
        persistence_parts.append(recording_persistence(scored, PERSISTENCE_THRESHOLD))
    persistence = pd.concat(persistence_parts) if persistence_parts else pd.DataFrame()
    features = sequential_features(data, persistence, onsets, cfg)
    counts = onset_counts(data, onsets, tuple(cfg["signal"]["ioi_range_s"]))
    extra = pd.DataFrame(index=data.index)
    for source in sources:
        kind, eid = _source_kind(source)
        if kind == "head":
            extra[f"head:{eid}"] = _secondary_head_oof(con, cfg, eid, data, assignment)
        else:
            extra[f"congeners:{eid}"] = congener_scores(con, eid, data)
    return Level1(
        encoder_id,
        data,
        y,
        groups,
        recordings,
        assignment,
        OOFScores(head_oof, tuple(folds), "logistic"),
        features,
        counts,
        extra,
        int(pd.Series(recordings).isin(list(onsets)).sum()),
    )


def _secondary_head_oof(
    con: sqlite3.Connection,
    cfg: dict,
    encoder_id: str,
    target: pd.DataFrame,
    assignment: dict[str, int],
) -> np.ndarray:
    """Score hors-pli d'un autre encodeur sur les fenêtres cibles : pour chaque pli commun, une
    tête apprise sur ses fenêtres des autres micros score toutes ses fenêtres des
    enregistrements testés, ramenées sur la grille cible."""
    head_cfg, bench = cfg["head"], cfg["benchmark"]
    params = encoder_params(con, encoder_id)
    store = EmbeddingStore(config_path(cfg, "embeddings"), encoder_id)
    data, X = embedded_training_set(
        con,
        store,
        params["window_s"],
        per_positive=bench["negatives_per_positive"],
        **pairing_options(cfg),
        seed=head_cfg["seed"],
        exclude_recordings=frozen_recordings(cfg),
    )
    y, points = data["y"].to_numpy(), data["point"].astype(str).to_numpy()
    fold_of = np.array([assignment.get(p, -1) for p in points])
    target_fold = target["point"].astype(str).map(assignment).fillna(-1).to_numpy()
    meta_all, emb_all = store_rows(store, set(target["recording_id"]))
    out = np.full(len(target), np.nan)
    open_rows = ~data["gated"].to_numpy()
    for fold in sorted(set(target_fold.tolist()) - {-1}):
        train = np.flatnonzero((fold_of != fold) & open_rows)
        rows = np.flatnonzero(target_fold == fold)
        if len(np.unique(y[train])) < 2 or not len(rows):
            continue
        C = _choose_C(
            X[train],
            y[train],
            points[train],
            head_cfg["C_grid"],
            head_cfg["n_splits"],
            head_cfg["seed"],
        )
        head = fit_logistic(X[train], y[train], C, head_cfg["seed"])
        tested = set(target["recording_id"].iloc[rows])
        mask = meta_all["recording_id"].isin(tested).to_numpy()
        if not mask.any():
            continue
        scored = meta_all[mask].assign(
            score=apply_gate(head.decision(emb_all[mask]), ~gated_mask(meta_all[mask])),
            dur_s=params["window_s"],
        )
        out[rows] = project_scores(target.iloc[rows], scored)
    return out


def congener_scores(con: sqlite3.Connection, encoder_id: str, target: pd.DataFrame) -> np.ndarray:
    """Plus grand logit de congénère (perch_v2, rangé par `embed`), ramené sur les fenêtres
    cibles. Aucun apprentissage : hors-pli par nature."""
    rows = pd.read_sql_query(
        "SELECT w.recording_id, w.offset_s, w.dur_s, MAX(s.score) AS score FROM scores s "
        "JOIN windows w USING (window_id) WHERE s.model_id LIKE ? GROUP BY s.window_id",
        con,
        params=(f"{encoder_id}:logit:%",),
    )
    rows = rows[rows["recording_id"].isin(set(target["recording_id"]))]
    if rows.empty:
        return np.full(len(target), np.nan)
    return project_scores(target, rows)


# --- Matrice de fusion et scores hors-pli ---------------------------------------------------------


def design_matrix(
    level1: Level1, position: list[str], cfg: dict, sources: list[str] | tuple[str, ...] = ()
) -> tuple[np.ndarray, list[str]]:
    """(entrées, noms) : la tête, les descripteurs de l'emplacement, les autres sources.

    Une fenêtre arrêtée par le seuillage en amont n'a pas de score de tête : NaN (la fusion
    la remplace par la moyenne), puis la porte est réappliquée à la sortie."""
    head = level1.head.values.astype(float).copy()
    head[head <= GATED_SCORE] = np.nan
    columns = ["head", *seq_columns(position, cfg)]
    parts = [head] + [level1.features[c].to_numpy(dtype=float) for c in columns[1:]]
    for source in sources:
        kind, eid = _source_kind(source)
        name = f"{kind}:{eid}"
        columns.append(name)
        parts.append(level1.sources[name].to_numpy(dtype=float))
    return np.column_stack(parts), columns


def fusion_options(cfg: dict, columns: list[str]) -> dict[str, Any]:
    """Réglages de `fit_fusion_model` tirés de la config. Poids fixes (`fusion.weights`) : les
    entrées sans poids se partagent à parts égales ce qui reste jusqu'à 1."""
    fcfg = cfg.get("fusion", {})
    given = {k: float(v) for k, v in (fcfg.get("weights") or {}).items() if k in columns}
    missing = [c for c in columns if c not in given]
    left = max(0.0, 1.0 - sum(given.values()))
    weights = given | {c: left / len(missing) for c in missing} if missing else given
    return {
        "C": float(fcfg.get("C", 1.0)),
        "weights": weights,
        "grid_step": float(fcfg.get("grid_step", 0.1)),
    }


def fused_oof(
    level1: Level1,
    method: str,
    position: list[str],
    cfg: dict,
    sources: list[str] | tuple[str, ...] = (),
) -> tuple[np.ndarray, list[str]]:
    """Score hors-pli de la chaîne (emplacement × méthode) ; tête seule s'il n'y a rien à
    fusionner. Portes réappliquées : amont (rythme) et seuillage en amont."""
    X, columns = design_matrix(level1, position, cfg, sources)
    if len(columns) == 1:
        out = level1.head.values.astype(float).copy()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fusion_model_oof(
                method,
                X,
                level1.y,
                level1.groups,
                columns,
                cfg["head"]["n_splits"],
                cfg["head"]["seed"],
                level1.assignment,
                **fusion_options(cfg, columns),
            ).values
    if "upstream" in position:
        out = apply_gate(out, upstream_pass(level1.counts, cfg))
    return apply_gate(out, ~level1.data["gated"].to_numpy()), columns


def fusion_benchmark(
    con: sqlite3.Connection,
    cfg: dict,
    encoder_id: str,
    methods: list[str] | None = None,
    positions: list[list[str]] | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Chaque emplacement du module séquentiel × chaque méthode de fusion, contre la tête
    seule : AP (fenêtres, enregistrements), écart apparié, part de chaque entrée."""
    fcfg = cfg["fusion"]
    methods = methods or list(fcfg.get("benchmark_methods") or FUSION_METHODS)
    if positions is None:
        positions = [positions_from(p) for p in fcfg.get("benchmark_positions", DEFAULT_POSITIONS)]
    sources = list(fcfg.get("sources") or []) if sources is None else sources
    needs_onsets = any(set(p) & {"upstream", "parallel"} for p in positions)
    level1 = build_level1(con, cfg, encoder_id, sources, require_onsets=needs_onsets)
    bench, seed = cfg["benchmark"], cfg["head"]["seed"]
    fingerprint = labels_fingerprint(con, cfg)
    reference = level1.head.values
    rows, comparisons, weights = [], [], []
    for position in positions:
        _, columns = design_matrix(level1, position, cfg, sources)
        for method in methods if len(columns) > 1 else ["none"]:
            if method == "weighted" and not fcfg.get("weights"):
                continue  # pas de poids fixés à la main : rien à évaluer
            values, _ = fused_oof(level1, method, position, cfg, sources)
            name = f"{position_tag(position)}/{method}"
            save_oof(
                cfg,
                oof_frame(
                    f"{encoder_id}/fusion:{name}",
                    "fusion",
                    level1.data,
                    values,
                    level1.assignment,
                    fingerprint,
                ),
            )
            for level in ("window", "recording"):
                metrics = evaluate(
                    values,
                    level1.y,
                    level1.recordings,
                    level=level,
                    precisions=tuple(bench["precisions"]),
                    n_boot=bench["n_boot"],
                    seed=seed,
                )
                rows.append(
                    {
                        "position": position_tag(position),
                        "method": method,
                        "inputs": len(columns),
                        **metrics,
                    }
                )
            rec = to_recordings(values, level1.y, level1.recordings)
            ref = to_recordings(reference, level1.y, level1.recordings).set_index("recording_id")
            comparisons.append(
                {
                    "position": position_tag(position),
                    "method": method,
                    **paired_bootstrap(
                        rec["y"].to_numpy(),
                        rec["score"].to_numpy(),
                        ref.loc[rec["recording_id"], "score"].to_numpy(),
                        rec["recording_id"].to_numpy(),
                        n_boot=bench["n_boot"],
                        seed=seed,
                    ),
                }
            )
            if len(columns) > 1:
                weights.append(
                    {
                        "position": position_tag(position),
                        "method": method,
                        **final_model(level1, method, position, cfg, sources).weights(),
                    }
                )
    table = pd.DataFrame(rows).sort_values(["level", "ap"], ascending=[True, False], kind="stable")
    return {
        "table": table.reset_index(drop=True),
        "comparisons": pd.DataFrame(comparisons),
        "weights": pd.DataFrame(weights),
        "n_windows": len(level1.y),
        "n_with_onsets": level1.n_with_onsets,
    }


def final_model(
    level1: Level1,
    method: str,
    position: list[str],
    cfg: dict,
    sources: list[str] | tuple[str, ...] = (),
) -> FusionModel:
    """Fusion apprise sur tout le jeu de développement (production, parts des entrées)."""
    X, columns = design_matrix(level1, position, cfg, sources)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_fusion_model(
            method, X, level1.y, columns, seed=cfg["head"]["seed"], **fusion_options(cfg, columns)
        )
