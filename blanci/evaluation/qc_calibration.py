"""Calibration du contrôle qualité sur les fenêtres étiquetées (§3, indices acoustiques).

Les seuils de `qc` dans la config sont provisoires. Un drapeau audio a une conséquence :
un enregistrement « micro dans sac » ou « silencieux » n'est jamais encodé
(`embed.select_recordings`), donc jamais scoré. Un seuil trop large ferait disparaître des
enregistrements où A. blanci chante, sans que personne le voie.

La calibration calcule les indices de chaque enregistrement étiqueté (et de chaque fenêtre
étiquetée), les range par groupe — cible du drapeau (`artefact_in_bag`, `rain` ou mention
de pluie) ; à protéger (A. blanci) ; autres — et propose pour chaque drapeau le seuil qui
signale le plus de cibles sans signaler aucun enregistrement à A. blanci. Elle n'écrit
rien dans la config : le seuil proposé se reporte à la main dans `config/local.yaml`.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from blanci.core.db import window_id_for
from blanci.embedding.grid import window_grid
from blanci.inputs.dataset import EXCLUDED_LABELS, interval_labels, load_spans
from blanci.inputs.labels import POSITIVE_LABELS
from blanci.inputs.qc import in_bag_runs, qc_flags, qc_indices

# (drapeau, indice, sens) : le drapeau se lève quand l'indice est sous (« below ») ou
# au-dessus (« above ») du seuil. Clé de config du seuil.
FLAGS = {
    "in_bag": ("hf_ratio", "below", "in_bag_hf_ratio"),
    "silent": ("rms_dbfs", "below", "silent_dbfs"),
}


def _group(label: pd.Series, tags: pd.Series) -> np.ndarray:
    return np.select(
        [
            label == "artefact_in_bag",
            (label == "rain") | tags.map(lambda t: "rain" in t),
            label.isin(POSITIVE_LABELS),
        ],
        ["in_bag", "rain", "blanci"],
        default="other",
    )


def labelled_windows(con: sqlite3.Connection) -> pd.DataFrame:
    """Dernier label de chaque fenêtre, avec ses conditions et le chemin de l'enregistrement ;
    plus les fenêtres couvertes par un extrait écouté (annotation par intervalles, `spans`),
    étiquetées comme `interval_labels` le fait pour l'entraînement, sur la grille w3 découpée
    dans les extraits (la table `windows` n'en a pas besoin). Un label de fenêtre prime sur celui
    de l'extrait quand ils s'accordent ; en désaccord (positif / négatif), la fenêtre est
    écartée, comme `_merge_labelled`.

    `attrs["series"]` : indices déjà rangés par le contrôle audio pour tous les enregistrements
    (voisins compris), dont la règle des suites d'`in_bag` a besoin."""
    cols = "r.path, r.dataset, r.site, r.mic_id, r.start_utc"
    df = pd.read_sql_query(
        f"""SELECT l.window_id, l.label, l.conditions, w.recording_id, w.offset_s, w.dur_s,
                  {cols}
           FROM labels l JOIN windows w USING (window_id) JOIN recordings r USING (recording_id)
           WHERE l.label_id IN (SELECT MAX(label_id) FROM labels GROUP BY window_id)""",
        con,
    )
    tags = df["conditions"].map(lambda c: set(json.loads(c).get("tags", [])) if c else set())
    df["group"] = _group(df["label"], tags)
    df = df.drop(columns="conditions")
    spans, intervals = load_spans(con)
    if not spans.empty:
        grid = _span_grid(con)
        found = interval_labels(spans, intervals, grid)
        # Même règle que `_merge_labelled` : une fenêtre que son label et un intervalle étiquettent
        # en désaccord (positive / négative) est écartée ; d'accord, le label de fenêtre prime.
        own_y = df.set_index("window_id")["label"].isin(POSITIVE_LABELS)
        own_y = own_y[~own_y.index.duplicated()]
        certain = ~found["label"].isin(EXCLUDED_LABELS)
        clash = (
            found["window_id"].isin(df.loc[~df["label"].isin(EXCLUDED_LABELS), "window_id"])
            & certain
            & (found["y"].to_numpy() != found["window_id"].map(own_y).fillna(False).to_numpy())
        )
        df = df[~df["window_id"].isin(found.loc[clash, "window_id"])]
        found = found[~found["window_id"].isin(df["window_id"]) & ~clash]
        if len(found):
            found = found[["window_id", "label"]].merge(grid, on="window_id", how="left")
            found["group"] = _group(found["label"], pd.Series([set()] * len(found)))
            df = pd.concat([df, found], ignore_index=True)
    df.attrs["series"] = _stored_indices(con)
    return df


def _span_grid(con: sqlite3.Connection, window_s: float = 3.0, hop_s: float = 1.5) -> pd.DataFrame:
    """Grille des fenêtres (w3 par défaut, comme `estimate_prevalence`) des enregistrements
    qui ont un extrait écouté. Elle ne dépend pas de la table `windows`, que `append_span`
    ne remplit pas : un enregistrement écouté mais jamais encodé y compte aussi."""
    recordings = pd.read_sql_query(
        """SELECT r.recording_id, r.path, r.dataset, r.site, r.mic_id, r.start_utc,
                  r.duration_s, MAX(s.end_s) AS last_end_s
           FROM recordings r JOIN spans s USING (recording_id) GROUP BY r.recording_id""",
        con,
    )
    rows = [
        (window_id_for(r.recording_id, offset, dur), r.recording_id, offset, dur, r.path,
         r.dataset, r.site, r.mic_id, r.start_utc)
        for r in recordings.itertuples()
        for offset, dur in window_grid(r.duration_s or r.last_end_s, window_s, hop_s)
    ]  # fmt: skip
    return pd.DataFrame(
        rows,
        columns=["window_id", "recording_id", "offset_s", "dur_s", "path", "dataset", "site",
                 "mic_id", "start_utc"],
    )  # fmt: skip


def _stored_indices(con: sqlite3.Connection) -> pd.DataFrame:
    """Indices du contrôle audio rangés avec chaque enregistrement (`recordings.qc_flags`)."""
    rows = []
    for rid, dataset, site, mic, start, qc in con.execute(
        "SELECT recording_id, dataset, site, mic_id, start_utc, qc_flags FROM recordings"
    ):
        idx = (json.loads(qc) if qc else {}).get("indices")
        if idx:
            rows.append((rid, dataset, site, mic, start, idx.get("hf_ratio"), idx.get("rms_dbfs")))
    return pd.DataFrame(
        rows,
        columns=[
            "recording_id",
            "dataset",
            "site",
            "mic_id",
            "start_utc",
            "r_hf_ratio",
            "r_rms_dbfs",
        ],
    )


def calibration_indices(
    windows: pd.DataFrame, raw_root: Path, channel: int | str = 0
) -> pd.DataFrame:
    """Indices QC de chaque fenêtre (`w_*`) et de son enregistrement entier (`r_*`).

    Chaque enregistrement est lu une fois, en entier (lecture seule).
    """
    rows = []
    for path, group in windows.groupby("path"):
        try:
            wav, sr = sf.read(Path(raw_root) / path, dtype="float32", always_2d=True)
        except Exception:  # fichier illisible : déjà signalé à l'inventaire
            continue
        if channel == "mean":
            x = wav.mean(axis=1)
        else:
            x = wav[:, min(int(channel), wav.shape[1] - 1)]
        whole = {f"r_{k}": v for k, v in qc_indices(x, sr).items()}
        for _, row in group.iterrows():
            start = round(row["offset_s"] * sr)
            cut = x[start : start + round(row["dur_s"] * sr)]
            part = {f"w_{k}": v for k, v in qc_indices(cut, sr).items()} if len(cut) else {}
            rows.append({**row.to_dict(), **whole, **part})
    out = pd.DataFrame(rows)
    out.attrs["series"] = windows.attrs.get("series")
    return out


def _suggest(targets: np.ndarray, protected: np.ndarray, direction: str) -> tuple[float, str]:
    """Seuil qui lève le drapeau sur le plus de cibles sans toucher aucun protégé."""
    if not len(targets):
        return float("nan"), "aucune cible étiquetée"
    if not len(protected):
        return float("nan"), "aucun enregistrement à protéger"
    if direction == "below":
        lo, hi = float(np.max(targets)), float(np.min(protected))
        if lo < hi:
            return (lo + hi) / 2, "séparation nette : milieu entre cibles et positifs"
        return hi, "chevauchement : juste sous le positif le plus bas"
    lo, hi = float(np.max(protected)), float(np.min(targets))
    if lo < hi:
        return (lo + hi) / 2, "séparation nette : milieu entre positifs et cibles"
    return lo, "chevauchement : juste au-dessus du positif le plus haut"


def _raised(values: pd.Series, threshold: float, direction: str) -> int:
    """Nombre de valeurs qui lèvent le drapeau à ce seuil (0 si le seuil est indéfini)."""
    if not np.isfinite(threshold):
        return 0
    return int((values < threshold).sum() if direction == "below" else (values > threshold).sum())


def bagged_recordings(indices: pd.DataFrame, thresholds: dict[str, Any]) -> set[str]:
    """Enregistrements que le drapeau `in_bag` retient vraiment aux seuils actuels : ratio sous
    le seuil (hors silencieux) **et** suite d'`in_bag_min_run` enregistrements consécutifs du
    même micro (`in_bag_runs`, comme `apply_audio_flags`, n° 194). Les voisins viennent de
    `indices.attrs["series"]` quand il existe ; sans lui (ou sans date), la règle se réduit
    aux enregistrements présents, et à un seuil par enregistrement s'ils n'ont pas de micro."""
    one = indices.groupby("recording_id").first()
    context = [c for c in ("dataset", "site", "mic_id", "start_utc") if c in one]
    if len(context) < 4:  # pas de micro ni d'heure : pas de suite à juger
        hf, rms = one["r_hf_ratio"], one["r_rms_dbfs"]
        ok = (rms >= thresholds["silent_dbfs"]) & (hf < thresholds["in_bag_hf_ratio"])
        return set(one.index[ok])
    cols = ["recording_id", *context, "r_hf_ratio", "r_rms_dbfs"]
    series = one.reset_index()[cols]
    stored = indices.attrs.get("series")
    if stored is not None and len(stored):
        series = pd.concat(
            [series, stored[~stored["recording_id"].isin(series["recording_id"])][cols]]
        )
    candidate = (series["r_rms_dbfs"] >= thresholds["silent_dbfs"]) & (
        series["r_hf_ratio"] < thresholds["in_bag_hf_ratio"]
    )
    return in_bag_runs(
        series.assign(candidate=candidate),
        int(thresholds.get("in_bag_min_run", 4)),
        float(thresholds.get("in_bag_max_gap_min", 60.0)),
    )


def _flagged_now(
    values: pd.Series, threshold: float, direction: str, flag: str, bagged: set[str]
) -> int:
    """Enregistrements signalés aux seuils actuels : `in_bag` suit la règle des suites."""
    if flag == "in_bag":
        return int(values.index.isin(bagged).sum())
    return _raised(values, threshold, direction)


def _flagged_suggested(
    targets: pd.Series,
    suggestion: float,
    direction: str,
    flag: str,
    indices: pd.DataFrame,
    thresholds: dict[str, Any],
    key: str,
) -> int:
    """Cibles signalées au seuil proposé, par la même règle que « aux seuils actuels » :
    pour `in_bag`, la règle des suites avec le seuil proposé à la place de l'actuel."""
    if flag != "in_bag":
        return _raised(targets, suggestion, direction)
    if not np.isfinite(suggestion):
        return 0
    bagged = bagged_recordings(indices, thresholds | {key: suggestion})
    return _flagged_now(targets, suggestion, direction, flag, bagged)


def suggest_thresholds(indices: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    """Par drapeau : effet du seuil actuel et seuil proposé, au niveau de l'enregistrement
    (celui où le drapeau décide de l'encodage). Un enregistrement est « à protéger » s'il
    contient au moins une fenêtre A. blanci ; « cible » s'il contient une fenêtre du drapeau."""
    recordings = indices.groupby("recording_id").agg(
        groups=("group", lambda g: set(g)), **{c: (c, "first") for c in indices if c[:2] == "r_"}
    )
    rows = []
    bagged = bagged_recordings(indices, thresholds)
    has_blanci = recordings["groups"].map(lambda g: "blanci" in g)
    for flag, (index, direction, key) in FLAGS.items():
        column = f"r_{index}"
        protected = recordings.loc[has_blanci, column]
        if flag == "in_bag":
            is_target = recordings["groups"].map(lambda g, f=flag: f in g)
            # Une cible qui contient aussi du chant reste à protéger.
            targets = recordings.loc[is_target & ~has_blanci, column]
        else:
            targets = pd.Series(dtype=float)
        current = float(thresholds[key])
        suggestion, why = _suggest(targets.to_numpy(), protected.to_numpy(), direction)
        rows.append(
            {
                "flag": flag,
                "index": index,
                "direction": "sous" if direction == "below" else "au-dessus",
                "config_key": key,
                "current": current,
                "targets": len(targets),
                "targets_flagged_now": _flagged_now(targets, current, direction, flag, bagged),
                "blanci_recordings": len(protected),
                "blanci_flagged_now": _flagged_now(protected, current, direction, flag, bagged),
                "suggested": suggestion,
                "targets_flagged_suggested": _flagged_suggested(
                    targets, suggestion, direction, flag, indices, thresholds, key
                ),
                "reason": why,
            }
        )
    return pd.DataFrame(rows)


def current_flags(indices: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    """Drapeaux que lèveraient les seuils actuels, enregistrement par enregistrement."""
    rows = []
    bagged = bagged_recordings(indices, thresholds)
    for rid, group in indices.groupby("recording_id"):
        values = {k[2:]: v for k, v in group.iloc[0].items() if k.startswith("r_")}
        flags = qc_flags(values, thresholds)
        rows.append(
            {
                "recording_id": rid,
                "groups": ",".join(sorted(set(group["group"]))),
                **{k: flags[k] for k in FLAGS},
                "in_bag": rid in bagged,
            }
        )
    return pd.DataFrame(rows)
