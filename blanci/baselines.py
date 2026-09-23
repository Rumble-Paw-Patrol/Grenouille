"""Baselines sans encodeur (§3, §6) : seuillage spectral, onsets et rythme, template matching.

Elles répondent à la question du go/no-go de P0 : si aucun encodeur n'atteint AP ≥ 0,5 et
rappel ≥ 0,8 à précision ≥ 0,1, le détecteur primaire sera « DSP + template matching ». Elles
sont jugées exactement comme les encodeurs : mêmes fenêtres annotées, mêmes négatifs appariés,
plis groupés par micro, mêmes métriques (`evaluate`).

Fenêtres évaluées : les fenêtres annotées (3 s, décalage de l'annotation) et
`negatives_per_positive` négatifs appariés présumés par enregistrement positif, tirés sur la
grille w3 des candidats de `benchmark_recordings`.

Scores (plus haut = plus probablement A. blanci) :
- `band_energy` : pic d'énergie en bande 4,4–5,5 kHz au-dessus de sa médiane dans la fenêtre,
  lissé sur la durée d'une note. Le « seuillage spectral » classique.
- `band_contrast` : même chose sur le contraste bande / bandes voisines (3,0–4,2 et 5,7–7,0 kHz) :
  un son large bande (pluie, insecte, cri d'oiseau) monte partout et ne compte pas.
- `notes` : nombre d'onsets en bande de durée compatible avec une note (`detect_onsets`).
- `rhythm` : nombre d'intervalles entre notes dans la plage d'A. blanci, départagés par `notes`.
- `template_mean`, `template_max` : corrélation normalisée maximale du spectrogramme avec le
  gabarit moyen des notes des positifs d'entraînement, ou avec le meilleur de `n_exemplars`
  notes prises une à une. Gabarits appris dans chaque pli, sans le micro testé : scores hors-pli.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter1d
from scipy.signal import spectrogram

from blanci.audio import resample
from blanci.dataset import (
    EXCLUDED_LABELS,
    benchmark_recordings,
    current_labels,
    paired_negatives,
    recordings_table,
)
from blanci.db import window_id_for
from blanci.evaluate import evaluate, grouped_folds
from blanci.frozen import frozen_recordings
from blanci.grid import window_grid
from blanci.labels import POSITIVE_LABELS
from blanci.sequential import detect_onsets

FIXED = ("band_energy", "band_contrast", "notes", "rhythm")
LEARNED = ("template_mean", "template_max")
BASELINES = FIXED + LEARNED

# Rééchantillonnage commun : les gabarits doivent avoir les mêmes cases de fréquence quel que
# soit l'enregistreur. 24 kHz couvre la bande de la note (≤ 5,5 kHz) et ses voisines.
SR = 24_000
NPERSEG = 512  # 21 ms, cases de 47 Hz
HOP = 256  # 10,7 ms
SPEC_BAND_HZ = (3000.0, 7000.0)  # spectrogramme conservé pour les gabarits
FLANKS_HZ = ((3000.0, 4200.0), (5700.0, 7000.0))


@dataclass
class WindowFeatures:
    """Descripteurs d'une fenêtre sur un canal : scores fixes + spectrogramme en bande (dB)."""

    fixed: dict[str, float]
    spec_db: np.ndarray  # (fréquences, temps), SPEC_BAND_HZ


def evaluation_windows(con: sqlite3.Connection, cfg: dict) -> pd.DataFrame:
    """Fenêtres annotées + négatifs appariés présumés : window_id, recording_id, path,
    offset_s, dur_s, label, y, presumed, point, site."""
    bench = cfg["benchmark"]
    offset_h = cfg["recorder"]["filename_utc_offset_h"]
    recordings = recordings_table(con)

    frozen = frozen_recordings(cfg)  # jeu gelé : jamais vu en développement (§6)
    labels = current_labels(con)
    labels = labels[~labels["label"].isin(EXCLUDED_LABELS)]
    labels = labels[~labels["recording_id"].isin(frozen)].copy()
    labels["y"] = labels["label"].isin(POSITIVE_LABELS).astype(int)
    labelled = labels[["window_id", "recording_id", "offset_s", "dur_s", "label", "y"]].assign(
        presumed=False
    )

    candidates = benchmark_recordings(con, bench["slot_tolerance_min"], offset_h)
    candidates = candidates[
        (candidates["role"] == "paired_candidate") & ~candidates["recording_id"].isin(frozen)
    ]
    w3 = cfg["grids"]["w3"]
    grid = pd.DataFrame(
        [
            (window_id_for(r.recording_id, o, w3["window_s"]), r.recording_id, o)
            for r in candidates.itertuples()
            for o, _ in window_grid(r.duration_s or 0.0, w3["window_s"], w3["hop_s"])
        ],
        columns=["window_id", "recording_id", "offset_s"],
    )
    positives = set(labelled.loc[labelled["y"] == 1, "recording_id"])
    negatives = paired_negatives(
        grid,
        recordings,
        positives,
        bench["negatives_per_positive"],
        bench["slot_tolerance_min"],
        offset_h,
        cfg["head"]["seed"],
    ).assign(dur_s=w3["window_s"], presumed=True)

    data = pd.concat([labelled, negatives], ignore_index=True)
    return data.merge(
        recordings[["recording_id", "path", "point", "site"]], on="recording_id", how="left"
    )


def read_windows(
    windows: pd.DataFrame, raw_root: Path, channels: tuple[int, ...]
) -> Iterator[tuple[int, dict[int, np.ndarray], int]]:
    """(indice de ligne, {canal: forme d'onde}, f_e) pour chaque fenêtre, fichier par fichier.

    Lecture partielle (seek) : seules les secondes utiles sont lues, jamais écrites. Un
    fichier mono donne son unique canal pour chaque canal demandé.
    """
    for path, group in windows.groupby("path", sort=True):
        with sf.SoundFile(Path(raw_root) / path) as f:
            for i, row in group.iterrows():
                f.seek(min(round(row["offset_s"] * f.samplerate), f.frames))
                wav = f.read(round(row["dur_s"] * f.samplerate), dtype="float32", always_2d=True)
                yield (
                    i,
                    {c: wav[:, min(c, wav.shape[1] - 1)] for c in channels},
                    f.samplerate,
                )


def _band_rows(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (freqs >= band[0]) & (freqs <= band[1])


def window_features(wav: np.ndarray, sr: int, signal_cfg: dict) -> WindowFeatures:
    """Scores fixes et spectrogramme en bande d'une fenêtre (forme d'onde mono)."""
    x = resample(wav, sr, SR).astype(np.float64)
    band = tuple(signal_cfg["band_hz"])
    if len(x) < NPERSEG:
        x = np.pad(x, (0, NPERSEG - len(x)))
    freqs, _, power = spectrogram(x, fs=SR, nperseg=NPERSEG, noverlap=NPERSEG - HOP)
    power = power + 1e-12
    in_band = power[_band_rows(freqs, band)].mean(axis=0)
    flanks = np.mean([power[_band_rows(freqs, f)].mean(axis=0) for f in FLANKS_HZ], axis=0)

    frames = max(1, round(signal_cfg["note_max_s"] * SR / HOP))  # une note ≈ 10 trames
    energy = uniform_filter1d(10 * np.log10(in_band), frames)
    contrast = uniform_filter1d(10 * np.log10(in_band / flanks), frames)

    onsets = detect_onsets(
        x,
        SR,
        band=band,
        note_dur_s=tuple(signal_cfg["note_dur_s"]),
        k_mad=signal_cfg["onset_k_mad"],
    )
    iois = np.diff(onsets)
    lo, hi = signal_cfg["ioi_range_s"]
    n_rhythm = int(((iois >= lo) & (iois <= hi)).sum())

    fixed = {
        "band_energy": float(energy.max() - np.median(energy)),
        "band_contrast": float(contrast.max() - np.median(contrast)),
        "notes": float(len(onsets)),
        "rhythm": n_rhythm + len(onsets) / 100,
    }
    spec = 10 * np.log10(power[_band_rows(freqs, SPEC_BAND_HZ)]).astype(np.float32)
    return WindowFeatures(fixed, spec)


# --- Template matching --------------------------------------------------------------------------


def _band_profile(spec_db: np.ndarray) -> np.ndarray:
    """Énergie par trame dans la bande de la note, pour caler un gabarit sur la note."""
    freqs = np.linspace(SPEC_BAND_HZ[0], SPEC_BAND_HZ[1], spec_db.shape[0])
    return spec_db[_band_rows(freqs, (4400.0, 5500.0))].mean(axis=0)


def _znorm(patches: np.ndarray) -> np.ndarray:
    """Centre-réduit chaque gabarit (dernier axe = gabarit aplati)."""
    patches = patches - patches.mean(axis=-1, keepdims=True)
    return patches / (np.linalg.norm(patches, axis=-1, keepdims=True) + 1e-9)


def note_patch(spec_db: np.ndarray, width: int) -> np.ndarray | None:
    """Gabarit de `width` trames centré sur la trame la plus forte en bande ; None si trop court."""
    if spec_db.shape[1] < width:
        return None
    profile = uniform_filter1d(_band_profile(spec_db), max(1, width // 2))
    start = int(np.clip(profile.argmax() - width // 2, 0, spec_db.shape[1] - width))
    return spec_db[:, start : start + width].ravel()


def template_scores(specs: list[np.ndarray], templates: np.ndarray, width: int) -> np.ndarray:
    """Corrélation normalisée maximale (dans le temps et entre gabarits) de chaque fenêtre."""
    templates = _znorm(np.atleast_2d(templates))
    out = np.full(len(specs), np.nan)
    for i, spec in enumerate(specs):
        if spec.shape[1] < width:
            continue
        # (positions, fréquences, largeur) → (positions, fréquences × largeur)
        view = sliding_window_view(spec, width, axis=1).transpose(1, 0, 2)
        out[i] = float((_znorm(view.reshape(len(view), -1)) @ templates.T).max())
    return out


def oof_template_scores(
    specs: list[np.ndarray],
    y: np.ndarray,
    presumed: np.ndarray,
    groups: np.ndarray,
    width: int,
    n_splits: int = 5,
    n_exemplars: int = 30,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Scores hors-pli des deux gabarits : appris sur les seuls positifs annotés du pli
    d'entraînement (jamais sur les négatifs présumés), appliqués au micro tenu à l'écart."""
    rng = np.random.default_rng(seed)
    out = {name: np.full(len(specs), np.nan) for name in LEARNED}
    for train, test in grouped_folds(y, groups, n_splits, seed):
        sources = [i for i in train if y[i] == 1 and not presumed[i]]
        patches = [p for p in (note_patch(specs[i], width) for i in sources) if p is not None]
        if not patches:
            continue
        patches = _znorm(np.stack(patches))
        subset = [specs[i] for i in test]
        out["template_mean"][test] = template_scores(subset, patches.mean(axis=0), width)
        chosen = rng.choice(len(patches), size=min(n_exemplars, len(patches)), replace=False)
        out["template_max"][test] = template_scores(subset, patches[chosen], width)
    return out


# --- Évaluation ---------------------------------------------------------------------------------


def run_baselines(
    con: sqlite3.Connection,
    cfg: dict,
    raw_root: Path,
    channels: tuple[int, ...] = (0,),
    progress_every: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tableau des métriques par baseline, canal et niveau ; scores par fenêtre)."""
    windows = evaluation_windows(con, cfg)
    if windows["y"].nunique() < 2:
        raise ValueError("il faut des positifs et des négatifs pour évaluer les baselines")
    windows = windows.reset_index(drop=True)
    features: dict[int, list[WindowFeatures | None]] = {c: [None] * len(windows) for c in channels}
    for n, (i, waves, sr) in enumerate(read_windows(windows, raw_root, channels), start=1):
        for c, wav in waves.items():
            features[c][i] = window_features(wav, sr, cfg["signal"])
        if progress_every and n % progress_every == 0:
            print(f"  {n}/{len(windows)} fenêtres lues", flush=True)

    width = max(3, round(2 * cfg["signal"]["note_max_s"] * SR / HOP))  # deux durées de note
    y = windows["y"].to_numpy()
    groups = windows["point"].to_numpy()
    recordings = windows["recording_id"].to_numpy()
    bench, head = cfg["benchmark"], cfg["head"]

    rows, scored = [], []
    for c in channels:
        feats = features[c]
        scores = {name: np.array([f.fixed[name] for f in feats]) for name in FIXED}
        scores |= oof_template_scores(
            [f.spec_db for f in feats],
            y,
            windows["presumed"].to_numpy(),
            groups,
            width,
            n_splits=head["n_splits"],
            seed=head["seed"],
        )
        for name, values in scores.items():
            scored.append(
                windows[["window_id", "recording_id", "y"]].assign(
                    baseline=name, channel=c, score=values
                )
            )
            finite = np.isfinite(values)
            for level in ("window", "recording"):
                metrics = evaluate(
                    values[finite],
                    y[finite],
                    recordings[finite],
                    level=level,
                    precisions=tuple(bench["precisions"]),
                    n_boot=bench["n_boot"],
                    seed=head["seed"],
                )
                rows.append(
                    {
                        "baseline": name,
                        "channel": c,
                        **metrics,
                        "n_mics": len(np.unique(groups[finite])),
                    }
                )
    table = pd.DataFrame(rows).sort_values(["level", "ap"], ascending=[True, False], kind="stable")
    return table.reset_index(drop=True), pd.concat(scored, ignore_index=True)


REPORT_COLUMNS = [
    "baseline",
    "channel",
    "level",
    "n_pos",
    "n_neg",
    "n_mics",
    "ap",
    "ap_lo",
    "ap_hi",
    "recall@p0.1",
    "recall@p0.5",
]


def write_baseline_report(
    table: pd.DataFrame, scores: pd.DataFrame, reports_dir: Path, stem: str = "baselines"
) -> dict[str, Path]:
    """CSV complet, scores par fenêtre et résumé markdown avec le critère du go/no-go."""
    from blanci.benchmark import to_markdown

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": reports_dir / f"{stem}.csv",
        "scores_csv": reports_dir / f"{stem}_scores.csv",
        "markdown": reports_dir / f"{stem}.md",
    }
    table.to_csv(paths["csv"], index=False)
    scores.to_csv(paths["scores_csv"], index=False)
    shown = [c for c in REPORT_COLUMNS if c in table.columns]
    text = [
        "# Baselines sans encodeur (§3)",
        "",
        "Critère du go/no-go de P0 (§9) : AP ≥ 0,5 et rappel ≥ 0,8 à précision ≥ 0,1, plis par "
        "micro. Les baselines fixes n'apprennent rien : leurs scores sont hors-pli par nature ; "
        "les gabarits sont appris dans chaque pli sans le micro testé.",
        "",
    ]
    for level in ("window", "recording"):
        part = table[table["level"] == level]
        if not part.empty:
            text += [f"## Niveau {level}", "", to_markdown(part[shown]), ""]
    paths["markdown"].write_text("\n".join(text), encoding="utf-8")
    return paths
