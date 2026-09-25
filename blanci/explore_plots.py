"""Graphiques et lecteurs audio des notebooks d'exploration (matplotlib, groupe `notebook`).

Les calculs sont dans `blanci/explore.py` ; ici, seulement l'affichage.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Audio

from blanci.audio import resample
from blanci.explore import WINDOW_COLUMNS, mean_spectrum_db, spectrogram_db
from blanci.sequential import GATES, band_envelope_db, detect_onsets, gate_threshold

LISTEN_SR = 24_000  # A. blanci chante sous 6 kHz : 24 kHz suffisent et allègent le notebook
FMAX_HZ = 12_000
COLORS = {
    "positive": "#2ca02c",
    "negative": "#d62728",
    "suspect_fn": "#ff7f0e",
    "paired": "#1f77b4",
    "chosen": "#ffffff",
}


def listen(wav: np.ndarray, sr: int, normalize: bool = True) -> Audio:
    """Lecteur audio, rééchantillonné à 24 kHz au plus. `normalize=False` garde le niveau
    réel (pour comparer deux fenêtres à l'oreille)."""
    wav = np.asarray(wav, dtype=np.float32)
    if sr > LISTEN_SR:
        wav, sr = resample(wav, sr, LISTEN_SR), LISTEN_SR
    if not normalize:
        wav = np.clip(wav, -1.0, 1.0)
    return Audio(wav, rate=sr, normalize=normalize)


def show_spectrogram(
    ax,
    wav: np.ndarray,
    sr: int,
    band_hz: tuple[float, float] | None = None,
    title: str | None = None,
    fmax: float = FMAX_HZ,
    offset_s: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
    nperseg: int = 512,
):
    """Spectrogramme (kHz, s) sur `ax`, bande d'A. blanci en pointillés."""
    freqs, times, power = spectrogram_db(wav, sr, nperseg)
    keep = freqs <= fmax
    if vmax is None:
        vmax = float(np.percentile(power[keep], 99.5))
    if vmin is None:
        vmin = vmax - 60
    image = ax.pcolormesh(
        times + offset_s,
        freqs[keep] / 1000,
        power[keep],
        shading="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    if band_hz is not None:
        for f in band_hz:
            ax.axhline(f / 1000, color="white", lw=0.7, ls="--", alpha=0.7)
    ax.set_ylabel("kHz")
    if title:
        ax.set_title(title, fontsize=10)
    return image


def _spans(ax, windows: pd.DataFrame, mask, color: str, label: str, **style) -> None:
    first = True
    for row in windows[mask].itertuples():
        ax.axvspan(
            row.offset_s,
            row.offset_s + row.dur_s,
            color=color,
            label=label if first else None,
            **({"alpha": 0.25, "lw": 0} | style),
        )
        first = False


def plot_recording(
    wav: np.ndarray,
    sr: int,
    windows: pd.DataFrame,
    cfg: dict,
    onsets: np.ndarray | None = None,
    negatives: pd.DataFrame | None = None,
    chosen: float | None = None,
    title: str | None = None,
):
    """Enregistrement entier : spectrogramme (débuts de notes en traits cyan, fenêtre choisie
    en cadre blanc) ; en dessous, une piste par sorte de fenêtre (annotées positives et
    négatives, faux négatifs suspects, négatifs appariés tirés dans l'enregistrement, fenêtre
    choisie), puis les scores des détecteurs importés (BlanciNet)."""
    band = tuple(cfg["signal"]["band_hz"])
    dur = float(windows["dur_s"].iloc[0])
    fig, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(15, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.1, 1]}
    )
    show_spectrogram(top, wav, sr, band, title, nperseg=1024)
    if onsets is not None and len(onsets):
        top.plot(onsets, np.full(len(onsets), band[1] / 1000 + 0.6), "|", color="cyan", ms=8)
    if chosen is not None:
        top.add_patch(
            plt.Rectangle(
                (chosen, 0.05), dur, FMAX_HZ / 1000 - 0.1, fill=False, ec=COLORS["chosen"], lw=1.5
            )
        )
    y = windows["y"] if "y" in windows else pd.Series(np.nan, index=windows.index)
    mine = windows.iloc[:0]
    if negatives is not None and len(negatives):
        mine = negatives[negatives["recording_id"] == windows["recording_id"].iloc[0]]
    tracks = {
        "annotée +": (windows[y == 1], COLORS["positive"]),
        "annotée −": (windows[y == 0], COLORS["negative"]),
        "faux nég. suspect": (windows[windows["suspect_fn"]], COLORS["suspect_fn"]),
        "négatif apparié": (mine, COLORS["paired"]),
        "fenêtre choisie": (windows[windows["offset_s"] == chosen], "black"),
    }
    for row, (frame, color) in enumerate(tracks.values()):
        spans = list(zip(frame["offset_s"], [dur] * len(frame), strict=True))
        middle.broken_barh(spans, (row - 0.35, 0.7), facecolors=color, alpha=0.6)
    middle.set_yticks(range(len(tracks)), list(tracks), fontsize=8)
    middle.set_ylim(len(tracks) - 0.5, -0.5)
    middle.grid(axis="x", alpha=0.3)
    detectors = [c for c in windows.columns if c not in WINDOW_COLUMNS]
    for name in detectors:
        bottom.step(windows["center_s"], windows[name].fillna(0), where="mid", label=name)
    bottom.set_ylim(0, 1.05)
    bottom.set_ylabel("score")
    bottom.set_xlabel("s")
    if detectors:
        bottom.legend(loc="upper right", fontsize=8)
    else:
        bottom.text(0.5, 0.5, "aucun détecteur importé", ha="center", transform=bottom.transAxes)
    fig.tight_layout()
    return fig


def plot_indices(windows: pd.DataFrame, values: pd.DataFrame, cfg: dict, upstream=None):
    """Valeur de chaque porte fenêtre par fenêtre, avec son seuil (trait plein si la porte est
    activée, pointillé sinon) ; annotations positives en vert."""
    fig, axes = plt.subplots(len(GATES), 1, figsize=(15, 7), sharex=True)
    enabled = set(upstream.gates) if upstream is not None else set()
    for ax, gate in zip(axes, GATES, strict=True):
        ax.plot(windows["center_s"], values[gate], ".-", lw=0.8)
        ax.axhline(gate_threshold(cfg, gate), color="k", ls="-" if gate in enabled else ":", lw=0.8)
        if "y" in windows:
            _spans(ax, windows, windows["y"] == 1, COLORS["positive"], None, alpha=0.2)
        ax.set_ylabel(gate, fontsize=8)
    axes[-1].set_xlabel("centre de la fenêtre (s)")
    fig.tight_layout()
    return fig


def plot_window(wav: np.ndarray, sr: int, cfg: dict, title: str | None = None):
    """Une fenêtre : spectrogramme, puis enveloppe en bande avec le seuil de détection des
    notes (médiane + k × MAD) et les débuts de notes retenus (durée de note compatible)."""
    signal = cfg["signal"]
    band = tuple(signal["band_hz"])
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    show_spectrogram(top, wav, sr, band, title)
    envelope = band_envelope_db(np.asarray(wav, dtype=np.float64), sr, band)
    median = np.median(envelope)
    mad = np.median(np.abs(envelope - median)) + 1e-6
    times = np.arange(len(envelope)) / sr
    bottom.plot(times, envelope, lw=0.6)
    bottom.axhline(
        median + signal["onset_k_mad"] * mad,
        color="k",
        ls="--",
        lw=0.8,
        label="seuil (médiane + k·MAD)",
    )
    onsets = detect_onsets(wav, sr, band, tuple(signal["note_dur_s"]), signal["onset_k_mad"])
    for t in onsets:
        bottom.axvline(t, color="cyan", lw=1)
    bottom.set_ylabel("dB en bande")
    bottom.set_xlabel(f"s — {len(onsets)} note(s) détectée(s) dans la fenêtre seule")
    bottom.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_stages(stages: list[tuple[str, np.ndarray]], sr: int, cfg: dict):
    """Spectrogramme à chaque étape du module en amont, même échelle de couleurs."""
    band = tuple(cfg["signal"]["band_hz"])
    fig, axes = plt.subplots(
        1, len(stages), figsize=(5 * len(stages), 3.5), sharey=True, squeeze=False
    )
    _, _, reference = spectrogram_db(stages[0][1], sr)
    vmax = float(np.percentile(reference, 99.5))
    for ax, (name, wav) in zip(axes[0], stages, strict=True):
        show_spectrogram(ax, wav, sr, band, name, vmin=vmax - 60, vmax=vmax)
        ax.set_xlabel("s")
    fig.tight_layout()
    return fig


def plot_pair(pos: np.ndarray, neg: np.ndarray, sr: int, cfg: dict, titles=("fenêtre", "négatif")):
    """Fenêtre et négatif apparié côte à côte (même échelle), leur différence en dB (ce que
    le prototype différentiel garde : rouge = plus fort dans la fenêtre), et leurs spectres
    moyens."""
    band = tuple(cfg["signal"]["band_hz"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    _, _, p = spectrogram_db(pos, sr)
    _, _, n = spectrogram_db(neg, sr)
    vmax = float(np.percentile(np.concatenate([p.ravel(), n.ravel()]), 99.5))
    show_spectrogram(axes[0, 0], pos, sr, band, titles[0], vmin=vmax - 60, vmax=vmax)
    show_spectrogram(axes[0, 1], neg, sr, band, titles[1], vmin=vmax - 60, vmax=vmax)
    freqs, times, _ = spectrogram_db(pos, sr)
    keep = freqs <= FMAX_HZ
    width = min(p.shape[1], n.shape[1])
    difference = (p[:, :width] - n[:, :width])[keep]
    image = axes[1, 0].pcolormesh(
        times[:width],
        freqs[keep] / 1000,
        difference,
        shading="auto",
        cmap="RdBu_r",
        vmin=-30,
        vmax=30,
    )
    fig.colorbar(image, ax=axes[1, 0], label="dB")
    for f in band:
        axes[1, 0].axhline(f / 1000, color="k", lw=0.7, ls="--")
    axes[1, 0].set_title(f"{titles[0]} − {titles[1]} (dB)", fontsize=10)
    axes[1, 0].set_ylabel("kHz")
    axes[1, 0].set_xlabel("s")
    for wav, label in ((pos, titles[0]), (neg, titles[1])):
        f, level = mean_spectrum_db(wav, sr)
        axes[1, 1].plot(f / 1000, level, label=label, lw=0.9)
    axes[1, 1].axvspan(band[0] / 1000, band[1] / 1000, color=COLORS["positive"], alpha=0.15)
    axes[1, 1].set_xlim(0, FMAX_HZ / 1000)
    axes[1, 1].set_xlabel("kHz")
    axes[1, 1].set_ylabel("dB")
    axes[1, 1].set_title("spectres moyens", fontsize=10)
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_embeddings(vectors: dict[str, np.ndarray]):
    """Embeddings normalisés superposés, différence des deux premiers (le prototype
    différentiel d'une paire) et matrice des cosinus."""
    names = list(vectors)
    E = np.stack([np.asarray(v, dtype=np.float32) for v in vectors.values()])
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    fig = plt.figure(figsize=(15, 6))
    grid = fig.add_gridspec(2, 3)
    top = fig.add_subplot(grid[0, :2])
    for name, e in zip(names, E, strict=True):
        top.plot(e, lw=0.6, label=name)
    top.set_title("embeddings normalisés", fontsize=10)
    top.legend(fontsize=8)
    bottom = fig.add_subplot(grid[1, :2], sharex=top)
    if len(E) >= 2:
        bottom.plot(E[0] - E[1], lw=0.6, color="k")
        bottom.set_title(f"{names[0]} − {names[1]}", fontsize=10)
    bottom.set_xlabel("dimension")
    matrix = fig.add_subplot(grid[:, 2])
    cosines = E @ E.T
    image = matrix.imshow(cosines, cmap="viridis", vmin=min(0.0, cosines.min()), vmax=1)
    matrix.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    matrix.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            matrix.text(
                j,
                i,
                f"{cosines[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="w" if cosines[i, j] < 0.7 else "k",
            )
    fig.colorbar(image, ax=matrix, fraction=0.046)
    matrix.set_title("cosinus", fontsize=10)
    fig.tight_layout()
    return fig


def plot_scores_along(
    centers: np.ndarray, scores: pd.DataFrame, windows: pd.DataFrame | None = None, title=None
):
    """Scores (une colonne par méthode) le long de l'enregistrement, annotations en vert."""
    fig, ax = plt.subplots(figsize=(15, 3))
    for name in scores.columns:
        ax.plot(centers, scores[name], ".-", lw=0.8, label=name)
    if windows is not None and "y" in windows:
        _spans(ax, windows, windows["y"] == 1, COLORS["positive"], "annotée positive", alpha=0.2)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("centre de la fenêtre (s)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def spectrogram_grid(segments: list[tuple[str, np.ndarray]], sr: int, cfg: dict, columns=5):
    """Petits spectrogrammes en grille (écoute d'un lot de fenêtres), même échelle."""
    band = tuple(cfg["signal"]["band_hz"])
    rows = int(np.ceil(len(segments) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 2.4 * rows), squeeze=False, sharey=True
    )
    for ax in axes.ravel()[len(segments) :]:
        ax.axis("off")
    for ax, (name, wav) in zip(axes.ravel(), segments, strict=False):
        show_spectrogram(ax, wav, sr, band, name, fmax=8000)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig
