"""Figures et extraits audio de la présentation de suivi n° 2.

Tout part de trois clips de l'ancien échantillon versionné (`echantillon/`, 66 clips de
l'ONF, retiré du dépôt le 01/10/2026 : les enregistrements ne sont pas publics). Pour relancer
le script, restaurer le dossier depuis l'historique git :

    git checkout ad43369 -- echantillon
    uv run --group notebook python documentation/prez/presentation-suivi-2/generer_figures.py

Les figures et les extraits audio déjà produits restent dans `figures/` et `audio/`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import Rectangle
from scipy.signal import resample_poly, stft

from blanci.heads.signal_processing import band_envelope_db, detect_onsets, rhythm_features

ROOT = Path(__file__).resolve().parents[3]
ECH = ROOT / "echantillon"
OUT = Path(__file__).resolve().parent / "figures"
AUDIO = Path(__file__).resolve().parent / "audio"

BAND = (4400, 5500)
INK = "#1C2B24"
MUTED = "#5E6E66"
ORANGE = "#F2A541"
CYAN = "#39D0E0"

# Clips retenus (voir la note orateur de chaque diapo)
CLIPS = {
    "blanci_net": "blanci/blanci_07_2LA04530_20260109_173000_6s.flac",
    "blanci_faible": "blanci/blanci_02_2LA03559_20260106_133000_66s.flac",
    "faux_ami": "amphibian/amphibian_04_2LA05191_20260110_150000_81s.flac",
}


def load(rel: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(ECH / rel)
    if wav.ndim > 1:
        wav = wav[:, 0]  # canal 0 (gain 6 dB), comme la chaîne (DECISIONS n° 50)
    return wav.astype(np.float64), sr


def spectro_db(wav: np.ndarray, sr: int, nperseg: int = 1024):
    f, t, z = stft(wav, fs=sr, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    s = 20 * np.log10(np.abs(z) + 1e-9)
    return f, t, s


def style(ax):
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_sequentiel(key: str, titre: str):
    wav, sr = load(CLIPS[key])
    f, t, s = spectro_db(wav, sr)
    on = detect_onsets(wav, sr, band=BAND)
    r = rhythm_features(on, len(wav) / sr)
    env = band_envelope_db(wav, sr, BAND)
    te = np.arange(len(env)) / sr
    med = np.median(env)
    thr = med + 4.0 * (np.median(np.abs(env - med)) + 1e-6)

    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(5.0, 3.3), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]}
    )
    keep = f <= 12000
    vmax = np.percentile(s[keep], 99.5)
    a1.pcolormesh(
        t, f[keep] / 1000, s[keep], shading="auto", cmap="magma", vmin=vmax - 60, vmax=vmax
    )
    for b in BAND:
        a1.axhline(b / 1000, color="white", ls=":", lw=1)
    for x in on:
        a1.axvline(x, color=CYAN, lw=1.4, alpha=0.9)
    a1.set_ylabel("kHz", color=MUTED)
    # Titre et descripteurs : dans la légende de la diapo (lisibles à la taille projetée).
    style(a1)

    a2.plot(te, env, color=INK, lw=0.6)
    a2.axhline(thr, color=ORANGE, lw=1.2)
    for x in on:
        a2.axvline(x, color=CYAN, lw=1.4)
    lo = np.percentile(env, 1)
    a2.set_ylim(lo, max(env.max(), thr) + 3)
    a2.set_ylabel("dB en bande", color=MUTED)
    a2.set_xlabel("temps (s)", color=MUTED)
    style(a2)
    fig.tight_layout()
    fig.savefig(OUT / f"seq_{key}.png", dpi=200)
    plt.close(fig)
    return r, len(on)


def fig_specaugment():
    wav, sr = load(CLIPS["blanci_net"])
    f, t, s = spectro_db(wav, sr)
    keep = f <= 12000
    s = s[keep]
    fk = f[keep] / 1000
    vmax = np.percentile(s, 99.5)
    rng = np.random.default_rng(3)
    masked = s.copy()
    masks = []
    for _ in range(2):  # masques de fréquence
        w = rng.integers(8, 20)
        f0 = rng.integers(0, s.shape[0] - w)
        masked[f0 : f0 + w] = vmax - 60
        masks.append(("f", f0, w))
    for _ in range(2):  # masques de temps
        w = rng.integers(20, 45)
        t0 = rng.integers(0, s.shape[1] - w)
        masked[:, t0 : t0 + w] = vmax - 60
        masks.append(("t", t0, w))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, m, ttl in zip(
        axes, [s, masked], ["original", "après SpecAugment (2 masques f, 2 masques t)"], strict=True
    ):
        ax.pcolormesh(t, fk, m, shading="auto", cmap="magma", vmin=vmax - 60, vmax=vmax)
        ax.set_title(ttl, loc="left", color=INK, fontsize=10)
        ax.set_xlabel("temps (s)", color=MUTED)
        style(ax)
    axes[0].set_ylabel("kHz", color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "specaugment.png", dpi=200)
    plt.close(fig)


def fig_patchs_attention():
    """Spectrogramme découpé en jetons (patchs) et poids d'une attention *illustrative* :
    softmax de l'énergie en bande de chaque patch, pas la sortie d'un vrai modèle."""
    wav, sr = load(CLIPS["blanci_net"])
    wav = wav[: int(5.0 * sr)]  # une fenêtre de 5 s, comme perch_v2
    f, t, s = spectro_db(wav, sr)
    keep = f <= 12000
    s = s[keep]
    fk = f[keep] / 1000
    n_t, n_f = 16, 4  # grille des jetons spatiaux de perch_v2 dans l'ONNX : 16 × 4
    ti = np.linspace(0, s.shape[1], n_t + 1).astype(int)
    fi = np.linspace(0, s.shape[0], n_f + 1).astype(int)
    energy = np.zeros((n_f, n_t))
    for i in range(n_f):
        for j in range(n_t):
            energy[i, j] = np.percentile(s[fi[i] : fi[i + 1], ti[j] : ti[j + 1]], 99)
    # Nouveauté locale : écart de chaque patch à la médiane de sa bande de fréquences, pour
    # qu'un fond constant (insectes à 8 kHz) ne reçoive pas de poids.
    nov = energy - np.median(energy, axis=1, keepdims=True)
    z = (nov - nov.mean()) / (nov.std() + 1e-9)
    w = np.exp(2.0 * z)
    w /= w.sum()

    vmax = np.percentile(s, 99.5)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax in (a1, a2):
        ax.pcolormesh(
            t,
            fk,
            s,
            shading="auto",
            cmap="gray_r" if ax is a2 else "magma",
            vmin=vmax - 60,
            vmax=vmax,
        )
        for x in ti[1:-1]:
            ax.axvline(t[min(x, len(t) - 1)], color="white" if ax is a1 else "#9DB3A6", lw=0.6)
        for y in fi[1:-1]:
            ax.axhline(fk[min(y, len(fk) - 1)], color="white" if ax is a1 else "#9DB3A6", lw=0.6)
        style(ax)
        ax.set_xlabel("temps (s)", color=MUTED)
    for i in range(n_f):
        for j in range(n_t):
            x0, x1 = t[min(ti[j], len(t) - 1)], t[min(ti[j + 1], len(t) - 1)]
            y0, y1 = fk[min(fi[i], len(fk) - 1)], fk[min(fi[i + 1], len(fk) - 1)]
            a2.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    color=ORANGE,
                    alpha=float(min(1.0, w[i, j] / w.max())) * 0.85,
                    lw=0,
                )
            )
    a1.set_title(
        f"une fenêtre de 5 s = {n_t} × {n_f} = {n_t * n_f} jetons",
        loc="left",
        color=INK,
        fontsize=10,
    )
    a2.set_title(
        "poids d'attention (illustration) : opacité ∝ poids", loc="left", color=INK, fontsize=10
    )
    a1.set_ylabel("kHz", color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "patchs_attention.png", dpi=200)
    plt.close(fig)
    return float(w.max()), float(1 / w.size)


def extraits_audio():
    """Mono, canal 0, 24 kHz, normalisé en crête : léger dans le .pptx, bande utile intacte."""
    for key, rel in CLIPS.items():
        wav, sr = load(rel)
        y = resample_poly(wav, 1, 2)
        y = 0.9 * y / (np.abs(y).max() + 1e-9)
        sf.write(AUDIO / f"{key}.wav", y.astype(np.float32), sr // 2, subtype="PCM_16")


def fig_titre():
    """Bandeau de la diapo de titre : spectrogramme nu (sans axes) d'un chant d'A. blanci."""
    wav, sr = load(CLIPS["blanci_net"])
    f, t, s = spectro_db(wav, sr)
    keep = (f >= 1000) & (f <= 9000)
    vmax = np.percentile(s[keep], 99.5)
    fig = plt.figure(figsize=(6, 7.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh(
        t[: len(t) * 6 // 10],
        f[keep],
        s[keep][:, : len(t) * 6 // 10],
        shading="auto",
        cmap="magma",
        vmin=vmax - 55,
        vmax=vmax,
    )
    ax.set_axis_off()
    fig.savefig(OUT / "titre.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    AUDIO.mkdir(exist_ok=True)
    fig_titre()
    print(fig_patchs_attention())
    print(fig_sequentiel("blanci_net", "A. blanci, chant net (Mataroni)"))
    print(fig_sequentiel("faux_ami", "Faux ami : Adenomera andreae (Mataroni)"))
    print(fig_sequentiel("blanci_faible", "A. blanci, chant faible (Mataroni, RB04)"))
    extraits_audio()
    # SpecAugment (retiré de la présentation) : fig_specaugment()
