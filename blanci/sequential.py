"""Module séquentiel et seuillage spectral, fusionnés (§3, DECISIONS n° 103) : descripteurs
du signal calculés hors encodeur, placés là où `sequential.position` le dit.

- Rythme : onsets de notes en bande (4,4–5,5 kHz), intervalles entre notes (IOI). A. blanci :
  note de 0,090–0,103 s, IOI 1,200–1,906 s (moyenne 1,414 s).
- Persistance : fraction de fenêtres positives, plus longue série, détections isolées (faux
  positifs suspects), trous (fenêtres négatives encadrées de positives : faux négatifs suspects).
- Solo contre chœur [HYPOTHÈSE H21] : IOI courts (< 0,6 s) et densité d'onsets élevée
  signalent plusieurs chanteurs.
- Spectre : énergie et contraste en bande (comme les baselines).

Emplacements (`sequential.position`, liste, vide = pas du tout) :
- `upstream` (amont, avant l'encodeur) : transformations du son et portes (seuillage
  spectral : énergie, contraste ; rythme : notes, intervalles), réglées dans
  `sequential.upstream` ;
- `parallel` : descripteurs de rythme fusionnés avec le score de la tête ;
- `downstream` (aval) : descripteurs de persistance, qui n'existent qu'après la tête.
En parallèle et en aval, le module module la décision (fusion), il ne met jamais de veto ; en
amont, une porte arrête la fenêtre (c'est son rôle), jugée d'abord par `upstream-bench`.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, istft, sosfiltfilt, stft


def band_envelope_db(
    wav: np.ndarray, sr: int, band: tuple[int, int], smooth_s: float = 0.01
) -> np.ndarray:
    """Enveloppe en dB du signal filtré dans la bande, lissée sur `smooth_s`."""
    high = min(band[1], 0.99 * sr / 2)
    sos = butter(4, [band[0], high], btype="bandpass", fs=sr, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, wav)))
    n = max(1, round(smooth_s * sr))
    env = np.convolve(env, np.ones(n) / n, mode="same")
    return 20 * np.log10(env + 1e-10)


def detect_onsets(
    wav: np.ndarray,
    sr: int,
    band: tuple[int, int] = (4400, 5500),
    note_dur_s: tuple[float, float] = (0.07, 0.13),
    k_mad: float = 4.0,
) -> np.ndarray:
    """Instants (s) de début des événements en bande dont la durée est celle d'une note.

    Seuil adaptatif : médiane + k × MAD de l'enveloppe en dB (robuste au fond de chaque
    enregistrement). Sert au module séquentiel, jamais de filtre amont (bande saturée, §3).
    """
    env = band_envelope_db(np.asarray(wav, dtype=np.float64), sr, band)
    median = np.median(env)
    mad = np.median(np.abs(env - median)) + 1e-6
    above = np.concatenate([[False], env > median + k_mad * mad, [False]])
    edges = np.flatnonzero(np.diff(above.astype(np.int8)))
    starts, ends = edges[::2], edges[1::2]
    durations = (ends - starts) / sr
    keep = (durations >= note_dur_s[0]) & (durations <= note_dur_s[1])
    return starts[keep] / sr


def rhythm_features(
    onsets: np.ndarray, duration_s: float, ioi_range_s: tuple[float, float] = (1.2, 1.906)
) -> dict[str, float]:
    iois = np.diff(np.sort(onsets))
    out = {
        "onset_rate_hz": len(onsets) / duration_s if duration_s > 0 else 0.0,
        "ioi_median_s": float(np.median(iois)) if len(iois) else float("nan"),
        "ioi_cv": float(iois.std() / iois.mean()) if len(iois) > 1 else float("nan"),
        "frac_ioi_blanci": float(np.mean((iois >= ioi_range_s[0]) & (iois <= ioi_range_s[1])))
        if len(iois)
        else 0.0,
        "frac_ioi_short": float(np.mean(iois < 0.6)) if len(iois) else 0.0,  # chœur ? (H21)
    }
    return out


GAP_RADIUS_S = 6.0  # ≈ 4 notes d'A. blanci (intervalle moyen 1,414 s)


def surrounded_by_positives(
    centers: np.ndarray, positive_centers: np.ndarray, radius: float = GAP_RADIUS_S
) -> np.ndarray:
    """Vrai pour chaque centre qui a un positif avant lui **et** un après lui, chacun à moins
    de `radius` : une fenêtre négative ainsi encadrée est un faux négatif suspect (DECISIONS
    n° 102), le miroir du « suspect » (positif isolé, faux positif probable)."""
    centers = np.asarray(centers, dtype=float)
    positives = np.sort(np.asarray(positive_centers, dtype=float))
    if not len(positives) or not len(centers):
        return np.zeros(len(centers), dtype=bool)
    left = np.searchsorted(positives, centers, side="left") - 1  # dernier positif < centre
    right = np.searchsorted(positives, centers, side="right")  # premier positif > centre
    has_left = (left >= 0) & (centers - positives[np.clip(left, 0, None)] <= radius)
    has_right = (right < len(positives)) & (
        positives[np.clip(right, None, len(positives) - 1)] - centers <= radius
    )
    return has_left & has_right


def persistence_features(
    window_scores: np.ndarray,
    threshold: float,
    offsets_s: np.ndarray | None = None,
    gap_radius_s: float = GAP_RADIUS_S,
) -> dict[str, float]:
    """Fraction de fenêtres positives, plus longue série, nombre de séries isolées (faux
    positifs suspects), étendue, nombre de trous (fenêtres négatives encadrées de positives à
    moins de `gap_radius_s` : faux négatifs suspects ; sans décalages, rayon de 2 fenêtres)."""
    positive = np.asarray(window_scores) >= threshold
    padded = np.concatenate([[False], positive, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    runs = edges[1::2] - edges[::2]
    if offsets_s is not None and positive.any():
        span = float(np.ptp(np.asarray(offsets_s)[positive]))
    else:
        span = float(np.ptp(np.flatnonzero(positive))) if positive.any() else 0.0
    where = np.asarray(offsets_s, dtype=float) if offsets_s is not None else None
    if where is None:
        where, radius = np.arange(len(positive), dtype=float), 2.0
    else:
        radius = gap_radius_s
    gaps = surrounded_by_positives(where[~positive], where[positive], radius)
    return {
        "frac_windows": float(positive.mean()) if len(positive) else 0.0,
        "n_positive": int(positive.sum()),
        "longest_run": int(runs.max()) if len(runs) else 0,
        "n_isolated": int((runs == 1).sum()),
        "span": span,
        "n_gaps": int(gaps.sum()),
    }


def sequential_features(
    onsets: np.ndarray,
    window_scores: np.ndarray,
    threshold: float = 0.5,
    duration_s: float = 120.0,
    offsets_s: np.ndarray | None = None,
    ioi_range_s: tuple[float, float] = (1.2, 1.906),
) -> dict[str, float]:
    """Descripteurs d'un enregistrement : rythme (audio) + persistance (scores de fenêtres)."""
    return rhythm_features(onsets, duration_s, ioi_range_s) | persistence_features(
        window_scores, threshold, offsets_s
    )


def note_snr_db(
    wav: np.ndarray,
    sr: int,
    band: tuple[int, int] = (4400, 5500),
    note_dur_s: tuple[float, float] = (0.07, 0.13),
    k_mad: float = 4.0,
    neighbour_s: float = 0.5,
) -> float:
    """RSB estimé du chant (§6) : énergie en bande pendant les notes détectées, contre les
    `neighbour_s` voisines de part et d'autre (notes exclues). NaN sans note détectée.

    Sert à ventiler le rappel par RSB : un détecteur qui ne rate que les chants à < 6 dB
    n'a pas le même défaut qu'un détecteur qui rate des chants nets.
    """
    wav = np.asarray(wav, dtype=np.float64)
    onsets = detect_onsets(wav, sr, band, note_dur_s, k_mad)
    if not len(onsets):
        return float("nan")
    env = 10 ** (band_envelope_db(wav, sr, band) / 10)  # puissance d'enveloppe
    note_n = round(note_dur_s[1] * sr)
    near_n = round(neighbour_s * sr)
    in_note = np.zeros(len(env), dtype=bool)
    near = np.zeros(len(env), dtype=bool)
    for t in onsets:
        i = round(t * sr)
        in_note[i : i + note_n] = True
        near[max(0, i - near_n) : i + note_n + near_n] = True
    near &= ~in_note
    if not near.any():
        return float("nan")
    return float(10 * np.log10(env[in_note].mean() / env[near].mean()))


# --- Débuts de notes stockés et descripteurs par fenêtre ---------------------------------------


def recording_onsets(wav: np.ndarray, sr: int, signal_cfg: dict) -> np.ndarray:
    """Débuts de notes d'un enregistrement entier, selon les réglages `signal` de la config."""
    return detect_onsets(
        wav,
        sr,
        band=tuple(signal_cfg["band_hz"]),
        note_dur_s=tuple(signal_cfg["note_dur_s"]),
        k_mad=signal_cfg["onset_k_mad"],
    )


def store_onsets(
    con: sqlite3.Connection, recording_id: str, onsets: np.ndarray, channel: int | str
) -> None:
    from blanci.db import utc_now

    con.execute(
        "INSERT OR REPLACE INTO onsets (recording_id, channel, onsets_json, computed_at) "
        "VALUES (?, ?, ?, ?)",
        (recording_id, str(channel), json.dumps([round(float(t), 4) for t in onsets]), utc_now()),
    )


def load_onsets(con: sqlite3.Connection, recording_ids=None) -> dict[str, np.ndarray]:
    """{recording_id: débuts de notes (s)} pour les enregistrements déjà traités."""
    rows = con.execute("SELECT recording_id, onsets_json FROM onsets").fetchall()
    wanted = None if recording_ids is None else set(recording_ids)
    return {
        rid: np.asarray(json.loads(text), dtype=float)
        for rid, text in rows
        if wanted is None or rid in wanted
    }


RHYTHM_COLUMNS = ("onset_rate_hz", "ioi_median_s", "ioi_cv", "frac_ioi_blanci", "frac_ioi_short")
PERSISTENCE_COLUMNS = (
    "frac_windows",
    "n_positive",
    "longest_run",
    "n_isolated",
    "span",
    "n_gaps",
)


def window_rhythm(
    windows: pd.DataFrame,
    onsets: dict[str, np.ndarray],
    ioi_range_s: tuple[float, float] = (1.2, 1.906),
) -> pd.DataFrame:
    """Rythme dans chaque fenêtre (recording_id, offset_s, dur_s) : débuts de notes tombant
    dans la fenêtre. NaN si l'enregistrement n'a pas de débuts de notes calculés."""
    rows = []
    for rid, offset, dur in zip(
        windows["recording_id"], windows["offset_s"], windows["dur_s"], strict=True
    ):
        if rid not in onsets:
            rows.append(dict.fromkeys(RHYTHM_COLUMNS, np.nan))
            continue
        times = onsets[rid]
        inside = times[(times >= offset) & (times < offset + dur)] - offset
        rows.append(rhythm_features(inside, dur, ioi_range_s))
    return pd.DataFrame(rows, index=windows.index, columns=list(RHYTHM_COLUMNS))


def recording_persistence(scores: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Persistance par enregistrement à partir des scores de **toutes** ses fenêtres
    (recording_id, offset_s, score) : fraction positive, plus longue série, isolées, étendue."""
    rows = {}
    for rid, group in scores.sort_values("offset_s").groupby("recording_id"):
        rows[rid] = persistence_features(
            group["score"].to_numpy(), threshold, group["offset_s"].to_numpy()
        )
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(PERSISTENCE_COLUMNS))


def compute_onsets(
    con: sqlite3.Connection,
    recordings: pd.DataFrame,
    raw_root,
    signal_cfg: dict,
    channel: int | str = 0,
    commit_every: int = 50,
) -> dict[str, int]:
    """Débuts de notes des enregistrements qui n'en ont pas encore (lecture seule de l'audio)."""
    from pathlib import Path

    from blanci.audio import load_audio

    done = {row[0] for row in con.execute("SELECT recording_id FROM onsets")}
    report = {"computed": 0, "skipped": 0, "errors": 0, "onsets": 0}
    for rec in recordings.itertuples():
        if rec.recording_id in done:
            report["skipped"] += 1
            continue
        try:
            wav, sr = load_audio(Path(raw_root) / rec.path, channel)
        except Exception:  # fichier illisible : déjà signalé à l'inventaire
            report["errors"] += 1
            continue
        found = recording_onsets(wav, sr, signal_cfg)
        store_onsets(con, rec.recording_id, found, channel)
        report["computed"] += 1
        report["onsets"] += len(found)
        if report["computed"] % commit_every == 0:
            con.commit()
    con.commit()
    return report


# --- En amont : seuillage spectral, transformations et portes (DECISIONS n° 90, 103) --------------
#
# Le module séquentiel et le seuillage spectral ne font qu'un : les mêmes descripteurs du signal
# (énergie et contraste en bande, débuts de notes, rythme, persistance) servent là où
# `sequential.position` les place. En amont (`upstream`), avant l'encodeur :
# - transformations (le son donné à l'encodeur change → autre encodeur, stock `<nom>+<étiquette>`) :
#   `bandpass` (passe-bande à phase nulle), `denoise` (soustraction spectrale du fond médian) ;
# - portes (une fenêtre arrêtée n'est pas encodée ; score `GATED_SCORE`, jamais apprise ; un
#   positif arrêté compte comme manqué) : `band_energy`, `band_contrast` (calculées sur le son
#   de la fenêtre, comme les baselines), `notes`, `rhythm` (comptées sur les débuts de notes de
#   l'enregistrement, les mêmes qu'en parallèle : une seule détection partout).

TRANSFORMS = ("bandpass", "denoise")
GATES = ("band_energy", "band_contrast", "notes", "rhythm")
ONSET_GATES = ("notes", "rhythm")
FEATURES = TRANSFORMS + GATES
GATE_THRESHOLD_KEY = {
    "band_energy": "min_db",
    "band_contrast": "min_db",
    "notes": "min_count",
    "rhythm": "min_count",
}
GATE_DEFAULTS = {"band_energy": 6.0, "band_contrast": 3.0, "notes": 1, "rhythm": 1}
GATED_SCORE = -1.0e6  # score d'une fenêtre arrêtée : fini (les métriques refusent −inf)


def bandpass(wav: np.ndarray, sr: int, band_hz: tuple[float, float], order: int = 4) -> np.ndarray:
    """Passe-bande à phase nulle sur le dernier axe ; la borne haute est ramenée sous Nyquist."""
    low, high = float(band_hz[0]), min(float(band_hz[1]), 0.99 * sr / 2)
    if not 0 < low < high:
        raise ValueError(f"bande {band_hz} impossible à {sr} Hz")
    sos = butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, np.asarray(wav, dtype=np.float64), axis=-1).astype(np.float32)


def denoise(
    wav: np.ndarray, sr: int, strength: float = 1.0, floor: float = 0.1, frame_s: float = 0.02
) -> np.ndarray:
    """Soustraction spectrale : |X| − strength × médiane de chaque fréquence, jamais sous
    `floor` × |X|. La phase est gardée. Fenêtre par fenêtre (dernier axe = temps)."""
    x = np.asarray(wav, dtype=np.float64)
    nperseg = int(2 ** round(np.log2(max(16, frame_s * sr))))
    if x.shape[-1] < nperseg:
        return x.astype(np.float32)
    noverlap = nperseg * 3 // 4
    _, _, spec = stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap, axis=-1)
    magnitude = np.abs(spec)
    noise = np.median(magnitude, axis=-1, keepdims=True)
    cleaned = np.maximum(magnitude - strength * noise, floor * magnitude)
    _, y = istft(cleaned * np.exp(1j * np.angle(spec)), fs=sr, nperseg=nperseg, noverlap=noverlap)
    out = np.zeros_like(x)
    n = min(x.shape[-1], y.shape[-1])
    out[..., :n] = y[..., :n]
    return out.astype(np.float32)


@dataclass
class Upstream:
    """Fonctionnalités actives en amont et leurs réglages.

    `transforms` : {nom: réglages} dans l'ordre d'application ; `gates` : {nom: seuil}.
    """

    transforms: dict[str, dict[str, Any]] = field(default_factory=dict)
    gates: dict[str, float] = field(default_factory=dict)
    combine: str = "all"
    signal_cfg: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return bool(self.transforms or self.gates)

    def transform_tag(self) -> str:
        """Étiquette des transformations, ajoutée au nom de l'encodeur (« bp3-7k+dn1 »)."""
        parts = []
        for name, opts in self.transforms.items():
            if name == "bandpass":
                lo, hi = opts["band_hz"]
                parts.append(f"bp{lo / 1000:g}-{hi / 1000:g}k")
            elif name == "denoise":
                parts.append(f"dn{opts.get('strength', 1.0):g}")
        return "+".join(parts)

    def gate_tag(self) -> str:
        """Étiquette des portes, ajoutée au nom du stock (« g-notes1-rhythm1-all »)."""
        if not self.gates:
            return ""
        body = "-".join(f"{name}{value:g}" for name, value in self.gates.items())
        return f"g-{body}-{self.combine}"

    def transform(self, wav: np.ndarray, sr: int) -> np.ndarray:
        for name, opts in self.transforms.items():
            if name == "bandpass":
                wav = bandpass(wav, sr, tuple(opts["band_hz"]), int(opts.get("order", 4)))
            elif name == "denoise":
                wav = denoise(
                    wav, sr, float(opts.get("strength", 1.0)), float(opts.get("floor", 0.1))
                )
        return np.asarray(wav, dtype=np.float32)

    def passes(self, values: pd.DataFrame) -> np.ndarray:
        """Fenêtres qui passent les portes actives (toutes si aucune porte)."""
        return gate_mask(values, self.gates, self.combine)


def gate_mask(values: pd.DataFrame, gates: dict[str, float], combine: str = "all") -> np.ndarray:
    """Vrai si la fenêtre passe : valeur ≥ seuil pour toutes (`all`) ou une (`any`) des portes.
    Une valeur manquante (NaN) laisse passer : dans le doute, on encode."""
    if combine not in ("all", "any"):
        raise ValueError(f"combinaison inconnue : {combine!r} (all ou any)")
    if not gates:
        return np.ones(len(values), dtype=bool)
    checks = [
        (values[name].isna() | (values[name] >= threshold)).to_numpy()
        for name, threshold in gates.items()
    ]
    return np.logical_and.reduce(checks) if combine == "all" else np.logical_or.reduce(checks)


def _upstream_section(cfg: dict) -> dict:
    """`sequential.upstream` ; une ancienne config à section `prefilter` reste lue."""
    section = (cfg.get("sequential", {}) or {}).get("upstream")
    return (section if section is not None else cfg.get("prefilter", {})) or {}


def upstream_from_cfg(cfg: dict, only: str | list[str] | None = None) -> Upstream:
    """Fonctionnalités amont activées (`enabled`) dans `sequential.upstream`.

    `only` (option `--upstream`) remplace les interrupteurs : liste de noms (« bandpass,notes »)
    ou « none » ; les réglages (bande, seuils) restent ceux de la config.
    """
    section = _upstream_section(cfg)
    gate_section = section.get("gates", {}) or {}
    if only is not None:
        names = [n.strip() for n in (only.split(",") if isinstance(only, str) else only)]
        names = [n for n in names if n and n != "none"]
        unknown = sorted(set(names) - set(FEATURES))
        if unknown:
            raise ValueError(f"fonctionnalités inconnues : {unknown} (connues : {FEATURES})")
        enabled = set(names)
    else:
        enabled = {n for n in TRANSFORMS if (section.get(n) or {}).get("enabled")}
        enabled |= {n for n in GATES if (gate_section.get(n) or {}).get("enabled")}
    defaults = {
        "bandpass": {"band_hz": [3000, 7000], "order": 4},
        "denoise": {"strength": 1.0, "floor": 0.1},
    }
    transforms = {
        n: defaults[n] | {k: v for k, v in (section.get(n) or {}).items() if k != "enabled"}
        for n in TRANSFORMS
        if n in enabled
    }
    gates = {n: gate_threshold(cfg, n) for n in GATES if n in enabled}
    return Upstream(
        transforms,
        gates,
        str(gate_section.get("combine", "all")),
        dict(cfg.get("signal", {})),
    )


def gate_threshold(cfg: dict, name: str) -> float:
    """Seuil de la porte `name` dans la config (défaut `GATE_DEFAULTS`)."""
    gate_section = _upstream_section(cfg).get("gates", {}) or {}
    return float((gate_section.get(name) or {}).get(GATE_THRESHOLD_KEY[name], GATE_DEFAULTS[name]))


def onset_gates(cfg: dict) -> tuple[dict[str, float], str]:
    """Portes « notes » et « rhythm » du module en amont, pour la chaîne de décision (fusion,
    score) : celles activées, sinon `notes` à son seuil (un module en amont sans porte de
    rythme activée garde au moins celle-là)."""
    upstream = upstream_from_cfg(cfg)
    gates = {n: v for n, v in upstream.gates.items() if n in ONSET_GATES}
    return (gates or {"notes": gate_threshold(cfg, "notes")}), upstream.combine


def onset_counts(
    windows: pd.DataFrame, onsets: dict[str, np.ndarray], ioi_range_s: tuple[float, float]
) -> pd.DataFrame:
    """Débuts de notes (`notes`) et intervalles d'A. blanci (`rhythm`) dans chaque fenêtre
    (recording_id, offset_s, dur_s), à partir des débuts de notes de l'enregistrement. NaN sans
    débuts de notes calculés (une porte laisse alors passer)."""
    rows = []
    for rid, offset, dur in zip(
        windows["recording_id"], windows["offset_s"], windows["dur_s"], strict=True
    ):
        times = onsets.get(rid)
        if times is None:
            rows.append((np.nan, np.nan))
            continue
        inside = np.sort(times[(times >= offset) & (times < offset + dur)])
        iois = np.diff(inside)
        rhythm = ((iois >= ioi_range_s[0]) & (iois <= ioi_range_s[1])).sum()
        rows.append((float(len(inside)), float(rhythm)))
    return pd.DataFrame(rows, columns=list(ONSET_GATES), index=windows.index)


def gate_values(
    windows: np.ndarray,
    sr: int,
    signal_cfg: dict,
    offsets_s: np.ndarray | None = None,
    onsets: np.ndarray | None = None,
    dur_s: float | None = None,
) -> pd.DataFrame:
    """Valeurs des quatre portes pour un lot de fenêtres (n, échantillons), à la f_e native.

    Énergie et contraste : calculés sur le son de chaque fenêtre, comme les baselines. Notes et
    rythme : comptés sur les débuts de notes de l'enregistrement (`onsets`, avec `offsets_s` et
    `dur_s` des fenêtres) — les mêmes que le module en parallèle ; à défaut, détectés dans la
    fenêtre seule.
    """
    from blanci.baselines import window_features

    rows = []
    for wav in np.atleast_2d(np.asarray(windows, dtype=np.float32)):
        fixed = window_features(wav, sr, signal_cfg).fixed
        rows.append(
            {
                "band_energy": fixed["band_energy"],
                "band_contrast": fixed["band_contrast"],
                "notes": fixed["notes"],
                "rhythm": float(np.floor(fixed["rhythm"])),  # intervalles dans la plage
            }
        )
    values = pd.DataFrame(rows, columns=list(GATES))
    if onsets is not None and offsets_s is not None and len(values):
        placed = pd.DataFrame(
            {"recording_id": "r", "offset_s": np.asarray(offsets_s, dtype=float), "dur_s": dur_s}
        )
        counts = onset_counts(
            placed, {"r": np.asarray(onsets, dtype=float)}, tuple(signal_cfg["ioi_range_s"])
        )
        values[list(ONSET_GATES)] = counts.to_numpy()
    return values


GATE_GRIDS = {
    "band_energy": (0.0, 3.0, 6.0, 9.0, 12.0, 15.0),
    "band_contrast": (0.0, 1.5, 3.0, 4.5, 6.0, 9.0),
    "notes": (1, 2, 3, 4),
    "rhythm": (1, 2, 3),
}


def gate_sweep(
    values: pd.DataFrame,
    y: np.ndarray,
    recordings: np.ndarray,
    grids: dict[str, tuple[float, ...]] | None = None,
) -> pd.DataFrame:
    """Une ligne par (porte, seuil) : fenêtres arrêtées (part des négatifs, calcul économisé),
    positifs perdus (fenêtres et enregistrements), rappel plafond.

    Un enregistrement positif est perdu si **toutes** ses fenêtres positives sont arrêtées : il
    ne peut plus être détecté, quel que soit l'encodeur.
    """
    from blanci.evaluate import wilson_interval

    y = np.asarray(y).astype(int)
    recordings = np.asarray(recordings)
    grids = grids or GATE_GRIDS
    rows = []
    positive_recordings = np.unique(recordings[y == 1])
    for name, thresholds in grids.items():
        for threshold in thresholds:
            passed = gate_mask(values, {name: threshold})
            rows.append(_sweep_row(name, threshold, passed, y, recordings, positive_recordings))
    table = pd.DataFrame(rows)
    lo_hi = [
        wilson_interval(int(k), int(n))
        for k, n in zip(table["pos_recordings_kept"], table["pos_recordings"], strict=True)
    ]
    table["recall_ceiling_lo"] = [lo for lo, _ in lo_hi]
    table["recall_ceiling_hi"] = [hi for _, hi in lo_hi]
    return table


def _sweep_row(name, threshold, passed, y, recordings, positive_recordings) -> dict[str, Any]:
    kept_recordings = set(recordings[(y == 1) & passed])
    n_kept = sum(r in kept_recordings for r in positive_recordings)
    negatives = y == 0
    return {
        "gate": name,
        "threshold": float(threshold),
        "neg_stopped": float((~passed[negatives]).mean()) if negatives.any() else float("nan"),
        "pos_windows_lost": int(((y == 1) & ~passed).sum()),
        "pos_windows": int((y == 1).sum()),
        "pos_recordings_kept": int(n_kept),
        "pos_recordings": int(len(positive_recordings)),
        "recall_ceiling": n_kept / len(positive_recordings)
        if len(positive_recordings)
        else float("nan"),
    }


def apply_gate(scores: np.ndarray, passed: np.ndarray) -> np.ndarray:
    """Scores après la porte : une fenêtre arrêtée prend `GATED_SCORE`."""
    out = np.asarray(scores, dtype=float).copy()
    out[~np.asarray(passed, dtype=bool)] = GATED_SCORE
    return out
