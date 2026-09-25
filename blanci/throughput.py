"""Relevé des encodeurs (§2) : débit, mémoire, dimension, jetons, projection sur la cible.

Mesuré sur du bruit synthétique à 48 kHz (f_e des Song Meter) : le rééchantillonnage vers la
f_e du modèle est compté, aucun enregistrement n'est lu ni encodé. Chaque encodeur tourne dans
son propre processus : le pic de mémoire est alors le sien, et un modèle qui échoue n'arrête
pas les autres.

Projection (§7) : une campagne d'une semaine ≈ 575 h d'audio ; la grille avance d'une
demi-fenêtre par défaut (`encoders.overlap`), donc une heure d'audio = 3 600 / pas fenêtres.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blanci.grid import overlap_from_cfg

CAMPAIGN_HOURS = 575.0  # une semaine de pose (§7)
PEAK_HOURS_SHARE = 1 / 3.6  # heures de pic seules (7–9 h, 15–17 h) : ÷ 3,6 (§7)
RECORDER_SR = 48_000


def peak_memory_mb() -> float:
    """Pic de mémoire résidente du processus, en Mo (Windows, macOS, Linux)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32")
        # Types déclarés : sans eux, ctypes tronque le pseudo-handle en 32 bits.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32.K32GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.PeakWorkingSetSize / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # octets / Kio


def measure_encoder(
    encoder: Any,
    n_windows: int = 64,
    hop_ratio: float = 0.5,
    seed: int = 0,
    load_s: float = float("nan"),
) -> dict[str, Any]:
    """Débit de `encoder.embed` sur `n_windows` fenêtres de bruit à 48 kHz, après un tour à
    vide (initialisations paresseuses, compilation)."""
    rng = np.random.default_rng(seed)
    n = round(encoder.window_s * RECORDER_SR)
    batch = rng.normal(0, 0.05, (n_windows, n)).astype(np.float32)
    encoder.embed(batch[: min(2, n_windows)], RECORDER_SR)  # tour à vide
    start = time.perf_counter()
    out = encoder.embed(batch, RECORDER_SR)
    elapsed = time.perf_counter() - start

    windows_per_s = n_windows / elapsed
    hop_s = encoder.window_s * hop_ratio
    realtime = windows_per_s * hop_s  # secondes d'audio couvertes par seconde de calcul
    campaign_h = CAMPAIGN_HOURS / realtime
    return {
        "encoder": encoder.name,
        "version": encoder.version,
        "sample_rate": int(encoder.sample_rate),
        "window_s": float(encoder.window_s),
        "dim": int(out.shape[1]),
        "has_tokens": bool(encoder.has_tokens),
        "load_s": float(load_s),
        "windows_per_s": float(windows_per_s),
        "realtime_factor": float(realtime),
        "min_per_audio_hour": float(60 / realtime),
        "campaign_h": float(campaign_h),
        "campaign_peak_hours_h": float(campaign_h * PEAK_HOURS_SHARE),
        "peak_memory_mb": float(peak_memory_mb()),
        "finite": bool(np.isfinite(out).all()),
    }


def measure_by_name(name: str, cfg: dict, n_windows: int = 64) -> dict[str, Any]:
    """Charge l'encodeur `name` (config) et le mesure, dans le processus courant."""
    from blanci.encoders import get_encoder

    start = time.perf_counter()
    encoder = get_encoder(name, cfg)
    load_s = time.perf_counter() - start
    return measure_encoder(
        encoder,
        n_windows=n_windows,
        hop_ratio=1.0 - overlap_from_cfg(cfg),
        load_s=load_s,
    )


def measure_in_subprocess(
    name: str, config: Path | None, n_windows: int = 64, timeout_s: float = 3600
) -> dict[str, Any]:
    """Mesure dans un processus neuf ; en cas d'échec, une ligne avec l'erreur."""
    command = [sys.executable, "-m", "blanci.throughput", name, "--n-windows", str(n_windows)]
    if config is not None:
        command += ["--config", str(config)]
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s, encoding="utf-8"
        )
    except subprocess.TimeoutExpired:
        return {"encoder": name, "error": f"plus de {timeout_s:.0f} s"}
    for line in reversed(done.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["sortie vide"]
    return {"encoder": name, "error": tail[0][:300]}


REPORT_COLUMNS = [
    "encoder",
    "sample_rate",
    "window_s",
    "dim",
    "has_tokens",
    "windows_per_s",
    "realtime_factor",
    "min_per_audio_hour",
    "campaign_h",
    "campaign_peak_hours_h",
    "peak_memory_mb",
    "load_s",
    "error",
]


def write_throughput_report(rows: list[dict], reports_dir: Path, machine: str) -> dict[str, Path]:
    from blanci.benchmark import to_markdown

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths = {"csv": reports_dir / "debit.csv", "markdown": reports_dir / "debit.md"}
    table = pd.DataFrame(rows).reindex(columns=REPORT_COLUMNS)
    if paths["csv"].exists():  # une mesure met à jour sa ligne, les autres encodeurs restent
        previous = pd.read_csv(paths["csv"]).reindex(columns=REPORT_COLUMNS)
        previous = previous[~previous["encoder"].isin(table["encoder"])]
        table = pd.concat([previous, table], ignore_index=True)
    for column in ("sample_rate", "dim"):
        table[column] = pd.to_numeric(table[column]).astype("Int64")
    table["has_tokens"] = table["has_tokens"].map(
        lambda v: v if pd.isna(v) else str(v).lower() in ("true", "1", "1.0")
    )
    table.to_csv(paths["csv"], index=False)
    shown = table.drop(columns=["error"] if table["error"].isna().all() else [])
    shown = shown.astype({"sample_rate": object, "dim": object}).fillna({"error": ""})
    text = [
        "# Débit des encodeurs (§2, §7)",
        "",
        f"Machine : {machine}. Bruit synthétique à 48 kHz, rééchantillonnage compris, CPU seul. "
        f"Temps réel : secondes d'audio traitées par seconde de calcul, grille au pas d'une "
        f"demi-fenêtre. Campagne : {CAMPAIGN_HOURS:.0f} h d'audio (une semaine de pose) ; "
        "heures de pic seules : ÷ 3,6.",
        "",
        to_markdown(shown, floatfmt="{:.1f}"),
    ]
    paths["markdown"].write_text("\n".join(text), encoding="utf-8")
    return paths


def machine_description() -> str:
    import os
    import platform

    return (
        f"{platform.processor() or platform.machine()}, {os.cpu_count()} fils, {platform.system()}"
    )


def main(argv: list[str] | None = None) -> None:
    """Point d'entrée du sous-processus : imprime une ligne JSON."""
    from blanci.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("encoder")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--n-windows", type=int, default=64)
    args = parser.parse_args(argv)
    row = measure_by_name(args.encoder, load_config(args.config), args.n_windows)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
