"""Les deux pistes d'un enregistreur, mesurées sur un échantillon : gains, dérive, défauts.

Chaque Song Meter a deux micros, donc deux pistes qui entendent la même scène. Leur écart de
niveau ne dépend que des gains réglés (6 ou 18 dB, lus dans l'en-tête GUANO) : s'il dérive au
fil des semaines, un des deux micros se dégrade sans forcément être étouffé (Turgeon et al.
2017). Leur cohérence (corrélation par bande de fréquence) dit la même chose autrement : un
micro étouffé, mort ou qui crépite ne ressemble plus à l'autre.

Échantillon (lecture seule de la base et des disques) :
- un enregistrement par micro et par jour local, le plus proche de 10 h (`--heure`) parmi ceux
  de durée normale ;
- les enregistrements où une piste bascule (voisins d'une suite `muffled_0` ou `muffled_1`) :
  le dernier sain avant la suite, le premier et le dernier de la suite, le premier sain après.

Mesures, par piste (0 et 1) : les indices de `blanci qc` (`qc_indices`), `hf_ratio` et RMS par
tranche de 10 s (une bascule en plein enregistrement), et des indices de défaut : facteur de
crête, aplatissement (kurtosis) de la dérivée, taux de clics, plus longue suite d'échantillons
identiques (trou numérique). Entre pistes : écart de niveau (dB) et cohérence moyenne sous et
au-dessus de 2 kHz (20 premières secondes). Plus le GUANO : modèle, gains, position, température.

    uv run python scripts/controle_qualite/echantillon_pistes.py            # reprenable
    options : --limite 50, --heure 10, --fils 6 (processus), --sortie <dossier>

Sortie : `paths.reports/pistes/echantillon_pistes.csv` (une ligne par enregistrement lu).
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import coherence

from blanci.core.config import config_path, default_user_config, load_config
from blanci.core.locate import locate
from blanci.inputs.ingest import read_guano
from blanci.inputs.qc import parse_flags, qc_indices

SEGMENT_S = 10.0
CLICK_K = 20.0  # clic : |dérivée| au-delà de 20 fois sa médiane absolue


def read_only(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)


def sample(con: sqlite3.Connection, cfg: dict, hour: float) -> pd.DataFrame:
    """Enregistrements à lire : un par micro et par jour, plus les bascules d'état."""
    rec = pd.read_sql_query(
        "SELECT recording_id, path, dataset, site, mic_id, start_utc, duration_s, channels, "
        "qc_flags FROM recordings",
        con,
    )
    flags = rec.pop("qc_flags").map(parse_flags)
    offset = pd.Timedelta(hours=cfg["recorder"]["filename_utc_offset_h"])
    rec["local"] = pd.to_datetime(rec["start_utc"], utc=True) + offset
    rec["day"] = rec["local"].dt.date
    rec["m0"] = flags.map(lambda f: bool(f.get("muffled_0")))
    rec["m1"] = flags.map(lambda f: bool(f.get("muffled_1")))
    normal = (rec["duration_s"] - cfg["qc"]["expected_duration_s"]).abs() <= 1
    ok = rec[normal & (rec["channels"] == 2)].copy()
    ok["gap"] = (ok["local"].dt.hour + ok["local"].dt.minute / 60 - hour).abs()
    daily = ok.sort_values("gap").groupby(["dataset", "site", "mic_id", "day"]).head(1)
    daily = daily.assign(motif="jour")

    edges = []
    for _, mic in ok.sort_values("local").groupby(["dataset", "site", "mic_id"], sort=False):
        for track in ("m0", "m1"):
            state = mic[track].to_numpy()
            change = np.flatnonzero(state[1:] != state[:-1])
            for i in change:  # i : dernier avant la bascule, i + 1 : premier après
                edges.append(mic.iloc[[i, i + 1]].assign(motif=f"bascule_{track}"))
    out = pd.concat([daily, *edges]) if edges else daily
    out = out.drop_duplicates("recording_id").drop(columns=["gap"])
    return out.sort_values(["dataset", "site", "mic_id", "local"]).reset_index(drop=True)


def defects(x: np.ndarray, sr: int) -> dict[str, float]:
    """Indices de défaut d'une piste : crête, clics, trous numériques."""
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x**2))) or 1e-12
    d = np.diff(x)
    mad = float(np.median(np.abs(d))) or 1e-12
    centred = d - d.mean()
    kurt = float(np.mean(centred**4) / (np.mean(centred**2) ** 2 + 1e-30))
    same = np.flatnonzero(np.diff(x) != 0)
    longest = int(np.max(np.diff(np.concatenate([[-1], same, [len(x) - 1]])))) if len(x) else 0
    return {
        "crest_db": 20 * float(np.log10(np.abs(x).max() / rms + 1e-12)),
        "diff_kurtosis": kurt,
        "clicks_per_s": float(np.sum(np.abs(d) > CLICK_K * mad)) / (len(x) / sr),
        "longest_flat_ms": 1000 * longest / sr,
    }


def segments(x: np.ndarray, sr: int) -> dict[str, str]:
    """`hf_ratio` et RMS par tranche de 10 s, en JSON (bascule en plein enregistrement)."""
    n = int(SEGMENT_S * sr)
    parts = [qc_indices(x[i : i + n], sr) for i in range(0, len(x) - n + 1, n)]
    return {
        "hf_seg": json.dumps([round(p["hf_ratio"], 4) for p in parts]),
        "rms_seg": json.dumps([round(p["rms_dbfs"], 2) for p in parts]),
    }


def guano_fields(source: io.BytesIO) -> dict[str, object]:
    g = read_guano(source)
    try:
        settings = json.loads(g.get("WA|Song Meter|Audio settings", "[]"))
    except ValueError:
        settings = []
    gains = [s.get("gain") for s in settings]
    lat, _, lon = g.get("Loc Position", "").partition(" ")
    return {
        "model": g.get("Model"),
        "firmware": g.get("Firmware Version"),
        "gain_0": gains[0] if len(gains) > 0 else None,
        "gain_1": gains[1] if len(gains) > 1 else None,
        "guano_timestamp": g.get("Timestamp"),
        "lat": float(lat) if lat else None,
        "lon": float(lon) if lon else None,
        "temperature_c": float(g["Temperature Int"]) if g.get("Temperature Int") else None,
    }


def measure(data: bytes) -> dict[str, object]:
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    out: dict[str, object] = {"sample_rate": sr} | guano_fields(io.BytesIO(data))
    for track in range(min(2, wav.shape[1])):
        x = wav[:, track]
        for k, v in (qc_indices(x, sr) | defects(x, sr) | segments(x, sr)).items():
            out[f"{k}_{track}"] = v
    if wav.shape[1] > 1:
        out["level_diff_db"] = out["rms_dbfs_0"] - out["rms_dbfs_1"]
        # Cohérence sur les 20 premières secondes : les deux micros sont à quelques centimètres,
        # elle reste faible au-dessus de 2 kHz même quand tout va bien (champ diffus).
        head = wav[: int(20 * sr)]
        f, c = coherence(head[:, 0], head[:, 1], fs=sr, nperseg=2048)
        out["coh_low"] = float(c[(f >= 100) & (f < 2000)].mean())
        out["coh_high"] = float(c[(f >= 2000) & (f <= min(10000, sr / 2))].mean())
    return out


def work(data: bytes | str) -> dict[str, object]:
    """Un enregistrement déjà lu, dans un processus à part (le calcul tient le verrou de
    Python). Une chaîne est une erreur de lecture, rendue telle quelle."""
    if isinstance(data, str):
        return {"erreur": data}
    try:
        return measure(data)
    except Exception as exc:  # fichier illisible : noté, pas bloquant
        return {"erreur": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--heure", type=float, default=10.0)
    ap.add_argument("--fils", type=int, default=6, help="processus")
    ap.add_argument("--sortie", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config or default_user_config())
    out_dir = args.sortie or config_path(cfg, "reports") / "pistes"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "echantillon_pistes.csv"

    todo = sample(read_only(config_path(cfg, "db")), cfg, args.heure)
    done = set(pd.read_csv(target)["recording_id"]) if target.is_file() else set()
    print(f"{len(todo)} enregistrements dans l'échantillon, {len(done)} déjà lus")
    todo = todo[~todo["recording_id"].isin(done)]
    if args.limite:
        todo = todo.head(args.limite)
    raw, renamed = str(config_path(cfg, "raw")), cfg["audio"].get("renamed")
    columns = list(pd.read_csv(target, nrows=0).columns) if target.is_file() else None

    def flush(rows: list[dict]) -> None:
        nonlocal columns
        frame = pd.DataFrame(rows)
        columns = columns or list(frame.columns)
        frame.reindex(columns=columns).to_csv(
            target, mode="a", header=not target.is_file(), index=False
        )

    rows, start = [], perf_counter()
    keep = ["recording_id", "dataset", "site", "mic_id", "start_utc", "motif", "m0", "m1"]
    recs = list(todo.itertuples())

    def read(rec) -> bytes | str:
        """Lecture en série : un disque USB à plateaux lu par plusieurs fils à la fois saute
        d'un fichier à l'autre et tombe à quelques Mo/s."""
        path = locate(rec.path, Path(raw), renamed=renamed)
        return "introuvable" if path is None else path.read_bytes()

    with ProcessPoolExecutor(max_workers=args.fils) as pool:
        pending: deque = deque()
        n = 0
        for rec in [*recs, None]:
            if rec is not None:
                pending.append((rec, pool.submit(work, read(rec))))
            while pending and (rec is None or len(pending) > 2 * args.fils):
                done_rec, future = pending.popleft()
                n += 1
                rows.append({k: getattr(done_rec, k) for k in keep} | future.result())
                if n % 100 == 0 or n == len(recs):
                    flush(rows)
                    rows = []
                    rate = n / (perf_counter() - start)
                    left = (len(recs) - n) / rate / 60
                    print(
                        f"  {n}/{len(recs)}, {rate:.1f}/s, reste environ {left:.0f} min", flush=True
                    )
    print(f"écrit : {target}")


if __name__ == "__main__":
    main()
