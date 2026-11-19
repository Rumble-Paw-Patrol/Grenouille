"""Le spectre des deux pistes, en tiers d'octave, sur l'échantillon des pistes.

Les deux micros d'un Song Meter entendent la même scène : bande par bande, leur écart ne dépend
que du gain réglé. Ce script mesure, pour chaque enregistrement de `echantillon_pistes.csv`
(3 607 : un par micro et par jour, plus les bascules d'état), le spectre moyen de chaque piste
(densité de Welch de `blanci qc`, 2 min entières) rangé en 24 tiers d'octave de 100 Hz à
20 kHz (`inventaire.track_band_levels`). Les mesures sont écrites une fois pour toutes : la
figure (`documentation/prez/scripts/fig_spectre_pistes.py`) ne relit plus le son.

Lecture seule de la base et des disques ; chemins par `blanci.core.locate.locate` (tous les
disques branchés). Reprenable : les enregistrements déjà mesurés sont sautés.

    uv run python scripts/controle_qualite/spectre_pistes.py            # ~45 min, deux disques
    options : --limite 50, --fils 6 (processus), --sortie <dossier>, --config <local.yaml>

Sortie : `paths.reports/pistes/spectre_pistes.csv` (une ligne par enregistrement lu :
`recording_id`, `sample_rate`, `L0_<bande>`, `L1_<bande>` en dB pleine échelle, `erreur`).
"""

from __future__ import annotations

import argparse
import io
import sqlite3
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import pandas as pd
import soundfile as sf

from blanci.core.config import config_path, default_user_config, load_config
from blanci.core.locate import locate
from blanci.exploration.inventaire import third_octave_bands, track_band_levels

BANDS = third_octave_bands(100, 20000)


def work(data: bytes | str) -> dict[str, object]:
    """Un enregistrement déjà lu, mesuré dans un processus à part ; une chaîne est une erreur
    de lecture, rendue telle quelle."""
    if isinstance(data, str):
        return {"erreur": data}
    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return {"sample_rate": sr} | track_band_levels(wav, sr, BANDS)
    except Exception as exc:  # fichier illisible : noté, pas bloquant
        return {"erreur": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--fils", type=int, default=6, help="processus")
    ap.add_argument("--sortie", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config or default_user_config())
    out_dir = args.sortie or config_path(cfg, "reports") / "pistes"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "spectre_pistes.csv"

    ids = pd.read_csv(config_path(cfg, "reports") / "pistes" / "echantillon_pistes.csv",
                      usecols=["recording_id"])["recording_id"]
    con = sqlite3.connect(f"file:{config_path(cfg, 'db').as_posix()}?mode=ro", uri=True)
    paths = pd.read_sql_query("SELECT recording_id, path FROM recordings", con)
    con.close()
    todo = paths[paths["recording_id"].isin(set(ids))].sort_values("path")
    done = set(pd.read_csv(target, usecols=["recording_id"])["recording_id"]) \
        if target.is_file() else set()
    print(f"{len(todo)} enregistrements dans l'échantillon, {len(done)} déjà mesurés")
    todo = todo[~todo["recording_id"].isin(done)]
    if args.limite:
        todo = todo.head(args.limite)
    raw, renamed = str(config_path(cfg, "raw")), cfg["audio"].get("renamed")
    columns = ["recording_id", "sample_rate",
               *[f"L{t}_{b}" for t in (0, 1) for b in BANDS["name"]], "erreur"]

    def flush(rows: list[dict]) -> None:
        pd.DataFrame(rows).reindex(columns=columns).to_csv(
            target, mode="a", header=not target.is_file(), index=False)

    def read(rid_path) -> bytes | str:
        """Lecture en série : un disque USB à plateaux lu par plusieurs fils à la fois saute
        d'un fichier à l'autre et tombe à quelques Mo/s (tri par chemin : lecture dans l'ordre)."""
        path = locate(rid_path, Path(raw), renamed=renamed)
        return "introuvable" if path is None else path.read_bytes()

    recs = list(todo.itertuples(index=False))
    rows, start, n = [], perf_counter(), 0
    with ProcessPoolExecutor(max_workers=args.fils) as pool:
        pending: deque = deque()
        for rec in [*recs, None]:
            if rec is not None:
                pending.append((rec.recording_id, pool.submit(work, read(rec.path))))
            while pending and (rec is None or len(pending) > 2 * args.fils):
                rid, future = pending.popleft()
                n += 1
                rows.append({"recording_id": rid} | future.result())
                if n % 100 == 0 or n == len(recs):
                    flush(rows)
                    rows = []
                    rate = n / (perf_counter() - start)
                    left = (len(recs) - n) / rate / 60
                    print(f"  {n}/{len(recs)}, {rate:.1f}/s, reste environ {left:.0f} min",
                          flush=True)
    print(f"écrit : {target}")


if __name__ == "__main__":
    main()
