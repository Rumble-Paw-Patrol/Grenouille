"""Trois files d'écoute sur la carte CUSUM du son étouffé : milieu, bords, pistes douteuses.

Un épisode est une suite d'enregistrements consécutifs d'un micro (dans l'ordre du programme,
`qc._programme_ranks`, un créneau manqué toléré) dont une même piste est retenue par la carte
(`muffled_0` ou `muffled_1`). Chaque candidat vient avec ses deux voisins au programme (celui
d'avant et celui d'après), pour entendre l'avant et l'après ; trois extraits de 30 s au milieu
du fichier, dans l'ordre du temps, par candidat.

- `etouffe_milieu` : enregistrements retenus alors que leur `hf_ratio` est **au-dessus** du seuil
  (`muffled_hf_ratio`, 5 %), en plein milieu d'un épisode (retenus des deux côtés). La carte les
  garde parce que la somme accumulée avant eux reste positive : l'écoute dit si le micro y est
  encore étouffé (alors la règle est juste) ou déjà libéré (alors elle retient trop).
- `etouffe_bordures` : premier et dernier enregistrement retenus de chaque épisode. Aucun n'est
  au-dessus du seuil (la traîne ne garde que ce qui est sous la référence) : ce sont les
  enregistrements les plus proches de l'enlèvement ou de la pose, et leur voisin hors épisode,
  non retenu, dit si la carte coupe au bon endroit.

- `etouffe_douteux` : enregistrements dont une piste est retenue et dont l'autre, lue, est sous
  le seuil sans être retenue (candidat isolé) : les deux pistes se ressemblent, la seconde n'est
  pas franchement saine. `recording_channel` n'y bascule pas de piste.

Lecture seule de la base ; n'écrit que les trois files dans `annotations/files_ecoute/`.

    uv run python scripts/son_etouffe/files_milieu_bordure.py [--config <fichier>]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from blanci.annotation.selection import write_queue
from blanci.core.config import config_path, default_user_config, load_config
from blanci.inputs.qc import _programme_ranks, cusum_settings

UTC_OFFSET = pd.Timedelta(hours=-3)  # heure locale de Guyane
EXTRACT_S = 30.0
TRACKS = (0, 1)


def load(db: Path) -> pd.DataFrame:
    """Une ligne par enregistrement : indices et drapeaux de chaque piste (NaN / False si la
    piste n'est pas lue)."""
    con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT recording_id, path, dataset, site, mic_id, start_utc, duration_s, qc_flags "
        "FROM recordings"
    ).fetchall()
    con.close()
    out = []
    for rid, path, dataset, site, mic, start, duration, qc in rows:
        flags = json.loads(qc) if qc else {}
        row = [rid, path, dataset, site, mic, start, duration]
        for t in TRACKS:
            ind = flags.get("indices" if t == 0 else "indices_1")
            row += [float(ind["hf_ratio"]) if ind else np.nan, bool(flags.get(f"muffled_{t}"))]
        out.append(row)
    columns = ["recording_id", "path", "dataset", "site", "mic_id", "start_utc", "duration_s"]
    columns += [f"{k}{t}" for t in TRACKS for k in ("hf", "ret")]
    df = pd.DataFrame(out, columns=columns)
    df["t"] = pd.to_datetime(df["start_utc"], utc=True, errors="coerce")
    return df.dropna(subset=["t"])


def series(df: pd.DataFrame):
    """Séries de chaque micro, dans l'ordre du temps, avec le rang au programme (`rank`)."""
    for _, mic in df.groupby(["dataset", "site", "mic_id"], sort=False):
        mic = mic.sort_values("t", kind="stable").reset_index(drop=True)
        yield mic.assign(rank=_programme_ranks(mic["t"]))


def runs(retained: np.ndarray, ranks: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Épisodes : suites maximales d'enregistrements retenus dont les rangs se suivent à
    `max_gap` près. Rend (premier, dernier) positions, bornes comprises."""
    found, start = [], None
    for i, kept in enumerate(retained):
        if kept and start is not None and ranks[i] - ranks[i - 1] > max_gap:
            found.append((start, i - 1))
            start = None
        if kept and start is None:
            start = i
        if not kept and start is not None:
            found.append((start, i - 1))
            start = None
    if start is not None:
        found.append((start, len(retained) - 1))
    return found


def neighbours(ranks: np.ndarray, i: int, max_gap: int) -> tuple[int | None, int | None]:
    """Positions du voisin d'avant et du voisin d'après au programme (None au-delà de `max_gap`
    créneaux ou en bout de série)."""
    before = i - 1 if i > 0 and ranks[i] - ranks[i - 1] <= max_gap else None
    after = i + 1 if i + 1 < len(ranks) and ranks[i + 1] - ranks[i] <= max_gap else None
    return before, after


def groups(
    mic: pd.DataFrame, ref: float, max_gap: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """Groupes « milieu », « bordure » et « douteux » d'un micro : chacun a `centre`, `voisins`
    (positions de la série) et de quoi écrire la cause. Douteux : piste retenue dont l'autre
    piste, lue, est sous le seuil sans être retenue (candidat isolé)."""
    middle, border, doubt = [], [], []
    ranks = mic["rank"].to_numpy()
    for t in TRACKS:
        hf, kept = mic[f"hf{t}"].to_numpy(), mic[f"ret{t}"].to_numpy(dtype=bool)
        for first, last in runs(kept, ranks, max_gap):
            span = {"track": t, "first": first, "last": last, "n": last - first + 1}
            for i in range(first + 1, last):
                if hf[i] >= ref:
                    middle.append({**span, "centre": i, "voisins": (i - 1, i + 1)})
            other = 1 - t
            for i in range(first, last + 1):
                o = mic.loc[i, f"hf{other}"]
                if not mic.loc[i, f"ret{other}"] and o < ref:  # NaN : piste non lue, exclue
                    doubt.append({**span, "centre": i, "voisins": neighbours(ranks, i, max_gap)})
            for i in sorted({first, last}):
                side = "seul" if first == last else ("début" if i == first else "fin")
                border.append(
                    {**span, "centre": i, "side": side, "voisins": neighbours(ranks, i, max_gap)}
                )
    return middle, border, doubt


def local(t: pd.Timestamp) -> str:
    return f"{t + UTC_OFFSET:%d/%m %H:%M}"


def role_label(r: pd.Series, candidate: bool, track: int, ref: float) -> str:
    """Rôle d'un extrait pour le poste d'annotation : « candidat retenu » ; pour un voisin,
    « voisin retenu » (retenu sur la piste du candidat), « voisin sain » (aucune piste lue n'est
    retenue ni sous le seuil) ou « voisin douteux » (non retenu, mais une piste lue sous le
    seuil : candidat isolé)."""
    if candidate:
        return "candidat retenu"
    if r[f"ret{track}"]:
        return "voisin retenu"
    hf = [r[f"hf{t}"] for t in TRACKS if not np.isnan(r[f"hf{t}"])]
    clear = bool(hf) and not any(r[f"ret{t}"] for t in TRACKS) and min(hf) >= ref
    return "voisin sain" if clear else "voisin douteux"


def queue_rows(
    mic: pd.DataFrame, found: list[dict], kind: str, first_group: int, ref: float
) -> list[dict]:
    """Trois extraits par groupe (voisin d'avant, candidat, voisin d'après), dans l'ordre du
    temps. Un enregistrement déjà pris (candidat d'un autre groupe, ou voisin) n'est pas répété."""
    rows = []
    for n, g in enumerate(found, start=first_group):
        t, c = g["track"], mic.loc[g["centre"]]
        episode = (
            f"épisode de la piste {t} : {g['n']} enregistrement{'s' if g['n'] > 1 else ''}, "
            f"{local(mic.loc[g['first'], 't'])} à {local(mic.loc[g['last'], 't'])}"
        )
        for pos in sorted(
            p for p in (g["voisins"][0], g["centre"], g["voisins"][1]) if p is not None
        ):
            r = mic.loc[pos]
            role = "candidat" if pos == g["centre"] else ("avant" if pos < g["centre"] else "après")
            inside = bool(r[f"ret{t}"])
            label = role_label(r, role == "candidat", t, ref)
            what = {
                "milieu": f"retenu à hf {c[f'hf{t}']:.3f}, au-dessus du seuil, au milieu d'un "
                f"{episode}",
                "bordures": f"{g.get('side')} d'épisode, retenu à hf {c[f'hf{t}']:.3f} ; {episode}",
                "douteux": f"piste {t} retenue (hf {c[f'hf{t}']:.3f}), piste {1 - t} lue sous le "
                f"seuil sans épisode (hf {c[f'hf{1 - t}']:.3f}) ; {episode}",
            }[kind]
            cause = (
                f"groupe {n} : {what}"
                if role == "candidat"
                else (
                    f"groupe {n} : voisin {role} ({'dans' if inside else 'hors'} l'épisode, "
                    f"piste {t} hf {r[f'hf{t}']:.3f}) du candidat de {local(c['t'])}"
                )
            )
            offset = float(
                np.clip(r["duration_s"] / 2 - EXTRACT_S / 2, 0, max(r["duration_s"] - EXTRACT_S, 0))
            )
            rows.append(
                {
                    "recording_id": r["recording_id"],
                    "path": r["path"],
                    "site": r["site"],
                    "mic_id": r["mic_id"],
                    "start_utc": r["start_utc"],
                    "offset_s": round(offset, 1),
                    "dur_s": min(EXTRACT_S, float(r["duration_s"])),
                    "score": np.nan,
                    "reason": f"etouffe_{kind}/{r['site']}/{label}",
                    "role": label,
                    "source": "flag",
                    "dataset": r["dataset"],
                    "piste_0_hf": round(r["hf0"], 4),
                    "piste_1_hf": round(r["hf1"], 4),
                    "cause": cause,
                    "_group": n,
                    "_role": role,
                }
            )
    return rows


def dedupe(rows: list[dict]) -> pd.DataFrame:
    """Un extrait par enregistrement : un candidat passe avant un voisin ; sinon le premier
    groupe garde la ligne, les autres sont notés dans `cause`."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df.drop(columns=["_group", "_role"], errors="ignore")
    df["_rank"] = (df["_role"] != "candidat").astype(int)
    order = df.sort_values(["_rank", "_group"], kind="stable")
    others = order.groupby("recording_id")["_group"].agg(lambda s: sorted(set(s.iloc[1:])))
    keep = order.drop_duplicates("recording_id").copy()
    extra = keep["recording_id"].map(others)
    keep["cause"] += extra.map(lambda g: f" (aussi groupes {', '.join(map(str, g))})" if g else "")
    return keep.sort_values(["_group", "start_utc"], kind="stable").drop(
        columns=["_group", "_role", "_rank"]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config or default_user_config())
    settings = cusum_settings(cfg["qc"])
    ref, max_gap = settings["ref_ratio"], int(settings["max_missed"]) + 1
    middle_rows, border_rows, doubt_rows = [], [], []
    n_middle = n_border = n_doubt = 0
    for mic in series(load(config_path(cfg, "db"))):
        middle, border, doubt = groups(mic, ref, max_gap)
        middle_rows += queue_rows(mic, middle, "milieu", n_middle + 1, ref)
        border_rows += queue_rows(mic, border, "bordures", n_border + 1, ref)
        doubt_rows += queue_rows(mic, doubt, "douteux", n_doubt + 1, ref)
        n_middle, n_border, n_doubt = (
            n_middle + len(middle),
            n_border + len(border),
            n_doubt + len(doubt),
        )
    about = {
        "origine": "scripts/son_etouffe/files_milieu_bordure.py (carte CUSUM, "
        f"référence {ref:g}, un créneau manqué toléré)",
    }
    for name, rows, content in (
        (
            "etouffe_milieu",
            middle_rows,
            "enregistrements retenus au-dessus du seuil, en plein milieu d'un épisode, avec "
            "leurs deux voisins au programme ; extraits de 30 s",
        ),
        (
            "etouffe_bordures",
            border_rows,
            "premier et dernier enregistrement retenus de chaque épisode, avec leurs deux "
            "voisins au programme (dont un hors épisode) ; extraits de 30 s",
        ),
        (
            "etouffe_douteux",
            doubt_rows,
            "enregistrements dont une piste est retenue et l'autre, lue, est sous le seuil sans "
            "être retenue (douteuse : le repli de piste ne s'y fait pas), avec leurs deux "
            "voisins au programme ; extraits de 30 s",
        ),
    ):
        queue = dedupe(rows)
        path = write_queue(cfg, queue, name, about={**about, "contenu": content})
        print(f"écrit : {path} ({len(queue)} extraits)")


if __name__ == "__main__":
    main()
