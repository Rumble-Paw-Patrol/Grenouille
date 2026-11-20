"""File d'écoute `piste_cassee` : les extraits annotés en écoutant une piste cassée, à réécouter
sur la piste valable.

Un extrait annoté s'encode sur la piste écoutée (`channels.annotation_channels`). Avant les
règles de piste et le repli du son étouffé, le poste ouvrait tout sur la piste 0 : des extraits
ont été annotés « rien » sur une piste 0 sourde (Kaw D SMA14079, règle « piste 0 sourde ») ou
étouffée avec une piste 1 saine (repli, n° 203). Leur label ne dit rien de la piste 1, et ils
seraient encodés sur la piste cassée. Le gain inversé (piste 0 à 18 dB) n'est pas une casse :
ces extraits gardent leur piste écoutée.

La file reprend chaque extrait aux bornes de son dernier envoi (`source` = plan), avec la
colonne `channel` : le poste l'ouvre sur cette piste (`workbench.queue_track`). Le réenvoi
remplace l'ancien et dit la nouvelle piste : l'enregistrement passe de lui-même sur la piste
valable à l'encodage.

Lecture seule de la base ; écrit la file dans `paths.queues`.

    uv run python scripts/encodage/file_piste_cassee.py [--config config/local.yaml]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from blanci.annotation.selection import write_queue
from blanci.core.config import config_path, default_user_config, load_config
from blanci.embedding.channels import (
    channel_rules,
    expected_channels,
    listened_channels,
    matching_rule,
)
from blanci.inputs.qc import parse_flags

NAME = "piste_cassee"
# Raisons de règle qui disent une piste cassée (les autres, comme le gain inversé, non).
BROKEN_REASONS = ("piste 0 sourde",)


def broken_listened(con: sqlite3.Connection, cfg: dict) -> pd.DataFrame:
    """Enregistrements annotés dont la piste écoutée est cassée : la règle ou le repli du son
    étouffé (sans la piste écoutée) en choisit une autre, pour une piste sourde ou étouffée.
    Colonnes `recording_id`, `listened`, `valid` (la piste valable), `cause`."""
    raw = cfg["audio"].get("channel_rules") or []
    rules = channel_rules(raw)
    channel = cfg["audio"]["channel"]
    heard = listened_channels(con).dropna(subset=["listened"])
    recs = pd.read_sql_query(
        "SELECT recording_id, dataset, site, mic_id, start_utc, channels, qc_flags FROM recordings",
        con,
    )
    recs = recs[recs["recording_id"].isin(heard["recording_id"])]
    valid = expected_channels(recs, channel, rules, cfg["qc"])  # sans la piste écoutée
    without_rules = expected_channels(recs, channel, [], cfg["qc"])
    rows = []
    for rec in recs.to_dict("records"):
        rid = rec["recording_id"]
        listened = heard.loc[heard["recording_id"] == rid, "listened"].iloc[0]
        if listened == valid[rid]:
            continue
        i = matching_rule(rec, rules)
        flags = parse_flags(rec["qc_flags"])
        if i is not None and raw[i].get("reason") in BROKEN_REASONS:
            cause = f"règle : {raw[i]['reason']}"
        elif flags.get(f"muffled_{listened}") and without_rules[rid] == valid[rid]:
            # repli du son étouffé : la piste écoutée est étouffée, l'autre lue et saine
            cause = f"piste {listened} étouffée, piste {valid[rid]} saine"
        else:
            continue  # gain inversé : la piste écoutée n'est pas cassée
        rows.append(
            {"recording_id": rid, "listened": listened, "valid": valid[rid], "cause": cause}
        )
    return pd.DataFrame(rows, columns=["recording_id", "listened", "valid", "cause"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, help="défaut : config/local.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config or default_user_config())
    db = config_path(cfg, "db")
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    broken = broken_listened(con, cfg)
    spans = pd.read_sql_query(
        "SELECT span_id, recording_id, start_s, end_s FROM spans ORDER BY span_id", con
    )
    spans = spans[spans["recording_id"].isin(broken["recording_id"])]
    spans = spans.assign(start_s=spans["start_s"].round(2), end_s=spans["end_s"].round(2))
    last = spans.drop_duplicates(["recording_id", "start_s", "end_s"], keep="last")
    queue = last.merge(broken, on="recording_id")
    queue = pd.DataFrame(
        {
            "recording_id": queue["recording_id"],
            "offset_s": queue["start_s"],
            "dur_s": (queue["end_s"] - queue["start_s"]).round(2),
            "channel": queue["valid"].astype(int),
            "reason": "piste_cassee/" + queue["cause"],
            "source": "plan",
        }
    ).sort_values(["reason", "recording_id", "offset_s"])
    write_queue(
        cfg,
        queue,
        NAME,
        {
            "origine": "scripts/encodage/file_piste_cassee.py (piste écoutée = piste encodée, "
            "20/11/2026)",
            "contenu": f"{len(queue)} extraits de {queue['recording_id'].nunique()} "
            "enregistrements annotés en écoutant une piste 0 cassée (sourde, ou étouffée avec "
            "la piste 1 saine) ; aux bornes de leur dernier envoi, ouverts sur la piste valable "
            "(colonne channel) ; le réenvoi les fait encoder sur cette piste",
        },
    )
    print(broken.groupby("cause").size().to_string())
    print(f"{len(queue)} extraits, {queue['recording_id'].nunique()} enregistrements")


if __name__ == "__main__":
    main()
