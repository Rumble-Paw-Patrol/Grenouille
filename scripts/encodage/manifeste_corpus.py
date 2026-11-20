"""Manifeste du corpus entériné : chaque enregistrement retenu, la piste sur laquelle il s'encode,
et l'empreinte de la base qui l'a produit.

Le contrôle qualité est clos (20/11/2026) : ce manifeste fixe le corpus une fois pour toutes. Un
enregistrement y est s'il est gardé par les drapeaux (`embed.select_recordings` : inventaire,
son étouffé, relevés, drapeaux confirmés à l'écoute, `keep`, commentaires « poubelle »…). Sa piste
est celle que `blanci embed` lira (`channels.expected_channels`, avec `annotation_channels`) :
règle `broken` (Kaw D), puis piste dite en commentaire, piste écoutée (cassée : la piste
valable), puis `audio.channel_rules` et le repli du son étouffé.

Colonnes de `corpus.csv` : recording_id, dataset, site, mic_id, start_utc, duration_s, piste
(« 0 », « 1 », comme `stock_channels`), origine (« règle broken », « commentaire », « piste
cassée », « intervalles », « envoi », « règle : <raison> », « repli son étouffé », « défaut »),
positif (A. blanci entendu). `LISEZMOI.md` : comptes, empreinte (`database_fingerprint`), commit.

Lecture seule de la base. Écrit `annotations/corpus_<aaaammjj>/`.

    uv run python scripts/encodage/manifeste_corpus.py [--config config/local.yaml]
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

from blanci.core.config import config_path, default_user_config, load_config
from blanci.embedding.channels import (
    channel_key,
    channel_rules,
    expected_channels,
    listened_channels,
    matching_rule,
)
from blanci.embedding.embed import select_recordings
from blanci.exploration.inventaire import database_fingerprint
from blanci.inputs.qc import positive_recordings

REPO = Path(__file__).resolve().parents[2]


def fr(x: float, digits: int = 1) -> str:
    """Nombre à la française : espace des milliers, virgule décimale."""
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def corpus(con: sqlite3.Connection, cfg: dict) -> pd.DataFrame:
    """Le corpus retenu et la piste d'encodage de chaque enregistrement, avec son origine."""
    raw = cfg["audio"].get("channel_rules") or []
    rules = channel_rules(raw)
    channel = cfg["audio"]["channel"]
    recs = select_recordings(con)
    heard = listened_channels(con, thresholds=cfg["qc"]).dropna(subset=["listened"])
    listened = {r: int(c) for r, c in zip(heard["recording_id"], heard["listened"], strict=True)}
    source = dict(zip(heard["recording_id"], heard["source"], strict=True))
    tracks = expected_channels(recs, channel, rules, cfg["qc"], listened)
    plain = expected_channels(recs, channel, [], cfg["qc"])
    origins = []
    for rec in recs.to_dict("records"):
        rid, i = rec["recording_id"], matching_rule(rec, rules)
        if i is not None and rules[i].get("broken"):
            origins.append("règle broken")
        elif rid in listened:
            origins.append(source[rid])
        elif i is not None:
            origins.append(f"règle : {raw[i].get('reason', i)}")
        elif plain[rid] != channel_key(channel, rec.get("channels")):
            origins.append("repli son étouffé")
        else:
            origins.append("défaut")
    out = recs[["recording_id", "dataset", "site", "mic_id", "start_utc", "duration_s"]].copy()
    out["piste"] = out["recording_id"].map(tracks)
    out["origine"] = origins
    out["positif"] = out["recording_id"].isin(positive_recordings(con))
    return out.sort_values(["dataset", "site", "mic_id", "start_utc"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, help="défaut : config/local.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config or default_user_config())
    db = config_path(cfg, "db")
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    table = corpus(con, cfg)
    fp = database_fingerprint(con)
    total = con.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    out = REPO / "annotations" / f"corpus_{date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "corpus.csv", index=False)
    hours = table["duration_s"].sum() / 3600
    by_set = table.groupby("dataset").agg(n=("recording_id", "size"), h=("duration_s", "sum"))
    lines = [
        f"# Corpus entériné du {date.today():%d/%m/%Y}",
        "",
        "Contrôle qualité clos. `corpus.csv` : les enregistrements que garde le corpus et la piste "
        "sur laquelle chacun s'encode (`scripts/encodage/manifeste_corpus.py`).",
        "",
        f"- {fr(len(table), 0)} enregistrements retenus sur {fr(total, 0)} en base, "
        f"{fr(hours)} h ; {int(table['positif'].sum())} avec A. blanci entendu.",
        *(
            f"- jeu {d} : {fr(r.n, 0)} enregistrements, {fr(r.h / 3600)} h"
            for d, r in by_set.iterrows()
        ),
        "",
        "## Piste d'encodage",
        "",
        *(
            f"- piste {p}, {o} : {fr(n, 0)}"
            for (p, o), n in table.groupby(["piste", "origine"]).size().items()
        ),
        "",
        "## Empreinte",
        "",
        f"- base : `{db}` (migration {fp['user_version']})",
        f"- inventaire : `{fp['inventory_sha256']}`",
        f"- drapeaux : `{fp['flags_sha256']}`",
        f"- dernière annotation : extrait {fp['last_span']}, fenêtre {fp['last_label']}",
        f"- code : commit `{commit}` ; configuration : `config/default.yaml` de ce commit",
        "",
        "Une base dont l'empreinte des drapeaux diffère n'a plus ce corpus : ne relancer ni "
        "`blanci flag`, ni `blanci qc`, ni `blanci ingest`. Une annotation sur l'autre piste d'un "
        "enregistrement déjà encodé change sa piste (`blanci pistes`).",
        "",
    ]
    (out / "LISEZMOI.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
