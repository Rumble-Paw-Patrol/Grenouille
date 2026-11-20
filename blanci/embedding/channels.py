"""Piste lue pour chaque enregistrement, et enregistrements d'un stock encodés sur une autre.

La piste par défaut est `audio.channel` (0 : premier micro, réglé à 6 dB). Deux choses la changent :

- une règle de `audio.channel_rules` : pour ces micros (sur ce site, ce jeu, cette période), lire
  telle piste. Micro sourd (Kaw D SMA14079 en 2024), gain inversé (piste 0 réglée à 18 dB) ;
- le repli du son étouffé : la piste choisie est étouffée et l'autre est lue et saine (indices
  rangés, non étouffée), on lit l'autre (`muffled_0` sans `muffled_1`, DECISIONS n° 203 ; et
  l'inverse quand la règle choisit la piste 1). Une piste non lue ne sert pas de repli.

Chaque stock range la piste lue de chaque enregistrement (table `stock_channels`, rangée par
`embed`). Un enregistrement encodé avant cette table a la piste que donnent les réglages rangés
avec son stock (`legacy_channels`). `channel_plan` compare, pour tout le stock, la piste rangée
à celle de la règle ; `embed` refuse un stock qui en contient d'autres (`check_stock_channels`),
et `drop_recordings` les retire de leur partition pour que la reprise de `embed` les réencode
(elle saute un enregistrement déjà présent dans le stock).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from blanci.embedding.store import PART_MARK, WRITE_OPTIONS, EmbeddingStore
from blanci.inputs.qc import parse_flags

# Piste 0 étouffée et piste 1 valable (`muffled_0` sans `muffled_1`, DECISIONS n° 203) :
# l'enregistrement est encodé sur la piste 1.
MUFFLED_0_CHANNEL = 1
RULE_KEYS = ("dataset", "site", "mic_id", "from", "to", "channel", "reason")


def _utc(value: Any) -> str:
    """Instant UTC au format de `recordings.start_utc` (« 2024-05-23T11:00:00Z ») : les chaînes
    se comparent alors dans l'ordre du temps. Une heure sans fuseau est lue en UTC."""
    t = pd.Timestamp(value)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def channel_rules(raw: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Règles de `audio.channel_rules`, vérifiées et mises sous une forme fixe : `mic_id` en
    liste triée, `from` / `to` en UTC, `reason` retirée (un commentaire ne change pas la piste).
    C'est cette forme qui entre dans l'identité du stock."""
    rules = []
    for i, rule in enumerate(raw or []):
        unknown = set(rule) - set(RULE_KEYS)
        if unknown:
            raise ValueError(f"audio.channel_rules[{i}] : clés inconnues {sorted(unknown)}")
        if "mic_id" not in rule or "channel" not in rule:
            raise ValueError(f"audio.channel_rules[{i}] : mic_id et channel sont obligatoires")
        if rule["channel"] not in (0, 1):
            raise ValueError(f"audio.channel_rules[{i}] : channel vaut 0 ou 1")
        mics = rule["mic_id"]
        out: dict[str, Any] = {
            "mic_id": sorted({str(m) for m in ([mics] if isinstance(mics, str) else mics)}),
            "channel": int(rule["channel"]),
        }
        for key in ("dataset", "site"):
            if rule.get(key) is not None:
                out[key] = str(rule[key])
        for key in ("from", "to"):
            if rule.get(key) is not None:
                out[key] = _utc(rule[key])
        if "from" in out and "to" in out and out["from"] >= out["to"]:
            raise ValueError(f"audio.channel_rules[{i}] : from doit précéder to")
        rules.append(out)
    return rules


def matching_rule(rec: Mapping[str, Any], rules: list[dict[str, Any]]) -> int | None:
    """Rang de la première règle qui s'applique à l'enregistrement (`mic_id`, et s'ils sont
    dits : `dataset`, `site`, début dans [`from`, `to`[), None si aucune."""
    for i, rule in enumerate(rules):
        if str(rec.get("mic_id")) not in rule["mic_id"]:
            continue
        if any(k in rule and str(rec.get(k)) != rule[k] for k in ("dataset", "site")):
            continue
        if "from" in rule or "to" in rule:
            if not rec.get("start_utc"):
                continue
            start = _utc(rec["start_utc"])
            if start < rule.get("from", start) or ("to" in rule and start >= rule["to"]):
                continue
        return i
    return None


def rule_channel(
    rec: Mapping[str, Any], rules: list[dict[str, Any]], default: int | str
) -> int | str:
    """Piste que donne la première règle qui s'applique à l'enregistrement ; `default` sinon."""
    i = matching_rule(rec, rules)
    return default if i is None else rules[i]["channel"]


def recording_channel(
    flags: Mapping[str, Any],
    channel: int | str,
    rec: Mapping[str, Any] | None = None,
    rules: list[dict[str, Any]] | None = None,
    fallback: bool = True,
) -> int | str:
    """Canal lu pour un enregistrement : celui de la règle qui s'applique (`rule_channel`), sinon
    `channel` ; puis, avec `fallback`, l'autre piste si celle-ci est étouffée et que l'autre est
    **lue et saine** (indices rangés, pas étouffée ; n° 203, précisé le 20/11/2026). Une piste
    1 que `blanci qc` n'a pas encore lue (`indices_1` absent) ne vaut pas la piste choisie :
    on ne change de piste qu'à bon escient. Un fichier mono est lu tel quel quel que soit le
    canal (`load_audio`)."""
    chosen = rule_channel(rec, rules, channel) if rec is not None and rules else channel
    if not fallback:
        return chosen
    m0, m1 = bool(flags.get("muffled_0")), bool(flags.get("muffled_1"))
    read0, read1 = "indices" in flags, "indices_1" in flags
    if chosen == 1:
        return 0 if m1 and not m0 and read0 else 1
    return MUFFLED_0_CHANNEL if m0 and not m1 and read1 else chosen


def channel_key(channel: int | str, channels: Any = 2) -> str:
    """Piste telle que rangée dans `stock_channels` : « 0 », « 1 » ou « mean » ; « 0 » pour un
    fichier mono, qui n'a qu'une piste quel que soit le canal demandé."""
    try:
        mono = int(channels) == 1
    except (TypeError, ValueError):
        mono = False
    return "0" if mono else str(channel)


def _channels_of(
    recordings: pd.DataFrame, channel: int | str, rules: list[dict[str, Any]], fallback: bool
) -> pd.Series:
    out = {
        rec["recording_id"]: channel_key(
            recording_channel(parse_flags(rec.get("qc_flags")), channel, rec, rules, fallback),
            rec.get("channels"),
        )
        for rec in recordings.to_dict("records")
    }
    return pd.Series(out, dtype=object)


def expected_channels(
    recordings: pd.DataFrame, channel: int | str, rules: list[dict[str, Any]]
) -> pd.Series:
    """Piste à lire pour chaque enregistrement (index : `recording_id`), règles et repli compris."""
    return _channels_of(recordings, channel, rules, fallback=True)


def legacy_channels(
    recordings: pd.DataFrame, stored: Mapping[str, Any], channel: int | str
) -> pd.Series:
    """Piste lue pour un enregistrement encodé avant la table `stock_channels`, d'après les
    réglages rangés avec son stock : son canal (`channel` s'il n'est pas rangé, comme le
    faisait `check_stock_identity`), ses règles (aucune avant leur création), le repli du son
    étouffé s'il y était (`muffled_0_channel`, rangé depuis le n° 203). Les drapeaux sont ceux
    d'aujourd'hui : un drapeau `muffled_0` posé après l'encodage n'est pas vu."""
    rules = stored.get("channel_rules") or []
    return _channels_of(
        recordings, stored.get("channel", channel), rules, "muffled_0_channel" in stored
    )


# --- Contenu d'un stock -----------------------------------------------------------------------


def stock_contents(store: EmbeddingStore) -> pd.DataFrame:
    """Enregistrements d'un stock et leurs fenêtres, partition par partition (morceaux d'un
    passage interrompu compris) : `recording_id`, `dataset`, `site`, `month`, `windows`. Ne lit
    que la colonne `recording_id`."""
    rows = []
    for path in store.fragments():
        ids = pq.read_table(path, columns=["recording_id"]).column("recording_id")
        counts = pd.Series(ids.to_pylist()).value_counts()
        month = path.stem.split(PART_MARK)[0]
        for rid, n in counts.items():
            rows.append((rid, path.parent.parent.name, path.parent.name, month, int(n)))
    columns = ["recording_id", "dataset", "site", "month", "windows"]
    df = pd.DataFrame(rows, columns=columns)
    return df.groupby(columns[:4], as_index=False)["windows"].sum()


def recorded_channels(con: sqlite3.Connection, encoder_id: str) -> pd.Series:
    """Piste rangée par `embed` pour chaque enregistrement du stock (index : `recording_id`) ;
    vide dans une base ouverte en lecture seule d'avant la table (migration 6)."""
    try:
        rows = con.execute(
            "SELECT recording_id, channel FROM stock_channels WHERE encoder_id = ?", (encoder_id,)
        ).fetchall()
    except sqlite3.OperationalError:  # no such table
        rows = []
    return pd.Series({r[0]: r[1] for r in rows}, dtype=object)


def record_channels(
    con: sqlite3.Connection, encoder_id: str, channels: Mapping[str, str], replace: bool = True
) -> None:
    """Range la piste lue de chaque enregistrement ; `replace=False` garde une piste déjà rangée."""
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    con.executemany(
        f"{verb} INTO stock_channels (encoder_id, recording_id, channel) VALUES (?, ?, ?)",
        [(encoder_id, rid, str(ch)) for rid, ch in channels.items()],
    )
    con.commit()


def _stored_identity(con: sqlite3.Connection, encoder_id: str) -> dict[str, Any]:
    row = con.execute(
        "SELECT params_json FROM models WHERE model_id = ? AND kind = 'encoder'", (encoder_id,)
    ).fetchone()
    return json.loads(row[0] or "{}") if row else {}


def channel_plan(
    con: sqlite3.Connection,
    store: EmbeddingStore,
    channel: int | str,
    rules: list[dict[str, Any]],
    contents: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pour chaque enregistrement du stock : la piste encodée (`encoded` : rangée, sinon
    `legacy_channels`), celle de la règle (`expected`), `recorded` (piste rangée en base),
    `stale` (les deux diffèrent), avec sa partition, ses fenêtres, son micro et sa date. Un
    enregistrement du stock absent de la table `recordings` est laissé de côté."""
    contents = stock_contents(store) if contents is None else contents
    columns = [*contents.columns, "mic_id", "start_utc", "encoded", "expected", "recorded", "stale"]
    if contents.empty:
        return pd.DataFrame(columns=columns)
    recs = pd.read_sql_query(
        "SELECT recording_id, dataset, site, mic_id, start_utc, channels, qc_flags FROM recordings",
        con,
    )
    recs = recs[recs["recording_id"].isin(contents["recording_id"])]
    recorded = recorded_channels(con, store.encoder_id)
    legacy = legacy_channels(
        recs[~recs["recording_id"].isin(recorded.index)],
        _stored_identity(con, store.encoder_id),
        channel,
    )
    expected = expected_channels(recs, channel, rules)
    out = contents.merge(recs[["recording_id", "mic_id", "start_utc"]], on="recording_id")
    out["recorded"] = out["recording_id"].isin(recorded.index)
    out["encoded"] = out["recording_id"].map(pd.concat([recorded, legacy]))
    out["expected"] = out["recording_id"].map(expected)
    out["stale"] = out["encoded"] != out["expected"]
    return out[columns]


def check_stock_channels(
    con: sqlite3.Connection,
    store: EmbeddingStore,
    channel: int | str,
    rules: list[dict[str, Any]],
    command: str = "blanci pistes",
) -> pd.DataFrame:
    """ValueError si le stock contient des enregistrements encodés sur une autre piste que celle
    de la règle : la reprise les sauterait, et le stock mêlerait les deux pistes. Sinon, range
    la piste des enregistrements encodés avant la table `stock_channels` (celle de
    `legacy_channels`) : un changement de règle ou de drapeau sera jugé contre elle. Rend le
    plan (`channel_plan`)."""
    plan = channel_plan(con, store, channel, rules)
    stale = plan[plan["stale"]]
    if len(stale):
        by = stale.groupby(["encoded", "expected"]).size()
        detail = ", ".join(f"{n} de la piste {a} à la piste {b}" for (a, b), n in by.items())
        raise ValueError(
            f"le stock {store.encoder_id} contient {stale['recording_id'].nunique()} "
            f"enregistrements encodés sur une autre piste que celle de la règle ({detail} ; "
            f"{int(stale['windows'].sum())} fenêtres) : la reprise les sauterait. "
            f"`{command}` montre ce qui serait retiré, `--apply` le retire ; puis relancer "
            "l'encodage, qui les réencode"
        )
    legacy = plan[~plan["recorded"]].drop_duplicates("recording_id")
    if len(legacy):
        found = dict(zip(legacy["recording_id"], legacy["encoded"], strict=True))
        record_channels(con, store.encoder_id, found, replace=False)
    return plan


# --- Retrait ----------------------------------------------------------------------------------


def archive_store(store: EmbeddingStore, day: str) -> EmbeddingStore:
    """Où vont les fenêtres retirées d'un stock : `<racine>/_mis_de_cote/<stock>_pistes_<jour>`,
    un stock comme un autre (mêmes partitions, `EmbeddingStore.load`), hors des stocks que lisent
    `embed`, les têtes et `blanci pistes`. Sert à comparer les scores sur les anciens et les
    nouveaux embeddings."""
    return EmbeddingStore(Path(store.root) / "_mis_de_cote", f"{store.encoder_id}_pistes_{day}")


def drop_from_partition(
    store: EmbeddingStore,
    dataset: str,
    site: str,
    month: str,
    recording_ids: Iterable[str],
    keep: EmbeddingStore | None = None,
) -> int:
    """Retire d'une partition les fenêtres de ces enregistrements ; rend le nombre retiré.

    Les morceaux d'un passage interrompu sont d'abord fondus (`consolidate`). La partition est
    recopiée groupe de lignes par groupe de lignes, sans être chargée, dans un fichier
    temporaire qui la remplace d'un coup : interrompu, le retrait laisse la partition entière.
    Une partition vidée est supprimée. Avec `keep`, les fenêtres retirées vont d'abord dans la
    même partition de ce stock (`archive_store`), écrite en entier avant que la partition
    d'origine ne soit remplacée ; un retrait refait après une coupure les y remet sans doublon
    (`consolidate` garde la dernière écriture d'une fenêtre)."""
    path = store.consolidate(dataset, site, month)
    wanted = pa.array(sorted(set(recording_ids)), type=pa.string())
    if not path.exists() or not len(wanted):
        return 0
    tmp = path.with_suffix(".parquet.tmp")
    kept_tmp = None
    if keep is not None:
        target = keep.partition_path(dataset, site, month)
        target.parent.mkdir(parents=True, exist_ok=True)
        kept_tmp = target.with_suffix(".parquet.tmp")
    removed = kept = 0
    with pq.ParquetFile(path) as source:
        schema = source.schema_arrow
        writer = pq.ParquetWriter(tmp, schema, **WRITE_OPTIONS)
        archive = pq.ParquetWriter(kept_tmp, schema, **WRITE_OPTIONS) if kept_tmp else None
        try:
            for group in range(source.num_row_groups):
                table = source.read_row_group(group)
                ids = table.column("recording_id")
                if pa.types.is_dictionary(ids.type):
                    ids = ids.cast(pa.string())
                drop = pc.is_in(ids, value_set=wanted)
                n = int(pc.sum(drop).as_py() or 0)
                if n and archive is not None:
                    archive.write_table(table.filter(drop))
                removed += n
                table = table.filter(pc.invert(drop))
                kept += table.num_rows
                if table.num_rows:
                    writer.write_table(table)
        finally:
            writer.close()
            if archive is not None:
                archive.close()
    if kept_tmp is not None:
        if removed:
            parts = keep.part_paths(dataset, site, month)
            number = int(parts[-1].stem.split(PART_MARK)[1]) + 1 if parts else 1
            kept_tmp.replace(target.with_name(f"{month}{PART_MARK}{number:06d}.parquet"))
            keep.consolidate(dataset, site, month)
        else:
            kept_tmp.unlink()
    if not removed:
        tmp.unlink()
    elif not kept:
        tmp.unlink()
        path.unlink()
    else:
        tmp.replace(path)
    return removed


def drop_recordings(
    con: sqlite3.Connection,
    store: EmbeddingStore,
    plan: pd.DataFrame,
    keep: EmbeddingStore | None = None,
) -> pd.DataFrame:
    """Retire du stock les enregistrements du plan (colonnes `recording_id`, `dataset`, `site`,
    `month`), partition par partition, puis leur piste rangée (`stock_channels`). Avec `keep`
    (`archive_store`), les fenêtres retirées y sont gardées, et le plan à côté (`retrait.csv` :
    piste encodée et piste de la règle de chaque enregistrement). La table `windows` n'est pas
    touchée : ses lignes ne dépendent pas de la piste, servent aux autres stocks de même grille
    et portent les labels. Rend, par partition, les fenêtres retirées."""
    if keep is not None and len(plan):
        keep.directory.mkdir(parents=True, exist_ok=True)
        listing = keep.directory / "retrait.csv"
        old = pd.read_csv(listing, dtype=str) if listing.exists() else pd.DataFrame()
        new = plan.drop(columns=["stale"], errors="ignore").astype(str)
        pd.concat([old, new], ignore_index=True).drop_duplicates(
            "recording_id", keep="last"
        ).to_csv(listing, index=False)
    rows = []
    for (dataset, site, month), part in plan.groupby(["dataset", "site", "month"]):
        removed = drop_from_partition(store, dataset, site, month, part["recording_id"], keep)
        rows.append((dataset, site, month, part["recording_id"].nunique(), removed))
    con.executemany(
        "DELETE FROM stock_channels WHERE encoder_id = ? AND recording_id = ?",
        [(store.encoder_id, rid) for rid in plan["recording_id"].unique()],
    )
    con.commit()
    return pd.DataFrame(rows, columns=["dataset", "site", "month", "recordings", "windows"])
