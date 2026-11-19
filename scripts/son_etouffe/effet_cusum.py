# ruff: noqa: E501  (le rapport Markdown écrit un paragraphe par ligne)
"""Effet de la carte CUSUM du son étouffé sur la base, avant de l'y écrire, et sa calibration.

Lecture seule : la table `recordings` est copiée en mémoire et `apply_audio_flags` (la fonction
même de `blanci flag` et de `blanci qc`, avec le réglage de `qc` de la configuration) y est
appliquée ; la base n'est jamais écrite. On compare les drapeaux rangés (règle d'avant) à ceux
que la carte poserait : enregistrements qui changent de statut (écarté / gardé), lesquels (site,
micro, jour), positifs touchés, piste encodée qui change.

Calibration (même copie) : pour une grille de références et de seuils, `muffled_cusum` sur la
piste 0 et sur les deux pistes, comparée aux cas écoutés :
- étouffés à l'oreille : extraits dont le commentaire dit « étouffé », « poubelle », « dans un
  sac »… (`annotations/extraits_annotes.csv`, « idem » résolu par le commentaire précédent,
  comme le § 4 du notebook 06), qui ont des indices audio ;
- tirage au hasard parmi les gardés (lots 1 et 2, jeu de test : 674 écoutés) : étouffés à
  l'oreille et jugés bons ;
- positifs (A. blanci entendu) ;
- micros sans aucun épisode connu (ni `muffled_0`, ni `muffled_1`, ni `confirmed_muffled`) :
  ce que la carte y retient, et ce qu'elle y retiendrait par hasard (séries mélangées, 20 fois).

    uv run python scripts/son_etouffe/effet_cusum.py
    options : --config <fichier>, --sortie <fichier .md> (défaut :
    paths.reports/son_etouffe/cusum.md)
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from blanci.core.config import config_path, default_user_config, load_config
from blanci.embedding.channels import channel_rules, recording_channel
from blanci.exploration import inventaire as inv
from blanci.inputs.qc import (
    _track_hf,
    apply_audio_flags,
    cusum_settings,
    is_excluded,
    muffled_cusum,
    parse_flags,
    positive_recordings,
)

ROOT = Path(__file__).resolve().parents[2]
BAD = re.compile(r"(?i)poubelle|muffl|étouff|etouff|dans (?:le |un )?sac|à écarter|dégager")
ONE_TRACK = re.compile(r"(?i)seul(?:e)? (?:le |la )?(?:micro|piste)")
DRAWS = ("lot1", "lot2", "test_v1")
GRID_REF = (0.05, 0.07, 0.10)
GRID_H = (2.0, 3.0, 4.0, 5.0)
PERMUTATIONS = 20
UTC_OFFSET = pd.Timedelta(hours=-3)  # heure locale de Guyane


def copy_recordings(con: sqlite3.Connection) -> sqlite3.Connection:
    """La table `recordings` seule, copiée dans une base en mémoire."""
    (sql,) = con.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'recordings'"
    ).fetchone()
    mem = sqlite3.connect(":memory:")
    mem.execute(sql)
    cols = [r[1] for r in con.execute("PRAGMA table_info(recordings)")]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM recordings").fetchall()
    mem.executemany(
        f"INSERT INTO recordings ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [tuple(r) for r in rows],
    )
    mem.commit()
    return mem


def table(con: sqlite3.Connection) -> pd.DataFrame:
    rows = con.execute(
        "SELECT recording_id, dataset, site, mic_id, start_utc, channels, qc_flags "
        "FROM recordings"
    ).fetchall()
    df = pd.DataFrame(
        [tuple(r) for r in rows],
        columns=["recording_id", "dataset", "site", "mic_id", "start_utc", "channels", "qc"],
    )
    df["fl"] = df["qc"].map(parse_flags)
    return df.drop(columns="qc")


def listened_sets() -> dict[str, set[str]]:
    """Étouffés à l'oreille, « une seule piste étouffée », tirages au hasard écoutés."""
    spans = pd.read_csv(ROOT / "annotations" / "extraits_annotes.csv")
    spans = spans[spans["remplace_par"].isna()].sort_values(["recording_id", "created_at"])
    resolved, last = [], {}
    for r in spans.itertuples():
        c = str(r.comment) if pd.notna(r.comment) else ""
        if re.match(r"(?i)^idem\b", c):
            c = last.get(r.recording_id, "") + " [idem]"
        if c:
            last[r.recording_id] = c
        resolved.append(c)
    spans = spans.assign(c=resolved)
    noted = spans["c"].str.contains(BAD)
    one = set(spans.loc[noted & spans["c"].str.contains(ONE_TRACK), "recording_id"])
    draws = set()
    for q in DRAWS:
        draws |= set(pd.read_csv(ROOT / "annotations" / "files_ecoute" / q / "candidats.csv")
                     ["recording_id"])
    return {"bad": set(spans.loc[noted, "recording_id"]) - one, "one": one,
            "draws": draws & set(spans["recording_id"])}


def series(df: pd.DataFrame, qc: dict) -> pd.DataFrame:
    """hf_ratio de chaque piste pour la carte (NaN : non lue ou muette), comme
    `apply_audio_flags`."""
    measured = df[df["fl"].map(lambda f: "indices" in f)].copy()
    measured["hf_0"] = measured["fl"].map(lambda f: _track_hf(f["indices"], qc))
    measured["hf_1"] = [
        h0 if ch == 1 else _track_hf(f.get("indices_1"), qc)
        for f, ch, h0 in zip(measured["fl"], measured["channels"], measured["hf_0"], strict=True)
    ]
    return measured


def retained(s: pd.DataFrame, track: int, ref: float, h: float, max_missed: int) -> set[str]:
    return muffled_cusum(s.assign(hf_ratio=s[f"hf_{track}"]), ref, h, max_missed)


def calibration(s: pd.DataFrame, sets: dict, positives: set[str], max_missed: int) -> str:
    ids = set(s["recording_id"])
    bad = sets["bad"] & ids
    draws_bad = sets["draws"] & bad
    draws_good = sets["draws"] - sets["bad"] - sets["one"]
    flagged = s["fl"].map(lambda f: bool(f.get("muffled_0") or f.get("muffled_1")
                                         or f.get("confirmed_muffled")))
    keys = ["dataset", "site", "mic_id"]
    sick = s[flagged].groupby(keys).size().index
    healthy = s[~s.set_index(keys).index.isin(sick)]
    rng = np.random.default_rng(0)
    rows = []
    for ref in GRID_REF:
        for h in GRID_H:
            k0 = retained(s, 0, ref, h, max_missed)
            both = k0 & retained(s, 1, ref, h, max_missed)
            k_healthy = retained(healthy, 0, ref, h, max_missed)
            shuffled = []
            for _ in range(PERMUTATIONS):
                mixed = healthy.copy()
                mixed["hf_0"] = mixed.groupby(keys)["hf_0"].transform(
                    lambda x: rng.permutation(x.to_numpy()))
                shuffled.append(len(retained(mixed, 0, ref, h, max_missed)))
            rows.append(
                f"| {ref:.2f} | {h:.0f} | {len(k0)} | {len(both)} | {len(bad & k0)} / {len(bad)} "
                f"| {len(bad & both)} | {len(draws_bad & k0)} / {len(draws_bad)} "
                f"| {len(draws_good & k0)} / {len(draws_good)} | {len(positives & k0)} "
                f"| {len(k_healthy)} | {np.mean(shuffled):.1f} |"
            )
    n_mics = healthy.groupby(keys).ngroups
    head = (
        f"Micros sans épisode connu : {n_mics} ({len(healthy)} enregistrements).\n\n"
        "| référence | h | piste 0 retenue | deux pistes | étouffés à l'oreille (piste 0) "
        "| (deux pistes) | tirage : étouffés (piste 0) | tirage : bons (piste 0) "
        "| positifs (piste 0) | micros sans épisode | idem, séries mélangées (moyenne) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    return head + "\n".join(rows)


def changes(
    before: pd.DataFrame, after: pd.DataFrame, positives: set[str], audio: dict
) -> pd.DataFrame:
    df = before[["recording_id", "dataset", "site", "mic_id", "start_utc", "channels", "fl"]]
    df = df.merge(after[["recording_id", "fl"]].rename(columns={"fl": "fl_new"}),
                  on="recording_id")
    df["pos"] = df["recording_id"].isin(positives)
    channel, rules = audio.get("channel", 0), channel_rules(audio.get("channel_rules"))
    keys = ["dataset", "site", "mic_id", "start_utc"]
    for tag, col in (("old", "fl"), ("new", "fl_new")):
        df[f"m_{tag}"] = df[col].map(lambda f: bool(f.get("muffled")))
        df[f"m0_{tag}"] = df[col].map(lambda f: bool(f.get("muffled_0")))
        df[f"m1_{tag}"] = df[col].map(lambda f: bool(f.get("muffled_1")))
        df[f"ex_{tag}"] = df[col].map(is_excluded) & ~df["pos"]
        df[f"ch_{tag}"] = [
            recording_channel(f, channel, rec, rules)
            for f, rec in zip(df[col], df[keys].to_dict("records"), strict=True)
        ]
    df["unread"] = df["fl_new"].map(
        lambda f: "indices" in f and bool(f.get("muffled_0")) and "indices_1" not in f
    ) & (df["channels"] != 1)
    df["hf0"] = df["fl"].map(lambda f: f.get("indices", {}).get("hf_ratio", np.nan))
    df["hf1"] = df["fl"].map(lambda f: f.get("indices_1", {}).get("hf_ratio", np.nan))
    t = pd.to_datetime(df["start_utc"], utc=True, errors="coerce") + UTC_OFFSET
    df["jour"] = t.dt.strftime("%d/%m/%Y")
    df["heure"] = t.dt.strftime("%H:%M")
    return df


def by_day(df: pd.DataFrame) -> str:
    if df.empty:
        return "aucun\n"
    g = df.groupby(["site", "mic_id", "jour"]).agg(
        n=("recording_id", "size"), de=("heure", "min"), a=("heure", "max"),
        hf0=("hf0", "median"), hf1=("hf1", "median"),
    ).reset_index()
    lines = ["| site | micro | jour (heure locale) | n | de | à | hf piste 0 (méd.) "
             "| hf piste 1 (méd.) |", "|---|---|---|---|---|---|---|---|"]
    for r in g.itertuples():
        hf1 = "—" if pd.isna(r.hf1) else f"{r.hf1:.4f}"
        lines.append(f"| {r.site} | {r.mic_id} | {r.jour} | {r.n} | {r.de} | {r.a} "
                     f"| {r.hf0:.4f} | {hf1} |")
    return "\n".join(lines) + "\n"


def commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "blanci/inputs/qc.py"],
                           cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    return out + (" (qc.py modifié)" if dirty else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=default_user_config())
    ap.add_argument("--sortie", type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    qc = cfg["qc"]
    settings = cusum_settings(qc)
    out = args.sortie or config_path(cfg, "reports") / "son_etouffe" / "cusum.md"

    con = inv.open_database(cfg)
    print("empreinte de la base…", flush=True)
    empreinte = inv.database_fingerprint(con)
    positives = positive_recordings(con)
    mem = copy_recordings(con)
    before = table(mem)
    counts = apply_audio_flags(mem, qc)
    after = table(mem)
    df = changes(before, after, positives, cfg["audio"])
    s = series(before, qc)
    print("calibration…", flush=True)
    calib = calibration(s, listened_sets(), positives, settings["max_missed"])

    enter = df[df["ex_new"] & ~df["ex_old"]]
    leave = df[~df["ex_new"] & df["ex_old"]]
    m_plus = df[df["m_new"] & ~df["m_old"]]
    m_minus = df[~df["m_new"] & df["m_old"]]
    kept = ~df["ex_new"] & ~df["ex_old"]
    channel = df[kept & (df["ch_new"] != df["ch_old"])]
    touched = df[df["pos"] & ((df["m_new"] != df["m_old"]) | (df["m0_new"] != df["m0_old"])
                              | (df["m1_new"] != df["m1_old"]))]
    changed = df[(df["m_new"] != df["m_old"]) | (df["m0_new"] != df["m0_old"])
                 | (df["m1_new"] != df["m1_old"])]
    per_site = Counter(changed["dataset"] + " " + changed["site"])

    text = f"""# Son étouffé : la carte CUSUM à la place de « 4 de suite »

Généré le {datetime.now():%d/%m/%Y %H:%M} par `scripts/son_etouffe/effet_cusum.py` (commit {commit()}), en lecture seule : `apply_audio_flags` appliquée à une copie en mémoire de `recordings`. Base : empreinte des drapeaux `{empreinte["flags_sha256"][:16]}`, de l'inventaire `{empreinte["inventory_sha256"][:16]}`.

## Le réglage

- **Série** : par micro (jeu, site, micro) et par piste, dans l'ordre du programme du micro (`qc._programme_ranks`) : créneaux lus dans les données (présents au moins un jour sur deux, grille au pas médian, 30 min), le dernier du soir et le premier du lendemain se suivent. La nuit ne coupe plus la série ; plus de {settings["max_missed"]} créneau manqué la coupe (comme l'ancien trou d'une heure). Une piste non lue ou muette compte comme manquée.
- **Carte** : CUSUM de Page (1954) sur y = log10(hf_ratio), plancher 1e-4 ; S = max(0, S + log10(r) − y).
- **Référence r = {settings["ref_ratio"]:g}** : milieu, en log, entre une piste saine (médiane 0,75) et une piste dans un sac (médiane 0,0035), soit 0,051 : la valeur de référence de Page pour ce saut. Le seuil de 5 % réglé à la main y trouve sa justification.
- **Seuil h = {settings["h"]:g}** : un sac typique (+1,16 décade par enregistrement) est retenu au 4e enregistrement, comme l'ancienne règle ; un enregistrement seul ne peut jamais l'être (apport au plus log10(0,05 / 1e-4) = 2,7). Sur les micros sans épisode connu, h = 3 ou 4 ne retient rien (tableau de calibration).
- **Épisode** : le cœur (excursion qui alarme, lue dans le sens du temps et à l'envers) en entier ; la traîne d'un seul sens, seulement sous la référence.
- **Deux pistes** : inchangé (n° 203), `muffled` si les deux pistes sont retenues ; la piste 1 encodée si seule la piste 0 l'est.

## Effet sur la base

| | avant | carte CUSUM |
|---|---|---|
| piste 0 retenue (`muffled_0`) | {int(df["m0_old"].sum())} | {counts["muffled_0"]} |
| piste 1 retenue (`muffled_1`) | {int(df["m1_old"].sum())} | {counts["muffled_1"]} |
| deux pistes (`muffled`) | {int(df["m_old"].sum())} | {counts["muffled"]} |
| écartés du corpus (tous drapeaux, positifs protégés) | {int(df["ex_old"].sum())} | {int(df["ex_new"].sum())} |

- `muffled` : +{len(m_plus)}, −{len(m_minus)} ; écartés : +{len(enter)}, −{len(leave)} (les autres changements de `muffled` portent sur des enregistrements déjà écartés par un autre drapeau, surtout `confirmed_muffled`).
- Enregistrements dont un drapeau du son étouffé change, par jeu et site : {", ".join(f"{k} {v}" for k, v in per_site.most_common()) or "aucun"}.
- Positifs connus touchés (un drapeau du son étouffé change) : {len(touched)} (jamais écartés de toute façon).
- Gardés dont la piste encodée change (`channels.recording_channel`, règles `audio.channel_rules` comprises) : {len(channel)}.
- Piste 0 retenue mais piste 1 jamais lue : {int(df["unread"].sum())} ; ils gardent leur `muffled` d'avant jusqu'à ce que `blanci qc` lise leur piste 1 (disque du relevé branché).

### Écartés en plus

{by_day(enter)}
### Plus écartés

{by_day(leave)}
### `muffled` en plus (dont déjà écartés par un autre drapeau)

{by_day(m_plus)}
### `muffled` en moins

{by_day(m_minus)}
### Gardés dont la piste encodée change

{by_day(channel)}
### Piste 0 retenue, piste 1 jamais lue

{by_day(df[df["unread"]])}
## Calibration

{calib}

Lecture : « étouffés à l'oreille » = extraits notés étouffés (commentaire), avec indices ; la plupart sont déjà `confirmed_muffled`, donc écartés quel que soit le calcul. Ceux que la carte ne prend pas sur les deux pistes ont une piste 1 entre 0,05 et 0,10 (étouffement partiel). « Tirage » = lots 1 et 2 et jeu de test, tirés au hasard parmi les gardés et écoutés.

## Limites

- La référence 0,05 vise le sac typique : un étouffement léger (vers 0,03) demande une vingtaine d'enregistrements avant l'alarme, un étouffement partiel (0,06 à 0,10) n'est pas pris. Une référence à 0,10 les prendrait, mais retient aussi des épisodes sur des micros sans épisode connu, jamais écoutés (tableau).
- Le programme est lu dans les données : un micro qui n'a jamais tenu son programme a pour programme ce qu'il a fait.
- La piste 1 n'est lue que si la piste 0 est candidate ou retenue, ou voisine d'un candidat piste 1 : la carte de la piste 1 ne voit pas le reste de la série.
- Le réglage est calibré sur peu de cas écoutés (une trentaine d'étouffés à l'oreille, six dans le tirage au hasard), presque tous à CDR en décembre 2025.
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"écrit : {out}")


if __name__ == "__main__":
    main()
