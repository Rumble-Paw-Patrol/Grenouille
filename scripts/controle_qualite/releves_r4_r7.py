"""Contrôle de l'inventaire des relevés arrivés le 23/11/2026 : R4 RHOZACE et R7 Mataroni.

R4 (`audio/R4_2026-02_RHOZACE`) : Gabaret (GAB_164, GAB_166) et sa sous-section Pai (PAI_124),
du 11/02 au 15/04/2026 ; R7 (`audio/R7_2026-05_Mataroni`) complet, du 19 au 28/05/2026. Copiés
du disque Seagate Basic de Biophonia sur BLANCI_2025-26 (`scripts/copie/07`).

Tous les chiffres sortent de `blanci.exploration.inventaire` (fonctions testées), sur la base
citée par son empreinte : effectifs, drapeaux et causes d'exclusion, complétude du programme,
jours vides, noms de fichiers, doublons de contenu, durées anormales, tailles, et sur un
échantillon d'en-têtes (un fichier par micro et par jour, le plus proche de 10 h) : gains,
horloge, position.

Lecture seule (base ouverte en lecture seule, disques lus). Sortie : `paths.reports/releves_r4_r7/`
(`RAPPORT.md` et les tableaux en CSV).

    uv run python scripts/controle_qualite/releves_r4_r7.py [--config config/local_F.yaml]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from blanci.core.config import PROJECT_ROOT, config_path, load_config
from blanci.exploration import inventaire as inv
from blanci.inputs.qc import parse_flags

SURVEYS = ("audio/R4_2026-02_RHOZACE", "audio/R7_2026-05_Mataroni")


def table(df: pd.DataFrame) -> str:
    """Tableau Markdown (sans dépendance : tabulate n'est pas installé)."""
    if not len(df):
        return "(aucun)"
    cells = df.astype(object).where(df.notna(), "").astype(str)
    cells = cells.apply(lambda c: c.str.replace("|", "/", regex=False).str.slice(0, 120))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    return "\n".join([head, rule, *("| " + " | ".join(r) + " |" for r in cells.values)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "local_F.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    db = config_path(cfg, "db")
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    out = config_path(cfg, "reports") / "releves_r4_r7"
    out.mkdir(parents=True, exist_ok=True)
    raw, renamed = config_path(cfg, "raw"), cfg["audio"].get("renamed")
    qc = cfg["qc"]

    every = inv.load_recordings(con, cfg)
    rec = every[every["survey"].isin(SURVEYS)].copy()
    lines = [f"# Contrôle des relevés R4 et R7 ({len(rec)} enregistrements)", ""]
    lines += ["Empreinte de la base : `" + json.dumps(inv.database_fingerprint(con)) + "`", ""]

    lines += ["## Effectifs", "", table(inv.counts_by(rec, ["survey", "site"]).round(1)), ""]
    lines += [table(inv.counts_by(rec, ["site", "mic_id"]).round(1)), ""]
    tot = inv.corpus_totals(rec)
    tot.pop("by_dataset")
    lines += ["Totaux : `" + json.dumps(tot) + "`", ""]

    flags = inv.flag_counts(rec)
    lines += ["## Drapeaux", "", "`" + json.dumps(flags, ensure_ascii=False) + "`", ""]
    causes = rec[rec["excluded"]].assign(cause=lambda d: d["fl"].map(inv.exclusion_cause))
    lines += [table(causes.groupby(["site", "cause"]).size().rename("n").reset_index()), ""]
    lines += [table(inv.track_flags_summary(rec, qc)), ""]

    prog = inv.recording_programme(rec)
    lines += ["## Programme et complétude", "", table(prog[["site", "mic_id", "label",
              "n_slots", "step_min", "inner_days"]]), ""]
    days = inv.mic_days(rec, prog)
    comp = inv.completeness_summary(days)
    comp.pop("by_dataset", None)
    lines += ["`" + json.dumps(comp, default=str) + "`", ""]
    partial = days[~days["edge"] & (days["completeness"] < 1)]
    lines += [table(partial[["site", "mic_id", "day", "n", "expected", "n_missing"]]), ""]
    empty = inv.empty_days(rec)
    lines += ["Jours vides entre pose et relève :", "", table(empty[["site", "mic_id",
              "empty_days", "followed_days", "longest_start", "longest_end"]]), ""]
    off = inv.off_programme_recordings(rec, prog)
    lines += ["Hors programme :", "", table(off[["path", "duration_s", "reason"]]), ""]

    lines += ["## Noms, doublons, durées, tailles", ""]
    names = inv.filename_conformity(rec)
    bad = names[~names["name_ok"] | ~names["prefix_is_mic"] | (names["suffix"] != "")
                | ~names["extension_ok"]]
    lines += ["Noms hors convention :", "", table(bad.merge(rec[["recording_id", "path"]])[
        ["path", "suffix", "prefix_is_mic"]]), ""]
    dup = inv.content_duplicates(every)["groups"]
    dup = dup[dup["paths"].map(lambda ps: any(p.startswith(SURVEYS) for p in ps))]
    lines += ["Même contenu (SHA-256) sous deux noms :", "", table(dup), ""]
    lines += ["Homonymes :", "", table(inv.homonyms(rec)), ""]
    dur = inv.duration_classes(rec, qc)
    lines += ["Durées anormales :", "", table(dur[["path", "duration_s", "duration_class",
              "on_slot"]]), ""]
    sizes = inv.file_size_check(rec, raw, renamed)
    lines += ["Tailles hors de la durée annoncée ou fichiers vides :", "",
              table(sizes[sizes["size_off"] | sizes["is_empty"] | ~sizes["found"]][
                  ["path", "size", "expected", "diff"]]), ""]

    sample = inv.sample_recordings(rec)
    headers = inv.read_headers(sample, raw, renamed)
    headers.to_csv(out / "entetes.csv", index=False)
    lines += [f"## En-têtes ({int(headers['found'].sum())} fichiers lus)", ""]
    lines += ["Gains :", "", table(inv.gain_settings(headers)), ""]
    no_guano = rec[rec["fl"].map(lambda f: not f)]  # garde-fou : drapeaux jamais calculés
    lines += [f"Enregistrements sans drapeaux : {len(no_guano)}", ""]
    offsets = inv.clock_offsets(headers, float(cfg["recorder"]["filename_utc_offset_h"]))
    clock = inv.clock_summary(offsets, float(qc.get("clock_tolerance_min", 5)))
    lines += ["Horloge (en-tête − nom) :", "", table(clock), ""]
    starts = inv.clock_summary(inv.name_start_offsets(rec), float(qc.get("clock_tolerance_min",
                                                                          5)))
    lines += ["Horloge, tous les fichiers (heure de début en base − nom) :", "",
              table(starts[starts["off"]]), ""]
    pos = inv.mic_positions(headers)
    lines += ["Positions :", "", table(pos), ""]
    lines += [table(inv.position_outliers(pos, 1.0)), ""]
    lines += ["Lieux :", "", table(inv.gps_places(headers)), ""]

    flagged = rec[rec["fl"].map(lambda f: any(
        k in f for k in ("clock_off", "survey_edge", "off_campaign", "duration_off")))]
    flagged = flagged.assign(drapeaux=flagged["qc_flags"].map(lambda q: ",".join(
        k for k, v in parse_flags(q).items() if v is True)))
    flagged[["path", "site", "mic_id", "start_utc", "duration_s", "drapeaux"]].to_csv(
        out / "signales.csv", index=False)
    (out / "RAPPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{out / 'RAPPORT.md'}")


if __name__ == "__main__":
    main()
