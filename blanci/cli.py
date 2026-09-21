"""Interface en ligne de commande `blanci` (§13.5). Jalon M0 : ingest, import-labels, check-grid."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer

from blanci.config import config_path, load_config
from blanci.db import connect
from blanci.grid import containing_windows, max_hop_without_cut, window_grid
from blanci.ingest import ingest as run_ingest
from blanci.labels import POSITIVE_LABELS, import_label_file

app = typer.Typer(help="Détection acoustique d'Anomaloglossus blanci.", no_args_is_help=True)


def _cfg(ctx: typer.Context) -> dict[str, Any]:
    return ctx.obj


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Fichier YAML surchargeant la config.")
    ] = None,
) -> None:
    ctx.obj = load_config(config)


@app.command()
def ingest(
    ctx: typer.Context,
    dataset: Annotated[str, typer.Option(help="Jeu : sous-dossier de la racine (2023, 2026).")],
    root: Annotated[Path | None, typer.Argument(help="Racine audio (défaut : paths.raw).")] = None,
    qc: Annotated[bool, typer.Option(help="Calculer les indices de contrôle qualité.")] = True,
    hash_files: Annotated[bool, typer.Option("--hash/--no-hash", help="SHA-256 du contenu.")] = True,
    force: Annotated[bool, typer.Option(help="Réinventorier les fichiers déjà connus.")] = False,
) -> None:
    """Inventaire des enregistrements + QC → table recordings."""
    cfg = _cfg(ctx)
    root = root or config_path(cfg, "raw")
    if root.resolve() != config_path(cfg, "raw").resolve():
        typer.echo(f"attention : racine {root} ≠ paths.raw ; les chemins stockés sont relatifs à {root}")
    con = connect(config_path(cfg, "db"))
    report = run_ingest(con, root, dataset, cfg, run_qc=qc, hash_file=hash_files, force=force)
    typer.echo(f"{report.added} ajoutés, {report.skipped} déjà inventoriés, {len(report.errors)} erreurs")
    if report.errors:
        out = config_path(cfg, "reports") / f"ingest_errors_{dataset}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([("path", "error"), *report.errors])
        typer.echo(f"détail des erreurs : {out}")


@app.command("import-labels")
def import_labels(
    ctx: typer.Context,
    files: Annotated[list[Path], typer.Argument(help="Fichiers CSV ou Excel reçus.")],
    kind: Annotated[
        str | None, typer.Option(help="positive | negative, si le fichier n'a pas de colonne label.")
    ] = None,
    annotator: Annotated[str | None, typer.Option(help="Défaut : labels.import.annotator.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Analyser sans rien écrire.")] = False,
    allow_partial: Annotated[
        bool, typer.Option(help="Importer les lignes résolues même si d'autres ne le sont pas.")
    ] = False,
) -> None:
    """Import des annotations (345 positifs + 158 faux amis) avec analyse des commentaires."""
    if kind not in (None, "positive", "negative"):
        raise typer.BadParameter("--kind attend positive ou negative")
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    failed = False
    for path in files:
        report = import_label_file(con, path, cfg, kind, annotator, dry_run, allow_partial)
        typer.echo(report.summary())
        if report.inserted:
            typer.echo(f"  {report.inserted} labels ajoutés")
        elif report.unresolved and not dry_run:
            typer.echo("  rien n'est importé (--allow-partial pour importer les lignes résolues)")
            failed = True
    if failed:
        raise typer.Exit(1)


@app.command("check-grid")
def check_grid(ctx: typer.Context) -> None:
    """Vérifie que chaque grille garde entières les notes des fenêtres positives annotées."""
    cfg = _cfg(ctx)
    note_s = cfg["signal"]["note_max_s"]
    con = connect(config_path(cfg, "db"))
    rows = con.execute(
        f"""SELECT DISTINCT w.window_id, w.offset_s, w.dur_s, r.duration_s
            FROM labels l JOIN windows w USING (window_id) JOIN recordings r USING (recording_id)
            WHERE l.label IN ({", ".join("?" * len(POSITIVE_LABELS))})""",
        POSITIVE_LABELS,
    ).fetchall()
    typer.echo(f"{len(rows)} fenêtres positives annotées")
    for name, grid in cfg["grids"].items():
        window_s, hop_s = grid["window_s"], grid["hop_s"]
        guaranteed = hop_s <= max_hop_without_cut(window_s, note_s) + 1e-9
        uncovered = [
            r["window_id"]
            for r in rows
            if not containing_windows(
                r["offset_s"], r["offset_s"] + r["dur_s"], window_grid(r["duration_s"], window_s, hop_s)
            )
        ]
        typer.echo(
            f"  {name} ({window_s} s / {hop_s} s) : note de {note_s} s jamais coupée : "
            f"{'oui' if guaranteed else 'NON'} ; annotations hors de toute fenêtre : "
            f"{len(uncovered)}/{len(rows)}"
        )
        for window_id in uncovered[:10]:
            typer.echo(f"    - {window_id}")


@app.command()
def status(ctx: typer.Context) -> None:
    """Résumé de la base : enregistrements, drapeaux QC, labels."""
    con = connect(config_path(_cfg(ctx), "db"))
    typer.echo("Enregistrements (jeu / site : fichiers, micros, heures) :")
    for r in con.execute(
        """SELECT dataset, site, COUNT(*) n, COUNT(DISTINCT mic_id) mics,
                  SUM(duration_s) / 3600.0 hours, MIN(sample_rate) sr_min, MAX(sample_rate) sr_max
           FROM recordings GROUP BY dataset, site ORDER BY dataset, site"""
    ):
        sr = f"{r['sr_min']} Hz" if r["sr_min"] == r["sr_max"] else f"{r['sr_min']}–{r['sr_max']} Hz"
        typer.echo(
            f"  {r['dataset']} / {r['site']} : {r['n']}, {r['mics']} micros, {r['hours']:.1f} h, {sr}"
        )
    flags: Counter[str] = Counter()
    for (qc,) in con.execute("SELECT qc_flags FROM recordings WHERE qc_flags IS NOT NULL"):
        flags.update(k for k, v in json.loads(qc).items() if v is True)
    if flags:
        typer.echo("Drapeaux QC (seuils provisoires) : " + ", ".join(f"{k}={v}" for k, v in flags.items()))
    typer.echo("Labels :")
    for r in con.execute(
        "SELECT source, label, COUNT(*) n FROM labels GROUP BY source, label ORDER BY source, n DESC"
    ):
        typer.echo(f"  {r['source']} / {r['label']} : {r['n']}")
