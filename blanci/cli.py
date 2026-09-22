"""Interface en ligne de commande `blanci` (§13.5).

Chaque commande se contente de lire la config, d'ouvrir la base et d'appeler `service.py` :
la future GUI appellera les mêmes fonctions (§4). `export-onnx` attend M5.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from blanci.benchmark import run_benchmark, write_report
from blanci.config import config_path, load_config
from blanci.db import connect
from blanci.embed import embed_recordings, select_recordings
from blanci.encoders import get_encoder
from blanci.grid import containing_windows, max_hop_without_cut, window_grid
from blanci.ingest import ingest as run_ingest
from blanci.labels import POSITIVE_LABELS, import_label_file
from blanci.qc import apply_metadata_flags
from blanci.service import (
    append_label,
    evaluate_holdout,
    make_queue,
    ranked_points,
    score_and_decide,
    similarity_search,
    train_and_register,
)

app = typer.Typer(help="Détection acoustique d'Anomaloglossus blanci.", no_args_is_help=True)


def _cfg(ctx: typer.Context) -> dict[str, Any]:
    return ctx.obj


def _split(value: str | None) -> list[str]:
    """« tresor,kaw » → ['tresor', 'kaw']."""
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _write_csv(frame: pd.DataFrame, path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    typer.echo(f"{message} : {path}")


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
    dataset: Annotated[str, typer.Option(help="Jeu (2023, 2026…) : une étiquette.")],
    folder: Annotated[
        Path | None,
        typer.Argument(help="Dossier à inventorier, sous paths.raw (défaut : paths.raw/<jeu>)."),
    ] = None,
    site: Annotated[
        str | None,
        typer.Option(help="Site de tous les fichiers du dossier (sinon lu dans l'arborescence)."),
    ] = None,
    qc: Annotated[bool, typer.Option(help="Calculer les indices de contrôle qualité.")] = True,
    hash_files: Annotated[
        bool, typer.Option("--hash/--no-hash", help="SHA-256 du contenu.")
    ] = True,
    force: Annotated[bool, typer.Option(help="Réinventorier les fichiers déjà connus.")] = False,
) -> None:
    """Inventaire des enregistrements + QC → table recordings.

    Inventorier les relevés dans l'ordre chronologique : un fichier déjà vu sous un autre
    chemin (reste de carte SD) est écarté comme doublon, le premier inventorié l'emporte.
    """
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    report = run_ingest(
        con,
        config_path(cfg, "raw"),
        dataset,
        cfg,
        run_qc=qc,
        hash_file=hash_files,
        force=force,
        scan=folder,
        site=site,
    )
    typer.echo(
        f"{report.added} ajoutés, {report.skipped} déjà inventoriés, "
        f"{report.relocated} déplacés, {len(report.duplicates)} doublons écartés, "
        f"{len(report.errors)} erreurs"
    )
    _echo_flags(report.flagged)
    # Un rapport par (jeu, site) : inventorier un relevé n'écrase pas le rapport du précédent.
    stem = f"{dataset}_{site}" if site else dataset
    reports = config_path(cfg, "reports")
    for kind, header, rows in (
        ("errors", ("path", "error"), report.errors),
        ("duplicates", ("ecarte", "conserve"), report.duplicates),
    ):
        if not rows:
            continue
        out = reports / f"ingest_{kind}_{stem}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([header, *rows])
        typer.echo(f"détail ({'erreurs' if kind == 'errors' else 'doublons'}) : {out}")


def _echo_flags(flagged: dict[str, int]) -> None:
    typer.echo(
        f"signalés (jamais encodés) : {flagged.get('duration_off', 0)} de durée anormale, "
        f"{flagged.get('off_campaign', 0)} hors relevé"
    )


@app.command()
def flag(ctx: typer.Context) -> None:
    """Recalcule les drapeaux d'inventaire (durée anormale, hors relevé) sans lire l'audio.

    Un enregistrement signalé reste dans la base et sur le disque ; il n'est simplement jamais
    encodé, donc jamais tiré comme négatif ni proposé à la vérification.
    """
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    _echo_flags(apply_metadata_flags(con, cfg["qc"]))


@app.command("import-labels")
def import_labels(
    ctx: typer.Context,
    files: Annotated[list[Path], typer.Argument(help="Fichiers CSV ou Excel reçus.")],
    kind: Annotated[
        str | None,
        typer.Option(help="positive | negative, si le fichier n'a pas de colonne label."),
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
                r["offset_s"],
                r["offset_s"] + r["dur_s"],
                window_grid(r["duration_s"], window_s, hop_s),
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
def embed(
    ctx: typer.Context,
    encoder: Annotated[str, typer.Option(help="Nom dans encoders.models ou paquet ONNX.")],
    dataset: Annotated[str | None, typer.Option(help="Restreindre à un jeu (2023, 2026).")] = None,
    site: Annotated[str | None, typer.Option(help="Restreindre à un site.")] = None,
    peak_hours: Annotated[
        bool, typer.Option(help="Ne traiter que les heures de pic locales (§5).")
    ] = False,
    batch: Annotated[int | None, typer.Option(help="Défaut : encoders.batch_size.")] = None,
) -> None:
    """Extraction des embeddings → stock Parquet. Reprenable : ce qui est fait est sauté."""
    cfg = _cfg(ctx)
    if batch:
        cfg["encoders"]["batch_size"] = batch
    con = connect(config_path(cfg, "db"))
    recordings = select_recordings(
        con,
        dataset=dataset,
        site=site,
        peak_hours=cfg["peak_hours_local"] if peak_hours else None,
        utc_offset_h=cfg["recorder"]["filename_utc_offset_h"],
    )
    if recordings.empty:
        typer.echo("aucun enregistrement retenu par ces filtres")
        raise typer.Exit(1)
    typer.echo(f"{len(recordings)} enregistrements à traiter avec {encoder}")
    model = get_encoder(encoder, cfg)
    report = embed_recordings(
        con,
        model,
        recordings,
        config_path(cfg, "raw"),
        config_path(cfg, "embeddings"),
        hop_ratio=cfg["encoders"]["grid_hop_ratio"],
        channel=cfg["audio"]["channel"],
    )
    typer.echo(
        f"{report.encoder_id} : {report.recordings} encodés, {report.skipped} déjà faits, "
        f"{report.errors} illisibles ; {report.windows} fenêtres, "
        f"{report.windows_per_s:.1f} fenêtres/s (×{report.realtime_factor:.0f} temps réel)"
    )


@app.command()
def benchmark(
    ctx: typer.Context,
    encoders: Annotated[str, typer.Option(help="Identifiants séparés par des virgules.")],
    site: Annotated[str | None, typer.Option(help="Restreindre à un site.")] = None,
    dataset: Annotated[str | None, typer.Option(help="Restreindre à un jeu.")] = None,
) -> None:
    """Tableau comparatif des encodeurs (§2), en plis groupés par micro."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    filters = {k: v for k, v in (("site", site), ("dataset", dataset)) if v}
    results, comparisons = run_benchmark(
        con, _split(encoders), config_path(cfg, "embeddings"), cfg, filters or None
    )
    paths = write_report(results, comparisons, config_path(cfg, "reports"))
    for level in ("window", "recording"):
        part = results[results["level"] == level]
        typer.echo(f"Niveau {level} (AP décroissante) :")
        for r in part.itertuples():
            typer.echo(
                f"  {r.encoder_id:<16} {r.probe:<10} AP {r.ap:.3f} [{r.ap_lo:.3f} ; {r.ap_hi:.3f}]"
            )
    typer.echo(f"rapport : {paths['markdown']}")


@app.command()
def train(
    ctx: typer.Context,
    encoder: Annotated[str, typer.Option(help="Identifiant d'encodeur (nom-version).")],
    min_precision: Annotated[
        float | None,
        typer.Option(help="Précision plancher du seuil. Défaut : benchmark.precisions[0]."),
    ] = None,
    site: Annotated[str | None, typer.Option(help="N'entraîner que sur un site.")] = None,
) -> None:
    """Entraîne la tête logistique et calibre son seuil sur les scores hors-pli."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    result = train_and_register(
        con, encoder, cfg, filters={"site": site} if site else None, min_precision=min_precision
    )
    typer.echo(result.summary())
    typer.echo(f"tête enregistrée : {result.directory}")


@app.command()
def score(
    ctx: typer.Context,
    encoder: Annotated[str, typer.Option(help="Identifiant d'encodeur.")],
    head: Annotated[str, typer.Option(help="Version de tête (v1, v2…) ou latest.")] = "latest",
    site: Annotated[str | None, typer.Option(help="Restreindre à un site.")] = None,
    dataset: Annotated[str | None, typer.Option(help="Restreindre à un jeu.")] = None,
) -> None:
    """Score toutes les fenêtres du stock, puis décide par enregistrement (§1)."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    filters = {k: v for k, v in (("site", site), ("dataset", dataset)) if v}
    result = score_and_decide(con, encoder, cfg, version=head, filters=filters or None)
    typer.echo(result.summary())
    points = ranked_points(con, encoder, result.version)
    _write_csv(
        points,
        config_path(cfg, "reports") / f"points_{encoder}_{result.version}.csv",
        "points classés",
    )
    for r in points.head(10).itertuples():
        typer.echo(
            f"  {r.site}/{r.mic_id:<6} {r.status:<18} {r.n_positive} positifs "
            f"sur {r.n_recordings} enregistrements, {r.n_days_positive} jours"
        )


@app.command()
def queue(
    ctx: typer.Context,
    encoder: Annotated[str, typer.Option(help="Identifiant d'encodeur.")],
    n: Annotated[int | None, typer.Option(help="Défaut : active.batch_recordings.")] = None,
    head: Annotated[str, typer.Option(help="Version de tête ou latest.")] = "latest",
    mix: Annotated[
        str | None, typer.Option(help="Proportions incertains,top,aléatoire. Défaut : active.mix.")
    ] = None,
) -> None:
    """File de vérification : 60 % incertains, 20 % meilleurs, 20 % aléatoire stratifié (§5)."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    proportions = tuple(float(x) for x in _split(mix)) if mix else None
    if proportions is not None and len(proportions) != 3:
        raise typer.BadParameter("--mix attend trois proportions, par exemple 0.6,0.2,0.2")
    rows = make_queue(con, encoder, cfg, n=n, version=head, mix=proportions)
    counts = rows["reason"].value_counts().to_dict()
    typer.echo(f"{len(rows)} enregistrements : " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    _write_csv(rows, config_path(cfg, "reports") / f"queue_{encoder}.csv", "file")


@app.command()
def search(
    ctx: typer.Context,
    encoder: Annotated[str, typer.Option(help="Identifiant d'encodeur.")],
    k: Annotated[int, typer.Option(help="Nombre de candidats à remonter.")] = 300,
    site: Annotated[str | None, typer.Option(help="Chercher dans ce site.")] = None,
    dataset: Annotated[str | None, typer.Option(help="Chercher dans ce jeu.")] = None,
    paired_negatives: Annotated[
        bool, typer.Option(help="Retrancher la similarité aux négatifs appariés (§3).")
    ] = True,
) -> None:
    """Fenêtres les plus proches des positifs annotés (§5, récolte hors Mataroni)."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    filters = {key: value for key, value in (("site", site), ("dataset", dataset)) if value}
    found = similarity_search(
        con, encoder, cfg, k=k, filters=filters or None, use_paired_negatives=paired_negatives
    )
    typer.echo(f"{len(found)} candidats")
    for r in found.head(10).itertuples():
        typer.echo(f"  {r.score:+.3f}  {r.path} @ {r.offset_s:.1f} s")
    _write_csv(found, config_path(cfg, "reports") / f"search_{encoder}.csv", "candidats")


@app.command()
def evaluate(
    ctx: typer.Context,
    encoder: Annotated[str, typer.Option(help="Identifiant d'encodeur.")],
    level: Annotated[str, typer.Option(help="window | recording.")] = "recording",
    holdout: Annotated[
        str | None, typer.Option(help="Sites tenus à l'écart, par exemple tresor,kaw (§6).")
    ] = None,
) -> None:
    """Évalue sur des sites tenus à l'écart, ou en plis groupés par micro si aucun n'est donné."""
    if level not in ("window", "recording"):
        raise typer.BadParameter("--level attend window ou recording")
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    metrics = evaluate_holdout(con, encoder, cfg, _split(holdout), level=level)
    typer.echo(f"{metrics['encoder_id']} — {metrics['protocol']}, niveau {metrics['level']}")
    typer.echo(f"  {metrics['n_pos']} positifs, {metrics['n_neg']} négatifs")
    typer.echo(f"  AP {metrics['ap']:.3f} [{metrics['ap_lo']:.3f} ; {metrics['ap_hi']:.3f}]")
    for p in cfg["benchmark"]["precisions"]:
        typer.echo(
            f"  rappel à P≥{p} : {metrics[f'recall@p{p}']:.3f} "
            f"[{metrics[f'recall@p{p}_lo']:.3f} ; {metrics[f'recall@p{p}_hi']:.3f}]"
        )


@app.command("label")
def label_window(
    ctx: typer.Context,
    window_id: Annotated[str, typer.Argument(help="Identifiant de fenêtre.")],
    label: Annotated[str, typer.Option(help="blanci_solo, bird, rain…")],
    source: Annotated[str, typer.Option(help="import | similarity | active | random | audit.")],
    quality: Annotated[str | None, typer.Option(help="A, B ou C.")] = None,
    species: Annotated[str | None, typer.Option(help="Espèce du faux ami.")] = None,
    annotator: Annotated[str | None, typer.Option(help="Qui a annoté.")] = None,
) -> None:
    """Ajoute un label à une fenêtre (les labels ne se modifient jamais, ils s'ajoutent)."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    label_id = append_label(con, window_id, label, source, quality, species, None, annotator)
    typer.echo(f"label {label_id} ajouté : {window_id} → {label}")


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
        sr = (
            f"{r['sr_min']} Hz" if r["sr_min"] == r["sr_max"] else f"{r['sr_min']}–{r['sr_max']} Hz"
        )
        typer.echo(
            f"  {r['dataset']} / {r['site']} : {r['n']}, {r['mics']} micros, "
            f"{r['hours']:.1f} h, {sr}"
        )
    flags: Counter[str] = Counter()
    for (qc,) in con.execute("SELECT qc_flags FROM recordings WHERE qc_flags IS NOT NULL"):
        flags.update(k for k, v in json.loads(qc).items() if v is True)
    if flags:
        typer.echo(
            "Drapeaux QC (seuils provisoires) : " + ", ".join(f"{k}={v}" for k, v in flags.items())
        )
    typer.echo("Labels :")
    for r in con.execute(
        "SELECT source, label, COUNT(*) n FROM labels "
        "GROUP BY source, label ORDER BY source, n DESC"
    ):
        typer.echo(f"  {r['source']} / {r['label']} : {r['n']}")
