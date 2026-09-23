"""Interface en ligne de commande `blanci` (§13.5).

Chaque commande se contente de lire la config, d'ouvrir la base et d'appeler `service.py` :
la future GUI appellera les mêmes fonctions (§4). `export-onnx` attend M5.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from blanci.activity import write_activity_report
from blanci.baselines import run_baselines, write_baseline_report
from blanci.benchmark import run_benchmark, write_report
from blanci.config import config_path, load_config
from blanci.dataset import benchmark_recordings, current_labels, recordings_table
from blanci.db import connect
from blanci.embed import embed_recordings, select_recordings
from blanci.encoders import get_encoder
from blanci.frozen import freeze as freeze_recordings
from blanci.grid import containing_windows, max_hop_without_cut, window_grid
from blanci.ingest import ingest as run_ingest
from blanci.labels import POSITIVE_LABELS, import_detections, import_label_file
from blanci.qc import (
    AUDIO_FLAGS,
    apply_annotation_flags,
    apply_audio_flags,
    apply_metadata_flags,
    parse_flags,
)
from blanci.sequential import compute_onsets
from blanci.service import (
    activity_curves,
    append_label,
    compute_tokens,
    evaluate_frozen,
    evaluate_holdout,
    make_queue,
    ranked_points,
    run_clustering,
    score_and_decide,
    similarity_search,
    train_and_register,
    train_fusion,
)
from blanci.service import retrain as retrain_head
from blanci.throughput import (
    machine_description,
    measure_in_subprocess,
    write_throughput_report,
)
from blanci.workbench import agreement as annotator_agreement
from blanci.workbench import (
    blancinet_candidates,
    congener_candidates,
    random_candidates,
    recording_candidates,
)

# Console Windows en cp1252 : « ≥ », « → » ou « é » y feraient planter l'affichage (aide
# comprise, écrite avant tout callback) quand la sortie est redirigée. La CLI écrit en UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

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


def _echo_annotated(counts: dict[str, int]) -> None:
    typer.echo(
        f"posés à l'écoute, sur {counts['annotated']} enregistrements annotés : "
        f"{counts['in_bag']} micro dans sac (jamais encodés), {counts['rain']} pluie (remarque)"
    )


@app.command()
def flag(ctx: typer.Context) -> None:
    """Recalcule tous les drapeaux, sans lire l'audio : inventaire (durée anormale, hors
    relevé), audio depuis les indices déjà calculés (seuils actuels de `qc`), écoute.

    Écartent du corpus : silencieux, micro dans sac, durée anormale, hors relevé. Pluie et
    saturation sont des remarques. Un enregistrement signalé reste dans la base et sur le
    disque ; s'il est écarté, il n'est jamais encodé, donc jamais tiré comme négatif ni
    proposé à la vérification. Un enregistrement où A. blanci a été entendu n'est jamais
    écarté.
    """
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    _echo_flags(apply_metadata_flags(con, cfg["qc"]))
    audio = apply_audio_flags(con, cfg["qc"])
    typer.echo(
        "audio (enregistrements déjà contrôlés) : "
        + ", ".join(f"{k} {audio[k]}" for k in AUDIO_FLAGS)
    )
    _echo_annotated(apply_annotation_flags(con))


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
            _echo_annotated(apply_annotation_flags(con))
        elif report.unresolved and not dry_run:
            typer.echo("  rien n'est importé (--allow-partial pour importer les lignes résolues)")
            failed = True
    if failed:
        raise typer.Exit(1)


@app.command("import-detections")
def import_detections_command(
    ctx: typer.Context,
    table: Annotated[Path, typer.Argument(help="Export des détections (Blancinet).")],
    model: Annotated[str, typer.Option(help="Nom du détecteur dans la base.")] = "blancinet",
) -> None:
    """Range les détections d'un détecteur indépendant comme scores (pas comme labels).

    Aucun négatif présumé n'est tiré à moins de 3 s d'une détection ≥ 0,5 que personne n'a
    écoutée : ce serait prendre pour négatif une fenêtre où Blancinet entend A. blanci
    (DECISIONS n° 80, 85).
    """
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    report = import_detections(con, table, cfg, model)
    typer.echo(
        f"{report.stored} détections rangées sous « {model} » sur {report.rows} lignes "
        f"({report.unverified} jamais écoutées) ; {report.not_found} fichiers hors inventaire, "
        f"{report.ambiguous} ambigus, {report.unreadable} illisibles"
    )


@app.command("export-labels")
def export_labels(ctx: typer.Context) -> None:
    """Fenêtres annotées, une ligne chacune : label, qualité, espèce, commentaire.

    Le commentaire est celui de l'annotateur, tel qu'écrit (import ou poste d'annotation).
    """
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    labels = current_labels(con)
    rec = recordings_table(con)[["recording_id", "path", "site", "mic_id", "start_utc"]]
    table = labels.merge(rec, on="recording_id", how="left")
    columns = [
        "window_id", "path", "site", "mic_id", "start_utc", "offset_s", "dur_s", "label",
        "quality", "species", "comment", "source",
    ]  # fmt: skip
    _write_csv(
        table[columns].sort_values(["path", "offset_s"]),
        config_path(cfg, "reports") / "fenetres_annotees.csv",
        f"{len(table)} fenêtres annotées",
    )


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
    subset: Annotated[
        str | None,
        typer.Option(help="« benchmark » : annotés + candidats aux négatifs appariés seulement."),
    ] = None,
    qc: Annotated[
        bool | None,
        typer.Option(
            "--qc/--no-qc",
            help="Contrôle audio au passage des enregistrements qui ne l'ont pas eu "
            "(défaut : qc.during_embed).",
        ),
    ] = None,
) -> None:
    """Extraction des embeddings → stock Parquet. Reprenable : ce qui est fait est sauté.

    Avec le contrôle audio (défaut), un enregistrement jamais contrôlé l'est sur l'audio déjà
    lu ; silencieux ou micro dans sac, il n'est pas encodé (sauf s'il contient un positif).
    """
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
    if subset == "benchmark":
        wanted = benchmark_recordings(
            con,
            cfg["benchmark"]["slot_tolerance_min"],
            cfg["recorder"]["filename_utc_offset_h"],
        )
        recordings = recordings[recordings["recording_id"].isin(wanted["recording_id"])]
    elif subset is not None:
        raise typer.BadParameter("attendu : benchmark", param_hint="--subset")
    if recordings.empty:
        typer.echo("aucun enregistrement retenu par ces filtres")
        raise typer.Exit(1)
    typer.echo(f"{len(recordings)} enregistrements à traiter avec {encoder}")
    model = get_encoder(encoder, cfg)
    check_qc = cfg["qc"].get("during_embed", True) if qc is None else qc
    report = embed_recordings(
        con,
        model,
        recordings,
        config_path(cfg, "raw"),
        config_path(cfg, "embeddings"),
        hop_ratio=cfg["encoders"]["grid_hop_ratio"],
        channel=cfg["audio"]["channel"],
        signal_cfg=cfg["signal"],
        qc_thresholds=cfg["qc"] if check_qc else None,
    )
    if report.qc_checked:
        typer.echo(
            f"contrôle audio : {report.qc_checked} enregistrements, {report.qc_excluded} écartés "
            "(silencieux ou micro dans sac)"
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
def baselines(
    ctx: typer.Context,
    channels: Annotated[
        str | None,
        typer.Option(help="Canaux à comparer, ex. « 0,1 ». Défaut : audio.channel."),
    ] = None,
) -> None:
    """Baselines sans encodeur (§3) sur les fenêtres annotées : lit l'audio, n'encode rien."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    if channels:
        chosen = tuple(int(c) for c in _split(channels))
    else:
        default = cfg["audio"]["channel"]
        chosen = (default if isinstance(default, int) else 0,)
    table, scores = run_baselines(con, cfg, config_path(cfg, "raw"), chosen)
    paths = write_baseline_report(table, scores, config_path(cfg, "reports"))
    for level in ("window", "recording"):
        typer.echo(f"Niveau {level} (AP décroissante) :")
        for r in table[table["level"] == level].to_dict("records"):
            typer.echo(
                f"  {r['baseline']:<14} canal {r['channel']}  AP {r['ap']:.3f} "
                f"[{r['ap_lo']:.3f} ; {r['ap_hi']:.3f}]  "
                f"rappel à P>=0,1 {r.get('recall@p0.1', float('nan')):.2f}"
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
    head: Annotated[
        str,
        typer.Option(
            help="Version de tête (v1, v2…), latest ou adopted. Défaut : la tête adoptée par "
            "`blanci retrain`, sinon la plus récente."
        ),
    ] = "default",
    site: Annotated[str | None, typer.Option(help="Restreindre à un site.")] = None,
    dataset: Annotated[str | None, typer.Option(help="Restreindre à un jeu.")] = None,
    fusion: Annotated[
        bool, typer.Option(help="Décider avec la fusion (tête + rythme + persistance).")
    ] = False,
) -> None:
    """Score toutes les fenêtres du stock, puis décide par enregistrement (§1)."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    filters = {k: v for k, v in (("site", site), ("dataset", dataset)) if v}
    result = score_and_decide(
        con, encoder, cfg, version=head, filters=filters or None, fusion=fusion
    )
    typer.echo(result.summary())
    points = ranked_points(con, encoder, result.version, result.threshold_id)
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
def retrain(
    ctx: typer.Context,
    encoder: Annotated[str, typer.Option(help="Identifiant d'encodeur.")],
    frozen: Annotated[
        str | None, typer.Option(help="Jeu gelé qui juge (défaut : le plus récent).")
    ] = None,
    tolerance: Annotated[
        float | None, typer.Option(help="Perte tolérée d'AP et de rappel (défaut : config).")
    ] = None,
    force: Annotated[bool, typer.Option(help="Adopter même si moins bonne.")] = False,
) -> None:
    """Réentraîne la tête sur tous les labels et ne l'adopte que si elle n'est pas moins
    bonne sur le jeu gelé (§4, M5). La tête adoptée est celle de `blanci score`."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    out = retrain_head(con, encoder, cfg, frozen, tolerance, force)
    typer.echo(
        f"nouvelle tête {out['new_version']} (tête adoptée avant : {out['previous_version']})"
    )
    for key, name in (("new", "nouvelle"), ("previous", "adoptée")):
        m = out.get(key)
        if m:
            typer.echo(
                f"  {name} {m['head_version']} sur le jeu gelé : AP {m['ap_recording']:.3f} "
                f"[{m['ap_lo']:.3f} ; {m['ap_hi']:.3f}], rappel au seuil "
                f"{m['recall_at_threshold']:.2f}, {m['false_alarms_per_hour']:.1f} fausses "
                "alarmes par heure"
            )
    if "previous_not_judged" in out:
        typer.echo(f"  adoptée non jugeable : {out['previous_not_judged']}")
    verdict = "ADOPTÉE" if out["adopted"] else "NON ADOPTÉE"
    typer.echo(f"{verdict} : {out['reason']}")
    if out["adopted"]:
        typer.echo(f"suite : blanci score --encoder {encoder}")


@app.command()
def activity(
    ctx: typer.Context,
    encoder: Annotated[str, typer.Option(help="Identifiant d'encodeur.")],
    head: Annotated[str, typer.Option(help="Version de tête (défaut : adoptée).")] = "default",
    fusion: Annotated[bool, typer.Option(help="Décisions de la fusion.")] = False,
    dataset: Annotated[str | None, typer.Option(help="Jeu, par exemple 2023.")] = None,
    by: Annotated[str, typer.Option(help="site | mic_id.")] = "site",
    reference_hours: Annotated[
        Path | None,
        typer.Option(help="Courbe horaire de référence numérisée (CSV : [site,] hour, value)."),
    ] = None,
    reference_months: Annotated[
        Path | None,
        typer.Option(help="Courbe mensuelle de référence (CSV : [site,] month, value)."),
    ] = None,
    suspects_detected: Annotated[
        bool, typer.Option(help="Compter les enregistrements « suspect » comme détectés.")
    ] = False,
) -> None:
    """Courbes d'activité journalières et saisonnières, confrontées aux patrons de Courtois
    et al. 2025 (§6 niveau 3, M4). Demande les décisions de `blanci score`."""
    if by not in ("site", "mic_id"):
        raise typer.BadParameter("--by attend site ou mic_id")
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    reference = {
        key: pd.read_csv(path)
        for key, path in (("hour", reference_hours), ("month", reference_months))
        if path is not None
    }
    out = activity_curves(
        con, encoder, cfg, head, fusion, dataset, by, reference or None, suspects_detected
    )
    stem = f"activite_{encoder}_{out['head_version']}" + (f"_{dataset}" if dataset else "")
    paths = write_activity_report(
        config_path(cfg, "reports"),
        stem,
        out["diel"],
        out["seasonal"],
        out["checks"],
        out["correlations"],
        cfg["activity"]["min_correlation"],
        f"{encoder} tête {out['head_version']}" + (f", jeu {dataset}" if dataset else ""),
    )
    for r in out["checks"].to_dict("records"):
        typer.echo(
            f"  {r[by]:<14} {r['recordings']:>6} enr.  pics/autres {r['peak_ratio']:.2f}  "
            f"mois forts/creux {r['season_ratio']:.2f}  "
            f"{'patrons retrouvés' if r['peaks_found'] and r['season_found'] else 'À EXAMINER'}"
        )
    typer.echo(f"rapport : {paths[0]}")


@app.command()
def evaluate(
    ctx: typer.Context,
    encoder: Annotated[str, typer.Option(help="Identifiant d'encodeur.")],
    level: Annotated[str, typer.Option(help="window | recording.")] = "recording",
    holdout: Annotated[
        str | None, typer.Option(help="Sites tenus à l'écart, par exemple tresor,kaw (§6).")
    ] = None,
    frozen: Annotated[
        str | None,
        typer.Option(help="Juger la dernière tête sur un jeu gelé (version, ou « last »)."),
    ] = None,
) -> None:
    """Évalue sur des sites tenus à l'écart, ou en plis groupés par micro si aucun n'est donné.

    Avec --frozen : la tête enregistrée, à son seuil, sur un jeu gelé jamais vu (§6).
    """
    if level not in ("window", "recording"):
        raise typer.BadParameter("--level attend window ou recording")
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    if frozen:
        result = evaluate_frozen(
            con, encoder, cfg, frozen_version=None if frozen == "last" else frozen
        )
        typer.echo(
            f"{encoder} tête {result['head_version']} — jeu gelé {result['frozen_version']} "
            f"({result['n_recordings']} enregistrements écoutés en entier)"
        )
        for lvl in ("recording", "window"):
            m = result[lvl]
            typer.echo(
                f"  {lvl} : {m['n_pos']} positifs, {m['n_neg']} négatifs, AP {m['ap']:.3f} "
                f"[{m['ap_lo']:.3f} ; {m['ap_hi']:.3f}]"
            )
        typer.echo(
            f"  au seuil de la tête ({result['threshold']:.3f}) : rappel fenêtres "
            f"{result['recall_at_threshold']:.2f}, "
            f"{result['false_alarms_per_hour']:.1f} fausses alarmes par heure"
        )
        for r in result["by_stratum"].to_dict("records"):
            typer.echo(
                f"    {r['by']} {r['stratum']:<12} {r['n_pos']:>4} positifs  "
                f"rappel {r['recall']:.2f}"
            )
        return
    metrics = evaluate_holdout(con, encoder, cfg, _split(holdout), level=level)
    typer.echo(f"{metrics['encoder_id']} — {metrics['protocol']}, niveau {metrics['level']}")
    typer.echo(f"  {metrics['n_pos']} positifs, {metrics['n_neg']} négatifs")
    typer.echo(f"  AP {metrics['ap']:.3f} [{metrics['ap_lo']:.3f} ; {metrics['ap_hi']:.3f}]")
    for p in cfg["benchmark"]["precisions"]:
        typer.echo(
            f"  rappel à P≥{p} : {metrics[f'recall@p{p}']:.3f} "
            f"[{metrics[f'recall@p{p}_lo']:.3f} ; {metrics[f'recall@p{p}_hi']:.3f}]"
        )
    typer.echo(
        f"  rappel par strate, fenêtres, au seuil de précision ≥ {metrics['stratum_precision']} :"
    )
    for r in metrics["by_stratum"].to_dict("records"):
        typer.echo(
            f"    {r['by']} {r['stratum']:<12} {r['n_pos']:>4} positifs  rappel {r['recall']:.2f} "
            f"[{r['recall_lo']:.2f} ; {r['recall_hi']:.2f}]"
        )


@app.command()
def freeze(
    ctx: typer.Context,
    source: Annotated[Path, typer.Argument(help="File CSV (recording_id) : candidats_gele.csv…")],
    version: Annotated[str, typer.Option(help="Nom de la version : v1, v2…")],
) -> None:
    """Gèle des enregistrements (§6) : jamais entraînés, seulement jugés. Irréversible."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    path, report = freeze_recordings(con, cfg, source, version)
    typer.echo(
        f"jeu gelé {version} : {report['n_recordings']} enregistrements ({path}, lecture seule)"
    )
    if report["labels_withdrawn"]:
        typer.echo(
            f"  {report['labels_withdrawn']} labels existants sortent de l'entraînement, dont "
            f"{report['positive_labels_withdrawn']} positifs"
        )
    if report["already_frozen"]:
        typer.echo(f"  {report['already_frozen']} déjà gelés dans une autre version")


@app.command()
def onsets(
    ctx: typer.Context,
    subset: Annotated[
        str | None, typer.Option(help="« benchmark » : annotés + négatifs appariés seulement.")
    ] = None,
    site: Annotated[str | None, typer.Option(help="Restreindre à un site.")] = None,
) -> None:
    """Débuts de notes par enregistrement (module séquentiel, §3). Lit l'audio, n'encode rien.

    `blanci embed` les calcule déjà au passage ; cette commande sert aux enregistrements
    qu'on ne veut pas encoder. Reprenable : les enregistrements traités sont sautés.
    """
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    recordings = select_recordings(con, site=site)
    if subset == "benchmark":
        wanted = benchmark_recordings(
            con, cfg["benchmark"]["slot_tolerance_min"], cfg["recorder"]["filename_utc_offset_h"]
        )
        recordings = recordings[recordings["recording_id"].isin(wanted["recording_id"])]
    report = compute_onsets(
        con, recordings, config_path(cfg, "raw"), cfg["signal"], cfg["audio"]["channel"]
    )
    typer.echo(
        f"{report['computed']} enregistrements traités, {report['skipped']} déjà faits, "
        f"{report['errors']} illisibles ; {report['onsets']} débuts de notes"
    )


@app.command()
def fusion(
    ctx: typer.Context,
    encoder: Annotated[str, typer.Option(help="Identifiant d'encodeur.")],
    head: Annotated[str, typer.Option(help="Version de tête (v1, v2…) ou latest.")] = "latest",
) -> None:
    """Fusion (§3) : évaluée hors-pli contre la tête seule, puis enregistrée pour `score`."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    result = train_fusion(con, encoder, cfg, head)
    typer.echo(
        f"{result['model_id']} : {result['n_windows']} fenêtres, "
        f"{result['n_with_onsets']} avec débuts de notes"
    )
    for level in ("recording", "window"):
        h, f = result["comparison"][f"head_{level}"], result["comparison"][f"fusion_{level}"]
        typer.echo(
            f"  {level} : AP tête {h['ap']:.3f} → fusion {f['ap']:.3f} ; "
            f"rappel à P≥0.1 {h['recall@p0.1']:.2f} → {f['recall@p0.1']:.2f}"
        )
    p = result["paired"]
    typer.echo(
        f"  écart d'AP par enregistrement {p['diff']:+.3f} [{p['lo']:+.3f} ; {p['hi']:+.3f}] : "
        f"{'significatif' if p['significant'] else 'non significatif'}"
    )
    coefs = ", ".join(f"{k} {v:+.2f}" for k, v in result["coefficients"].items())
    typer.echo(f"  coefficients (standardisés) : {coefs}")
    typer.echo(
        f"  seuil {result['threshold']:.3f} (rappel {result['recall_at_threshold']:.2f}) ; "
        "décider avec : blanci score --fusion"
    )


@app.command("qc-calibrate")
def qc_calibrate(ctx: typer.Context) -> None:
    """Seuils du contrôle qualité mesurés sur les fenêtres étiquetées (lit l'audio, n'écrit
    rien dans la config) : effet des seuils actuels et seuils proposés."""
    from blanci.qc_calibration import (
        calibration_indices,
        current_flags,
        labelled_windows,
        suggest_thresholds,
    )

    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    windows = labelled_windows(con)
    typer.echo(
        f"{len(windows)} fenêtres étiquetées, {windows['recording_id'].nunique()} "
        f"enregistrements lus en entier : "
        + ", ".join(f"{g} {n}" for g, n in windows["group"].value_counts().items())
    )
    indices = calibration_indices(windows, config_path(cfg, "raw"), cfg["audio"]["channel"])
    table = suggest_thresholds(indices, cfg["qc"])
    for r in table.to_dict("records"):
        typer.echo(
            f"  {r['flag']:<10} {r['index']} {r['direction']} {r['current']:g} : "
            f"cibles signalées {r['targets_flagged_now']}/{r['targets']}, enregistrements à "
            f"A. blanci signalés {r['blanci_flagged_now']}/{r['blanci_recordings']} ; "
            f"proposé {r['suggested']:.4g} → cibles {r['targets_flagged_suggested']}/"
            f"{r['targets']} ({r['reason']})"
        )
    reports = config_path(cfg, "reports")
    _write_csv(indices.drop(columns=["path"]), reports / "qc_indices.csv", "indices")
    _write_csv(table, reports / "qc_calibration.csv", "seuils")
    _write_csv(current_flags(indices, cfg["qc"]), reports / "qc_drapeaux_actuels.csv", "drapeaux")


@app.command()
def tokens(
    ctx: typer.Context,
    encoder: Annotated[str, typer.Option(help="Nom dans encoders.models (perch_v2).")],
) -> None:
    """Jetons des fenêtres du benchmark pour la sonde attentive (§3). Encode ~1 500 fenêtres."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    report = compute_tokens(con, get_encoder(encoder, cfg), cfg)
    typer.echo(
        f"{report['windows']} fenêtres ({report['recordings']} enregistrements), "
        f"{report['skipped']} déjà faites ; `blanci benchmark` ajoute alors la sonde attentive"
    )


@app.command("anuraset-prepare")
def anuraset_prepare(ctx: typer.Context) -> None:
    """AnuraSet (§2) : extrait raw_data.zip et l'inventorie (--config config/anuraset.yaml)."""
    from blanci.anuraset import prepare

    cfg = _cfg(ctx)
    if "anuraset" not in cfg:
        raise typer.BadParameter("lancer avec --config config/anuraset.yaml")
    report = prepare(connect(config_path(cfg, "db")), cfg)
    typer.echo(
        f"{report['extracted']} fichiers extraits, {report['added']} inventoriés, "
        f"{report['errors']} illisibles"
    )


@app.command("anuraset-profile")
def anuraset_profile(
    ctx: typer.Context,
    audio: Annotated[
        bool, typer.Option(help="Mesurer la fréquence dominante (lit quelques chants).")
    ] = True,
) -> None:
    """Profil des espèces d'AnuraSet et suggestion d'espèces proches d'A. blanci (§2)."""
    from blanci.anuraset import (
        dominant_frequencies,
        read_strong_labels,
        species_profile,
        suggest_species,
    )

    cfg = _cfg(ctx)
    acfg = cfg["anuraset"]
    calls = read_strong_labels(Path(acfg["labels"]))
    profile = species_profile(calls, acfg["max_call_s"])
    if audio:
        con = connect(config_path(cfg, "db"))
        recordings = pd.read_sql_query("SELECT path FROM recordings", con)
        freqs = dominant_frequencies(
            calls, config_path(cfg, "raw"), recordings, acfg["profile_per_species"]
        )
        profile = profile.merge(freqs, left_on="species", right_index=True, how="left")
        chosen = suggest_species(profile)  # note brève, 3–6 kHz, ≥ 300 chants, ≥ 2 sites
        typer.echo("Espèces proches d'A. blanci (note brève, 3–6 kHz, ≥ 2 sites) :")
        for r in chosen.to_dict("records"):
            typer.echo(
                f"  {r['species']:<8} {r['n_calls']:>6} chants, {r['n_sites']} sites, "
                f"{r['duration_median_s']:.2f} s, {r['dominant_hz']:.0f} Hz"
            )
    _write_csv(profile, config_path(cfg, "reports") / "anuraset_especes.csv", "profil")


@app.command("anuraset-benchmark")
def anuraset_benchmark(
    ctx: typer.Context,
    encoders: Annotated[str, typer.Option(help="Identifiants séparés par des virgules.")],
    species: Annotated[
        str | None, typer.Option(help="Codes AnuraSet ; défaut : anuraset.species.")
    ] = None,
) -> None:
    """Pré-benchmark AnuraSet (§2) : sondes par encodeur et par espèce, plis par site."""
    from blanci.anuraset import (
        read_strong_labels,
        run_anuraset_benchmark,
        write_anuraset_report,
    )

    cfg = _cfg(ctx)
    chosen = _split(species) or list(cfg["anuraset"]["species"])
    if not chosen:
        raise typer.BadParameter("aucune espèce : --species ou anuraset.species (voir profile)")
    con = connect(config_path(cfg, "db"))
    calls = read_strong_labels(Path(cfg["anuraset"]["labels"]))
    results, comparisons = run_anuraset_benchmark(con, cfg, _split(encoders), chosen, calls)
    path = write_anuraset_report(results, comparisons, config_path(cfg, "reports"))
    for r in results[results["level"] == "window"].to_dict("records"):
        typer.echo(f"  {r['encoder_id']:<24} {r['species']:<8} {r['probe']:<16} AP {r['ap']:.3f}")
    typer.echo(f"rapport : {path}")


@app.command()
def cluster(
    ctx: typer.Context,
    encoder: Annotated[str, typer.Option(help="Identifiant d'encodeur (nom-version).")],
    mode: Annotated[str, typer.Option(help="c0 : exploration globale ; c1 : détection ?")] = "c0",
    n: Annotated[
        int | None, typer.Option(help="C0 : fenêtres tirées ; C1 : négatifs appariés.")
    ] = None,
) -> None:
    """Clustering HDBSCAN sur ACP (§5 bis) : groupes, AMI avec les micros, verdict C1."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    if mode not in ("c0", "c1"):
        raise typer.BadParameter("c0 ou c1", param_hint="--mode")
    summary, table, windows = run_clustering(con, encoder, cfg, mode, n)
    typer.echo(
        f"{summary['n_windows']} fenêtres, {summary['n_clusters']} groupes, "
        f"{summary['noise_share']:.0%} de bruit, AMI groupes/micros {summary['ami_mic']:.2f}"
    )
    if mode == "c1":
        typer.echo(
            f"meilleur groupe : rappel {summary['best_recall']:.2f}, enrichissement "
            f"{summary['best_enrichment']:.1f} → C1 {'réussi' if summary['passed'] else 'échoué'}"
        )
    reports = config_path(cfg, "reports")
    _write_csv(table, reports / f"cluster_{mode}_{encoder}.csv", "groupes")
    _write_csv(windows, reports / f"cluster_{mode}_{encoder}_fenetres.csv", "fenêtres")


@app.command()
def agreement(
    ctx: typer.Context,
    annotators: Annotated[str, typer.Option(help="Deux annotateurs, ex. « léonard,tuteur ».")],
) -> None:
    """Accord de deux annotateurs sur les fenêtres écoutées par les deux (§5, calibration)."""
    names = _split(annotators)
    if len(names) != 2:
        raise typer.BadParameter("deux noms séparés par une virgule", param_hint="--annotators")
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    summary, table = annotator_agreement(con, *names)
    if not summary["n_windows"]:
        typer.echo("aucune fenêtre écoutée par les deux")
        raise typer.Exit(1)
    typer.echo(
        f"{summary['n_windows']} fenêtres communes : même label {summary['label_agreement']:.0%}, "
        f"même réponse blanci/non {summary['blanci_agreement']:.0%}, accord sur les positifs "
        f"{summary['positive_agreement']:.2f} ({summary['n_blanci_first']} contre "
        f"{summary['n_blanci_second']} positifs)"
    )
    _write_csv(
        table.reset_index(),
        config_path(cfg, "reports") / f"accord_{names[0]}_{names[1]}.csv",
        "tableau croisé",
    )


@app.command()
def throughput(
    ctx: typer.Context,
    encoders: Annotated[str, typer.Option(help="Noms séparés par des virgules.")],
    n_windows: Annotated[int, typer.Option(help="Fenêtres de bruit par mesure.")] = 64,
) -> None:
    """Débit, mémoire, dimension et jetons de chaque encodeur (§2), projetés sur une campagne.

    Bruit synthétique : aucun enregistrement n'est lu. Un processus par encodeur.
    """
    cfg = _cfg(ctx)
    config = ctx.parent.params.get("config") if ctx.parent else None
    rows = []
    for name in _split(encoders):
        typer.echo(f"{name} : chargement et mesure…")
        row = measure_in_subprocess(name, Path(config).resolve() if config else None, n_windows)
        rows.append(row)
        if "error" in row:
            typer.echo(f"  échec : {row['error']}")
        else:
            typer.echo(
                f"  {row['windows_per_s']:.1f} fenêtres/s, ×{row['realtime_factor']:.0f} temps "
                f"réel, campagne {row['campaign_h']:.0f} h, {row['peak_memory_mb']:.0f} Mo, "
                f"dim {row['dim']}{' (jetons)' if row['has_tokens'] else ''}"
            )
    paths = write_throughput_report(rows, config_path(cfg, "reports"), machine_description())
    typer.echo(f"rapport : {paths['markdown']}")


@app.command()
def candidates(
    ctx: typer.Context,
    from_table: Annotated[
        Path | None,
        typer.Option("--from", help="Export Blancinet : ses détections jamais écoutées."),
    ] = None,
    congeners: Annotated[
        str | None,
        typer.Option(help="Encodeur perch_v2 encodé : fenêtres où il entend un congénère."),
    ] = None,
    per_site: Annotated[int, typer.Option(help="Candidats Blancinet ou Perch par site.")] = 30,
    random: Annotated[int, typer.Option(help="Fenêtres tirées au hasard (heures de pic).")] = 10,
    whole: Annotated[
        int,
        typer.Option(
            "--entiers",
            help="Enregistrements à écouter en entier, par micro et heure (audit, jeu gelé).",
        ),
    ] = 0,
    reason: Annotated[
        str, typer.Option(help="Motif des enregistrements entiers : audit_aleatoire, jeu_gele…")
    ] = "audit_aleatoire",
    sites: Annotated[str | None, typer.Option(help="Sites, ex. « CDR,PatawaOuest ».")] = None,
    name: Annotated[str, typer.Option(help="Nom de la file : candidats_<nom>.csv.")] = "lot1",
    seed: Annotated[int, typer.Option(help="Graine du tirage.")] = 0,
) -> None:
    """File d'écoute pour le poste d'annotation (§5) : Blancinet réparti, strate aléatoire,
    enregistrements entiers (audit aléatoire et jeu gelé, §6)."""
    cfg = _cfg(ctx)
    con = connect(config_path(cfg, "db"))
    wanted = _split(sites) or None
    parts = []
    if from_table is not None:
        parts.append(blancinet_candidates(con, from_table, cfg, per_site, wanted, seed))
    if congeners:
        parts.append(congener_candidates(con, congeners, per_site, wanted))
    if random:
        parts.append(random_candidates(con, cfg, random, wanted, seed=seed))
    if whole:
        parts.append(recording_candidates(con, cfg, whole, wanted, reason=reason, seed=seed))
    if not parts:
        raise typer.BadParameter("rien à tirer : --from, --congeners, --random ou --entiers")
    queue = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    if queue.empty:
        typer.echo("aucun candidat")
        raise typer.Exit(1)
    for (site, reason), n in queue.groupby(["site", "reason"]).size().items():
        typer.echo(f"  {site:<14} {reason:<20} {n}")
    _write_csv(
        queue.reset_index(drop=True),
        config_path(cfg, "reports") / f"candidats_{name}.csv",
        f"{len(queue)} candidats",
    )


@app.command()
def annotate(ctx: typer.Context) -> None:
    """Ouvre le poste d'annotation dans le navigateur (groupe `app` : uv sync --group app)."""
    import subprocess

    app_file = Path(__file__).with_name("app.py")
    command = [sys.executable, "-m", "streamlit", "run", str(app_file), "--"]
    config = ctx.parent.params.get("config") if ctx.parent else None
    if config:
        command += ["--config", str(Path(config).resolve())]
    raise typer.Exit(subprocess.call(command))


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
    heard: Counter[str] = Counter()
    for (qc,) in con.execute("SELECT qc_flags FROM recordings WHERE qc_flags IS NOT NULL"):
        parsed = parse_flags(qc)
        flags.update(k for k, v in parsed.items() if v is True)
        if "annotated" in parsed:
            heard.update(["annotated", *parsed["annotated"]])
    if flags:
        typer.echo("Drapeaux calculés : " + ", ".join(f"{k}={v}" for k, v in flags.items()))
    if heard:
        typer.echo(
            f"Drapeaux posés à l'écoute ({heard.pop('annotated')} enregistrements annotés) : "
            + (", ".join(f"{k}={v}" for k, v in heard.items()) or "aucun")
        )
    typer.echo("Labels :")
    for r in con.execute(
        "SELECT source, label, COUNT(*) n FROM labels "
        "GROUP BY source, label ORDER BY source, n DESC"
    ):
        typer.echo(f"  {r['source']} / {r['label']} : {r['n']}")
