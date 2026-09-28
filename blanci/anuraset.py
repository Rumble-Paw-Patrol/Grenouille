"""Pré-benchmark AnuraSet (§2) : classer les encodeurs sur des anoures néotropicaux.

AnuraSet (Cañas et al. 2023, Zenodo 8342596, licence CC BY) : 1 612 enregistrements d'une
minute, 4 sites du Brésil, 42 espèces, chaque chant daté (début, fin, qualité L/M/H). On y
choisit deux ou trois anoures à note brève en 3–6 kHz, comme A. blanci, et on juge chaque
encodeur exactement comme sur les données ONF : mêmes sondes, mêmes métriques, plis groupés
par site. Indicateur, pas garantie : autres espèces, autre forêt, autres enregistreurs.

Tout vit à part des données ONF (`config/anuraset.yaml` : base, stocks, rapports séparés).

- `prepare` : extrait `raw_data.zip` sous <racine>/anuraset/<site>/ et l'inventorie ;
- `read_strong_labels` : un chant par ligne (fichier, site, début, fin, espèce, qualité) ;
- `species_profile` (+ `dominant_frequencies`) : de quoi choisir les espèces ;
- `window_labels` : fenêtre positive si elle contient un chant entier de l'espèce (ou tient
  dans un chœur annoté), négative si aucun chant de l'espèce ne la touche ; les fenêtres qui
  coupent un chant sont écartées ;
- `run_anuraset_benchmark` : sondes du §3 par encodeur et par espèce, comparaisons appariées.
"""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from blanci.benchmark import compare_encoders, probe_table
from blanci.config import config_path
from blanci.db import encoder_params
from blanci.head import calibration_options
from blanci.ingest import ingest
from blanci.store import EmbeddingStore

DATASET = "anuraset"
QUALITIES = {"L": "low", "M": "medium", "H": "high"}


def file_key(name: str) -> str:
    return Path(str(name).replace("\\", "/")).stem.lower()


# --- Préparation -------------------------------------------------------------------------------


def extract_raw(archive: Path, root: Path) -> int:
    """Extrait les fichiers audio de `raw_data.zip` sous <root>/anuraset/<site>/<fichier>.

    Le premier niveau de l'archive (« raw_data/ ») est retiré ; un fichier déjà extrait est
    sauté (reprise possible). Renvoie le nombre de fichiers extraits.
    """
    target = Path(root) / DATASET
    n = 0
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            parts = Path(member.filename).parts
            if member.is_dir() or Path(member.filename).suffix.lower() not in (".wav", ".flac"):
                continue
            if len(parts) < 2 or parts[-1].startswith("."):
                continue
            site = parts[-2]
            out = target / site / parts[-1]
            if out.exists() and out.stat().st_size == member.file_size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(out, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            n += 1
    return n


def prepare(con: sqlite3.Connection, cfg: dict) -> dict[str, Any]:
    """Extraction (si besoin) puis inventaire, sans contrôle qualité ni empreinte."""
    root = config_path(cfg, "raw")
    acfg = cfg["anuraset"]
    extracted = extract_raw(Path(acfg["archive"]), root)
    report = ingest(con, root, DATASET, cfg, run_qc=False, hash_file=False)
    return {"extracted": extracted, "added": report.added, "errors": len(report.errors)}


# --- Labels ------------------------------------------------------------------------------------


def read_strong_labels(source: Path) -> pd.DataFrame:
    """Chants datés depuis `strong_labels.zip` (ou son dossier extrait).

    Fichier texte par enregistrement : « début<TAB>fin<TAB>ESPÈCE_Q » (Q = L, M, H).
    """
    rows = []

    def parse(name: str, text: str) -> None:
        site = Path(name).parent.name
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            code, _, quality = parts[2].rpartition("_")
            if not code:
                code, quality = parts[2], ""
            rows.append(
                {
                    "file_key": file_key(name),
                    "site": site,
                    "start_s": float(parts[0]),
                    "end_s": float(parts[1]),
                    "species": code,
                    "quality": QUALITIES.get(quality, quality),
                }
            )

    source = Path(source)
    if source.is_dir():
        for path in sorted(source.rglob("*.txt")):
            parse(str(path), path.read_text(encoding="utf-8", errors="replace"))
    else:
        with zipfile.ZipFile(source) as z:
            for name in z.namelist():
                if name.endswith(".txt"):
                    parse(name, z.read(name).decode("utf-8", errors="replace"))
    return pd.DataFrame(
        rows, columns=["file_key", "site", "start_s", "end_s", "species", "quality"]
    )


def species_profile(calls: pd.DataFrame, max_call_s: float = 5.0) -> pd.DataFrame:
    """Par espèce : chants, enregistrements, sites, durée des chants (médiane, 90e centile).

    Les annotations plus longues que `max_call_s` sont des chœurs continus (« 0–60 s ») :
    comptées à part, exclues des durées.
    """
    calls = calls.assign(duration=calls["end_s"] - calls["start_s"])
    brief = calls[calls["duration"] <= max_call_s]
    table = calls.groupby("species").agg(
        n_calls=("file_key", "size"),
        n_recordings=("file_key", "nunique"),
        n_sites=("site", "nunique"),
    )
    table["n_long"] = (calls["duration"] > max_call_s).groupby(calls["species"]).sum()
    table["duration_median_s"] = brief.groupby("species")["duration"].median()
    table["duration_p90_s"] = brief.groupby("species")["duration"].quantile(0.9)
    return table.sort_values("n_calls", ascending=False).reset_index()


def dominant_frequencies(
    calls: pd.DataFrame,
    raw_root: Path,
    recordings: pd.DataFrame,
    per_species: int = 30,
    max_call_s: float = 1.0,
    fmin_hz: float = 500.0,
    seed: int = 0,
) -> pd.Series:
    """Fréquence dominante médiane (Hz) des chants brefs de chaque espèce, sur un échantillon
    de `per_species` chants de qualité moyenne ou haute (lecture seule de l'audio)."""
    rng = np.random.default_rng(seed)
    paths = recordings.assign(file_key=recordings["path"].map(file_key)).set_index("file_key")
    brief = calls[
        (calls["end_s"] - calls["start_s"] <= max_call_s)
        & calls["quality"].isin(["medium", "high"])
        & calls["file_key"].isin(paths.index)
    ]
    out = {}
    for species, group in brief.groupby("species"):
        sample = group.iloc[rng.permutation(len(group))[:per_species]]
        peaks = []
        for call in sample.itertuples():
            with sf.SoundFile(Path(raw_root) / paths.at[call.file_key, "path"]) as f:
                f.seek(int(call.start_s * f.samplerate))
                n = max(int((call.end_s - call.start_s) * f.samplerate), 256)
                x = f.read(n, dtype="float32", always_2d=True)[:, 0]
                sr = f.samplerate
            spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
            freqs = np.fft.rfftfreq(len(x), 1 / sr)
            keep = freqs >= fmin_hz
            if keep.any():
                peaks.append(freqs[keep][spectrum[keep].argmax()])
        if peaks:
            out[species] = float(np.median(peaks))
    return pd.Series(out, name="dominant_hz")


def suggest_species(
    profile: pd.DataFrame,
    band_hz: tuple[float, float] = (3000.0, 6000.0),
    max_duration_s: float = 0.3,
    min_calls: int = 300,
    min_sites: int = 2,
) -> pd.DataFrame:
    """Espèces proches d'A. blanci : note brève, dominante en 3–6 kHz, assez de chants, sur
    au moins deux sites (sinon les plis par site n'ont pas de sens)."""
    ok = (
        profile["dominant_hz"].between(*band_hz)
        & (profile["duration_median_s"] <= max_duration_s)
        & (profile["n_calls"] >= min_calls)
        & (profile["n_sites"] >= min_sites)
    )
    return profile[ok].sort_values("n_calls", ascending=False)


def window_labels(
    windows: pd.DataFrame, calls: pd.DataFrame, species: str, eps: float = 1e-6
) -> np.ndarray:
    """1 si la fenêtre contient un chant entier de l'espèce (ou tient dans un chœur annoté
    d'un seul tenant), 0 si aucun ne la touche, NaN sinon (chant coupé : écarté).
    `windows` : file_key, offset_s, dur_s."""
    y = np.zeros(len(windows))
    target = calls[calls["species"] == species]
    by_file = {k: g[["start_s", "end_s"]].to_numpy() for k, g in target.groupby("file_key")}
    for i, (key, offset, dur) in enumerate(
        zip(windows["file_key"], windows["offset_s"], windows["dur_s"], strict=True)
    ):
        spans = by_file.get(key)
        if spans is None:
            continue
        start, end = offset, offset + dur
        overlap = (spans[:, 0] < end) & (spans[:, 1] > start)
        inside = (spans[:, 0] >= start - eps) & (spans[:, 1] <= end + eps)
        within = (spans[:, 0] <= start + eps) & (spans[:, 1] >= end - eps)  # chœur continu
        if inside.any() or within.any():
            y[i] = 1.0
        elif overlap.any():
            y[i] = np.nan
    return y


# --- Benchmark ---------------------------------------------------------------------------------


def _encoder_windows(
    con: sqlite3.Connection, cfg: dict, encoder_id: str
) -> tuple[pd.DataFrame, np.ndarray]:
    store = EmbeddingStore(config_path(cfg, "embeddings"), encoder_id)
    meta, emb = store.load()
    if not len(meta):
        raise ValueError(f"aucun embedding AnuraSet pour {encoder_id} (blanci embed)")
    recordings = pd.read_sql_query("SELECT recording_id, path, site FROM recordings", con)
    recordings["file_key"] = recordings["path"].map(file_key)
    # Jointure à gauche : l'ordre des lignes reste celui des embeddings.
    meta = meta.merge(
        recordings[["recording_id", "file_key", "site"]], on="recording_id", how="left"
    )
    window_s = encoder_params(con, encoder_id)["window_s"]
    return meta.assign(dur_s=window_s), emb


def species_rows(
    meta: pd.DataFrame,
    y: np.ndarray,
    species: str,
    negatives_per_positive: int,
    rng: np.random.Generator,
    encoder_id: str = "",
) -> np.ndarray:
    """Fenêtres jugées pour une espèce : tous ses positifs, et `negatives_per_positive`
    négatifs par positif, tirés dans chaque site (au moins autant qu'un positif)."""
    known = ~np.isnan(y)
    pos = np.flatnonzero(known & (y == 1))
    if len(pos) < 10:
        raise ValueError(f"{species} : {len(pos)} fenêtres positives pour {encoder_id}")
    neg = []
    for site, idx in meta[known & (y == 0)].groupby("site").groups.items():
        n_site_pos = int(((y == 1) & (meta["site"] == site).to_numpy()).sum())
        k = min(len(idx), max(n_site_pos, 1) * negatives_per_positive)
        neg.extend(rng.choice(np.asarray(idx), size=k, replace=False).tolist())
    return np.sort(np.r_[pos, np.asarray(neg, dtype=int)])


# R78 (DECISIONS n° 125) : têtes comparées sur AnuraSet quand on les choisit ailleurs que sur
# Mataroni. Toutes les têtes et régularisations sans jetons ; les tokens d'AnuraSet ne sont pas
# calculés. Le groupe des régularisations est le site (R13, R19–R21, R37, R66).
HEADS = [
    "logistic",
    "prototype",
    "knn:k=5",
    "lda_shrunk",
    "logistic+R19",
    "logistic+R20",
    "logistic+R21",
    "logistic+R37",
    "logistic+R13",
    "loss:focal",
    "loss:sigmoid",
    "dann",
]


def run_anuraset_heads(
    con: sqlite3.Connection,
    cfg: dict,
    encoder_id: str,
    species: list[str],
    calls: pd.DataFrame,
    methods: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """R78 : le benchmark des têtes sur AnuraSet, **un pli par site** (leave-one-site-out).

    Chaque tête est apprise sur 3 sites et jugée sur le 4ᵉ ; son C, et le weight decay des
    têtes torch, sont choisis par des plis internes eux aussi par site : les réglages retenus
    sont ceux qui passent d'un site à l'autre, pas d'un micro à l'autre d'un même site. En
    attendant des positifs hors de Mataroni, c'est là qu'on juge ce qui généralise.

    Renvoie `table` (espèce × tête × niveau : AP, IC, rappels), `sites` (AP de chaque tête sur
    chaque site tenu à l'écart), `comparisons` (contre `head.reference`, apparié) et
    `selection` (R74 : la gagnante doit-elle sa place à la chance ?)."""
    import importlib.util

    from blanci.evaluate import average_precision
    from blanci.head import oof_scores
    from blanci.head_benchmark import _inputs, compare_to_reference, expand_methods, level_rows
    from blanci.regularization import (
        Context,
        canonical,
        domain_statistics,
        fold_ids,
        needs_domain,
        needs_pool,
        regularizer_for,
        selection_estimate,
    )

    acfg, head_cfg = cfg["anuraset"], cfg["head"]
    methods = expand_methods(methods) or list(acfg.get("heads") or HEADS)
    if importlib.util.find_spec("torch") is None:
        methods = [m for m in methods if not m.startswith(("dann", "gated"))]
    rng = np.random.default_rng(head_cfg["seed"])
    meta, emb = _encoder_windows(con, cfg, encoder_id)
    sites = meta["site"].astype(str).to_numpy()
    domain = domain_statistics(emb, sites, "site") if needs_domain(methods) else None
    reference = canonical(head_cfg.get("reference", "logistic"))
    tables, per_site, comparisons, selections = [], [], [], []
    for sp in species:
        y_all = window_labels(meta, calls, sp)
        rows = species_rows(meta, y_all, sp, acfg["negatives_per_positive"], rng, encoder_id)
        X, y = emb[rows].astype(np.float32), y_all[rows].astype(int)
        groups, recordings = sites[rows], meta["recording_id"].to_numpy()[rows]
        context = Context(
            groups,
            domain=domain,
            domain_rows=None if domain is None else domain.rows(groups),
            classes=np.where(y == 1, "blanci", "fond").astype(object),
        )
        if needs_pool(methods):  # R81 : les fenêtres que ce tirage n'a pas retenues
            size = int(((cfg.get("regularization") or {}).get("R81") or {}).get("pool", 20000))
            others = np.setdiff1d(np.arange(len(meta)), rows)
            pool = rng.choice(others, size=min(size, len(others)), replace=False)
            context.pool, context.pool_groups = emb[pool].astype(np.float32), sites[pool]
            if domain is not None:
                context.pool_domain_rows = domain.rows(sites[pool])
        n_sites = len(np.unique(groups))
        scores, folds = {}, None
        for spec in methods:
            name, base, regularizer = regularizer_for(spec, cfg, context)
            method, inputs = _inputs(base, X, None)
            oof = oof_scores(
                inputs,
                y,
                groups,
                n_splits=n_sites,  # un pli par site
                method=method,
                C_grid=head_cfg["C_grid"],
                seed=head_cfg["seed"],
                regularizer=regularizer,
                **calibration_options(cfg),
            )
            scores[name], folds = oof.values, oof.folds
            labels = {"species": sp, "head": name, "n_sites": n_sites}
            tables += level_rows(oof, y, recordings, cfg, labels)
            for site in np.unique(groups):
                mask = groups == site
                per_site.append(
                    {
                        "species": sp,
                        "head": name,
                        "held_out_site": site,
                        "n_pos": int(y[mask].sum()),
                        "ap": average_precision(y[mask], oof.values[mask]),
                    }
                )
        if reference in scores:
            comparisons.append(
                compare_to_reference(scores, reference, y, recordings, cfg).assign(species=sp)
            )
        if len(scores) > 1:
            chosen = selection_estimate(scores, y, fold_ids(len(y), folds), recordings)
            selections.append({"species": sp, **{k: v for k, v in chosen.items()}})
    return {
        "table": pd.DataFrame(tables),
        "sites": pd.DataFrame(per_site),
        "comparisons": pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame(),
        "selection": pd.DataFrame(selections),
    }


def write_anuraset_heads_report(out: dict, encoder_id: str, reports_dir: Path) -> Path:
    from blanci.benchmark import to_markdown

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"anuraset_tetes_{encoder_id}".replace(":", "_")
    for key, frame in out.items():
        frame.to_csv(reports_dir / f"{stem}_{key}.csv", index=False)
    table = out["table"]
    shown = ["species", "head", "level", "n_pos", "n_neg", "ap", "ap_lo", "ap_hi"]
    shown += ["ap_fold_mean", "ap_raw", "recall@p0.1"]  # par site ; avant recalibration
    text = [
        f"# Têtes sur AnuraSet, un pli par site (R78) : {encoder_id}",
        "",
        "Chaque tête est apprise sur les autres sites et jugée sur le site tenu à l'écart ; ses "
        "réglages sont choisis de même. Indicateur de généralisation entre sites, sur d'autres "
        "espèces que A. blanci.",
        "",
    ]
    for level in ("window", "recording"):
        part = table[table["level"] == level]
        part = part.sort_values(["species", "ap"], ascending=[True, False])
        text += [f"## Niveau {level}", "", to_markdown(part[[c for c in shown if c in part]]), ""]
    if not out["sites"].empty:
        pivot = (
            out["sites"]
            .pivot_table(index=["species", "head"], columns="held_out_site", values="ap")
            .reset_index()
        )
        text += ["## AP sur chaque site tenu à l'écart", "", to_markdown(pivot), ""]
    if not out["comparisons"].empty:
        compared = to_markdown(out["comparisons"])
        text += ["## Contre la référence (enregistrements)", "", compared, ""]
    if not out["selection"].empty:
        text += ["## Sélection honnête (R74, R80)", "", to_markdown(out["selection"]), ""]
    path = reports_dir / f"{stem}.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path


def run_anuraset_benchmark(
    con: sqlite3.Connection,
    cfg: dict,
    encoder_ids: list[str],
    species: list[str],
    calls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tableau encodeur × espèce × sonde × niveau, comparaisons appariées par espèce).

    Négatifs : `negatives_per_positive` fenêtres par positif, tirées dans chaque site ; plis
    groupés par site ; niveau « recording » = l'enregistrement d'une minute.
    """
    acfg, bench, head = cfg["anuraset"], cfg["benchmark"], cfg["head"]
    rng = np.random.default_rng(head["seed"])
    tables, comparisons = [], []
    for sp in species:
        per_encoder = {}
        for encoder_id in encoder_ids:
            meta, emb = _encoder_windows(con, cfg, encoder_id)
            y = window_labels(meta, calls, sp)
            rows = species_rows(meta, y, sp, acfg["negatives_per_positive"], rng, encoder_id)
            table, scores = probe_table(
                emb[rows].astype(np.float32),
                y[rows].astype(int),
                meta["site"].to_numpy()[rows],
                meta["recording_id"].to_numpy()[rows],
                n_splits=head["n_splits"],
                C_grid=head["C_grid"],
                precisions=tuple(bench["precisions"]),
                n_boot=bench["n_boot"],
                seed=head["seed"],
                calibration=calibration_options(cfg),
            )
            table.insert(0, "species", sp)
            table.insert(0, "encoder_id", encoder_id)
            table["n_sites"] = meta["site"].iloc[rows].nunique()
            tables.append(table)
            per_encoder[encoder_id] = (
                scores["logistic"],
                y[rows].astype(int),
                meta["recording_id"].to_numpy()[rows],
            )
        if len(per_encoder) > 1:
            comparisons.append(compare_encoders(per_encoder, cfg).assign(species=sp))
    results = pd.concat(tables, ignore_index=True)
    compared = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    return results, compared


def write_anuraset_report(
    results: pd.DataFrame, comparisons: pd.DataFrame, reports_dir: Path
) -> Path:
    from blanci.benchmark import to_markdown

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(reports_dir / "anuraset_benchmark.csv", index=False)
    comparisons.to_csv(reports_dir / "anuraset_comparaisons.csv", index=False)
    columns = [
        "encoder_id",
        "species",
        "probe",
        "level",
        "n_pos",
        "n_neg",
        "n_sites",
        "ap",
        "ap_lo",
        "ap_hi",
        "recall@p0.1",
        "recall@p0.5",
    ]
    text = [
        "# Pré-benchmark AnuraSet (§2)",
        "",
        "Plis groupés par site. Indicateur du classement des encodeurs sur des anoures "
        "néotropicaux, pas une mesure sur A. blanci.",
        "",
    ]
    for level in ("window", "recording"):
        part = results[results["level"] == level].sort_values(
            ["species", "ap"], ascending=[True, False]
        )
        text += [f"## Niveau {level}", "", to_markdown(part[[c for c in columns if c in part]]), ""]
    if not comparisons.empty:
        text += [
            "## Comparaisons appariées (sonde logistique, enregistrements)",
            "",
            to_markdown(comparisons),
            "",
        ]
    path = reports_dir / "anuraset_benchmark.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path
