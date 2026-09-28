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
  coupent un chant sont écartées, sauf pour un chant qu'aucune fenêtre ne contient entier
  (fenêtres jointives) : la fenêtre qui en porte la plus grande part le garde ; les fichiers où
  les labels faibles signalent l'espèce sans chant daté sont écartés (`weak_only_files`) ;
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
from blanci.config import config_path, project_path
from blanci.db import encoder_params
from blanci.evaluate import with_holm
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
    extracted = extract_raw(project_path(acfg["archive"]), root)
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
    windows: pd.DataFrame,
    calls: pd.DataFrame,
    species: str,
    eps: float = 1e-6,
    unsure_files: set[str] | frozenset[str] = frozenset(),
) -> np.ndarray:
    """1 si la fenêtre contient un chant entier de l'espèce (ou tient dans un chœur annoté
    d'un seul tenant), 0 si aucun ne la touche, NaN sinon (chant coupé : écarté).

    Un chant qu'aucune fenêtre ne contient entier (à cheval sur la jonction de deux fenêtres
    jointives, `encoders.overlap: 0`, ou plus long que le pas) rend positive la fenêtre qui en
    porte la plus grande part, la première à égalité : sans cela il n'aurait aucune fenêtre
    positive, et les chants centrés seraient seuls jugés (DECISIONS n° 138). `unsure_files`
    (`weak_only_files`) : fichiers où les labels faibles signalent l'espèce sans chant daté ;
    toutes leurs fenêtres sont écartées (NaN) plutôt que comptées négatives.
    `windows` : file_key, offset_s, dur_s."""
    y = np.zeros(len(windows))
    keys = windows["file_key"].to_numpy()
    starts = windows["offset_s"].to_numpy(dtype=float)
    ends = starts + windows["dur_s"].to_numpy(dtype=float)
    target = calls[calls["species"] == species]
    by_file = {
        k: g[["start_s", "end_s"]].to_numpy(dtype=float) for k, g in target.groupby("file_key")
    }
    for key, rows in pd.DataFrame({"key": keys}).groupby("key").indices.items():
        spans = by_file.get(key)
        if spans is None:
            continue
        a, b = spans[:, :1], spans[:, 1:]  # (chants, 1) face aux fenêtres (1, n)
        s, e = starts[rows][None, :], ends[rows][None, :]
        shared = np.minimum(b, e) - np.maximum(a, s)  # durée commune chant × fenêtre
        touches = shared > 0
        whole = ((a >= s - eps) & (b <= e + eps)) | ((a <= s + eps) & (b >= e - eps))  # chœur
        label = np.where(whole.any(axis=0), 1.0, np.where(touches.any(axis=0), np.nan, 0.0))
        orphan = touches.any(axis=1) & ~whole.any(axis=1)  # aucune fenêtre ne le tient entier
        if orphan.any():
            label[np.argmax(np.where(touches[orphan], shared[orphan], -np.inf), axis=1)] = 1.0
        y[rows] = label
    if unsure_files:
        y[np.isin(keys, list(unsure_files))] = np.nan
    return y


def read_weak_labels(path: Path | str | None) -> pd.DataFrame | None:
    """Labels faibles d'AnuraSet (`weak_labels.csv` : un fichier par ligne, une colonne
    `SPECIES_<code>` par espèce), avec `file_key` ; None sans chemin. Un chemin donné mais
    absent est une erreur : sans ces labels, ni vrais négatifs à encoder, ni fichiers douteux
    à écarter (DECISIONS n° 138)."""
    if not path:
        return None
    path = project_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"labels faibles introuvables : {path} (anuraset.weak_labels ; retirer la clé pour "
            "s'en passer, sans vrais négatifs ni fichiers douteux écartés)"
        )
    weak = pd.read_csv(path)
    return weak.assign(file_key=weak["AUDIO_FILE_ID"].map(file_key))


def weak_only_files(weak: pd.DataFrame | None, calls: pd.DataFrame, species: str) -> set[str]:
    """Fichiers où les labels faibles signalent `species` sans aucun chant daté de l'espèce :
    ses chants y sont peut-être, non datés. Leurs fenêtres ne sont ni positives ni négatives
    pour cette espèce (`window_labels(unsure_files=…)`, DECISIONS n° 138). Vide sans labels
    faibles ou si l'espèce n'y figure pas."""
    column = f"SPECIES_{species}"
    if weak is None or column not in weak:
        return set()
    flagged = set(weak.loc[weak[column] > 0, "file_key"])
    return flagged - set(calls.loc[calls["species"] == species, "file_key"])


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
    weak = read_weak_labels(acfg.get("weak_labels"))
    meta, emb = _encoder_windows(con, cfg, encoder_id)
    sites = meta["site"].astype(str).to_numpy()
    domain = domain_statistics(emb, sites, "site") if needs_domain(methods) else None
    reference = canonical(head_cfg.get("reference", "logistic"))
    tables, per_site, comparisons, selections = [], [], [], []
    for sp in species:
        y_all = window_labels(meta, calls, sp, unsure_files=weak_only_files(weak, calls, sp))
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
                compare_to_reference(scores, reference, y, recordings, cfg).assign(
                    species=sp, level="recording"
                )
            )
            comparisons.append(
                window_comparisons(scores, reference, y, recordings, cfg).assign(species=sp)
            )
        if len(scores) > 1:
            chosen = selection_estimate(scores, y, fold_ids(len(y), folds), recordings)
            selections.append({"species": sp, **{k: v for k, v in chosen.items()}})
    compared = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    return {
        "table": pd.DataFrame(tables),
        "sites": pd.DataFrame(per_site),
        "comparisons": _holm_by_level(compared),
        "selection": pd.DataFrame(selections),
    }


def _holm_by_level(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Holm sur toutes les têtes et toutes les espèces d'un même niveau (fenêtre,
    enregistrement) : 13 têtes × 4 espèces jugées à 5 % donnent 2 à 3 « victoires » par hasard
    (DECISIONS n° 139)."""
    if comparisons.empty or "level" not in comparisons:
        return with_holm(comparisons)
    parts = [with_holm(part) for _, part in comparisons.groupby("level", sort=False)]
    return pd.concat(parts).sort_index()


def window_comparisons(
    scores: dict[str, np.ndarray],
    reference: str,
    y: np.ndarray,
    recordings: np.ndarray,
    cfg: dict,
) -> pd.DataFrame:
    """Chaque tête contre la référence au niveau **fenêtre**, bootstrap apparié qui tire des
    enregistrements entiers. Sur AnuraSet, une espèce commune chante dans presque chaque
    enregistrement d'une minute : l'AP par enregistrement n'y est plus définie, celle par
    fenêtre si (n° 136). Les plis sont des sites, mais 2 à 4 sites ne suffisent pas à tirer
    des sites : l'intervalle reste optimiste, les p-valeurs sont corrigées de Holm
    (`_holm_by_level`, n° 139)."""
    from blanci.evaluate import paired_bootstrap

    rows = []
    for method, values in scores.items():
        if method == reference:
            continue
        result = paired_bootstrap(
            y,
            values,
            scores[reference],
            recordings,
            n_boot=cfg["benchmark"]["n_boot"],
            seed=cfg["head"]["seed"],
        )
        rows.append({"head": method, "reference": reference, **result, "level": "window"})
    return pd.DataFrame(rows)


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
    weak = read_weak_labels(acfg.get("weak_labels"))
    tables, comparisons = [], []
    for sp in species:
        per_encoder = {}
        for encoder_id in encoder_ids:
            meta, emb = _encoder_windows(con, cfg, encoder_id)
            y = window_labels(meta, calls, sp, unsure_files=weak_only_files(weak, calls, sp))
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
    return results, with_holm(compared)  # toutes les paires de toutes les espèces (n° 139)


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


# --- Campagne d'un seul tenant (28/09) -----------------------------------------------------------

# Têtes de la campagne, par ordre de priorité (DECISIONS n° 136). Chacune répond à une question
# sur la généralisation d'un site à l'autre ; toutes sont jugées sur un site tenu à l'écart.
CAMPAIGN_HEADS = [
    "logistic",  # référence
    "prototype",  # amorcer un site avec quelques exemples, sans apprentissage
    "knn:k=5",
    "logistic+R19",  # fond du site retiré par le stock non annoté (amorçage d'un site)
    "logistic+R20",
    "logistic+R21",  # directions du site effacées
    "dann",  # le même, appris
    "logistic+R37",  # biais par site, σ fixé
    "logistic+R37=glmm",  # biais par site, σ estimé (GLMM)
    "logistic+R19+R37",
    "logistic+R13",  # chaque site pèse autant
    "loss:focal",  # déséquilibre des classes
    "logistic+R18=64",  # ACP : jette-t-elle le chant ?
    "lda_shrunk",
]

# Choix des espèces, du plus strict au plus large : on descend d'un cran tant qu'il manque
# des espèces. (étiquette, bande de fréquence dominante en Hz ou None, durée médiane max en s,
# chants min, sites min)
SPECIES_LEVELS = [
    ("proche d'A. blanci : note ≤ 0,3 s, 3–6 kHz, ≥ 300 chants", (3000.0, 6000.0), 0.3, 300, 2),
    ("élargi : note ≤ 0,5 s, 2–7 kHz, ≥ 150 chants", (2000.0, 7000.0), 0.5, 150, 2),
    ("multi-sites : note ≤ 1 s, ≥ 100 chants, fréquence libre", None, 1.0, 100, 2),
]


def choose_species(profile: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Jusqu'à `n` espèces pour la campagne, par niveaux (`SPECIES_LEVELS`) : d'abord celles
    qui ressemblent à A. blanci (note brève en 3–6 kHz), puis on élargit. Toujours au moins
    deux sites : sans cela, un pli par site n'a pas de sens. Colonne `criterion` : le niveau
    qui a retenu l'espèce."""
    chosen: list[pd.DataFrame] = []
    taken: set[str] = set()
    for label, band, max_duration, min_calls, min_sites in SPECIES_LEVELS:
        ok = (
            (profile["duration_median_s"] <= max_duration)
            & (profile["n_calls"] >= min_calls)
            & (profile["n_sites"] >= min_sites)
            & ~profile["species"].isin(taken)
        )
        if band is not None:
            if "dominant_hz" not in profile:
                continue
            ok &= profile["dominant_hz"].between(*band)
        found = profile[ok].sort_values(["n_sites", "n_calls"], ascending=False)
        found = found.head(n - len(taken)).assign(criterion=label)
        chosen.append(found)
        taken |= set(found["species"])
        if len(taken) >= n:
            break
    return pd.concat(chosen, ignore_index=True) if chosen else profile.head(0)


def rank_encoders(results: pd.DataFrame, level: str = "window") -> pd.DataFrame:
    """Encodeurs classés par l'AP moyenne de la sonde logistique sur les espèces (un pli par
    site), avec l'AP moyenne par pli et le nombre d'espèces. Niveau fenêtre par défaut : une
    espèce commune chante dans presque tous les enregistrements d'une minute, et l'AP par
    enregistrement n'y est plus définie (répétition générale du n° 136)."""
    part = results[(results["probe"] == "logistic") & (results["level"] == level)]
    columns = {"ap": "mean"} | ({"ap_fold_mean": "mean"} if "ap_fold_mean" in part else {})
    table = part.groupby("encoder_id").agg(columns | {"species": "nunique"})
    return table.rename(columns={"species": "n_species"}).sort_values("ap", ascending=False)


def rank_heads(table: pd.DataFrame, level: str = "window") -> pd.DataFrame:
    """Têtes classées par leur AP moyenne sur les espèces (poolée et par site), avec leur rang
    moyen : une tête qui gagne sur une espèce et s'effondre sur une autre descend."""
    part = table[table["level"] == level].copy()
    part["rank"] = part.groupby("species")["ap"].rank(ascending=False)
    columns = {"ap": "mean", "rank": "mean"}
    if "ap_fold_mean" in part:
        columns["ap_fold_mean"] = "mean"
    out = part.groupby("head").agg(columns).rename(columns={"rank": "mean_rank"})
    return out.sort_values(["mean_rank", "ap"], ascending=[True, False]).reset_index()


def campaign_recordings(
    recordings: pd.DataFrame, calls: pd.DataFrame, weak_labels: Path | None = None
) -> pd.DataFrame:
    """Enregistrements à encoder pour la campagne : ceux qui ont au moins un chant daté
    (strong labels), et ceux que les labels faibles disent sans aucune espèce (vrais négatifs,
    un quart d'AnuraSet). Écartés : ceux qui n'ont aucun chant daté mais où les labels faibles
    signalent une espèce (13 sur 1 612).

    Un fichier gardé pour les chants datés d'une espèce peut en signaler une autre sans chant
    daté : c'est à l'évaluation, espèce par espèce, que ses fenêtres sont alors écartées
    (`weak_only_files`, DECISIONS n° 138). `weak_labels` donné mais absent : erreur
    (`read_weak_labels`)."""
    keys = recordings["path"].map(file_key)
    keep = keys.isin(set(calls["file_key"]))
    weak = read_weak_labels(weak_labels)
    if weak is not None:
        species = [c for c in weak.columns if c.startswith("SPECIES_")]
        keep |= keys.isin(set(weak.loc[weak[species].sum(axis=1) == 0, "file_key"]))
    return recordings[keep.to_numpy()]


def run_anuraset_campaign(
    con: sqlite3.Connection,
    cfg: dict,
    encoder_ids: list[str],
    calls: pd.DataFrame,
    species: pd.DataFrame,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    """Campagne AnuraSet (DECISIONS n° 136) : les encodeurs sur les espèces choisies (sonde
    logistique et autres sondes du §3, un pli par site), puis toutes les têtes de
    `CAMPAIGN_HEADS` sur le meilleur encodeur. `species` : sortie de `choose_species`."""
    codes = list(species["species"])
    results, comparisons = run_anuraset_benchmark(con, cfg, encoder_ids, codes, calls)
    encoders = rank_encoders(results)
    best = str(encoders.index[0])
    heads = run_anuraset_heads(con, cfg, best, codes, calls, methods or CAMPAIGN_HEADS)
    return {
        "species": species,
        "encoder_results": results,
        "encoder_comparisons": comparisons,
        "encoders": encoders.reset_index(),
        "best_encoder": best,
        "heads": heads,
        "head_ranking": rank_heads(heads["table"]),
    }


def write_campaign_report(out: dict[str, Any], reports_dir: Path) -> Path:
    """`anuraset_campagne.md` : espèces et pourquoi, encodeurs, têtes (classement moyen, AP par
    site tenu à l'écart, écarts appariés à la logistique), et les rapports détaillés."""
    from blanci.benchmark import to_markdown

    reports_dir = Path(reports_dir)
    write_anuraset_report(out["encoder_results"], out["encoder_comparisons"], reports_dir)
    heads_path = write_anuraset_heads_report(out["heads"], out["best_encoder"], reports_dir)
    species_cols = ["species", "criterion", "n_calls", "n_recordings", "n_sites"]
    species_cols += ["duration_median_s", "dominant_hz"]
    species = out["species"][[c for c in species_cols if c in out["species"]]]
    text = [
        "# Campagne AnuraSet",
        "",
        "Chaque modèle est appris sur les autres sites et jugé sur un site qu'il n'a jamais vu "
        "(un pli par site) : c'est la généralisation d'un site à l'autre qu'on mesure. "
        "Indicateur sur d'autres anoures que A. blanci, pas un verdict (DECISIONS n° 136).",
        "",
        "## Espèces",
        "",
        to_markdown(species),
        "",
        "## Encodeurs (sonde logistique, fenêtres, moyenne sur les espèces)",
        "",
        to_markdown(out["encoders"]),
        "",
        f"Meilleur : **{out['best_encoder']}** ; les têtes sont jugées sur ses embeddings.",
        "",
        "## Têtes : classement sur les espèces (fenêtres)",
        "",
        "Rang moyen (1 = la meilleure pour chaque espèce), AP poolée moyenne, AP moyenne par "
        "site tenu à l'écart (`ap_fold_mean`).",
        "",
        to_markdown(out["head_ranking"]),
        "",
    ]
    comparisons = out["heads"]["comparisons"]
    if not comparisons.empty:
        text += [
            "## Écart apparié à la logistique (bootstrap par enregistrement)",
            "",
            "Une tête ne bat la logistique que si `significant_holm` : p-valeur corrigée de "
            "Holm sur toutes les têtes et toutes les espèces (DECISIONS n° 139). Les "
            "enregistrements d'un site sont tirés un à un, alors que les plis sont des sites : "
            "l'intervalle reste optimiste ; l'AP par site tenu à l'écart dit la variabilité.",
            "",
            to_markdown(comparisons),
            "",
        ]
    sites = out["heads"]["sites"]
    if not sites.empty:
        pivot = sites.pivot_table(index=["species", "head"], columns="held_out_site", values="ap")
        text += ["## AP sur chaque site tenu à l'écart", "", to_markdown(pivot.reset_index()), ""]
    text += [f"Détails : `{heads_path.name}`, `anuraset_benchmark.md`.", ""]
    path = reports_dir / "anuraset_campagne.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path
