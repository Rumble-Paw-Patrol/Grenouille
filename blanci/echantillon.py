"""Banc d'essai sur l'échantillon versionné (`echantillon/`, DECISIONS n° 113).

66 clips FLAC de ~10 s (48 kHz stéréo), un par fenêtre étiquetée, 3,5 s de contexte de part
et d'autre (`echantillon/LISEZMOI.md`). Sert à faire tourner la chaîne sur du vrai son sans le
disque : encodeurs, têtes, régularisations, pertes. **Pas à départager les têtes** : 10
positifs, 5 micros, un site ; les intervalles le disent.

- Fenêtre de chaque clip : la durée de l'encodeur, centrée sur la fenêtre étiquetée (3 s).
- Plis groupés par micro ; un clip = un enregistrement (niveau fenêtre = niveau enregistrement).
- R19/R20/R21 : groupe `site` par défaut (1 à 3 clips par micro : centrer par micro revient à
  retirer le clip à lui-même). Statistiques calculées sur l'échantillon, sans labels.
- Diagnostic du site (prémisse de R19–R21) : sur les négatifs, une logistique reconnaît-elle
  le site d'un micro qu'elle n'a jamais vu ?
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blanci.labels import POSITIVE_LABELS

SAMPLE_DIR = Path("echantillon")
EXCLUDED = ("uncertain", "blanci_uncertain")


def load_labels(root: Path = SAMPLE_DIR) -> pd.DataFrame:
    """labels.csv, sans les labels incertains, avec y, point (site/micro) et recording_id."""
    df = pd.read_csv(Path(root) / "labels.csv")
    df = df[~df["label"].isin(EXCLUDED)].reset_index(drop=True)
    df["y"] = df["label"].isin(POSITIVE_LABELS).astype(int)
    df["point"] = df["site"].astype(str) + "/" + df["mic_id"].astype(str)
    df["recording_id"] = df["source"].astype(str)
    df["window_id"] = df["fichier"]
    df["presumed"] = False
    df["gated"] = False
    return df


def cut_window(path: Path, center_s: float, window_s: float, channel: int = 0):
    """(forme d'onde mono, f_e) : `window_s` secondes centrées sur `center_s`, complétées de
    zéros si le clip est trop court."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = wav.mean(axis=1) if channel == "mean" else wav[:, int(channel)]
    n = round(window_s * sr)
    start = round((center_s - window_s / 2) * sr)
    out = np.zeros(n, dtype=np.float32)
    lo, hi = max(start, 0), min(start + n, len(mono))
    if hi > lo:
        out[lo - start : hi - start] = mono[lo:hi]
    return out, sr


def encode(
    cfg: dict,
    encoder_name: str,
    df: pd.DataFrame,
    root: Path = SAMPLE_DIR,
    cache: Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """(embeddings, jetons ou None) des clips, en cache (`cache/<encodeur>.npz`)."""
    cache_file = None if cache is None else Path(cache) / f"{encoder_name}.npz"
    if cache_file is not None and cache_file.exists():
        stored = np.load(cache_file, allow_pickle=False)
        if list(stored["files"]) == df["fichier"].tolist():
            return stored["X"], stored["tokens"] if stored["tokens"].size else None
    from blanci.encoders import get_encoder

    encoder = get_encoder(encoder_name, cfg)
    channel = cfg.get("audio", {}).get("channel", 0)
    rows, token_rows = [], []
    for row in df.itertuples():
        center = row.fenetre_debut_dans_clip_s + row.fenetre_dur_s / 2
        wav, sr = cut_window(Path(root) / row.fichier, center, encoder.window_s, channel)
        rows.append(encoder.embed(wav[None, :], sr)[0])
        tokens = encoder.embed_tokens(wav[None, :], sr)
        if tokens is not None:
            token_rows.append(tokens[0])
    X = np.stack(rows).astype(np.float32)
    T = np.stack(token_rows).astype(np.float32) if len(token_rows) == len(rows) else None
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_file,
            X=X,
            tokens=T if T is not None else np.zeros(0, np.float32),
            files=np.array(df["fichier"].tolist()),
        )
    return X, T


def site_diagnostic(X: np.ndarray, df: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """Le site se lit-il dans l'embedding d'un micro jamais vu ? (négatifs, plis par micro)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    neg = df["y"].to_numpy() == 0
    sites, groups = df.loc[neg, "site"].to_numpy(), df.loc[neg, "point"].to_numpy()
    names, counts = np.unique(sites, return_counts=True)
    n_folds = min(5, len(np.unique(groups)))
    if len(names) < 2 or n_folds < 2:
        return {}
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    predicted = cross_val_predict(model, X[neg], sites, groups=groups, cv=GroupKFold(n_folds))
    accuracy = float((predicted == sites).mean())
    chance = float(counts.max() / counts.sum())
    return {
        "n_negatives": int(neg.sum()),
        "sites": dict(zip(names.tolist(), counts.tolist(), strict=True)),
        "accuracy": accuracy,
        "chance": chance,
        "verdict": (
            "le site se lit dans l'embedding, même pour un micro jamais vu"
            if accuracy > chance + 0.15
            else "pas de signature de site nette à cet effectif"
        ),
    }


def run(
    cfg: dict,
    encoder_name: str,
    methods: list[str],
    root: Path = SAMPLE_DIR,
    cache: Path | None = None,
    by: str = "site",
) -> dict[str, Any]:
    """Têtes (avec régularisations et pertes) sur l'échantillon, plis groupés par micro."""
    from blanci.evaluate import evaluate
    from blanci.head import oof_scores
    from blanci.head_benchmark import _inputs, expand_methods
    from blanci.regularization import Context, domain_statistics, needs_domain, regularizer_for

    df = load_labels(root)
    X, tokens = encode(cfg, encoder_name, df, root, cache)
    methods = expand_methods(methods) or ["logistic"]
    groups = df["point"].to_numpy()
    y = df["y"].to_numpy()
    context_groups = df[by].astype(str).to_numpy()
    context = Context(context_groups, (y == 0))
    if needs_domain(methods):
        context.domain = domain_statistics(X, context_groups, by)
        context.domain_rows = context.domain.rows(context_groups)
    head_cfg, bench = cfg["head"], cfg["benchmark"]
    n_splits = int(min(head_cfg["n_splits"], len(np.unique(groups[y == 1]))))
    rows, scores = [], {}
    for spec in methods:
        name, base, regularizer = regularizer_for(spec, cfg, context)
        method, inputs = _inputs(base, X, tokens)
        values = oof_scores(
            inputs,
            y,
            groups,
            n_splits=n_splits,
            method=method,
            C_grid=head_cfg["C_grid"],
            seed=head_cfg["seed"],
            tokens=tokens,
            regularizer=regularizer,
        ).values
        scores[name] = values
        metrics = evaluate(
            values,
            y,
            df["recording_id"].to_numpy(),
            level="window",
            precisions=tuple(bench["precisions"]),
            n_boot=bench["n_boot"],
            seed=head_cfg["seed"],
        )
        rows.append({"head": name, **metrics})
    table = pd.DataFrame(rows).sort_values("ap", ascending=False, kind="stable")
    return {
        "table": table.reset_index(drop=True),
        "scores": scores,
        "site": site_diagnostic(X, df, head_cfg["seed"]),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "n_mics_pos": int(len(np.unique(groups[y == 1]))),
        "n_splits": n_splits,
        "dim": int(X.shape[1]),
        "tokens": None if tokens is None else list(tokens.shape[1:]),
    }
