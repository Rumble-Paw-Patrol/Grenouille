"""Benchmark des têtes (§3, DECISIONS n° 92–93) : toutes les têtes, mêmes fenêtres, mêmes plis.

Têtes comparées sur les embeddings gelés d'un encodeur (`head.METHODS`, `pooling.POOLINGS`) :

| tête | choix de w |
|---|---|
| exemplar_medoid | l'embedding d'une seule fenêtre de référence (recherche par l'exemple) |
| exemplar | le positif d'entraînement le plus proche (recherche par les exemples) |
| knn | plus proche positif − plus proche négatif |
| simple_prototype | μ₊, moyenne des positifs |
| prototype | μ₊ − μ₋, négatifs appariés (prototype différentiel) |
| logistic | appris (régression logistique) sur l'embedding par défaut |
| logistic_to_prototype | R30 : logistique tirée vers le prototype différentiel |
| lda_shrunk | R31 : LDA à covariance rétrécie (Ledoit-Wolf) |
| logistic:<pooling> | appris sur les jetons résumés par `pooling` (max, moyenne + max, gem…) |
| attentive | appris sur les jetons, pondérés par une requête apprise |
| cascade | logistic, puis attentive sur les meilleurs candidats |

Régularisations (`blanci/regularization.py`, DECISIONS n° 108) : dans le nom de la tête,
`logistic+R18=16+R19`. Le nom canonique figure dans les tableaux et les scores hors-pli ;
`regularization.variants` ajoute des têtes régularisées à la liste par défaut.

Deux mesures :

1. `run_head_benchmark` : chaque tête en validation croisée groupée par micro, plis communs à
   tous les modèles ; AP et rappel aux précisions plancher, niveaux fenêtre et enregistrement ;
   comparaisons appariées contre la tête de référence (`head.reference`) ; diagnostic du fond
   capté (prototype différentiel contre prototype simple, §3) ; scores hors-pli enregistrés
   (`blanci/oof.py`).
2. `annotation_curve` : AP sur un site cible selon le nombre k d'enregistrements positifs de ce
   site ajoutés à l'entraînement (k = 0 : transfert pur). Hypothèse à tester (notes.md) : le
   prototype différentiel, qui retire la moyenne du site, ferait mieux que le linear probe
   quand le site est peu annoté ; le linear probe reprendrait l'avantage avec k. Protocole :
   - pour chaque cible (micro tant que tous les positifs viennent de Mataroni, site ensuite),
     ses enregistrements sont coupés en deux moitiés tirées au hasard (positifs, négatifs
     annotés, négatifs présumés séparément) : une moitié de test, fixe pour tous les k, et une
     réserve ;
   - l'entraînement = toutes les autres cibles + les négatifs présumés de la réserve (fond du
     site, gratuit : aucune annotation) + k enregistrements positifs de la réserve, et une part
     proportionnelle de ses négatifs annotés. Les k sont emboîtés (k = 2 contient k = 1) ;
   - `repeats` tirages par cible ; C des têtes logistiques choisi une fois par cible, sur les
     autres cibles ;
   - écart de chaque tête à la référence, apparié par (cible, tirage), avec intervalle
     bootstrap : l'hypothèse tient si l'écart différentiel − logistique est > 0 aux petits k.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from blanci.attentive import TokenStore
from blanci.config import config_path
from blanci.dataset import embedded_training_set, folds_for, pairing_options
from blanci.db import encoder_params
from blanci.evaluate import average_precision, evaluate, paired_bootstrap, to_recordings
from blanci.frozen import frozen_recordings
from blanci.head import METHODS, choose_C, fit_and_score, oof_scores
from blanci.oof import labels_fingerprint, oof_frame, save_oof
from blanci.pooling import as_grid, available_poolings, pool
from blanci.regularization import (
    Context,
    canonical,
    needs_domain,
    regularizer_for,
    store_domain_statistics,
)
from blanci.store import EmbeddingStore

TOKEN_METHODS = ("attentive", "cascade")
LEVELS = ("window", "recording")


def head_methods(tokens: np.ndarray | None, variants: list[str] | None = None) -> list[str]:
    """Têtes possibles : celles des embeddings, plus celles des jetons s'il y en a, plus les
    variantes régularisées de la config (`regularization.variants`)."""
    base = [m for m in METHODS if m not in TOKEN_METHODS]
    if tokens is not None:
        base += [f"logistic:{p}" for p in available_poolings(tokens)] + list(TOKEN_METHODS)
    return base + [v for v in variants or [] if v not in base]


def regularization_context(
    con: sqlite3.Connection,
    cfg: dict,
    encoder_id: str,
    data: pd.DataFrame,
    specs: list[str],
    filters: dict | None = None,
) -> Context:
    """Contexte des fenêtres pour les régularisations : groupe, négatifs annotés (R15) et,
    si une tête le demande, statistiques du stock par micro (R19, R20)."""
    by = (cfg.get("regularization", {}) or {}).get("by", "point")
    groups = data[by].astype(str).to_numpy()
    hard = (data["y"].to_numpy() == 0) & ~data["presumed"].to_numpy(dtype=bool)
    context = Context(groups, hard)
    if needs_domain(specs):
        store = EmbeddingStore(config_path(cfg, "embeddings"), encoder_id)
        context.domain = store_domain_statistics(con, store, filters, by)
        context.domain_rows = context.domain.rows(groups)
    return context


def _inputs(method: str, X: np.ndarray, tokens: np.ndarray | None) -> tuple[str, np.ndarray]:
    """(méthode de `head`, entrée) : `logistic:max` = logistic sur les jetons résumés."""
    if method.startswith("logistic:"):
        if tokens is None:
            raise ValueError(f"{method} demande les jetons de l'encodeur (blanci tokens)")
        return "logistic", pool(tokens, method.split(":", 1)[1])
    if method == "attentive":
        if tokens is None:
            raise ValueError("attentive demande les jetons de l'encodeur (blanci tokens)")
        return "attentive", tokens
    return method, X


def load_tokens(
    con: sqlite3.Connection, cfg: dict, encoder_id: str, window_ids: list[str]
) -> np.ndarray | None:
    """Jetons des fenêtres, remis en grille temps × fréquence si la config le dit
    (`encoders.models.<nom>.token_grid: {time, freq, order}`)."""
    tokens = TokenStore(config_path(cfg, "tokens"), encoder_id).load(window_ids)
    if tokens is None or tokens.ndim != 3:
        return tokens
    row = con.execute("SELECT name FROM models WHERE model_id = ?", (encoder_id,)).fetchone()
    spec = cfg["encoders"]["models"].get(row[0] if row else "", {}) or {}
    grid = spec.get("token_grid")
    if not grid:
        return tokens
    return as_grid(tokens, int(grid["time"]), int(grid["freq"]), grid.get("order", "time_major"))


def benchmark_data(
    con: sqlite3.Connection, cfg: dict, encoder_id: str, filters: dict | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """(fenêtres évaluées, embeddings, jetons ou None) : les mêmes que le benchmark des
    encodeurs (labels, négatifs appariés, jeu gelé exclu)."""
    params = encoder_params(con, encoder_id)
    data, X = embedded_training_set(
        con,
        EmbeddingStore(config_path(cfg, "embeddings"), encoder_id),
        params["window_s"],
        per_positive=cfg["benchmark"]["negatives_per_positive"],
        **pairing_options(cfg),
        seed=cfg["head"]["seed"],
        filters=filters,
        exclude_recordings=frozen_recordings(cfg),
    )
    if data["y"].nunique() < 2:
        raise ValueError(f"{encoder_id} : une seule classe dans le jeu étiqueté")
    data = data.assign(dur_s=params["window_s"])
    return data, X, load_tokens(con, cfg, encoder_id, data["window_id"].tolist())


# --- 1. Toutes les têtes en validation croisée ---------------------------------------------------


def run_head_benchmark(
    con: sqlite3.Connection,
    cfg: dict,
    encoder_id: str,
    methods: list[str] | None = None,
    filters: dict | None = None,
) -> dict[str, Any]:
    """Tableau des têtes, comparaisons à la référence, diagnostic du fond capté."""
    head_cfg, bench = cfg["head"], cfg["benchmark"]
    data, X, tokens = benchmark_data(con, cfg, encoder_id, filters)
    variants = (cfg.get("regularization", {}) or {}).get("variants", [])
    methods = methods or head_methods(tokens, variants)
    context = regularization_context(con, cfg, encoder_id, data, methods, filters)
    y = data["y"].to_numpy()
    groups = data["point"].to_numpy()
    recordings = data["recording_id"].to_numpy()
    gated = data["gated"].to_numpy()
    assignment = folds_for(con, cfg)
    fingerprint = labels_fingerprint(con, cfg)

    rows, scores = [], {}
    for spec in methods:
        method, head, regularizer = regularizer_for(spec, cfg, context)
        base, inputs = _inputs(head, X, tokens)
        oof = oof_scores(
            inputs,
            y,
            groups,
            n_splits=head_cfg["n_splits"],
            method=base,
            C_grid=head_cfg["C_grid"],
            seed=head_cfg["seed"],
            gated=gated,
            assignment=assignment,
            tokens=tokens,
            cascade_fraction=head_cfg.get("cascade_fraction", 0.2),
            regularizer=regularizer,
        )
        scores[method] = oof.values
        save_oof(
            cfg,
            oof_frame(
                f"{encoder_id}/{method}", "encoder_head", data, oof.values, assignment, fingerprint
            ),
        )
        for level in LEVELS:
            metrics = evaluate(
                oof.values,
                y,
                recordings,
                level=level,
                precisions=tuple(bench["precisions"]),
                n_boot=bench["n_boot"],
                seed=head_cfg["seed"],
            )
            rows.append({"encoder_id": encoder_id, "head": method, **metrics})
    table = pd.DataFrame(rows).sort_values(["level", "ap"], ascending=[True, False], kind="stable")

    reference = canonical(head_cfg.get("reference", "logistic"))
    comparisons = compare_to_reference(scores, reference, y, recordings, cfg)
    return {
        "table": table.reset_index(drop=True),
        "comparisons": comparisons,
        "background": background_diagnostic(scores, y, recordings, cfg),
        "scores": scores,
        "n_mics": int(len(np.unique(groups))),
    }


def compare_to_reference(
    scores: dict[str, np.ndarray],
    reference: str,
    y: np.ndarray,
    recordings: np.ndarray,
    cfg: dict,
) -> pd.DataFrame:
    """Chaque tête contre la référence, au niveau enregistrement, bootstrap apparié (§6)."""
    if reference not in scores:
        return pd.DataFrame()
    rows = []
    for method, values in scores.items():
        if method == reference:
            continue
        labels, a, b, units = _pair(values, scores[reference], y, recordings)
        result = paired_bootstrap(
            labels, a, b, units, n_boot=cfg["benchmark"]["n_boot"], seed=cfg["head"]["seed"]
        )
        rows.append({"head": method, "reference": reference, **result})
    return pd.DataFrame(rows)


def background_diagnostic(
    scores: dict[str, np.ndarray], y: np.ndarray, recordings: np.ndarray, cfg: dict
) -> dict[str, Any]:
    """Le fond sonore pollue-t-il l'embedding ? (§3, notes.md)

    Si prototype simple et différentiel classent pareil, l'ambiance des micros ne pollue pas ;
    si le différentiel fait nettement mieux (intervalle apparié au-dessus de zéro), elle pollue.
    """
    if not {"prototype", "simple_prototype"} <= set(scores):
        return {}
    labels, a, b, units = _pair(scores["prototype"], scores["simple_prototype"], y, recordings)
    result = paired_bootstrap(
        labels, a, b, units, n_boot=cfg["benchmark"]["n_boot"], seed=cfg["head"]["seed"]
    )
    if result["significant"] and result["diff"] > 0:
        verdict = "le fond sonore pollue l'embedding : retrancher les négatifs appariés aide"
    elif result["significant"]:
        verdict = "le prototype simple fait mieux : le différentiel retire aussi du chant"
    else:
        verdict = "pas d'écart net : le fond sonore ne pollue pas le classement"
    return result | {"verdict": verdict}


def _pair(a: np.ndarray, b: np.ndarray, y: np.ndarray, recordings: np.ndarray):
    rec_a = to_recordings(a, y, recordings)
    rec_b = to_recordings(b, y, recordings).set_index("recording_id")
    return (
        rec_a["y"].to_numpy(),
        rec_a["score"].to_numpy(),
        rec_b.loc[rec_a["recording_id"], "score"].to_numpy(),
        rec_a["recording_id"].to_numpy(),
    )


# --- 2. Courbe selon le nombre d'annotations du site cible ----------------------------------------


def _stable_id(text: str) -> int:
    """Entier stable d'un nom (hash() de Python change à chaque lancement)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _half(items: list[str], rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """(test, réserve) : moitié au hasard, le test prend l'élément impair."""
    items = list(items)
    rng.shuffle(items)
    cut = (len(items) + 1) // 2
    return items[:cut], items[cut:]


def annotation_curve(
    con: sqlite3.Connection,
    cfg: dict,
    encoder_id: str,
    methods: list[str] | None = None,
    k_grid: list[int] | None = None,
    repeats: int | None = None,
    by: str | None = None,
    level: str = "recording",
) -> dict[str, pd.DataFrame]:
    """Courbe d'apprentissage par cible (voir le module). Renvoie `runs` (une ligne par
    tête, cible, tirage et k), `summary` (moyenne par tête et k) et `gaps` (écart apparié de
    chaque tête à la référence, par k, avec intervalle bootstrap)."""
    head_cfg = cfg["head"]
    curve = head_cfg.get("curve", {}) or {}
    methods = methods or list(curve.get("methods", ["prototype", "logistic"]))
    k_grid = sorted(k_grid or curve.get("k", [0, 1, 2, 5, 10, 20]))
    repeats = repeats or int(curve.get("repeats", 5))
    by = by or curve.get("by", "point")
    seed = head_cfg["seed"]

    data, X, tokens = benchmark_data(con, cfg, encoder_id)
    open_rows = ~data["gated"].to_numpy()
    data, X = data[open_rows].reset_index(drop=True), X[open_rows]
    tokens = tokens[open_rows] if tokens is not None else None
    context = regularization_context(con, cfg, encoder_id, data, methods)
    heads = {spec: regularizer_for(spec, cfg, context) for spec in methods}
    y = data["y"].to_numpy()
    targets_of = data[by].astype(str).to_numpy()
    recordings = data["recording_id"].to_numpy()
    groups = data["point"].to_numpy()

    kind = np.where(y == 1, "pos", np.where(data["presumed"].to_numpy(), "presumed", "neg"))
    rec_kind = (
        pd.DataFrame({"recording_id": recordings, "kind": kind, "target": targets_of})
        .assign(rank=lambda d: d["kind"].map({"pos": 0, "neg": 1, "presumed": 2}))
        .sort_values("rank")
        .drop_duplicates("recording_id")  # un enregistrement à positif est « pos »
    )
    runs = []
    for target in sorted(set(targets_of)):
        mine = rec_kind[rec_kind["target"] == target]
        pos = mine.loc[mine["kind"] == "pos", "recording_id"].tolist()
        if len(pos) < 2:  # il faut au moins un positif en test et un en réserve
            continue
        others = np.flatnonzero(targets_of != target)
        if len(np.unique(y[others])) < 2:
            continue
        # C de chaque tête, choisi une fois par cible sur les autres cibles (régularisations
        # comprises : ACP et INLP ajustées sur ces mêmes fenêtres).
        C_of = {}
        for spec, (_, head, regularizer) in heads.items():
            name, inputs = _inputs(head, X, tokens)
            C_of[spec] = choose_C(
                name,
                inputs,
                y,
                groups,
                others,
                head_cfg["C_grid"],
                head_cfg["n_splits"],
                seed,
                regularizer,
            )
        for r in range(repeats):
            rng = np.random.default_rng([seed, r, _stable_id(target)])
            test_pos, pool_pos = _half(pos, rng)
            neg = mine.loc[mine["kind"] == "neg", "recording_id"].tolist()
            test_neg, pool_neg = _half(neg, rng)
            presumed = mine.loc[mine["kind"] == "presumed", "recording_id"].tolist()
            test_bg, pool_bg = _half(presumed, rng)
            test = np.flatnonzero(np.isin(recordings, test_pos + test_neg + test_bg))
            if len(np.unique(y[test])) < 2:
                continue
            base = np.concatenate([others, np.flatnonzero(np.isin(recordings, pool_bg))])
            for k in k_grid:
                if k > len(pool_pos):
                    break
                n_neg = round(len(pool_neg) * k / len(pool_pos)) if pool_pos else 0
                support = pool_pos[:k] + pool_neg[:n_neg]
                train = np.concatenate([base, np.flatnonzero(np.isin(recordings, support))])
                for spec, (method, head, regularizer) in heads.items():
                    name, inputs = _inputs(head, X, tokens)
                    values = fit_and_score(
                        name,
                        inputs,
                        y,
                        groups,
                        train,
                        test,
                        C=C_of[spec],
                        seed=seed,
                        tokens=tokens,
                        cascade_fraction=head_cfg.get("cascade_fraction", 0.2),
                        regularizer=regularizer,
                    )
                    runs.append(
                        {
                            "head": method,
                            "target": target,
                            "repeat": r,
                            "k": k,
                            "ap": _ap(values, y[test], recordings[test], level),
                            "n_test_pos_recordings": len(test_pos),
                            "n_support_neg_recordings": n_neg,
                        }
                    )
    runs = pd.DataFrame(runs)
    if runs.empty:
        raise ValueError(f"aucune cible ({by}) n'a au moins deux enregistrements positifs")
    reference = canonical(head_cfg.get("reference", "logistic"))
    return {
        "runs": runs,
        "summary": _curve_summary(runs),
        "gaps": _curve_gaps(runs, reference, cfg["benchmark"]["n_boot"], seed),
    }


def _ap(scores: np.ndarray, y: np.ndarray, recordings: np.ndarray, level: str) -> float:
    if level == "recording":
        rec = to_recordings(scores, y, recordings)
        return average_precision(rec["y"].to_numpy(), rec["score"].to_numpy())
    return average_precision(y, scores)


def _curve_summary(runs: pd.DataFrame) -> pd.DataFrame:
    return (
        runs.groupby(["head", "k"])
        .agg(
            ap_mean=("ap", "mean"),
            ap_sd=("ap", "std"),
            n_runs=("ap", "count"),
            n_targets=("target", "nunique"),
        )
        .reset_index()
    )


def _curve_gaps(runs: pd.DataFrame, reference: str, n_boot: int, seed: int) -> pd.DataFrame:
    """Écart moyen AP(tête) − AP(référence) par k, apparié par (cible, tirage) ; intervalle
    bootstrap en rééchantillonnant les cibles (unités indépendantes)."""
    if reference not in set(runs["head"]):
        return pd.DataFrame()
    wide = runs.pivot_table(index=["target", "repeat", "k"], columns="head", values="ap")
    rng = np.random.default_rng(seed)
    rows = []
    for method in [m for m in wide.columns if m != reference]:
        diff = (wide[method] - wide[reference]).dropna().rename("gap").reset_index()
        for k, part in diff.groupby("k"):
            per_target = part.groupby("target")["gap"].mean().to_numpy()
            if not len(per_target):
                continue
            boots = [
                rng.choice(per_target, len(per_target), replace=True).mean()
                for _ in range(max(1, n_boot))
            ]
            lo, hi = np.quantile(boots, [0.025, 0.975])
            rows.append(
                {
                    "head": method,
                    "reference": reference,
                    "k": int(k),
                    "gap": float(per_target.mean()),
                    "lo": float(lo),
                    "hi": float(hi),
                    "n_targets": len(per_target),
                    "significant": bool(lo > 0 or hi < 0),
                }
            )
    return pd.DataFrame(rows)


def plot_curve(summary: pd.DataFrame, path) -> bool:
    """Courbe AP moyenne contre k, une ligne par tête (PNG). Faux si matplotlib manque."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    for head, part in summary.groupby("head"):
        ax.errorbar(
            part["k"], part["ap_mean"], yerr=part["ap_sd"], marker="o", capsize=3, label=head
        )
    ax.set_xlabel("enregistrements positifs annotés du site cible (k)")
    ax.set_ylabel("AP moyenne (site cible)")
    ax.set_xscale("symlog", linthresh=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True
