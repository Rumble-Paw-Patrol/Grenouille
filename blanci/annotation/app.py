"""Poste d'annotation Streamlit (§5, M2, DECISIONS n° 100) : outil de travail, pas le livrable.

    uv sync --group app
    uv run blanci annotate   (les enregistrements sont cherchés sur les disques branchés)

Le poste sert à n'importe quel projet d'écoute : l'en-tête (titre, espèce, chant et sa bande)
se règle dans la fenêtre « Nouveau projet… » ou « Modifier ce projet… » du menu « Projet » ;
les étiquettes et les qualités dans le visualiseur ; tout se garde dans `annotations/`.

- **File d'écoute** (panneau de gauche) : une file déjà écrite, ou une nouvelle file tirée sur
  place par n'importe quelle méthode de l'outil de sélection (`blanci/annotation/selection.py` :
  60-20-20 à proportions réglables, similarité, couverture, groupes, audit, hasard, negative
  mining, phénologie, suspects, trous dans un chant), ou la **carte des embeddings** (YAPAT
  fait maison) : on entoure une zone de points, on l'écoute.
- **Navigation** : sous la file, la liste de ses extraits (numérotés dans la file entière,
  ✓ et label pour ceux déjà écoutés) ; on peut revenir sur n'importe lequel et le
  réécouter, une nouvelle réponse s'ajoute à l'ancienne (correction, jamais écrasée).
  « Sauter les extraits que j'ai déjà écoutés » et « … écoutés par un collègue » : ◀ ▶ et la
  validation vont au précédent ou au prochain extrait qui reste à écouter ; tout est écouté :
  ◀ ▶ sont grisés (« Vous avez tout écouté »). Aucune case cochée : on va au voisin.
- **Annotateurs** : on se choisit dans le menu « Annotateur » (liste du projet,
  `annotations/annotateurs.txt`, où « Ajouter quelqu'un… » inscrit un nouveau venu). En
  rouvrant un extrait, on retrouve ses propres intervalles, modifiables, et ceux du dernier
  envoi de chaque autre annotateur, en lecture seule (pointillés) ; un envoi sans intervalle
  est dit en toutes lettres sous le spectrogramme. Un envoi fait sur un extrait plus long qui
  contient celui-ci compte aussi (enregistrement entier annoté avant un découpage).
  « Masquer les annotations de » cache ceux de certains (ou de tout le monde).
- **Écoute** : l'enregistrement entier sur un spectrogramme zoomable (`viewer.py` : molette,
  glisser, barre de lecture qui suit l'un ou l'autre micro), volume et bande d'écoute
  réglable ; filtre dynamique et contraste réglés micro par micro ; rien de cela ne touche
  l'audio d'origine.
- **Étiquettes et qualités** : celles du projet (`annotations/etiquettes.json`,
  `etiquettes.py`) ; « + » en propose une (celles déjà connues d'abord : retirées, ou vues dans
  les annotations ; ou un nom nouveau), « − » en retire de la liste sans l'oublier.
- **Découpage** : l'extrait se découpe en fenêtres de longueur choisie, calées sur le
  candidat ; on les annote une à une sans quitter l'enregistrement (suspendu).
- **Qualité** : en intervalles, chaque intervalle a la sienne (barre au-dessus du
  spectrogramme) ; en fenêtres, une qualité pour la fenêtre.
- **Validation** : commentaire facultatif, puis « Valider l'annotation » (sous le commentaire,
  à droite du spectrogramme, ou Ctrl + Entrée) : la réponse est ajoutée (jamais écrasée), copiée
  dans le dossier des annotations validées, et l'extrait suivant s'affiche.
- **Groupes** : une file tirée par groupes montre, groupe par groupe, ce qui a été entendu ; un
  groupe homogène s'étiquette en entier d'un clic (source « bulk »).
- **Habillage** : celui du tableau de bord (`style.py`, bandeau dans `bandeau.py`), clair ou
  sombre selon le système ; les thèmes de Streamlit sont passés par `blanci annotate`, la
  feuille de style est injectée ici. Le mode d'emploi est en tête du panneau de gauche et
  s'ouvre en haut de la page.

Toute la logique est dans `workbench.py` et `selection.py` ; ce fichier ne fait qu'afficher.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from blanci.annotation import projets
from blanci.annotation.bandeau import (
    DEFAULT_HEADER,
    bandeau,
    read_header,
    reset_header,
    save_header,
)
from blanci.annotation.etiquettes import (
    add_label,
    add_quality,
    known_labels,
    read_set,
    remove_label,
    remove_quality,
    suggestions,
)
from blanci.annotation.habillage import read_prefs, write_prefs
from blanci.annotation.selection import NEEDS_ENCODER
from blanci.annotation.style import BRAND, CHART_COLORS, LABEL_COLORS, UNHEARD_COLOR, css
from blanci.annotation.viewer import HELP, asset, asset_src, viewer, viewer_args
from blanci.annotation.workbench import (
    ANSWERS,
    INTERVAL_ANSWERS,
    add_annotator,
    extract_comments,
    extract_intervals,
    latest_labels,
    load_candidates,
    local_time,
    local_times,
    mic_names,
    next_position,
    ordered_classes,
    progress,
    read_annotators,
    read_channels,
    save_answer,
    save_span,
    split_windows,
    visible_annotators,
    wav_bytes,
)
from blanci.core.config import (
    DEFAULT_CONFIG,
    config_path,
    default_user_config,
    load_config,
    project_path,
)
from blanci.core.db import connect, window_id_for
from blanci.core.locate import locate, remember_root, remembered_roots, variants
from blanci.service import export_annotations

NAMES = dict(ANSWERS) | {"edge": "bord d'un intervalle (écarté)"}
SHOW_METHOD = False  # rubrique « Méthode » du bandeau latéral (annotation par fenêtres)
EXISTING, MAP = "existing", "map"
MODES = {
    EXISTING: "File d'écoute déjà prête",
    "active": "File 60-20-20 (incertains, meilleurs, hasard)",
    "similarity": "Récolte par similarité",
    "coverage": "Couverture (YAPAT maison, automatique)",
    MAP: "Carte des embeddings (YAPAT maison, à la main)",
    "cluster": "Groupes, puis étiquetage en bloc",
    "audit": "Audit aléatoire (enregistrements entiers)",
    "random": "Fenêtres au hasard",
    "negative_mining": "Negative mining",
    "phenology": "Échantillonnage phénologique",
    "suspects": "Détections isolées (« suspect »)",
    "gaps": "Trous dans un chant (faux négatifs suspects)",
}
ENCODER_MODES = (*NEEDS_ENCODER, MAP)  # la carte lit aussi un stock d'embeddings


def _config_file() -> Path | None:
    """`--config` après `--` sur la ligne de commande streamlit, sinon $BLANCI_CONFIG, sinon
    `config/local.yaml` s'il existe. Pas indispensable pour écouter : les enregistrements sont
    aussi cherchés sur les disques branchés (`blanci.core.locate`)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    env = os.environ.get("BLANCI_CONFIG")
    return args.config or (Path(env) if env else None) or default_user_config()


def _setup(config: Path | None):
    """Config et connexion, rouvertes à chaque rafraîchissement : Streamlit change de fil
    d'exécution entre deux clics, et une connexion SQLite ne se partage pas entre fils. La
    config n'est relue que si un des deux YAML a changé (l'analyser coûtait 0,3 s par clic)."""
    files = (DEFAULT_CONFIG, *([Path(config)] if config else []))
    stamps = tuple(p.stat().st_mtime_ns if p.is_file() else 0 for p in files)
    cfg = _cached_config(str(config) if config else None, stamps)
    return cfg, connect(config_path(cfg, "db"))


@st.cache_data(show_spinner=False)
def _cached_config(config: str | None, stamps: tuple) -> dict:
    """`load_config`, gardé tant que les fichiers ne changent pas (`stamps` : leurs dates de
    modification, dans la clé du cache ; copie à chaque appel)."""
    return load_config(Path(config) if config else None)


def _queues(cfg) -> list[Path]:
    """Files de candidats : un dossier par file (`<paths.queues>/<nom>/candidats.csv`)."""
    from blanci.annotation.selection import queue_dir

    found = queue_dir(cfg).glob("*/candidats.csv")
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _queue_name(path: Path) -> str:
    return path.parent.name


def _encoders(con) -> list[str]:
    return [
        r[0]
        for r in con.execute("SELECT model_id FROM models WHERE kind = 'encoder' ORDER BY model_id")
    ]


def _open_queue(path: Path) -> None:
    """Bascule sur une file qui vient d'être écrite, au prochain affichage : un widget déjà
    affiché ne peut plus changer de valeur pendant cette exécution."""
    st.session_state["pending_queue"] = str(path)
    st.rerun()


def _apply_pending_queue() -> None:
    """Début d'exécution, avant tout widget : la file demandée devient la file ouverte."""
    pending = st.session_state.pop("pending_queue", None)
    if pending:
        st.session_state["mode"] = EXISTING
        st.session_state["queue_path"] = st.session_state["queue_pick"] = pending


# --- Panneau de gauche : sélection -------------------------------------------------------------


def _selection_panel(cfg, con) -> tuple[str, Path | None, str | None]:
    """(mode, file choisie ou None, encodeur ou None)."""
    from blanci.annotation.selection import select_candidates, write_queue

    st.header("File d'écoute")
    mode = st.selectbox("Mode de sélection", list(MODES), format_func=MODES.get, key="mode")
    encoder = None
    encoders = _encoders(con)
    if mode in ENCODER_MODES and not encoders:
        st.info("Aucun encodeur encodé : `blanci embed` d'abord.")
        return mode, None, None
    if (mode in ENCODER_MODES or mode == "gaps") and encoders:
        encoder = st.selectbox("Encodeur", encoders, key="encoder")
    if mode == EXISTING:
        queues = _queues(cfg)
        if not queues:
            st.warning("Aucune file d'écoute. En tirer une avec un autre mode.")
            return mode, None, encoder
        # Menu à clé, et file retenue à part (`queue_path`) : sans clé, le menu revenait à la
        # première file (la plus récente, « doutes ») dès qu'un autre menu relançait la page
        # (changer de projet), et Streamlit oublie la valeur d'un menu qui n'est pas affiché.
        paths = [str(q) for q in queues]
        if st.session_state.get("queue_pick") not in paths:
            wanted = st.session_state.get("queue_path")
            st.session_state["queue_pick"] = wanted if wanted in paths else paths[0]
        chosen = Path(
            st.selectbox(
                "File d'écoute",
                paths,
                key="queue_pick",
                format_func=lambda p: _queue_name(Path(p)),
            )
        )
        st.session_state["queue_path"] = str(chosen)
        return mode, chosen, encoder
    if mode == MAP:
        return mode, None, encoder

    options: dict = {}
    options["n"] = st.number_input(
        "Extraits (par groupe pour les groupes)",
        1,
        2000,
        10 if mode == "cluster" else 40,
        key=f"n::{mode}",
    )
    if mode == "active":
        st.caption("Proportions de la file (normalisées à 100 %)")
        mix = [
            st.slider("incertains", 0, 100, 60, 5),
            st.slider("meilleurs scores", 0, 100, 20, 5),
            st.slider("hasard stratifié", 0, 100, 20, 5),
        ]
        total = sum(mix) or 1
        options["mix"] = [m / total for m in mix]
    if mode == "negative_mining":
        options["mode"] = st.radio(
            "Négatifs durs",
            ["unlikely", "false_friends"],
            horizontal=True,
            format_func={
                "unlikely": "scores hauts là où elle est improbable",
                "false_friends": "proches des faux amis annotés",
            }.get,
        )
    if mode == "gaps":
        choices = ["scores", "labels"] if encoder else ["labels"]
        options["mode"] = st.radio(
            "Trous",
            choices,
            horizontal=True,
            format_func={
                "scores": "dans les scores du modèle",
                "labels": "négatifs annotés entre deux positifs",
            }.get,
        )
    if mode in ("similarity", "coverage", "cluster", "audit", "random", "phenology"):
        site = st.text_input("Site (vide : tous)", key=f"site::{mode}").strip()
        if site:
            options["site"], options["sites"] = site, [site]
    if mode == "phenology":
        options["whole"] = st.checkbox("Enregistrements entiers")
    if st.button("Générer la file", type="primary", use_container_width=True):
        try:
            queue = select_candidates(con, cfg, mode, encoder, **options)
        except (ValueError, NotImplementedError) as exc:
            st.error(str(exc))
            return mode, None, encoder
        if queue.empty:
            st.warning("Aucun extrait.")
            return mode, None, encoder
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        about = {"mode": MODES[mode], "encodeur": encoder} | options
        about["annotateur"] = st.session_state.get("annotator")
        _open_queue(write_queue(cfg, queue, f"{mode}_{stamp}", about))
    return mode, None, encoder


# --- Carte des embeddings -----------------------------------------------------------------------


@st.cache_data(show_spinner="Projection des embeddings…")
def _map_points(config: str | None, encoder: str, n: int, method: str):
    from blanci.annotation.selection import embedding_map

    cfg, con = _setup(Path(config) if config else None)
    return embedding_map(con, cfg, encoder, n=n, method=method)


def _map_page(cfg, con, encoder: str | None, config: Path | None) -> None:
    import altair as alt

    from blanci.annotation.selection import map_selection, write_queue

    st.subheader("Carte des embeddings")
    if encoder is None:
        st.info("Choisir un encodeur dans le panneau de gauche.")
        return
    cols = st.columns(2)
    n = cols[0].number_input("Fenêtres sur la carte", 200, 50000, 5000, 500)
    method = cols[1].selectbox("Projection", ["auto", "umap", "tsne", "pca"])
    try:
        points = _map_points(str(config) if config else None, encoder, int(n), method)
    except (ValueError, ImportError) as exc:
        st.error(str(exc))
        return
    st.caption("Entourer une zone (cliquer-glisser), puis l'écouter. Couleur : label entendu.")
    heard = sorted(set(points["label"]) - {"non écouté"})
    domain = heard + ["non écouté"] * bool((points["label"] == "non écouté").any())
    spare = itertools.cycle(CHART_COLORS)
    palette = [LABEL_COLORS[x] if x in LABEL_COLORS else next(spare) for x in heard]
    chart = (
        alt.Chart(points)
        .mark_circle(size=16, opacity=0.7)
        .encode(
            x=alt.X("x", axis=None),
            y=alt.Y("y", axis=None),
            color=alt.Color(
                "label",
                legend=alt.Legend(title="label"),
                scale=alt.Scale(domain=domain, range=palette + [UNHEARD_COLOR]),
            ),
            tooltip=["site", "recording_id", "offset_s", "label"],
        )
        .add_params(alt.selection_interval(name="zone"))
    )
    event = st.altair_chart(chart, on_select="rerun", key="carte", use_container_width=True)
    zone = (getattr(event, "selection", None) or {}).get("zone") or {}
    if "x" in zone and "y" in zone:
        (x0, x1), (y0, y1) = sorted(zone["x"]), sorted(zone["y"])
        chosen = points[points["x"].between(x0, x1) & points["y"].between(y0, y1)]
    else:
        chosen = points.iloc[:0]
    unheard = chosen[chosen["label"] == "non écouté"]
    st.write(f"{len(chosen)} fenêtres dans la zone, dont {len(unheard)} jamais écoutées.")
    if len(unheard) and st.button(f"Écouter la zone ({len(unheard)} fenêtres)", type="primary"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        about = {
            "mode": MODES[MAP],
            "encodeur": encoder,
            "projection": method,
            "zone": f"x {x0:.2f}…{x1:.2f}, y {y0:.2f}…{y1:.2f}",
        }
        _open_queue(write_queue(cfg, map_selection(con, unheard), f"carte_{stamp}", about))


# --- Groupes : étiquetage en bloc ----------------------------------------------------------------


def _cluster_panel(cfg, con, queue) -> None:
    from blanci.annotation.selection import cluster_status, label_cluster

    encoder = str(queue["encoder_id"].dropna().iloc[0])
    min_checked = int(cfg.get("selection", {}).get("cluster_min_checked", 10))
    with st.expander("Groupes : étiquetage en bloc", expanded=False):
        status = cluster_status(con, cfg, encoder, min_checked)
        st.dataframe(status, hide_index=True, use_container_width=True)
        ready = status[status["homogeneous"]]
        if ready.empty:
            st.caption(
                f"Un groupe s'étiquette en bloc après {min_checked} écoutes d'un même label."
            )
            return
        group = st.selectbox(
            "Groupe homogène",
            ready["cluster"].tolist(),
            format_func=lambda g: f"{g} : {ready.set_index('cluster').loc[g, 'label']}",
        )
        if st.button("Étiqueter tout le groupe"):
            written = label_cluster(
                con,
                cfg,
                encoder,
                int(group),
                annotator=st.session_state.get("annotator") or None,
                min_checked=min_checked,
            )
            st.success(f"{written} fenêtres étiquetées en bloc.")


# --- Écoute et réponse ---------------------------------------------------------------------------


def _goto(key: str, value: int) -> None:
    """Rappel de bouton ou de liste : la position change avant le prochain affichage."""
    st.session_state[key] = value


def _follow(key: str, widget: str) -> None:
    st.session_state[key] = st.session_state[widget]


def _colleagues(con, me: str, hidden: list[str], everyone: bool) -> list[str | None]:
    """Les autres annotateurs affichés (tous sauf soi et les masqués), tels qu'écrits dans la
    base ; None : les lignes sans annotateur (import)."""
    shown = visible_annotators(con, hidden, everyone)
    if shown is None:
        shown = [
            a
            for (a,) in con.execute(
                "SELECT DISTINCT annotator FROM labels UNION SELECT DISTINCT annotator FROM spans"
            )
        ]
    return [a for a in shown if a is None or not me or a.casefold() != me.casefold()]


def _to_skip(con, queue, me: str, colleagues: list, skip_mine: bool, skip_others: bool):
    """Extraits que ◀ ▶ et la validation sautent : écoutés par soi (`skip_mine`), par un
    collègue (`skip_others`). Série de la file : le label entendu pour ceux à sauter, None pour
    les autres (`next_position`)."""
    skip = None
    if skip_mine and me:
        skip = progress(con, queue, [me])
    if skip_others and colleagues:
        theirs = progress(con, queue, colleagues)
        skip = theirs if skip is None else skip.fillna(theirs)
    if skip is None:  # rien à sauter
        import pandas as pd

        skip = pd.Series([None] * len(queue), index=queue.index, dtype=object)
    return skip


def _advance(con, queue, pos, nav_args, key, end) -> None:
    """Après une réponse : extrait suivant (qui reste à écouter si une case « Sauter… » est
    cochée), ou fin de file si cette réponse vient d'écouter le dernier."""
    skipping = nav_args[2] or nav_args[3]
    nav = _to_skip(con, queue, *nav_args)
    st.session_state[key] = next_position(nav, pos, skipping)
    st.session_state[end] = skipping and bool(nav.notna().all())


def _pick(key: str, widget: str, end: str) -> None:
    """Candidat choisi dans la liste : on le rouvre, même dans une file terminée."""
    if st.session_state[widget] is not None:
        st.session_state[key] = st.session_state[widget]
        st.session_state[end] = False


def _candidate_strip(queue, done, chosen, key, end, offset_h) -> None:
    """Liste déroulante des extraits de la file ouverte, sous la liste des files : numéro dans
    la file entière, site, micro, heure, et ✓ label s'il a déjà été écouté. File terminée
    (`end`) : aucun extrait n'est sélectionné, n'importe lequel se rouvre."""
    rows = [
        f"{i + 1}. {site} · {mic} · {start} · {offset:.0f} s"
        + (f" · ✓ {NAMES.get(heard, heard)}" if heard else "")
        for i, (site, mic, start, offset, heard) in enumerate(
            zip(
                queue["site"],
                queue["mic_id"],
                local_times(queue["start_utc"], offset_h),
                queue["offset_s"],
                done,
                strict=True,
            )
        )
    ]
    widget = f"strip::{chosen}"
    st.session_state[widget] = None if st.session_state.get(end) else st.session_state[key]
    st.selectbox(
        f"Extraits de la file ({int(done.notna().sum())} / {len(queue)} écoutés)",
        range(len(queue)),
        format_func=rows.__getitem__,
        key=widget,
        on_change=_pick,
        args=(key, widget, end),
        placeholder="Choisir un extrait à réécouter",
    )


@st.cache_resource
def _prefetcher() -> tuple[ThreadPoolExecutor, dict]:
    """Un fil de lecture en arrière-plan et les lectures lancées, par (fichier, début, durée)."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch"), {}


def _prefetch(full: str, extract: tuple[float, float] | None = None, whole: bool = True) -> None:
    """Lit à l'avance le candidat suivant pendant qu'on écoute celui-ci : quand on l'ouvre,
    `_media` reprend le résultat au lieu de relire le disque. Une seule lecture d'avance.
    `whole` : l'enregistrement entier aussi (inutile s'il est déjà en cache : extrait suivant
    du même enregistrement)."""
    pool, pending = _prefetcher()
    spans = ([(0.0, math.inf)] if whole else []) + ([extract] if extract else [])
    todo = [(full, o, d) for o, d in spans if (full, o, d) not in pending]
    for stale in [k for k in pending if k not in todo and k[0] != full]:
        pending.pop(stale, None)
    for args in todo:
        pending[args] = pool.submit(_build_media, *args)


def _media_key(full: str, offset_s: float, dur_s: float, channel: int) -> str:
    """Nom du fichier d'écoute d'un micro (`asset_src`), connu avant de lire l'enregistrement."""
    return f"v1|{full}|{offset_s}|{dur_s}|{channel}"


def _next_audios(full: str, offset_s: float, dur_s: float) -> list[tuple[str, str, float]]:
    """Fichiers d'écoute que `_build_media` écrira pour l'extrait suivant, sans attendre qu'il
    les écrive : le navigateur les guette et prépare l'extrait d'avance."""
    return [
        (name, asset_src(_media_key(full, offset_s, dur_s, c), ".wav"), max(0.0, offset_s))
        for c, name in enumerate(mic_names(full))
    ]


def _build_media(full: str, offset_s: float, dur_s: float) -> dict:
    """Une piste d'écoute par micro, nommée d'après son gain (`mic_names` : il change d'un
    enregistreur à l'autre, et un fichier mono n'a qu'un micro)."""
    path = Path(full)
    wav, sr, start = read_channels(path.parent, path.name, offset_s, dur_s, 0.0)
    names = mic_names(full)[: wav.shape[1]]
    return {
        "t0": start,
        "t1": start + len(wav) / sr,
        "audios": [
            (
                name,
                asset(
                    wav_bytes(wav[:, c], sr),
                    ".wav",
                    name_key=_media_key(full, offset_s, dur_s, c),
                ),
                start,
            )
            for c, name in enumerate(names)
        ],
    }


@st.cache_data(max_entries=6, show_spinner="Lecture de l'enregistrement…")
def _media(full: str, offset_s: float = 0.0, dur_s: float = math.inf) -> dict:
    """Enregistrement entier (ou l'extrait [offset_s, offset_s + dur_s]), des deux micros,
    rangé à côté du visualiseur et gardé en cache : changer de fenêtre ou tracer un
    intervalle ne relit pas le disque. Le spectrogramme, le volume et la bande d'écoute se
    font dans le navigateur. Reprend la lecture d'avance (`_prefetch`) si elle existe."""
    ahead = _prefetcher()[1].pop((full, offset_s, dur_s), None)
    if ahead is not None:
        try:
            return ahead.result()
        except Exception:  # lecture d'avance ratée : on relit, l'erreur éventuelle remontera
            pass
    return _build_media(full, offset_s, dur_s)


# Streamlit < 1.37 : sans fragment, toute la page se réaffiche.
_fragment = getattr(st, "fragment", lambda f: f)


@_fragment
def _viewer_fragment(args: dict, extract: str, ikey: str, ckey: str, interval_mode: bool):
    """Visualiseur et liste des intervalles : tracer, ajuster ou effacer un intervalle ne
    réaffiche que ce fragment, pas toute la page. `ikey` : intervalles de qui annote, avec
    ceux qu'il masque (une valeur renvoyée pour un autre réglage est ignorée)."""
    value = viewer(args)
    if value and value.get("key") == extract and value.get("ivkey") == ikey:
        st.session_state[ikey] = value["intervals"]
        st.session_state[ckey] = int(value.get("channel", 0))
        request = value.get("action")
        if request and st.session_state.get("last_action_nonce") != request.get("nonce"):
            st.session_state["last_action_nonce"] = request.get("nonce")
            if request.get("type") == "submit":
                st.session_state["submit_request"] = extract
                st.rerun(scope="app")  # l'envoi se fait dans `main`, avec le commentaire
            try:
                _apply(request)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.rerun(scope="app")  # listes relues par `main`
    if interval_mode:
        st.caption(_interval_caption(st.session_state[ikey]))


def _apply(request: dict) -> None:
    """Étiquette ou qualité proposée ou retirée depuis la barre du visualiseur."""
    kind = request.get("type")
    if kind == "add_label":
        add_label(request.get("name") or "", request.get("color"), request.get("code"))
    elif kind == "remove_label":
        remove_label(str(request.get("code")))
    elif kind == "add_quality":
        add_quality(str(request.get("quality") or ""))
    elif kind == "remove_quality":
        remove_quality(str(request.get("quality")))


def _classes(form_key: str, answers) -> list[str]:
    columns = st.columns(5)
    return [
        label
        for i, (label, name) in enumerate(answers)
        if columns[i % 5].checkbox(name, key=f"{form_key}::{label}")
    ]


# Le bouton de validation reste grisé sans annotateur.
_NO_ANNOTATOR = "Choisir l'annotateur dans le panneau de gauche."
VALIDATE = "Valider l'annotation ▶"


def _details(qualities: list[str], with_quality: bool = True) -> tuple[str | None, str | None]:
    """Qualité (sauf `with_quality` faux : en intervalles, chacun a la sienne, au-dessus du
    spectrogramme) et commentaire. L'espèce d'un faux ami est une étiquette (« + »)."""
    quality = st.radio("Qualité", ["—", *qualities], horizontal=True) if with_quality else "—"
    comment = st.text_input("Commentaire")
    return None if quality == "—" else quality, comment.strip() or None


def _comment_box(form_key: str, annotator: str) -> tuple[bool, str | None]:
    """Commentaire facultatif et « Valider l'annotation », côte à côte, sous le spectrogramme.
    Hors formulaire : « Valider » à droite du spectrogramme prend aussi le commentaire tapé.
    Renvoie (bouton cliqué, commentaire)."""
    comment_key = f"comment::{form_key}::{st.session_state.get('validated', 0)}"
    with st.container(border=True):
        cols = st.columns([5, 2], vertical_alignment="bottom")
        comment = cols[0].text_input(
            "Commentaire", key=comment_key, placeholder="Facultatif : ce qui mérite d'être dit"
        )
        clicked = cols[1].button(
            VALIDATE,
            type="primary",
            use_container_width=True,
            disabled=not annotator.strip(),
            help=None if annotator.strip() else _NO_ANNOTATOR,
        )
    return clicked, (comment or "").strip() or None


def _save_extract(con, candidate, t0, t1, intervals, annotator, channel, comment) -> bool:
    """Enregistre l'extrait écouté et ses intervalles ; vrai si c'est fait."""
    if not annotator.strip():
        st.error(_NO_ANNOTATOR)
        return False
    try:
        save_span(
            con,
            candidate,
            t0,
            t1,
            [tuple(i) for i in intervals],
            [],
            annotator.strip(),
            comment=comment,
            channel=channel,
        )
    except ValueError as exc:
        st.error(str(exc))
        return False
    st.session_state["validated"] = st.session_state.get("validated", 0) + 1
    _send_copy(con, annotator.strip())
    return True


def _answer_form(con, target, form_key, annotator, channel, qualities) -> bool:
    """Annotation par fenêtres (suspendue) : classes (plusieurs possibles), qualité, espèce,
    commentaire ; vrai si un label a été enregistré."""
    with st.form(key=form_key, clear_on_submit=True):
        st.markdown("**Classes entendues** (plusieurs possibles ; aucune cochée : rien)")
        ticked = _classes(form_key, INTERVAL_ANSWERS)
        quality, comment = _details(qualities)
        sent = st.form_submit_button(
            "Valider ▶",
            type="primary",
            use_container_width=True,
            disabled=not annotator.strip(),
            help=None if annotator.strip() else _NO_ANNOTATOR,
        )
    if not sent:
        return False
    if not annotator.strip():
        st.error(_NO_ANNOTATOR)
        return False
    label, extra = ordered_classes(ticked)
    save_answer(
        con,
        target,
        label,
        annotator.strip(),
        quality=quality,
        comment=comment,
        channel=channel,
        extra_labels=extra,
    )
    _send_copy(con, annotator.strip())
    return True


def _interval_caption(intervals) -> str:
    if not intervals:
        return "Aucun intervalle tracé : tout l'extrait sera négatif."
    return f"{len(intervals)} intervalle(s) : " + " · ".join(
        f"{a:.1f}–{b:.1f} s ({NAMES.get(c, c)}{f', qualité {q}' if q else ''})"
        for a, b, c, q in sorted(((*i, None)[:4] for i in intervals), key=lambda t: t[:3])
    )


# Pendant qu'une case ou un menu relance le script, Streamlit grise toute la page (éléments
# « stale ») : on la laisse telle quelle, le visualiseur garde son état et l'audio.
NO_GREY = """<style>
[data-stale="true"], [data-stale="true"] * { opacity: 1 !important; transition: none !important; }
</style>"""


# Les menus déroulants de Streamlit sont aussi des champs de saisie : on y tape du texte qui
# filtre la liste. Ici les listes sont courtes et choisies à la souris, on rend le champ
# non éditable. Le script tourne dans un cadre de même origine, d'où il atteint la page ; un
# observateur le refait pour les menus créés après coup.
# Il recopie aussi le thème choisi par Streamlit (clair ou sombre : le color-scheme qu'il pose
# sur .stApp) dans `data-gr-theme` sur la page, que lit la feuille de style (`style.py`) ; le
# menu de Streamlit change le thème sans rien recharger, d'où la vérification régulière.
READONLY_MENUS = """<script>
(() => {
  const root = window.parent.document;
  if (window.parent.__blanciMenus) return;
  window.parent.__blanciMenus = true;
  const fix = () => root.querySelectorAll('div[data-baseweb="select"] input').forEach(i => {
    if (i.readOnly) return;
    i.readOnly = true; i.setAttribute("inputmode", "none"); i.style.caretColor = "transparent";
    i.style.cursor = "pointer";
  });
  const theme = () => {
    const app = root.querySelector(".stApp");
    if (!app) return;
    const scheme = window.parent.getComputedStyle(app).colorScheme || "";
    const t = scheme.includes("light") ? "light" : scheme.includes("dark") ? "dark" : null;
    if (t && root.documentElement.dataset.grTheme !== t) root.documentElement.dataset.grTheme = t;
  };
  // Un menu ouvert dans le panneau de gauche se fermait dès qu'on faisait défiler le panneau :
  // le défilement du panneau n'est plus transmis au menu, qui reste ouvert (et suit son champ).
  window.parent.addEventListener("scroll", e => {
    const t = e.target;
    if (t && t.closest && t.closest('[data-testid="stSidebar"]')
        && root.querySelector('[role="listbox"]')) e.stopImmediatePropagation();
  }, true);
  new MutationObserver(fix).observe(root.body, {childList: true, subtree: true});
  window.parent.setInterval(theme, 400);
  fix();
  theme();
})();
</script>"""


_ADD_ANNOTATOR = "➕ Ajouter quelqu'un…"


def _annotator_input(names: list[str]) -> str:
    """Menu « Annotateur » parmi `names` (`annotations/annotateurs.txt`), vide à chaque
    ouverture du poste (plusieurs personnes annotent sur la même base) : sans lui, l'envoi est
    impossible. « Ajouter quelqu'un… » inscrit un nouveau nom dans la liste du projet."""
    if "annotator_added" in st.session_state:  # un menu ne se change qu'avant d'être affiché
        st.session_state["annotator_pick"] = st.session_state.pop("annotator_added")
    picked = st.selectbox(
        "Annotateur",
        [*names, _ADD_ANNOTATOR],
        index=None,
        key="annotator_pick",
        placeholder="Qui annote ?",
    )
    if picked == _ADD_ANNOTATOR:
        with st.form("new_annotator", border=False):
            name = st.text_input("Nom à ajouter à la liste des annotateurs")
            if st.form_submit_button("Ajouter"):
                try:
                    st.session_state["annotator_added"] = add_annotator(name)
                except ValueError:
                    st.error("Écrire un nom.")
                else:
                    st.rerun()
        picked = None
    annotator = picked or ""
    st.session_state["annotator"] = annotator
    return annotator


# Copie des annotations de chacun, réécrite après chaque validation : le dossier où arrivent
# ses annotations (de quoi vérifier qu'elles sont parties, ou les transmettre : fichiers
# « Nom_*.csv », que `blanci import-labels` relit sur une autre machine).
SEND_FOLDER = project_path("annotations/a_envoyer")


def _send_prefix(annotator: str) -> str:
    return re.sub(r"[^\w-]+", "_", annotator).strip("_") + "_"


@st.cache_resource
def _exporter() -> tuple[ThreadPoolExecutor, threading.Lock, set, dict]:
    """Un fil d'écriture des copies à envoyer, les annotateurs dont la copie attend son tour,
    et la dernière erreur de chacun (à montrer au clic suivant)."""
    return (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="export"),
        threading.Lock(),
        set(),
        {},
    )


def _export_job(db: str, annotator: str) -> None:
    """Copie d'`annotator`, avec sa propre connexion (une connexion SQLite ne se partage pas
    entre fils). Elle relit la base au moment de tourner : elle couvre tous les envois faits
    pendant qu'elle attendait."""
    _, lock, queued, errors = _exporter()
    with lock:
        queued.discard(annotator)
    con = connect(Path(db))
    try:
        export_annotations(con, [annotator], SEND_FOLDER, _send_prefix(annotator))
    except OSError as exc:
        errors[annotator] = f"{exc} : fermer le CSV ouvert"
    except Exception as exc:  # sans quoi le fil l'avalerait : on la montre au clic suivant
        errors[annotator] = f"{type(exc).__name__} : {exc}"
    else:
        errors.pop(annotator, None)
    finally:
        con.close()


def _send_copy(con, annotator: str) -> None:
    """Réécrit les CSV d'`annotator` dans `SEND_FOLDER` (`export_annotations`), en arrière-plan :
    l'extrait suivant s'affiche sans attendre. Une copie déjà en attente pour lui suffit. Un
    fichier ouvert dans Excel ne s'écrit pas : l'annotation est enregistrée quand même, la copie
    attend l'envoi suivant (on le signale au clic d'après)."""
    pool, lock, queued, errors = _exporter()
    if error := errors.pop(annotator, None):
        st.toast(f"Copie à envoyer non mise à jour ({error}).")
    db = con.execute("PRAGMA database_list").fetchone()[2]
    with lock:
        if annotator in queued:
            return
        queued.add(annotator)
    pool.submit(_export_job, db, annotator)


def _send_panel(annotator: str) -> None:
    """Le dossier où arrivent ses annotations validées, et un bouton qui l'ouvre (Windows)."""
    st.header("Annotations validées")
    st.caption("Chaque annotation validée est copiée dans ce dossier avec le nom de l'annotateur.")
    if hasattr(os, "startfile") and st.button(
        "📂 Ouvrir le dossier des annotations", use_container_width=True, help=str(SEND_FOLDER)
    ):
        SEND_FOLDER.mkdir(parents=True, exist_ok=True)
        os.startfile(SEND_FOLDER)


def _masking_panel(names: list[str]) -> tuple[list[str], bool, bool]:
    """Cases « Masquer les annotations de » : une par annotateur de la liste, et « tout le
    monde » ; puis « Masquer les commentaires ». Renvoie (noms masqués, tout le monde,
    commentaires masqués). Les commentaires d'un annotateur masqué le sont aussi."""
    st.header("Affichage")
    st.markdown("**Masquer les annotations de**")
    # Grille de deux colonnes, rangée par rangée (« Tout le monde » en tête) : les cases d'une
    # rangée sont à la même hauteur, et aucune ne reste seule dans sa colonne.
    everyone, hidden = False, []
    items = [None, *names]  # None : « Tout le monde »
    for row in range(0, len(items), 2):
        for column, name in zip(st.columns(2), items[row : row + 2], strict=False):
            if name is None:
                everyone = column.checkbox("Tout le monde", key="hide_everyone")
            elif column.checkbox(name, key=f"hide::{name}", disabled=everyone):
                hidden.append(name)
    no_comments = st.checkbox("Masquer les commentaires", key="hide_comments")
    return hidden, everyone, no_comments


def _comments_panel(comments: list[dict], t0: float, t1: float) -> None:
    """Commentaires déjà laissés sur l'extrait, sous le spectrogramme : qui, quand, et la
    partie de l'enregistrement quand ce n'est pas tout l'extrait (fenêtre, autre découpage)."""
    if not comments:
        return
    lines = []
    for c in comments:
        where = (
            ""
            if abs(c["start_s"] - t0) < 0.05 and abs(c["end_s"] - t1) < 0.05
            else f" · {c['start_s']:.1f}–{c['end_s']:.1f} s"
        )
        date = str(c["date"] or "")[:10]
        lines.append(f"- **{c['who'] or '(sans nom)'}** ({date}{where}) : {c['text']}")
    st.markdown("💬 **Commentaires déjà laissés**\n" + "\n".join(lines))


def _recordings_root(cfg) -> list[Path]:
    """Dossiers collés à la main (retenus à côté de la base), en plus des disques branchés que
    `locate` parcourt seul."""
    return remembered_roots(config_path(cfg, "db"))


def _missing_recording(cfg, path: str) -> None:
    """Enregistrement introuvable sur tous les disques : on le dit, et on propose de coller le
    dossier où il se trouve (retenu pour les fois suivantes)."""
    st.error(
        f"Enregistrement introuvable sur les disques branchés : {path}. Brancher le disque, "
        "ou coller ci-dessous le dossier qui contient ce chemin."
    )
    folder = (
        st.text_input(
            "Dossier des enregistrements",
            placeholder="/Volumes/MonDisque  ou  D:\\",
            help="Le dossier d'où part le chemin ci-dessus (souvent la racine du disque). Retenu "
            "pour les prochaines fois.",
        )
        .strip()
        .strip('"')
    )
    if folder:
        if any((Path(folder) / v).is_file() for v in variants(path, cfg["audio"].get("renamed"))):
            remember_root(config_path(cfg, "db"), Path(folder))
            st.rerun()
        st.warning(f"Pas de {path} sous {folder}.")


def _notes(saved: dict, me: str, heard) -> list[str]:
    """Ce qui a déjà été dit de l'extrait sans intervalle à montrer : chaque envoi vide (de soi
    ou d'un autre), ou une ancienne réponse par fenêtres."""
    notes = []
    for who, ivs in saved.items():
        if ivs:
            continue
        if me and who is not None and who.casefold() == me.casefold():
            notes.append(
                "Votre dernier envoi sur cet extrait ne comptait aucun intervalle : rien noté."
            )
        else:
            notes.append(
                f"{who or '(sans nom)'} a écouté cet extrait et n'y a rien noté (aucun intervalle)."
            )
    if not saved and heard:
        notes.append(
            f"Déjà écouté, sans intervalle (réponse par fenêtres) : {NAMES.get(heard, heard)}."
        )
    return notes


def _project_fields(kind: str, active: str) -> None:
    """Contenu de la fenêtre « Nouveau projet » ou « Modifier le projet » : nom, puis l'en-tête
    du poste (titre, espèce, sous-titre, chant de l'espèce et sa bande d'écoute, légende), qui
    s'écrit dans le dossier du projet (`bandeau.save_header`). Un nouveau projet part des
    étiquettes d'origine ; sans espèce ni chant, il garde aussi l'en-tête d'origine."""
    new = kind == "new"
    folder = None if new else projets.header_dir(active)
    current = None if new else read_header(folder)
    base = current or DEFAULT_HEADER
    band = list(base.get("band_khz") or [None, None]) if not new else [None, None]
    default = not new and active == projets.DEFAULT_PROJECT
    with st.form(f"project::{kind}", border=False):
        name = st.text_input(
            "Nom du projet",
            value="" if new else projets.label(active),
            disabled=default,
            help="Le projet de départ ne se renomme pas." if default else None,
        )
        st.markdown("**En-tête du poste**")
        cols = st.columns(2)
        title = cols[0].text_input("Titre", value=base["title"])
        species = cols[1].text_input(
            "Espèce", value="" if new else base["species"], placeholder="Anomaloglossus blanci"
        )
        subtitle = st.text_input("Sous-titre", value=base["subtitle"])
        song = st.file_uploader(
            "Chant de l'espèce" + ("" if new else " (vide : garder celui du projet)"),
            type=["wav", "mp3", "flac", "ogg"],
            help="Ses 30 premières secondes sont gardées : « Écouter le chant » les joue, le "
            "spectrogramme de l'en-tête les montre.",
        )
        cols = st.columns([1, 1, 2])
        lo = cols[0].number_input("Bande d'écoute de (kHz)", 0.0, 96.0, band[0], 0.1)
        hi = cols[1].number_input("à (kHz)", 0.0, 96.0, band[1], 0.1)
        legend = cols[2].text_input(
            "Légende du chant (lieu, date…)",
            value=(current or {}).get("legend_text", ""),
            placeholder="facultatif",
        )
        st.caption("La bande d'écoute est tracée en pointillés sur le spectrogramme de l'en-tête.")
        sent = st.form_submit_button(
            "Créer le projet" if new else "Enregistrer", type="primary", use_container_width=True
        )
    if not new and current and st.button("Revenir à l'en-tête d'origine", use_container_width=True):
        reset_header(folder)
        st.rerun()
    if not sent:
        return
    unchanged = (title, species, subtitle, song, lo, hi, legend) == (
        base["title"],
        base["species"],
        base["subtitle"],
        None,
        band[0],
        band[1],
        (current or {}).get("legend_text", ""),
    )
    if not new and unchanged and (default or name.strip() == active):
        st.rerun()  # rien de changé : la fenêtre se ferme
    has_header = bool(species.strip() or song is not None or current) and not (
        not new and unchanged
    )
    if has_header and (lo is None or hi is None):
        st.error("Donner la bande d'écoute du chant (de … à … kHz).")
        return
    try:
        if new:
            created = projets.create_project(name)
            folder = projets.header_dir(created)
        if has_header:
            try:
                save_header(
                    title,
                    species,
                    subtitle,
                    (lo, hi),
                    song.getvalue() if song is not None else None,
                    legend,
                    folder=folder,
                )
            except ValueError:
                if new:  # projet à moitié créé : on le retire, rien n'y a été écrit d'autre
                    shutil.rmtree(projets.project_dir(created), ignore_errors=True)
                raise
        target = created if new else active
        if not new and not default:
            target = projets.rename_project(active, name)
    except ValueError as exc:
        st.error(str(exc))
        return
    write_prefs(projet=target)
    st.session_state.pop("project_pick", None)
    st.rerun()


@st.dialog("Nouveau projet", width="large")
def _new_project_dialog(active: str) -> None:
    _project_fields("new", active)


@st.dialog("Modifier le projet", width="large")
def _edit_project_dialog(active: str) -> None:
    _project_fields("edit", active)


def _project_input() -> None:
    """Menu « Projet » : le projet actif (préréglage : en-tête, étiquettes, files d'écoute et
    style) ; en choisir un autre recharge la page avec ses fichiers. « Modifier ce projet… » et
    « Nouveau projet… » ouvrent une fenêtre au milieu de la page (`_project_fields`), et le menu
    revient au projet actif."""
    names = projets.list_projects()
    active = str(read_prefs()["projet"])
    if active not in names:
        active = names[0]
    if st.session_state.get("project_pick") not in names:
        st.session_state["project_pick"] = active
    st.selectbox(
        "Projet",
        [*names, _EDIT_PROJECT, _NEW_PROJECT],
        key="project_pick",
        format_func=lambda n: (
            f"✏️ Modifier « {projets.label(active)} »…"
            if n == _EDIT_PROJECT
            else n
            if n == _NEW_PROJECT
            else projets.label(n)
        ),
        on_change=_choose_project,
        args=(active,),
    )
    dialog = st.session_state.pop("project_dialog", None)
    if dialog == "new":
        _new_project_dialog(active)
    elif dialog == "edit":
        _edit_project_dialog(active)


_NEW_PROJECT = "➕ Nouveau projet…"
_EDIT_PROJECT = "✏️ Modifier ce projet…"


def _choose_project(active: str) -> None:
    picked = st.session_state.get("project_pick")
    if picked in (_NEW_PROJECT, _EDIT_PROJECT):
        st.session_state["project_dialog"] = "new" if picked == _NEW_PROJECT else "edit"
        st.session_state["project_pick"] = active
    elif picked:
        write_prefs(projet=picked)


def main() -> None:
    prefs = read_prefs()
    queues_dir = projets.activate(str(prefs["projet"]))
    species = (read_header() or DEFAULT_HEADER)["species"]
    st.set_page_config(page_title=f"Annotation · {species}", page_icon="🐸", layout="wide")
    st.markdown(NO_GREY + "\n" + css(prefs["theme"], prefs["style"]), unsafe_allow_html=True)
    if hasattr(st, "iframe"):  # `components.html` est retiré des Streamlit récents
        st.iframe(READONLY_MENUS, height=1)
    else:
        components.html(READONLY_MENUS, height=0)
    choice = bandeau()  # boutons ronds de l'en-tête : mode jour / nuit, style
    if choice and choice.get("jeton") != st.session_state.get("look_token"):
        st.session_state["look_token"] = choice.get("jeton")
        write_prefs(theme=choice.get("theme"), style=choice.get("style"))
        st.rerun()
    config = _config_file()
    cfg, con = _setup(config)
    if queues_dir is not None:  # les files d'écoute du projet, pas celles du projet de départ
        queues_dir.mkdir(parents=True, exist_ok=True)
        cfg = {**cfg, "paths": {**cfg["paths"], "queues": str(queues_dir)}}
    _apply_pending_queue()
    raw = config_path(cfg, "raw")
    extra = _recordings_root(cfg)
    band_hz = tuple(float(f) for f in cfg["signal"]["band_hz"])
    spectro_default = int(cfg["audio"]["channel"] == 1)

    # Étiquettes et qualités du projet ; noms et couleurs de toutes les étiquettes connues.
    label_set = read_set()
    known = known_labels()
    NAMES.update({c: v["name"] for c, v in known.items()})
    used = [r[0] for r in con.execute("SELECT DISTINCT label FROM intervals ORDER BY label")]
    for code in used:
        NAMES.setdefault(code, code)

    with st.sidebar:
        # Marque, puis le mode d'emploi juste dessous (sa fenêtre s'ouvre en haut de la page).
        with st.container(key="marque"):
            st.markdown(BRAND, unsafe_allow_html=True)
            with st.popover("❓ Mode d'emploi"):
                st.markdown(HELP)
        st.header("Session")
        names = read_annotators()
        who_col, project_col = st.columns(2)
        with who_col:
            annotator = _annotator_input(names)
        with project_col:
            _project_input()
        mode, chosen, encoder = _selection_panel(cfg, con)
        candidate_slot = st.container()
        st.header("Navigation")
        skip_mine = st.checkbox("Sauter les extraits que j'ai déjà écoutés", True, key="skip_mine")
        skip_others = st.checkbox(
            "Sauter les extraits écoutés par un collègue", False, key="skip_others"
        )
        hidden, everyone, no_comments = _masking_panel(names)
        _send_panel(annotator)
        # Rubrique « Méthode » masquée (annotation par fenêtres suspendue) : mettre
        # SHOW_METHOD à True pour la réafficher.
        window_mode, split_s = False, 3.0
        if SHOW_METHOD:
            st.header("Méthode")
            window_mode = st.checkbox(
                "Annoter par fenêtres (suspendu)",
                value=False,
                help="Ancienne méthode : un label par fenêtre de longueur fixe, sur la même "
                "page. Par défaut, on trace les intervalles où l'espèce chante (§5.6).",
            )
            split_s = st.number_input(
                "Longueur des fenêtres (s)", 0.5, 60.0, 3.0, 0.5, disabled=not window_mode
            )

    if mode == MAP:
        _map_page(cfg, con, encoder, config)
        return
    if chosen is None:
        st.info("Choisir une file d'écoute, ou en générer une avec le mode de sélection.")
        return

    try:
        queue = load_candidates(chosen, con)
    except ValueError as exc:
        st.error(str(exc))
        return
    if queue.empty:
        st.warning("File vide.")
        return
    if "cluster" in queue.columns and "encoder_id" in queue.columns:
        _cluster_panel(cfg, con, queue)
    # Annotateurs affichés (`shown`), et ceux dont les réponses font « déjà écouté » dans la
    # liste (`who`) : les mêmes, plus soi-même, même masqué.
    me = annotator.strip()
    shown = visible_annotators(con, hidden, everyone)
    who = visible_annotators(con, hidden, everyone, keep=(me,) if me else ())
    done = progress(con, queue, who)
    n = len(queue)
    offset_h = cfg["recorder"]["filename_utc_offset_h"]
    # Ce que ◀ ▶ et la validation sautent : ses écoutes, celles des collègues, ou rien.
    nav_args = (me, _colleagues(con, me, hidden, everyone), skip_mine, skip_others)
    skipping = skip_mine or skip_others
    nav = _to_skip(con, queue, *nav_args)
    all_heard = skipping and bool(nav.notna().all())

    key = f"pos::{chosen}"
    end = f"end::{chosen}"  # file finie : plus aucun extrait affiché, jusqu'au choix d'un autre
    # Ouverture de la file : premier extrait qui reste à écouter, sinon le premier de la file
    # (file de réécoute) ; l'écran de fin n'apparaît qu'à la réponse qui finit la file.
    if key not in st.session_state:
        st.session_state[key] = max(next_position(nav, -1, skipping), 0)
        st.session_state[end] = False
    pos = min(max(int(st.session_state[key]), 0), n - 1)
    st.session_state[key] = pos
    if not all_heard:
        st.session_state[end] = False
    with candidate_slot:  # liste des extraits, puis, en dernier, ce qu'on sait de la file
        _candidate_strip(queue, done, chosen, key, end, offset_h)
        readme = chosen.parent / "LISEZMOI.md"
        if readme.is_file():
            with st.expander("Informations sur cette file d'écoute"):
                st.markdown(readme.read_text(encoding="utf-8"))
    if all_heard and st.session_state.get(end):
        st.success(
            "File terminée : tous les extraits sont écoutés. Pour en réécouter un, le choisir "
            "dans la liste de la file (panneau de gauche)."
        )
        return

    candidate = queue.iloc[pos].to_dict()
    st.subheader(  # « en-ecoute » : la feuille de style met le point rouge devant
        f"{candidate['site']} · {candidate['mic_id']} · "
        f"{local_time(candidate['start_utc'], offset_h)} (heure locale)",
        anchor="en-ecoute",
    )
    score = candidate.get("score")
    heard = done.iloc[pos]
    st.caption(
        f"Extrait {pos + 1} / {n} · {Path(str(candidate['path'])).name} · fenêtre "
        f"{candidate['offset_s']:.1f}–{candidate['offset_s'] + candidate['dur_s']:.1f} s · "
        f"{candidate.get('reason') or ''}"
        + (f" · score précédent {score:.2f}" if score == score and score is not None else "")
        + (f" · déjà écouté : {NAMES.get(heard, heard)}" if heard else "")
    )

    media = None
    full = locate(str(candidate["path"]), raw, extra, cfg["audio"].get("renamed"))
    if full is None:  # disque débranché, fichier déplacé
        _missing_recording(cfg, str(candidate["path"]))
    else:
        try:
            media = _media(str(full))
        except Exception as exc:  # fichier ouvert ailleurs, abîmé
            st.error(f"Lecture impossible : {full} ({exc}). Fichier ouvert ailleurs ou abîmé ?")

    # ◀ ▶ : voisin dans la file, ou précédent / prochain qui reste à écouter (sans faire le tour).
    before = next_position(nav, pos, skipping, step=-1, wrap=False)
    after = next_position(nav, pos, skipping, wrap=False)
    # Chargement d'avance de l'extrait suivant, pendant qu'on écoute celui-ci : lu sur le disque
    # ici (`_prefetch`), puis décodé et dessiné par le navigateur (`next_audios`), avant le rendu
    # du visualiseur pour que les deux commencent tout de suite.
    next_audios = None
    if media is not None and after != pos:
        upcoming = queue.iloc[after].to_dict()
        next_full = locate(str(upcoming["path"]), raw, extra, cfg["audio"].get("renamed"))
        planned_next = upcoming.get("source") == "plan" and not window_mode
        span = (float(upcoming["offset_s"]), float(upcoming["dur_s"]))
        # Même enregistrement : seul l'extrait suivant (jeu de test en extraits de 30 s) est à lire.
        same = next_full is not None and str(next_full) == str(full)
        if next_full is not None and (not same or planned_next):
            _prefetch(str(next_full), span if planned_next else None, whole=not same)
            next_audios = _next_audios(str(next_full), *(span if planned_next else (0.0, math.inf)))

    # Fenêtres de la page : celle du candidat, ou le découpage de l'extrait (mode fenêtres).
    cand_offset, cand_dur = float(candidate["offset_s"]), float(candidate["dur_s"])
    windows = [(cand_offset, cand_dur)]
    if window_mode and media is not None:
        offsets = split_windows(media["t0"], media["t1"], cand_offset, float(split_s))
        windows = [(o, float(split_s)) for o in offsets] or windows
    wkey = f"win::{chosen}::{pos}::{len(windows)}"
    if wkey not in st.session_state:
        st.session_state[wkey] = min(
            range(len(windows)), key=lambda i: abs(windows[i][0] - cand_offset)
        )
    w = min(int(st.session_state[wkey]), len(windows) - 1)
    ids = [window_id_for(candidate["recording_id"], o, d) for o, d in windows]
    window_labels = latest_labels(con, ids, shown) if window_mode else {}

    intervals: list = []
    channel = spectro_default
    form_key = f"span::{chosen}::{pos}"
    # Extrait écouté : l'enregistrement entier (n° 182), sauf pour une file tirée par plan,
    # dont l'extrait est la fenêtre du candidat (30 s du lot 1, n° 196). L'extrait enregistré
    # est ce qu'on a écouté : tout ce qui y est sans intervalle devient négatif.
    t0, t1 = (media["t0"], media["t1"]) if media is not None else (0.0, 0.0)
    planned = candidate.get("source") == "plan"
    if media is not None and planned and not window_mode and cand_dur < t1 - t0 - 0.05:
        try:
            media = _media(str(full), cand_offset, cand_dur)  # le lecteur ne joue que l'extrait
        except Exception as exc:
            st.error(f"Lecture impossible : {full} ({exc}). Fichier ouvert ailleurs ou abîmé ?")
            media = None
        else:
            t0, t1 = media["t0"], media["t1"]
    extract = f"{candidate['path']}:{t0:.2f}:{t1:.2f}"
    submit_now = media is not None and st.session_state.pop("submit_request", None) == extract
    if media is not None:
        # Ses intervalles (son dernier envoi, sauf s'il se masque) et ceux des autres affichés.
        saved = extract_intervals(con, candidate["recording_id"], t0, t1, shown)
        notes = _notes(saved, me, heard)
        comments = (
            [] if no_comments else extract_comments(con, candidate["recording_id"], t0, t1, shown)
        )
        mine = next((k for k in saved if k is not None and k.casefold() == me.casefold()), None)
        own = saved.pop(mine, []) if me else []
        others = {k or "(sans nom)": v for k, v in saved.items()}
        masked = "all" if everyone else ",".join(sorted(h.casefold() for h in hidden))
        ikey = f"iv::{extract}::{me.casefold()}::{masked}"
        if ikey not in st.session_state:
            # Intervalles tracés avant de s'identifier : repris à son nom, pas perdus.
            anon = f"iv::{extract}::::{masked}"
            draft = st.session_state.pop(anon, None) if me else None
            st.session_state[ikey] = [list(i) for i in (draft or own)]
        shown_windows = [
            {
                "t0": o,
                "t1": o + d,
                "label": NAMES.get(window_labels.get(i), window_labels.get(i)),
                "current": window_mode and j == w,
                "candidate": abs(o - cand_offset) < 0.005 and abs(d - cand_dur) < 0.005,
            }
            for j, ((o, d), i) in enumerate(zip(windows, ids, strict=True))
        ]
        ckey = f"ch::{extract}"
        st.session_state.setdefault(ckey, spectro_default)
        _viewer_fragment(
            viewer_args(
                t0,
                t1,
                media["audios"],
                shown_windows,
                band_hz,
                key=extract,
                labels=[(e["code"], e["name"]) for e in label_set["labels"]],
                colors={c: v["color"] for c, v in known.items() if c not in LABEL_COLORS},
                intervals=st.session_state[ikey],
                interval_mode=not window_mode,
                channel=spectro_default,
                next_audios=next_audios,
                others=others,
                intervals_key=ikey,
                qualities=label_set["qualities"],
                names=dict(NAMES),
                suggestions=[(e["code"], e["name"], e["color"]) for e in suggestions(used)],
                submit=None
                if window_mode
                else {
                    "label": "Valider\n▶",
                    "disabled": not me,
                    "help": "Valider l'annotation (Ctrl + Entrée)" if me else _NO_ANNOTATOR,
                },
                notes=notes,
            ),
            extract,
            ikey,
            ckey,
            not window_mode,
        )
        _comments_panel(comments, t0, t1)
        intervals = st.session_state[ikey]
        channel = st.session_state[ckey]

    if not window_mode:
        if media is not None:
            clicked, comment = _comment_box(form_key, annotator)
            if (clicked or submit_now) and _save_extract(
                con, candidate, t0, t1, intervals, annotator, channel, comment
            ):
                _advance(con, queue, pos, nav_args, key, end)
                st.rerun()
    else:
        offset, dur = windows[w]
        if len(windows) > 1:
            cols = st.columns([1, 6, 1])
            cols[0].button("◀ Fenêtre", disabled=w == 0, on_click=_goto, args=(wkey, w - 1))
            widget = f"{wkey}::select"
            st.session_state[widget] = w
            cols[1].selectbox(
                "Fenêtre",
                range(len(windows)),
                format_func=lambda j: (
                    f"Fenêtre {j + 1} / {len(windows)} : "
                    f"{windows[j][0]:.1f}–{windows[j][0] + windows[j][1]:.1f} s"
                    + (
                        f" · ✓ {NAMES.get(window_labels[ids[j]], window_labels[ids[j]])}"
                        if ids[j] in window_labels
                        else ""
                    )
                ),
                key=widget,
                on_change=_follow,
                args=(wkey, widget),
                label_visibility="collapsed",
            )
            cols[2].button(
                "Fenêtre ▶", disabled=w >= len(windows) - 1, on_click=_goto, args=(wkey, w + 1)
            )
        target = dict(candidate, offset_s=offset, dur_s=dur)
        if (offset, dur) != (cand_offset, cand_dur):
            target["score"] = None  # le score précédent est celui de la fenêtre du candidat
        qualities = label_set["qualities"]
        if _answer_form(con, target, f"form::{chosen}::{pos}::{w}", annotator, channel, qualities):
            if w < len(windows) - 1:
                st.session_state[wkey] = w + 1
            else:
                _advance(con, queue, pos, nav_args, key, end)
            st.rerun()

    # ◀ ▶ grisés quand il n'y a plus rien à écouter dans ce sens ; le survol dit pourquoi.
    if all_heard:
        why = (
            "Vous avez tout écouté."
            if skip_mine and not skip_others
            else "Tous les extraits de la file sont déjà écoutés."
        ) + " Décocher « Sauter… » pour parcourir la file."
        why_before = why_after = why
    else:
        why_before = "Rien à écouter avant celui-ci." if before == pos else None
        why_after = "Rien à écouter après celui-ci." if after == pos else None
    # Chacun une moitié de la largeur, en forme de flèche (`style.py`, `nav_prev`, `nav_next`).
    nav_cols = st.columns(2, gap="small")
    nav_cols[0].button(
        "Extrait précédent",
        key="nav_prev",
        disabled=before == pos,
        on_click=_goto,
        args=(key, before),
        help=why_before,
        use_container_width=True,
    )
    nav_cols[1].button(
        "Extrait suivant",
        key="nav_next",
        disabled=after == pos,
        on_click=_goto,
        args=(key, after),
        help=why_after,
        use_container_width=True,
    )


main()
