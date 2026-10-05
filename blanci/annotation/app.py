"""Poste d'annotation Streamlit (§5, M2, DECISIONS n° 100) : outil de travail, pas le livrable.

    uv sync --group app
    uv run blanci --config config/local.yaml annotate

- **Mode de sélection** (panneau de gauche) : une file déjà écrite, ou une nouvelle file tirée
  sur place par n'importe quelle méthode de l'outil de sélection (`blanci/annotation/selection.py` :
  60-20-20 à proportions réglables, similarité, couverture, groupes, audit, hasard, negative
  mining, phénologie, suspects, congénères), ou la **carte des embeddings** (YAPAT fait
  maison) : on entoure une zone de points, on l'écoute.
- **Navigation** : sous la file, la liste de ses candidats (numérotés dans la file entière,
  ✓ et label pour ceux déjà écoutés) ; on peut revenir sur n'importe lequel et le
  réécouter, une nouvelle réponse s'ajoute à l'ancienne (correction, jamais écrasée).
- **Écoute** : spectrogramme zoomable (`viewer.py` : molette, glisser, barre de lecture qui
  suit l'un ou l'autre micro), extrait autour du candidat ou enregistrement entier, volume, et
  bande d'écoute réglable (position et largeur) avec « N'écouter que la bande » ; rien de cela
  ne touche l'audio d'origine.
- **Découpage** : l'extrait se découpe en fenêtres de longueur choisie, calées sur le
  candidat ; on les annote une à une sans quitter l'enregistrement.
- **Réponse** : une ou plusieurs classes, qualité, espèce, commentaire, puis « Envoyer ▶ »
  (ou Entrée dans un champ) : le label est ajouté (jamais écrasé) et la fenêtre suivante
  s'affiche (fenêtre suivante du découpage, sinon candidat suivant).
- **Groupes** : une file tirée par groupes montre, groupe par groupe, ce qui a été entendu ; un
  groupe homogène s'étiquette en entier d'un clic (source « bulk »).

Toute la logique est dans `workbench.py` et `selection.py` ; ce fichier ne fait qu'afficher.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

from blanci.annotation.viewer import viewer_html
from blanci.annotation.workbench import (
    ANSWERS,
    latest_labels,
    load_candidates,
    local_time,
    next_position,
    ordered_classes,
    progress,
    read_clip,
    save_answer,
    spectrogram_png,
    split_windows,
    wav_bytes,
)
from blanci.core.config import config_path, load_config
from blanci.core.db import connect, window_id_for
from blanci.inputs.labels import QUALITIES

CHANNELS = {0: "micro 1 (gain 6 dB)", 1: "micro 2 (gain 18 dB)"}
NAMES = dict(ANSWERS)
CONTEXT, WHOLE = "context", "whole"
EXTENTS = {CONTEXT: "candidat et contexte", WHOLE: "enregistrement entier"}
# Au-delà, l'écoute est rééchantillonnée : un enregistrement entier reste léger dans la page.
LONG_EXTRACT_S, LISTEN_MAX_SR = 20.0, 24_000
EXISTING, MAP = "existing", "map"
MODES = {
    EXISTING: "File déjà écrite",
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
NEEDS_ENCODER = (
    "active",
    "similarity",
    "coverage",
    MAP,
    "cluster",
    "negative_mining",
    "suspects",
)


def _config_file() -> Path | None:
    """`--config` après `--` sur la ligne de commande streamlit, sinon $BLANCI_CONFIG."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    env = os.environ.get("BLANCI_CONFIG")
    return args.config or (Path(env) if env else None)


def _setup(config: Path | None):
    """Config et connexion, rouvertes à chaque rafraîchissement : Streamlit change de fil
    d'exécution entre deux clics, et une connexion SQLite ne se partage pas entre fils."""
    cfg = load_config(config)
    return cfg, connect(config_path(cfg, "db"))


def _queues(reports: Path) -> list[Path]:
    patterns = ("candidats_*.csv", "queue_*.csv", "search_*.csv")
    found = {p for pattern in patterns for p in reports.glob(pattern)}
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


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
        st.session_state["queue_path"] = pending


# --- Panneau de gauche : sélection -------------------------------------------------------------


def _selection_panel(cfg, con, reports: Path) -> tuple[str, Path | None, str | None]:
    """(mode, file choisie ou None, encodeur ou None)."""
    from blanci.annotation.selection import select_candidates, write_queue

    st.header("Sélection")
    mode = st.selectbox("Mode de sélection", list(MODES), format_func=MODES.get, key="mode")
    encoder = None
    encoders = _encoders(con)
    if mode in NEEDS_ENCODER and not encoders:
        st.info("Aucun encodeur encodé : `blanci embed` d'abord.")
        return mode, None, None
    if (mode in NEEDS_ENCODER or mode == "gaps") and encoders:
        encoder = st.selectbox("Encodeur", encoders, key="encoder")
    if mode == EXISTING:
        queues = _queues(reports)
        if not queues:
            st.warning(f"Aucune file dans {reports}. En tirer une avec un autre mode.")
            return mode, None, encoder
        wanted = st.session_state.get("queue_path")
        index = next((i for i, q in enumerate(queues) if str(q) == wanted), 0)
        chosen = st.selectbox(
            "File de candidats", queues, index=index, format_func=lambda p: p.name
        )
        return mode, chosen, encoder
    if mode == MAP:
        return mode, None, encoder

    options: dict = {}
    options["n"] = st.number_input(
        "Candidats (par groupe pour les groupes)",
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
            st.warning("Aucun candidat.")
            return mode, None, encoder
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _open_queue(write_queue(cfg, queue, f"{mode}_{stamp}"))
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
    chart = (
        alt.Chart(points)
        .mark_circle(size=16, opacity=0.7)
        .encode(
            x=alt.X("x", axis=None),
            y=alt.Y("y", axis=None),
            color=alt.Color("label", legend=alt.Legend(title="label")),
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
        _open_queue(write_queue(cfg, map_selection(con, unheard), f"carte_{stamp}"))


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


def _candidate_strip(queue, done, chosen, key, offset_h) -> None:
    """Liste déroulante des candidats de la file ouverte, sous la liste des files : numéro dans
    la file entière, site, micro, heure, et ✓ label s'il a déjà été écouté."""
    rows = [
        f"{i + 1}. {site} · {mic} · {local_time(start, offset_h)} · {offset:.0f} s"
        + (f" · ✓ {NAMES.get(heard, heard)}" if heard else "")
        for i, (site, mic, start, offset, heard) in enumerate(
            zip(
                queue["site"],
                queue["mic_id"],
                queue["start_utc"],
                queue["offset_s"],
                done,
                strict=True,
            )
        )
    ]
    widget = f"strip::{chosen}"
    st.session_state[widget] = st.session_state[key]
    st.selectbox(
        f"Candidats de la file ({int(done.notna().sum())} / {len(queue)} écoutés)",
        range(len(queue)),
        format_func=rows.__getitem__,
        key=widget,
        on_change=_follow,
        args=(key, widget),
    )


@st.cache_data(max_entries=6, show_spinner="Lecture de l'enregistrement…")
def _media(
    raw: str,
    path: str,
    offset_s: float,
    dur_s: float,
    context_s: float | None,
    spectro_channel: int,
    gain_db: float,
    band_hz: tuple[float, float] | None,
) -> dict:
    """Spectrogramme et écoute des deux micros, gardés en cache : changer de fenêtre dans le
    découpage ne relit pas le disque. `context_s` None : l'enregistrement entier."""
    clips = {
        c: read_clip(
            Path(raw),
            path,
            0.0 if context_s is None else offset_s,
            math.inf if context_s is None else dur_s,
            context_s or 0.0,
            c,
        )
        for c in CHANNELS
    }
    wav, sr, start = clips[spectro_channel]
    stop = start + len(wav) / sr
    max_sr = LISTEN_MAX_SR if stop - start > LONG_EXTRACT_S else None
    png, fmax = spectrogram_png(wav, sr)
    return {
        "png": png,
        "fmax": fmax,
        "t0": start,
        "t1": stop,
        "audios": [
            (CHANNELS[c], wav_bytes(w, rate, gain_db, band_hz, max_sr), s0)
            for c, (w, rate, s0) in clips.items()
        ],
    }


def _iframe(html: str, height: int) -> None:
    """Page HTML avec scripts : `st.iframe` (Streamlit ≥ 1.5x), sinon l'ancien
    `components.html`, retiré des versions récentes."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components

        components.html(html, height=height)


def _answer_form(con, target, form_key, annotator, channel) -> bool:
    """Classes (plusieurs possibles), qualité, espèce, commentaire ; vrai si un label a été
    enregistré."""
    with st.form(key=form_key, clear_on_submit=True):
        st.markdown("**Classes entendues** (plusieurs possibles ; aucune cochée : rien)")
        columns = st.columns(5)
        ticked = [
            label
            for i, (label, name) in enumerate(ANSWERS)
            if columns[i % 5].checkbox(name, key=f"{form_key}::{label}")
        ]
        quality = st.radio("Qualité (si A. blanci)", ["—", *QUALITIES], horizontal=True)
        species = st.text_input("Espèce entendue (faux ami, congénère…)")
        comment = st.text_input("Commentaire (conditions, chant lointain, pluie…)")
        sent = st.form_submit_button("Envoyer ▶", type="primary", use_container_width=True)
    if not sent:
        return False
    if not annotator.strip():
        st.error("Indiquer l'annotateur dans le panneau de gauche.")
        return False
    label, extra = ordered_classes(ticked)
    save_answer(
        con,
        target,
        label,
        annotator.strip(),
        quality=None if quality == "—" else quality,
        comment=comment.strip() or None,
        channel=channel,
        species=species.strip() or None,
        extra_labels=extra,
    )
    return True


def main() -> None:
    st.set_page_config(page_title="Annotation blanci", layout="wide")
    config = _config_file()
    cfg, con = _setup(config)
    _apply_pending_queue()
    raw, reports = config_path(cfg, "raw"), config_path(cfg, "reports")
    band_default = tuple(f / 1000 for f in cfg["signal"]["band_hz"])

    with st.sidebar:
        st.header("Session")
        annotator = st.text_input("Annotateur", value=st.session_state.get("annotator", ""))
        st.session_state["annotator"] = annotator
        mode, chosen, encoder = _selection_panel(cfg, con, reports)
        candidate_slot = st.container()
        st.header("Écoute")
        skip_done = st.checkbox(
            "Après une réponse, sauter les candidats déjà écoutés",
            value=True,
            help="Les candidats déjà écoutés restent accessibles par la liste et ◀ ▶.",
        )
        calibration = st.checkbox(
            "Calibration : ne compter que mes réponses",
            value=False,
            help="Deux annotateurs écoutent la même file sans voir les réponses de l'autre (§5).",
        )
        channel = st.radio(
            "Spectrogramme",
            list(CHANNELS),
            format_func=CHANNELS.get,
            index=int(cfg["audio"]["channel"] == 1),
        )
        extent = st.radio(
            "Étendue affichée", list(EXTENTS), format_func=EXTENTS.get, horizontal=True
        )
        context_s = (
            st.slider("Contexte autour de la fenêtre (s)", 0.0, 30.0, 3.0, 0.5)
            if extent == CONTEXT
            else None
        )
        split = st.checkbox(
            "Découper en fenêtres",
            help="Fenêtres calées sur le candidat, annotées une à une sur la même page.",
        )
        split_s = st.number_input(
            "Longueur des fenêtres (s)", 0.5, 60.0, 3.0, 0.5, disabled=not split
        )
        gain_db = st.slider("Volume d'écoute (dB, n'agit que sur l'écoute)", 0, 30, 0, 3)
        band_khz = st.slider(
            "Bande d'écoute (kHz)",
            0.1,
            12.0,
            band_default,
            0.1,
            help="Position et largeur de la bande : pointillés du spectrogramme.",
        )
        band_hz = (band_khz[0] * 1000, band_khz[1] * 1000)
        band_only = st.checkbox(
            "N'écouter que la bande",
            value=False,
            help="Passe-bande entre les pointillés du spectrogramme ; n'agit que sur l'écoute.",
        )

    if mode == MAP:
        _map_page(cfg, con, encoder, config)
        return
    if chosen is None:
        st.info("Choisir une file, ou en générer une avec le mode de sélection.")
        return

    queue = load_candidates(chosen, con)
    if queue.empty:
        st.warning("File vide.")
        return
    if "cluster" in queue.columns and "encoder_id" in queue.columns:
        _cluster_panel(cfg, con, queue)
    who = (annotator.strip() or None) if calibration else None
    done = progress(con, queue, who)
    n = len(queue)
    offset_h = cfg["recorder"]["filename_utc_offset_h"]

    key = f"pos::{chosen}"
    if key not in st.session_state:  # ouverture de la file : premier candidat pas écouté
        st.session_state[key] = max(next_position(done, -1, True), 0)
    pos = min(max(int(st.session_state[key]), 0), n - 1)
    st.session_state[key] = pos
    with candidate_slot:
        _candidate_strip(queue, done, chosen, key, offset_h)
    if done.notna().all():
        st.success("File terminée : tous les candidats ont été écoutés.")

    candidate = queue.iloc[pos].to_dict()
    st.subheader(
        f"{candidate['site']} · {candidate['mic_id']} · "
        f"{local_time(candidate['start_utc'], offset_h)} (heure locale)"
    )
    score = candidate.get("score")
    heard = done.iloc[pos]
    st.caption(
        f"Candidat {pos + 1} / {n} · {Path(str(candidate['path'])).name} · fenêtre "
        f"{candidate['offset_s']:.1f}–{candidate['offset_s'] + candidate['dur_s']:.1f} s · "
        f"{candidate.get('reason') or ''}"
        + (f" · score précédent {score:.2f}" if score == score and score is not None else "")
        + (f" · déjà écouté : {NAMES.get(heard, heard)}" if heard else "")
    )

    try:
        media = _media(
            str(raw),
            str(candidate["path"]),
            float(candidate["offset_s"]),
            float(candidate["dur_s"]),
            context_s,
            channel,
            gain_db,
            band_hz if band_only else None,
        )
    except Exception as exc:  # disque débranché, fichier déplacé
        st.error(f"Lecture impossible : {exc}")
        media = None

    # Fenêtres de la page : celle du candidat, ou le découpage de l'extrait.
    cand_offset, cand_dur = float(candidate["offset_s"]), float(candidate["dur_s"])
    windows = [(cand_offset, cand_dur)]
    if split and media is not None:
        offsets = split_windows(media["t0"], media["t1"], cand_offset, float(split_s))
        windows = [(o, float(split_s)) for o in offsets] or windows
    wkey = f"win::{chosen}::{pos}::{len(windows)}"
    if wkey not in st.session_state:
        st.session_state[wkey] = min(
            range(len(windows)), key=lambda i: abs(windows[i][0] - cand_offset)
        )
    w = min(int(st.session_state[wkey]), len(windows) - 1)
    ids = [window_id_for(candidate["recording_id"], o, d) for o, d in windows]
    window_labels = latest_labels(con, ids, who)

    if media is not None:
        shown = [
            {
                "t0": o,
                "t1": o + d,
                "label": NAMES.get(window_labels.get(i), window_labels.get(i)),
                "current": j == w,
                "candidate": abs(o - cand_offset) < 0.005 and abs(d - cand_dur) < 0.005,
            }
            for j, ((o, d), i) in enumerate(zip(windows, ids, strict=True))
        ]
        _iframe(
            viewer_html(
                media["png"],
                media["t0"],
                media["t1"],
                media["fmax"],
                media["audios"],
                shown,
                band_hz,
                key=f"{candidate['path']}:{media['t0']:.2f}:{media['t1']:.2f}",
            ),
            410,
        )

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
    else:
        st.caption(
            f"Fenêtre annotée : {offset:.1f}–{offset + dur:.1f} s"
            + (
                f" · ✓ {NAMES.get(window_labels[ids[w]], window_labels[ids[w]])}"
                if ids[w] in window_labels
                else ""
            )
        )

    target = dict(candidate, offset_s=offset, dur_s=dur)
    if (offset, dur) != (cand_offset, cand_dur):
        target["score"] = None  # le score précédent est celui de la fenêtre du candidat
    if _answer_form(con, target, f"form::{chosen}::{pos}::{w}", annotator, channel):
        if w < len(windows) - 1:
            st.session_state[wkey] = w + 1
        else:
            st.session_state[key] = next_position(progress(con, queue, who), pos, skip_done)
        st.rerun()

    nav = st.columns(3)
    nav[0].button("◀ Candidat précédent", disabled=pos == 0, on_click=_goto, args=(key, pos - 1))
    nav[1].button("Candidat suivant ▶", disabled=pos >= n - 1, on_click=_goto, args=(key, pos + 1))
    following = next_position(done, pos, True)
    nav[2].button(
        "Prochain jamais écouté ⏭",
        disabled=following == pos,
        on_click=_goto,
        args=(key, following),
    )


main()
