"""Poste d'annotation Streamlit (§5, M2, DECISIONS n° 100) : outil de travail, pas le livrable.

    uv sync --group app
    uv run blanci --config config/local.yaml annotate

- **Mode de sélection** (panneau de gauche) : une file déjà écrite, ou une nouvelle file tirée
  sur place par n'importe quelle méthode de l'outil de sélection (`blanci/selection.py` :
  60-20-20 à proportions réglables, similarité, couverture, groupes, audit, hasard, negative
  mining, phénologie, suspects, congénères), ou la **carte des embeddings** (YAPAT fait
  maison) : on entoure une zone de points, on l'écoute.
- **Réponse** : classe, qualité, espèce, commentaire, puis « Envoyer ▶ » (ou Entrée dans un
  champ) : le label est ajouté (jamais écrasé) et la fenêtre suivante s'affiche.
- **Groupes** : une file tirée par groupes montre, groupe par groupe, ce qui a été entendu ; un
  groupe homogène s'étiquette en entier d'un clic (source « bulk »).

Toute la logique est dans `workbench.py` et `selection.py` ; ce fichier ne fait qu'afficher.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from blanci.config import config_path, load_config  # noqa: E402
from blanci.db import connect  # noqa: E402
from blanci.labels import QUALITIES  # noqa: E402
from blanci.workbench import (  # noqa: E402
    ANSWERS,
    clip_spectrogram,
    load_candidates,
    local_time,
    progress,
    read_clip,
    save_answer,
    wav_bytes,
)

CHANNELS = {0: "micro 1 (gain 6 dB)", 1: "micro 2 (gain 18 dB)"}
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
    "congeners": "Congénères (Perch)",
}
NEEDS_ENCODER = (
    "active",
    "similarity",
    "coverage",
    MAP,
    "cluster",
    "negative_mining",
    "suspects",
    "congeners",
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


def _figure(wav, sr, start_s, offset_s, dur_s, band_hz):
    freqs, times, db = clip_spectrogram(wav, sr)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    vmax = float(db.max())
    ax.pcolormesh(
        times + start_s, freqs / 1000, db, shading="auto", cmap="magma", vmin=vmax - 60, vmax=vmax
    )
    ax.axvspan(offset_s, offset_s + dur_s, color="cyan", alpha=0.12)
    for edge in (offset_s, offset_s + dur_s):
        ax.axvline(edge, color="cyan", lw=1)
    for f in band_hz:
        ax.axhline(f / 1000, color="white", lw=0.6, ls="--", alpha=0.6)
    ax.set_xlabel("temps dans l'enregistrement (s)")
    ax.set_ylabel("kHz")
    fig.tight_layout()
    return fig


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
    from blanci.selection import select_candidates, write_queue

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
    from blanci.selection import embedding_map

    cfg, con = _setup(Path(config) if config else None)
    return embedding_map(con, cfg, encoder, n=n, method=method)


def _map_page(cfg, con, encoder: str | None, config: Path | None) -> None:
    import altair as alt

    from blanci.selection import map_selection, write_queue

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
    from blanci.selection import cluster_status, label_cluster

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


def _answer_form(con, candidate, chosen, pos, annotator, channel, skip_done, key) -> None:
    labels = [label for label, _ in ANSWERS]
    names = dict(ANSWERS)
    with st.form(key=f"form::{chosen}::{pos}", clear_on_submit=True):
        columns = st.columns([2, 1])
        label = columns[0].radio(
            "Classe",
            labels,
            format_func=names.get,
            horizontal=True,
            index=labels.index("background"),
        )
        quality = columns[1].radio("Qualité (si A. blanci)", ["—", *QUALITIES], horizontal=True)
        species = st.text_input("Espèce entendue (faux ami, congénère…)")
        comment = st.text_input("Commentaire (conditions, chant lointain, pluie…)")
        sent = st.form_submit_button("Envoyer ▶", type="primary", use_container_width=True)
    if sent:
        if not annotator.strip():
            st.error("Indiquer l'annotateur dans le panneau de gauche.")
            return
        save_answer(
            con,
            candidate,
            label,
            annotator.strip(),
            quality=None if quality == "—" else quality,
            comment=comment.strip() or None,
            channel=channel,
            species=species.strip() or None,
        )
        if not skip_done:
            st.session_state[key] = pos + 1
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Annotation blanci", layout="wide")
    config = _config_file()
    cfg, con = _setup(config)
    _apply_pending_queue()
    raw, reports = config_path(cfg, "raw"), config_path(cfg, "reports")

    with st.sidebar:
        st.header("Session")
        annotator = st.text_input("Annotateur", value=st.session_state.get("annotator", ""))
        st.session_state["annotator"] = annotator
        mode, chosen, encoder = _selection_panel(cfg, con, reports)
        st.header("Écoute")
        skip_done = st.checkbox("Masquer les candidats déjà écoutés", value=True)
        calibration = st.checkbox(
            "Calibration : ne masquer que mes réponses",
            value=False,
            help="Deux annotateurs écoutent la même file sans voir les réponses de l'autre (§5).",
        )
        channel = st.radio(
            "Spectrogramme",
            list(CHANNELS),
            format_func=CHANNELS.get,
            index=int(cfg["audio"]["channel"] == 1),
        )
        context_s = st.slider("Contexte autour de la fenêtre (s)", 0.0, 10.0, 3.0, 0.5)
        gain_db = st.slider("Volume d'écoute (dB, n'agit que sur l'écoute)", 0, 30, 0, 3)

    if mode == MAP:
        _map_page(cfg, con, encoder, config)
        return
    if chosen is None:
        st.info("Choisir une file, ou en générer une avec le mode de sélection.")
        return

    queue = load_candidates(chosen, con)
    if "cluster" in queue.columns and "encoder_id" in queue.columns:
        _cluster_panel(cfg, con, queue)
    done = progress(con, queue, (annotator.strip() or None) if calibration else None)
    st.sidebar.metric("Écoutés", f"{int(done.notna().sum())} / {len(queue)}")
    todo = queue[done.isna()] if skip_done else queue
    if todo.empty:
        st.success("File terminée.")
        return

    key = f"pos::{chosen}"
    pos = min(st.session_state.get(key, 0), len(todo) - 1)
    candidate = todo.iloc[pos].to_dict()

    offset_h = cfg["recorder"]["filename_utc_offset_h"]
    st.subheader(
        f"{candidate['site']} · {candidate['mic_id']} · "
        f"{local_time(candidate['start_utc'], offset_h)} (heure locale)"
    )
    score = candidate.get("score")
    st.caption(
        f"Candidat {pos + 1} / {len(todo)} · {Path(str(candidate['path'])).name} · fenêtre "
        f"{candidate['offset_s']:.1f}–{candidate['offset_s'] + candidate['dur_s']:.1f} s · "
        f"{candidate.get('reason') or ''}"
        + (f" · score précédent {score:.2f}" if score == score and score is not None else "")
        + (
            f" · déjà écouté : {done.loc[todo.index[pos]]}"
            if not skip_done and done.loc[todo.index[pos]]
            else ""
        )
    )

    try:
        clips = {
            c: read_clip(
                raw, candidate["path"], candidate["offset_s"], candidate["dur_s"], context_s, c
            )
            for c in CHANNELS
        }
    except Exception as exc:  # disque débranché, fichier déplacé
        st.error(f"Lecture impossible : {exc}")
        return
    wav, sr, start = clips[channel]
    st.pyplot(
        _figure(wav, sr, start, candidate["offset_s"], candidate["dur_s"], cfg["signal"]["band_hz"])
    )
    players = st.columns(2)
    for column, (c, (w, rate, _)) in zip(players, clips.items(), strict=True):
        column.markdown(f"**{CHANNELS[c]}**")
        column.audio(wav_bytes(w, rate, gain_db), format="audio/wav")

    _answer_form(con, candidate, chosen, pos, annotator, channel, skip_done, key)

    nav = st.columns(2)
    if nav[0].button("◀ Précédent", disabled=pos == 0):
        st.session_state[key] = pos - 1
        st.rerun()
    if nav[1].button("Passer ▶", disabled=pos >= len(todo) - 1):
        st.session_state[key] = pos + 1
        st.rerun()


main()
