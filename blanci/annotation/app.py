"""Poste d'annotation Streamlit (§5, M2, DECISIONS n° 100) : outil de travail, pas le livrable.

    uv sync --group app
    uv run blanci annotate   (les enregistrements sont cherchés sur les disques branchés)

- **Mode de sélection** (panneau de gauche) : une file déjà écrite, ou une nouvelle file tirée
  sur place par n'importe quelle méthode de l'outil de sélection (`blanci/annotation/selection.py` :
  60-20-20 à proportions réglables, similarité, couverture, groupes, audit, hasard, negative
  mining, phénologie, suspects, congénères), ou la **carte des embeddings** (YAPAT fait
  maison) : on entoure une zone de points, on l'écoute.
- **Navigation** : sous la file, la liste de ses candidats (numérotés dans la file entière,
  ✓ et label pour ceux déjà écoutés) ; on peut revenir sur n'importe lequel et le
  réécouter, une nouvelle réponse s'ajoute à l'ancienne (correction, jamais écrasée).
  « Sauter les candidats déjà écoutés » : ◀ ▶ et l'envoi d'une réponse vont au précédent ou
  au prochain jamais écouté, sinon au voisin dans la file.
- **Écoute** : l'enregistrement entier sur un spectrogramme zoomable (`viewer.py` : molette,
  glisser, barre de lecture qui suit l'un ou l'autre micro), volume et bande d'écoute
  réglable (position et largeur) ; rien de cela ne touche l'audio d'origine.
- **Multi-classe** (décoché par défaut) : on ne note qu'A. blanci et ses faux amis (en
  intervalles, comme elle) ; coché, on dit aussi les autres classes entendues.
- **Découpage** : l'extrait se découpe en fenêtres de longueur choisie, calées sur le
  candidat ; on les annote une à une sans quitter l'enregistrement.
- **Qualité** : en intervalles, chaque intervalle a la sienne (A, B, C : barre au-dessus du
  spectrogramme) ; en fenêtres, une qualité pour la fenêtre.
- **Réponse** : une ou plusieurs classes, qualité, espèce, commentaire, puis « Envoyer ▶ »
  (ou Entrée dans un champ) : le label est ajouté (jamais écrasé) et la fenêtre suivante
  s'affiche (fenêtre suivante du découpage, sinon candidat suivant).
- **Groupes** : une file tirée par groupes montre, groupe par groupe, ce qui a été entendu ; un
  groupe homogène s'étiquette en entier d'un clic (source « bulk »).
- **Habillage** : celui du tableau de bord (`style.py`, bandeau dans `bandeau.py`), clair ou
  sombre selon le système ; les thèmes de Streamlit sont passés par `blanci annotate`, la
  feuille de style est injectée ici.
  Le mode d'emploi est en tête du panneau de gauche et s'ouvre en haut de la page.

Toute la logique est dans `workbench.py` et `selection.py` ; ce fichier ne fait qu'afficher.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from blanci.annotation.bandeau import bandeau
from blanci.annotation.style import BRAND, CHART_COLORS, CSS, LABEL_COLORS, UNHEARD_COLOR
from blanci.annotation.viewer import HELP, asset, viewer, viewer_args
from blanci.annotation.workbench import (
    ANSWERS,
    INTERVAL_ANSWERS,
    latest_labels,
    load_candidates,
    local_time,
    next_position,
    ordered_classes,
    progress,
    read_clip,
    save_answer,
    save_span,
    span_intervals,
    split_windows,
    wav_bytes,
)
from blanci.core.config import config_path, default_user_config, load_config
from blanci.core.db import connect, window_id_for
from blanci.core.locate import locate, remember_root, remembered_roots
from blanci.inputs.labels import QUALITIES

CHANNELS = {0: "micro 1 (gain 6 dB)", 1: "micro 2 (gain 18 dB)"}
NAMES = dict(ANSWERS) | {"edge": "bord d'un intervalle (écarté)"}
# Classes de l'extrait hors A. blanci, dont la présence se dit par les intervalles.
OTHER_ANSWERS = tuple(a for a in ANSWERS if a not in INTERVAL_ANSWERS)
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
    d'exécution entre deux clics, et une connexion SQLite ne se partage pas entre fils."""
    cfg = load_config(config)
    return cfg, connect(config_path(cfg, "db"))


def _queues(reports: Path) -> list[Path]:
    """Files de candidats : un dossier par file (`files/<nom>/candidats.csv`), et les anciens
    CSV posés à plat dans `paths.reports`."""
    patterns = ("files/*/candidats.csv", "candidats_*.csv", "queue_*.csv", "search_*.csv")
    found = {p for pattern in patterns for p in reports.glob(pattern)}
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _queue_name(path: Path) -> str:
    return path.parent.name if path.name == "candidats.csv" else path.name


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
        chosen = st.selectbox("File de candidats", queues, index=index, format_func=_queue_name)
        readme = chosen.parent / "LISEZMOI.md"
        if chosen.name == "candidats.csv" and readme.is_file():
            with st.expander("D'où vient cette file ?"):
                st.markdown(readme.read_text(encoding="utf-8"))
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
def _media(full: str, offset_s: float = 0.0, dur_s: float = math.inf) -> dict:
    """Enregistrement entier (ou l'extrait [offset_s, offset_s + dur_s]), des deux micros,
    rangé à côté du visualiseur et gardé en cache : changer de fenêtre ou tracer un
    intervalle ne relit pas le disque. Le spectrogramme, le volume et la bande d'écoute se
    font dans le navigateur."""
    full = Path(full)
    clips = {c: read_clip(full.parent, full.name, offset_s, dur_s, 0.0, c) for c in CHANNELS}
    wav, sr, start = clips[0]
    return {
        "t0": start,
        "t1": start + len(wav) / sr,
        "audios": [
            (CHANNELS[c], asset(wav_bytes(w, rate), ".wav"), s0)
            for c, (w, rate, s0) in clips.items()
        ],
    }


# Streamlit < 1.37 : sans fragment, toute la page se réaffiche.
_fragment = getattr(st, "fragment", lambda f: f)


@_fragment
def _viewer_fragment(args: dict, extract: str, ikey: str, ckey: str, interval_mode: bool):
    """Visualiseur et liste des intervalles : tracer, ajuster ou effacer un intervalle ne
    réaffiche que ce fragment, pas toute la page."""
    value = viewer(args)
    if value and value.get("key") == extract:
        st.session_state[ikey] = value["intervals"]
        st.session_state[ckey] = int(value.get("channel", 0))
    if interval_mode:
        st.caption(_interval_caption(st.session_state[ikey]))


def _classes(form_key: str, answers) -> list[str]:
    columns = st.columns(5)
    return [
        label
        for i, (label, name) in enumerate(answers)
        if columns[i % 5].checkbox(name, key=f"{form_key}::{label}")
    ]


def _details(
    multiclass: bool, with_quality: bool = True
) -> tuple[str | None, str | None, str | None]:
    """Qualité (sauf `with_quality` faux : en intervalles, chacun a la sienne, au-dessus du
    spectrogramme), espèce, commentaire."""
    quality = (
        st.radio("Qualité (si A. blanci)", ["—", *QUALITIES], horizontal=True)
        if with_quality
        else "—"
    )
    species = st.text_input(
        "Espèce entendue (faux ami, congénère…)" if multiclass else "Espèce du faux ami (si connue)"
    )
    comment = st.text_input("Commentaire (conditions, chant lointain, pluie…)")
    return (
        None if quality == "—" else quality,
        species.strip() or None,
        comment.strip() or None,
    )


def _span_form(con, candidate, t0, t1, intervals, form_key, annotator, channel, multiclass) -> bool:
    """Annotation par intervalles : les intervalles viennent du spectrogramme, le formulaire
    dit le reste de l'extrait (les autres classes seulement en multi-classe). Vrai si
    l'extrait a été enregistré."""
    with st.form(key=form_key, clear_on_submit=True):
        ticked = []
        if multiclass:
            st.markdown(
                "**Autres classes entendues dans l'extrait** (A. blanci : par les "
                "intervalles ; aucune cochée : rien)"
            )
            ticked = _classes(form_key, OTHER_ANSWERS)
        quality, species, comment = _details(multiclass, with_quality=False)
        sent = st.form_submit_button(
            "Envoyer l'extrait ▶", type="primary", use_container_width=True
        )
    if not sent:
        return False
    if not annotator.strip():
        st.error("Indiquer l'annotateur dans le panneau de gauche.")
        return False
    try:
        save_span(
            con,
            candidate,
            t0,
            t1,
            [tuple(i) for i in intervals],
            ticked,
            annotator.strip(),
            quality=quality,
            comment=comment,
            channel=channel,
            species=species,
            multiclass=multiclass,
        )
    except ValueError as exc:
        st.error(str(exc))
        return False
    return True


def _answer_form(con, target, form_key, annotator, channel, multiclass) -> bool:
    """Annotation par fenêtres (suspendue) : classes (plusieurs possibles ; A. blanci seul
    hors multi-classe), qualité, espèce, commentaire ; vrai si un label a été enregistré."""
    with st.form(key=form_key, clear_on_submit=True):
        st.markdown("**Classes entendues** (plusieurs possibles ; aucune cochée : rien)")
        ticked = _classes(form_key, ANSWERS if multiclass else INTERVAL_ANSWERS)
        quality, species, comment = _details(multiclass)
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
        quality=quality,
        comment=comment,
        channel=channel,
        species=species,
        extra_labels=extra,
    )
    return True


def _interval_caption(intervals) -> str:
    if not intervals:
        return (
            "Aucun intervalle tracé : tout l'extrait sera négatif. Clic gauche + glisser sur le "
            "spectrogramme là où A. blanci chante (un faux ami : étiquette « faux ami » ou "
            "touche 4, avant de tracer)."
        )
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
  new MutationObserver(fix).observe(root.body, {childList: true, subtree: true});
  window.parent.setInterval(theme, 400);
  fix();
  theme();
})();
</script>"""


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
        if (Path(folder) / path).is_file():
            remember_root(config_path(cfg, "db"), Path(folder))
            st.rerun()
        st.warning(f"Pas de {path} sous {folder}.")


def main() -> None:
    st.set_page_config(page_title="Annotation blanci", page_icon="🐸", layout="wide")
    st.markdown(NO_GREY + "\n" + CSS, unsafe_allow_html=True)
    if hasattr(st, "iframe"):  # `components.html` est retiré des Streamlit récents
        st.iframe(READONLY_MENUS, height=1)
    else:
        components.html(READONLY_MENUS, height=0)
    bandeau()
    config = _config_file()
    cfg, con = _setup(config)
    _apply_pending_queue()
    raw, reports = config_path(cfg, "raw"), config_path(cfg, "reports")
    extra = _recordings_root(cfg)
    band_hz = tuple(float(f) for f in cfg["signal"]["band_hz"])
    spectro_default = int(cfg["audio"]["channel"] == 1)

    with st.sidebar:
        st.markdown(BRAND, unsafe_allow_html=True)
        # Fenêtre flottante, plus large que le panneau : le tableau des gestes y tient.
        with st.popover("❓ Mode d'emploi : souris et clavier"):
            st.markdown(HELP)
        st.header("Session")
        annotator = st.text_input("Annotateur", value=st.session_state.get("annotator", ""))
        st.session_state["annotator"] = annotator
        mode, chosen, encoder = _selection_panel(cfg, con, reports)
        candidate_slot = st.container()
        st.header("Écoute")
        skip_done = st.checkbox(
            "Sauter les candidats déjà écoutés",
            value=True,
            key="skip_done",
            help="◀ ▶ et l'envoi d'une réponse vont au précédent ou au prochain candidat jamais "
            "écouté ; décoché, au voisin dans la file. La liste des candidats donne toujours "
            "accès à tous.",
        )
        calibration = st.checkbox(
            "Masquer les annotations d'autres personnes",
            value=False,
            key="calibration",
            help="Seules vos réponses comptent comme « déjà écouté » : deux annotateurs "
            "écoutent la même file sans voir les réponses de l'autre (calibration, §5).",
        )
        multiclass = st.checkbox(
            "Multi-classe : noter aussi les autres espèces",
            value=False,
            key="multiclass",
            help="Décoché : on ne note qu'A. blanci (intervalles, qualité, commentaire). "
            "Coché : on dit aussi les autres classes entendues (oiseau, insecte, pluie…) et "
            "l'espèce.",
        )
        st.caption(
            "Étiquette et qualité des intervalles, bande d'écoute, volume, filtre dynamique, "
            "contraste et micro du spectrogramme : dans la barre au-dessus du spectrogramme."
        )
        st.header("Méthode")
        window_mode = st.checkbox(
            "Annoter par fenêtres (suspendu)",
            value=False,
            help="Ancienne méthode : un label par fenêtre de longueur fixe, sur la même page. "
            "Par défaut, on trace les intervalles où A. blanci chante (§5.6).",
        )
        split_s = st.number_input(
            "Longueur des fenêtres (s)", 0.5, 60.0, 3.0, 0.5, disabled=not window_mode
        )

    if mode == MAP:
        _map_page(cfg, con, encoder, config)
        return
    if chosen is None:
        st.info("Choisir une file, ou en générer une avec le mode de sélection.")
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
    st.subheader(  # « en-ecoute » : la feuille de style met le point rouge devant
        f"{candidate['site']} · {candidate['mic_id']} · "
        f"{local_time(candidate['start_utc'], offset_h)} (heure locale)",
        anchor="en-ecoute",
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

    media = None
    full = locate(str(candidate["path"]), raw, extra)
    if full is None:  # disque débranché, fichier déplacé
        _missing_recording(cfg, str(candidate["path"]))
    else:
        try:
            media = _media(str(full))
        except Exception as exc:  # fichier ouvert ailleurs, abîmé
            st.error(f"Lecture impossible : {full} ({exc}). Fichier ouvert ailleurs ou abîmé ?")

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
    window_labels = latest_labels(con, ids, who) if window_mode else {}

    intervals: list = []
    channel = spectro_default
    # Extrait écouté : l'enregistrement entier (n° 182), sauf pour une file tirée par plan,
    # dont l'extrait est la fenêtre du candidat (30 s du lot 1, n° 196). L'extrait enregistré
    # est ce qu'on a écouté : tout ce qui y est sans intervalle devient négatif.
    t0, t1 = (media["t0"], media["t1"]) if media is not None else (0.0, 0.0)
    planned = candidate.get("source") == "plan"
    if media is not None and planned and not window_mode and cand_dur < t1 - t0 - 0.05:
        media = _media(str(full), cand_offset, cand_dur)  # le lecteur ne joue que l'extrait
        t0, t1 = media["t0"], media["t1"]
    if media is not None:
        extract = f"{candidate['path']}:{t0:.2f}:{t1:.2f}"
        ikey = f"iv::{extract}"
        if ikey not in st.session_state:
            saved = span_intervals(con, candidate["recording_id"], t0, t1, who)
            st.session_state[ikey] = [list(i) for i in saved or []]
        shown = [
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
                shown,
                band_hz,
                key=extract,
                labels=list(INTERVAL_ANSWERS),
                intervals=st.session_state[ikey],
                interval_mode=not window_mode,
                channel=spectro_default,
            ),
            extract,
            ikey,
            ckey,
            not window_mode,
        )
        intervals = st.session_state[ikey]
        channel = st.session_state[ckey]

    if not window_mode:
        form_key = f"span::{chosen}::{pos}"
        if media is not None and _span_form(
            con,
            candidate,
            t0,
            t1,
            intervals,
            form_key,
            annotator,
            channel,
            multiclass,
        ):
            st.session_state[key] = next_position(progress(con, queue, who), pos, skip_done)
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
        if _answer_form(con, target, f"form::{chosen}::{pos}::{w}", annotator, channel, multiclass):
            if w < len(windows) - 1:
                st.session_state[wkey] = w + 1
            else:
                st.session_state[key] = next_position(progress(con, queue, who), pos, skip_done)
            st.rerun()

    # ◀ ▶ : voisin dans la file, ou précédent / prochain jamais écouté (sans faire le tour).
    before = next_position(done, pos, skip_done, step=-1, wrap=False)
    after = next_position(done, pos, skip_done, wrap=False)
    nav = st.columns(2)
    nav[0].button(
        "◀ Candidat précédent", disabled=before == pos, on_click=_goto, args=(key, before)
    )
    nav[1].button("Candidat suivant ▶", disabled=after == pos, on_click=_goto, args=(key, after))


main()
