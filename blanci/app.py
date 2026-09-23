"""Poste d'annotation Streamlit (§5, M2) : outil de travail, pas le livrable.

    uv sync --group app
    uv run blanci --config config/local.yaml annotate

Toute la logique est dans `blanci/workbench.py` ; ce fichier ne fait qu'afficher.
"""

from __future__ import annotations

import argparse
import os
import sys
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


def main() -> None:
    st.set_page_config(page_title="Annotation blanci", layout="wide")
    cfg, con = _setup(_config_file())
    raw, reports = config_path(cfg, "raw"), config_path(cfg, "reports")

    with st.sidebar:
        st.header("Session")
        annotator = st.text_input("Annotateur", value=st.session_state.get("annotator", ""))
        st.session_state["annotator"] = annotator
        queues = _queues(reports)
        if not queues:
            st.warning(f"Aucune file dans {reports}. Lancer `blanci candidates`.")
            return
        chosen = st.selectbox("File de candidats", queues, format_func=lambda p: p.name)
        skip_done = st.checkbox("Masquer les candidats déjà écoutés", value=True)
        channel = st.radio(
            "Spectrogramme",
            list(CHANNELS),
            format_func=CHANNELS.get,
            index=int(cfg["audio"]["channel"] == 1),
        )
        context_s = st.slider("Contexte autour de la fenêtre (s)", 0.0, 10.0, 3.0, 0.5)
        gain_db = st.slider("Volume d'écoute (dB, n'agit que sur l'écoute)", 0, 30, 0, 3)

    queue = load_candidates(chosen, con)
    done = progress(con, queue)
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

    quality = st.radio(
        "Qualité (si A. blanci)", ["—", *QUALITIES], horizontal=True, key=f"q::{chosen}::{pos}"
    )
    comment = st.text_input("Commentaire (espèce entendue, conditions…)", key=f"c::{chosen}::{pos}")

    buttons = st.columns(len(ANSWERS) // 2)
    for i, (label, text) in enumerate(ANSWERS):
        if buttons[i % len(buttons)].button(text, key=f"b::{label}", use_container_width=True):
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
            )
            if not skip_done:
                st.session_state[key] = pos + 1
            st.rerun()

    nav = st.columns(2)
    if nav[0].button("◀ Précédent", disabled=pos == 0):
        st.session_state[key] = pos - 1
        st.rerun()
    if nav[1].button("Passer ▶", disabled=pos >= len(todo) - 1):
        st.session_state[key] = pos + 1
        st.rerun()


main()
