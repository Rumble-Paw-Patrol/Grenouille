"""Prototype de vérification (§5, M2) : outil de travail, pas livrable.

File de fenêtres à trancher (schéma de labels §5), spectrogramme 2–8 kHz, boutons, raccourcis.
Chaque décision est écrite comme un label `source="active"` (ou "audit" via --audit),
en ajout seul.

Lancement : `uv run --group app streamlit run app/streamlit_app.py -- --config config/default.yaml`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blanci.audio import load_audio  # noqa: E402
from blanci.config import config_path, load_config  # noqa: E402
from blanci.db import connect, utc_now  # noqa: E402
from blanci.labels import LABELS, POSITIVE_LABELS, QUALITIES  # noqa: E402

st.set_page_config(page_title="blanci — vérification", layout="wide")


def spectrogram(wav: np.ndarray, sr: int, band_hz: tuple[int, int] = (2000, 8000)):
    from scipy.signal import spectrogram as scipy_spectrogram

    f, t, sxx = scipy_spectrogram(wav, sr, nperseg=1024, noverlap=768)
    keep = (f >= band_hz[0]) & (f <= band_hz[1])
    return t, f[keep], 10 * np.log10(sxx[keep] + 1e-12)


@st.cache_resource
def get_connection(config_path_str: str | None):
    cfg = load_config(Path(config_path_str) if config_path_str else None)
    return connect(config_path(cfg, "db")), cfg


def next_window(con) -> dict | None:
    """Prochaine fenêtre non tranchée de la file (source active/random, la plus récente)."""
    row = con.execute(
        """SELECT w.window_id, w.recording_id, w.offset_s, w.dur_s, r.path, r.duration_s
           FROM windows w JOIN recordings r USING (recording_id)
           WHERE w.window_id NOT IN (SELECT window_id FROM labels)
           ORDER BY RANDOM() LIMIT 1"""
    ).fetchone()
    return dict(row) if row else None


def record_label(con, window_id: str, label: str, quality: str | None, annotator: str, source: str) -> None:
    con.execute(
        "INSERT INTO labels (window_id, label, quality, species, conditions, annotator, source, created_at) "
        "VALUES (?, ?, ?, NULL, '{}', ?, ?, ?)",
        (window_id, label, quality, annotator, source, utc_now()),
    )
    con.commit()


def main() -> None:
    config_arg = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else None
    con, cfg = get_connection(config_arg)
    st.sidebar.text_input("Annotateur", key="annotator", value="expert")

    item = next_window(con)
    if item is None:
        st.success("File vide : plus aucune fenêtre à trancher.")
        return

    margin = 2.0  # secondes de contexte autour de la fenêtre, pour entendre les bords
    raw_root = Path(cfg["paths"]["raw"])
    wav, sr = load_audio(raw_root / item["path"])
    start = max(0.0, item["offset_s"] - margin)
    end = min(item["duration_s"], item["offset_s"] + item["dur_s"] + margin)
    clip = wav[round(start * sr) : round(end * sr)]

    st.subheader(f"{item['path']} — {item['offset_s']:.2f} s")
    st.audio(clip, sample_rate=sr)

    import matplotlib.pyplot as plt

    t, f, db = spectrogram(clip, sr, tuple(cfg["signal"]["band_hz"][:1] + [8000]))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.pcolormesh(t, f, db, shading="auto")
    ax.axvspan(item["offset_s"] - start, item["offset_s"] - start + item["dur_s"], color="white", alpha=0.15)
    ax.set(xlabel="s", ylabel="Hz")
    st.pyplot(fig)

    st.write("Positifs :")
    cols = st.columns(len(POSITIVE_LABELS))
    for col, label in zip(cols, POSITIVE_LABELS):
        if col.button(label, use_container_width=True):
            quality = st.session_state.get("quality", "A")
            record_label(con, item["window_id"], label, quality, st.session_state["annotator"], "active")
            st.rerun()
    st.radio("Qualité (Courtois et al. 2025)", QUALITIES, key="quality", horizontal=True)

    st.write("Négatifs / autre :")
    negatives = [label for label in LABELS if label not in POSITIVE_LABELS]
    cols = st.columns(4)
    for i, label in enumerate(negatives):
        if cols[i % 4].button(label, key=f"neg_{label}", use_container_width=True):
            record_label(con, item["window_id"], label, None, st.session_state["annotator"], "active")
            st.rerun()


if __name__ == "__main__":
    main()
