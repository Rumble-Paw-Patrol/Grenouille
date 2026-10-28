"""Paquet d'écoute à envoyer à un encadrant qui n'a ni GitHub ni les disques (DECISIONS n° 205,
206) : une page web autonome, rien à installer.

    <nom>/
      Ouvrir le poste.html          double-clic : le poste s'ouvre dans le navigateur
      LISEZ-MOI.txt                 la marche à suivre, pour l'encadrant
      donnees/
        visualiseur.html            le visualiseur du poste (`viewer._page`), tel quel
        paquet.js                   la file, nos intervalles, les listes du poste
        audio/NNN.js                l'audio d'un extrait, une piste par micro, en FLAC

La page (`annotation/web/poste.html`) parle au visualiseur comme Streamlit. L'audio et les
données sont des scripts (`window.AUDIO`, `window.PAQUET`) : une page ouverte depuis le disque
ne peut pas lire un fichier par fetch, mais charge un script. L'audio est en data: URL, que le
visualiseur lit comme une adresse ordinaire.

Les réponses reviennent dans les CSV de `service.export_annotations` (mêmes colonnes, avec
`recording_id`), relus par `blanci import-labels`.
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import sqlite3
from pathlib import Path

import pandas as pd
import soundfile as sf

from blanci.core.locate import locate

PAGE = Path(__file__).with_name("web") / "poste.html"

README = """Paquet d'écoute « {name} », envoyé par {author}
{line}

{message}

1. Dézipper le paquet s'il ne s'est pas dézippé automatiquement, puis double-cliquer sur
   « Ouvrir le poste.html ». Il s'ouvre dans le navigateur : rien à installer et pas besoin
   d'internet. Chrome ou Edge de préférence (ils écrivent vos réponses directement dans un
   dossier).
2. En haut à gauche, « Annotateur » : s'identifier (ou « Ajouter quelqu'un… » si votre nom
   n'est pas dans la liste).
3. Il y a {n} extraits à écouter. Les intervalles tracés par {author} sont en pointillés, à
   son nom, en dessous du spectrogramme, et déjà repris comme point de départ de votre
   réponse. Pour chaque extrait :
   - un avis : le champ « Commentaire » ;
   - si vous êtes d'accord : cliquez sur « Envoyer l'extrait », sans rien toucher ;
   - sinon, tracer ses propres intervalles (clic gauche + glisser), changer l'étiquette, en
     ajouter une (bouton +, au-dessus du spectrogramme) ;
   - cliquer sur « Envoyer l'extrait » enregistre votre choix et passe à l'extrait suivant.
4. Au premier envoi, Chrome et Edge demandent le dossier où écrire vos réponses : choisir
   par exemple ce dossier-ci. Elles y sont réécrites après chaque extrait. Avec un autre
   navigateur, vos annotations et commentaires ne sont pas sauvegardés automatiquement :
   cliquez sur « Télécharger mes réponses » à la fin.
5. À la fin, la page dit que c'est terminé et où est votre fichier (votre nom suivi de
   « _reponses.json ») : l'envoyer par mail à {author}.
"""


def extracts_with_label(
    con: sqlite3.Connection, queue: pd.DataFrame, label: str, annotator: list[str] | None = None
) -> pd.DataFrame:
    """Candidats de `queue` dont le dernier envoi (de `annotator`, ou de n'importe qui) sur
    l'extrait [offset_s, offset_s + dur_s] a au moins un intervalle `label`."""
    from blanci.annotation.workbench import extract_intervals

    keep = []
    for row in queue.itertuples(index=False):
        start, end = float(row.offset_s), float(row.offset_s) + float(row.dur_s)
        saved = extract_intervals(con, row.recording_id, start, end, annotator)
        keep.append(any(i[2] == label for ivs in saved.values() for i in ivs))
    return queue[keep].reset_index(drop=True)


def _flac(wav, sr: int) -> str:
    """Une piste en FLAC, en data: URL."""
    out = io.BytesIO()
    sf.write(out, wav, sr, format="FLAC", subtype="PCM_16")
    return "data:audio/flac;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def _tracks(path: Path, start: float, dur: float) -> list[str]:
    """Les deux micros de l'extrait [start, start + dur] (le même deux fois pour un fichier
    mono, comme le poste)."""
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        f.seek(round(start * sr))
        wav = f.read(round(dur * sr), dtype="int16", always_2d=True)
    return [_flac(wav[:, min(c, wav.shape[1] - 1)], sr) for c in (0, 1)]


def build_package(
    con: sqlite3.Connection,
    cfg: dict,
    queue: pd.DataFrame,
    name: str,
    author: str,
    out_root: Path,
    message: str = "",
) -> Path:
    """Écrit le paquet `out_root/<name>/` (voir le module) et `out_root/<name>.zip` ; renvoie
    le chemin du .zip. `queue` : recording_id, offset_s, dur_s (et source, reason). Refuse un
    paquet qui existe déjà et un enregistrement introuvable sur les disques branchés."""
    from blanci.annotation.etiquettes import read_custom
    from blanci.annotation.viewer import _page
    from blanci.annotation.workbench import (
        CHANNELS,
        INTERVAL_ANSWERS,
        extract_intervals,
        local_time,
        read_annotators,
    )
    from blanci.core.config import config_path
    from blanci.inputs.labels import QUALITIES

    folder = out_root / name
    if folder.exists():
        raise FileExistsError(f"le paquet existe déjà : {folder}")
    ids = list(dict.fromkeys(queue["recording_id"]))
    recordings = pd.read_sql_query(
        "SELECT recording_id, path, site, mic_id, start_utc FROM recordings "
        f"WHERE recording_id IN ({', '.join('?' * len(ids))})",
        con,
        params=ids,
    ).set_index("recording_id")
    renamed = cfg["audio"].get("renamed")
    sources = {
        rid: locate(str(recordings.loc[rid, "path"]), config_path(cfg, "raw"), [], renamed)
        for rid in ids
    }
    missing = [str(recordings.loc[r, "path"]) for r, s in sources.items() if s is None]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} enregistrements introuvables (disque débranché ?), dont {missing[0]}"
        )

    audio = folder / "donnees" / "audio"
    audio.mkdir(parents=True)
    offset_h = cfg["recorder"]["filename_utc_offset_h"]
    authors = ["Léonard", "L"] if author in ("Léonard", "L") else [author]
    items = []
    for i, row in enumerate(queue.itertuples(index=False), start=1):
        rid, start, dur = row.recording_id, float(row.offset_s), float(row.dur_s)
        rec = recordings.loc[rid]
        key = f"{rid}:{start:.2f}:{start + dur:.2f}"
        script = f"{i:03d}.js"
        tracks = json.dumps(_tracks(sources[rid], start, dur))
        (audio / script).write_text(
            f"window.AUDIO = window.AUDIO || {{}};\nwindow.AUDIO[{json.dumps(key)}] = {tracks};\n",
            encoding="utf-8",
        )
        others = extract_intervals(con, rid, start, start + dur)
        mine = extract_intervals(con, rid, start, start + dur, authors)
        items.append(
            {
                "key": key,
                "audio": script,
                "recording_id": rid,
                "path": Path(str(rec["path"]).replace("\\", "/")).name,
                "site": rec["site"],
                "mic_id": rec["mic_id"],
                "start_utc": rec["start_utc"],
                "local_time": local_time(rec["start_utc"], offset_h),
                "offset_s": start,
                "dur_s": dur,
                "source": getattr(row, "source", None) or "plan",
                "reason": getattr(row, "reason", None) or "",
                # point de départ de l'encadrant : envoyer sans rien toucher = être d'accord
                "proposal": [list(iv) for iv in next((v for v in mine.values() if v), [])],
                "others": {
                    (who or "(sans nom)"): [list(iv) for iv in ivs] for who, ivs in others.items()
                },
            }
        )
    package = {
        "nom": name,
        "de": author,
        "message": message,
        "annotateurs": read_annotators(),
        "labels": [list(a) for a in INTERVAL_ANSWERS],
        "qualities": list(QUALITIES),
        "channels": [CHANNELS[c] for c in sorted(CHANNELS)],
        "channel": int(cfg["audio"]["channel"] == 1),
        "band": [float(f) for f in cfg["signal"]["band_hz"]],
        "fview": 10_000.0,
        "custom": read_custom(),
        "items": items,
    }
    (folder / "donnees" / "paquet.js").write_text(
        "window.PAQUET = " + json.dumps(package, ensure_ascii=False, indent=1) + ";\n",
        encoding="utf-8",
    )
    (folder / "donnees" / "visualiseur.html").write_text(_page(), encoding="utf-8")
    shutil.copyfile(PAGE, folder / "Ouvrir le poste.html")
    (folder / "LISEZ-MOI.txt").write_text(
        README.format(
            name=name,
            author=author,
            line="=" * (len(name) + len(author) + 30),
            message=message or "Merci pour votre écoute !",
            n=len(items),
        ),
        encoding="utf-8",
    )
    return Path(shutil.make_archive(str(folder), "zip", out_root, name))
