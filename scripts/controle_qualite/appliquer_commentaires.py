"""Applique à la base ce que disent les commentaires d'écoute, et les enregistrements gardés.

- Drapeaux posés à l'écoute (`qc.apply_annotation_flags`) recalculés pour toute la base : un
  commentaire qui dit l'enregistrement inutilisable (« poubelle », « dans sac », « muffled sur
  les deux pistes », « en train d'être posé par l'équipe »… : étiquette `discard`,
  `labels.CONDITION_TAGS`) l'écarte comme un son étouffé, même s'il a été écrit avant cette
  règle (20/11/2026) ; le poste le fait à chaque envoi, ce script rattrape les anciens.
- `--garder` : `keep` sur ces enregistrements, écoutés et jugés bons (lève tout écartement).

La piste dite en commentaire (« piste 1 valable », « garder piste 1 ») n'écrit rien en base :
l'encodage la relit (`channels.comment_channel`).

Sans `--ecrire`, montre seulement ce qui changerait.

    uv run python scripts/controle_qualite/appliquer_commentaires.py [--garder ID ...] [--ecrire]
"""

from __future__ import annotations

import argparse
import sqlite3

from blanci.core.config import config_path, default_user_config, load_config
from blanci.inputs.qc import (
    annotation_flags,
    apply_annotation_flags,
    is_excluded,
    keep_flags,
    parse_flags,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--garder", nargs="*", default=[], help="recording_id à garder (keep)")
    ap.add_argument("--ecrire", action="store_true", help="écrire dans la base")
    args = ap.parse_args()
    cfg = load_config(args.config or default_user_config())
    db = config_path(cfg, "db")
    uri = f"file:{db.as_posix()}" + ("" if args.ecrire else "?mode=ro")
    con = sqlite3.connect(uri, uri=True)
    before = dict(con.execute("SELECT recording_id, qc_flags FROM recordings"))
    found = annotation_flags(con)
    changed = []
    for rid, qc in before.items():
        flags = parse_flags(qc)
        old = sorted(flags.get("annotated", [])) if "annotated" in flags else None
        new = found.get(rid)
        if old != new:
            after = {k: v for k, v in flags.items() if k != "annotated"}
            if new is not None:
                after["annotated"] = new
            if rid in args.garder:
                after["keep"] = True
            changed.append((rid, old, new, is_excluded(flags), is_excluded(after)))
    path = dict(con.execute("SELECT recording_id, path FROM recordings"))
    print(f"drapeaux d'écoute qui changent : {len(changed)} enregistrements")
    for rid, old, new, was, now in changed:
        move = "écarté" if now and not was else "repris" if was and not now else "inchangé"
        print(f"  {rid} {path[rid]} : {old} → {new} ({move})")
    for rid in args.garder:
        if rid not in before:
            raise SystemExit(f"enregistrement absent de la base : {rid}")
        state = "écarté" if is_excluded(before[rid]) else "gardé"
        print(f"  keep : {rid} {path[rid]} (aujourd'hui {state})")
    if not args.ecrire:
        print("rien n'a été écrit (--ecrire pour écrire)")
        return
    counts = apply_annotation_flags(con)
    kept = keep_flags(con, args.garder)
    print(f"écrit : {counts} ; keep posé sur {kept} enregistrements")


if __name__ == "__main__":
    main()
