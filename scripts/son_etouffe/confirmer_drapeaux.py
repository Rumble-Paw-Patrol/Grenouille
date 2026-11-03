"""Confirme à l'écoute les drapeaux des files `autres_drapeaux`, `muffled_douteux` et
`muffled_perdus` : pose `confirmed_<drapeau>` dans `qc_flags` (qui écarte du corpus, comme le
drapeau calculé, mais que ni `blanci flag` ni `blanci qc` ne réécrivent).

- `autres_drapeaux` : `confirmed_duration_off` et/ou `confirmed_off_campaign`, selon ce que
  l'enregistrement porte ;
- `muffled_douteux`, `muffled_perdus` : `confirmed_muffled`.

Sauf `--garder` (enregistrements écoutés et jugés bons, par `recording_id`) et les enregistrements
où A. blanci est entendu (jamais écartés). Sans `--ecrire`, ne fait que compter.
"""

from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from blanci.core.config import config_path, default_user_config, load_config
from blanci.inputs.qc import confirm_flags, flag_raised, parse_flags, positive_recordings

QUEUES = {
    "autres_drapeaux": ("duration_off", "off_campaign"),
    "muffled_douteux": ("muffled",),
    "muffled_perdus": ("muffled",),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--garder", nargs="*", default=[], help="recording_id à ne pas confirmer")
    ap.add_argument("--ecrire", action="store_true", help="écrire dans la base")
    args = ap.parse_args()
    cfg = load_config(args.config or default_user_config())
    queues = config_path(cfg, "queues")
    con = sqlite3.connect(config_path(cfg, "db"))
    protected = positive_recordings(con)
    for name, flags in QUEUES.items():
        ids = pd.read_csv(queues / name / "candidats.csv")["recording_id"].drop_duplicates()
        kept = ids[ids.isin(args.garder) | ids.isin(protected)]
        todo = ids.drop(kept.index)
        n = 0
        for rid in todo:
            qc = con.execute("SELECT qc_flags FROM recordings WHERE recording_id = ?", (rid,))
            have = parse_flags(qc.fetchone()[0])
            # Les files muffled n'ont pas toujours `muffled` levé (douteux : une piste, ou noté
            # à l'écoute) ; autres_drapeaux : ce que l'enregistrement porte.
            mine = [f for f in flags if name.startswith("muffled") or flag_raised(have, f)]
            n += bool(mine)
            if args.ecrire and mine:
                confirm_flags(con, [rid], mine)
        print(f"{name} : {len(ids)} candidats, {len(kept)} gardés, {n} confirmés")
        for rid in kept:
            print("   gardé :", rid, "(A. blanci entendu)" if rid in protected else "")
    con.close()


if __name__ == "__main__":
    main()
