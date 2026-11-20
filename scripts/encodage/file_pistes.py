"""File d'encodage « bonnes pistes » : ce que chaque stock doit retirer puis (ré)encoder.

Trois groupes d'enregistrements ne sont pas, ou pas tous, encodés sur la bonne piste :
(a) piste 0 étouffée et piste 1 valable (`muffled_0` sans `muffled_1`, n° 203) ; (b) Kaw D
SMA14079 au micro 0 sourd ; (c) les points au gain inversé (piste 0 à 18 dB). La piste de chacun
vient de `audio.channel_rules` et du repli du son étouffé (`blanci/embedding/channels.py`).

Pour chaque stock : les enregistrements à retirer (encodés sur une autre piste, `channel_plan`),
puis la file : les enregistrements gardés par les drapeaux (`select_recordings`) qui ne sont pas
dans le stock sur la bonne piste. Durée estimée = fenêtres / débit mesuré (`RATES`).

Lecture seule : la base est ouverte en lecture seule, les stocks ne sont lus que par leur colonne
`recording_id`. Écrit sous `rapports/encodage_pistes/` (jamais committé) : `<modèle>.csv`
(recording_id, groupe, à retirer), `resume.csv`, `extraits_annotes.csv`, `LISEZMOI.md`.

Usage : uv run python scripts/encodage/file_pistes.py [--config config/local.yaml]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from blanci.core.config import LOCAL_CONFIG, PROJECT_ROOT, config_path, load_config
from blanci.embedding.channels import (
    annotation_channels,
    channel_key,
    channel_plan,
    channel_rules,
    expected_channels,
    matching_rule,
)
from blanci.embedding.embed import select_recordings
from blanci.embedding.grid import window_grid
from blanci.embedding.store import EmbeddingStore
from blanci.inputs.qc import parse_flags, positive_recordings

# Débit mesuré sur ce PC (fenêtres par seconde de durée réelle, lecture comprise, quand elle
# est connue ; sinon débit de l'encodeur seul, et la durée réelle sera un peu plus longue).
RATES = {
    "perch_v2": (20.3, "débit de l'encodeur seul, rapports/embed_perch_v2.log (12 637 enr.)"),
    "birdnet": (40.4, "durée réelle cumulée, table models : 442 084 fenêtres en 10 948 s"),
    "birdnet_v3": (10.8, "durée réelle cumulée, table models : 3 232 127 fenêtres en 298 911 s"),
    "audioprotopnet": (
        5.9,
        "durée réelle du passage du 29/10 (OpenVINO f16) : 4 747 fenêtres "
        "de 101 enr. en 13 min 30 ; encodeur seul : 7,2",
    ),
}
OTHER = "autre : absent du stock"
# Où est chaque stock : perch_v2 n'est que sur F: (data/ du dépôt, jonction vers F:\data) ;
# les autres sous paths.embeddings de config/local.yaml (C:/Users/…/embeddings_pc).
STOCKS = {
    "perch_v2": ("data/embeddings", "perch_v2-bacpipe1.3.5"),
    "birdnet": (None, "birdnet-bacpipe1.3.5"),
    "birdnet_v3": (None, "birdnet_v3-bacpipe1.3.5"),
    "audioprotopnet": (None, "audioprotopnet-bacpipe1.3.5"),
}
TEST_QUEUE = Path("annotations") / "files_ecoute" / "test_v1" / "candidats.csv"


def group_of(
    rec: dict, flags: dict, rules: list[dict], reasons: list[str], channel, listened=None
) -> str:
    """Groupe d'un enregistrement de la file : la piste écoutée ou dite en commentaire
    (enregistrement annoté, `listened`), la règle qui le touche, le repli du son étouffé, ou
    « autre » (absent du stock pour une autre raison)."""
    if listened and rec["recording_id"] in listened:
        return "piste écoutée ou commentée"
    i = matching_rule(rec, rules)
    if i is not None:
        return f"règle : {reasons[i]}"
    if flags.get("muffled_0") and not flags.get("muffled_1"):
        return "piste 0 étouffée (n° 203)"
    return OTHER


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--repo",
        type=Path,
        default=PROJECT_ROOT,
        help="dossier du dépôt (rapports/, data/, annotations/, config/local.yaml)",
    )
    ap.add_argument("--config", type=Path, help="défaut : <repo>/config/local.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config or args.repo / LOCAL_CONFIG.relative_to(PROJECT_ROOT))
    out = args.repo / "rapports" / "encodage_pistes"
    raw_rules = cfg["audio"].get("channel_rules") or []
    rules = channel_rules(raw_rules)
    reasons = [r.get("reason", f"règle {i}") for i, r in enumerate(raw_rules)]
    channel = cfg["audio"]["channel"]
    db = config_path(cfg, "db")
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    recs = pd.read_sql_query("SELECT * FROM recordings", con)
    flags = {r: parse_flags(q) for r, q in zip(recs["recording_id"], recs["qc_flags"], strict=True)}
    selected = select_recordings(con)
    listened = annotation_channels(con)  # extraits annotés : sur la piste écoutée
    print(f"base {db} : {len(recs)} enregistrements, {len(selected)} gardés par les drapeaux")
    params = {
        r["model_id"]: json.loads(r["params_json"] or "{}")
        for r in con.execute("SELECT model_id, params_json FROM models WHERE kind = 'encoder'")
    }

    out.mkdir(parents=True, exist_ok=True)
    summary, lines = [], []
    for model, (root, stock_id) in STOCKS.items():
        root = args.repo / root if root else config_path(cfg, "embeddings")
        store = EmbeddingStore(root, stock_id)
        if not store.directory.is_dir():
            print(f"{model} : stock absent ({store.directory}), sauté")
            continue
        plan = channel_plan(con, store, channel, rules)
        stale = plan[plan["stale"]]
        ok = set(plan.loc[~plan["stale"], "recording_id"])
        todo = selected[~selected["recording_id"].isin(ok)].copy()
        todo["groupe"] = [
            group_of(rec, flags[rec["recording_id"]], rules, reasons, channel, listened)
            for rec in todo.to_dict("records")
        ]
        todo["retirer"] = todo["recording_id"].isin(set(stale["recording_id"]))
        p = params.get(stock_id, {})
        window_s, hop_s = p.get("window_s", 5.0), p.get("hop_s", 2.5)
        todo["fenetres"] = [
            len(window_grid(d if d == d else 120.0, window_s, hop_s)) for d in todo["duration_s"]
        ]
        expected = expected_channels(todo, channel, rules, listened=listened)
        todo["piste"] = todo["recording_id"].map(expected)
        columns = [
            "recording_id",
            "dataset",
            "site",
            "mic_id",
            "start_utc",
            "piste",
            "groupe",
            "retirer",
            "fenetres",
        ]
        todo[columns].to_csv(out / f"{model}.csv", index=False)  # tout ce qui manque
        # Les groupes, et tout ce qui est retiré du stock (à réencoder, quel que soit son groupe).
        tracks = todo[(todo["groupe"] != OTHER) | todo["retirer"]]
        tracks[columns].to_csv(out / f"{model}_pistes.csv", index=False)
        rate, source = RATES[model]
        hours = todo["fenetres"].sum() / rate / 3600
        hours_tracks = tracks["fenetres"].sum() / rate / 3600
        for group, g in todo.groupby("groupe"):
            summary.append(
                {
                    "modele": model,
                    "groupe": group,
                    "enregistrements": len(g),
                    "dont_a_retirer": int(g["retirer"].sum()),
                    "fenetres": int(g["fenetres"].sum()),
                    "heures": round(g["fenetres"].sum() / rate / 3600, 1),
                }
            )
        lines.append(
            f"- **{model}** (`{store.directory}`) : à retirer d'abord du stock "
            f"{stale['recording_id'].nunique()} enregistrements ({int(stale['windows'].sum())} "
            f"fenêtres) ; trois groupes (`{model}_pistes.csv`) : {len(tracks)} enregistrements, "
            f"{int(tracks['fenetres'].sum())} fenêtres, ≈ {hours_tracks:.1f} h ; tout ce qui "
            f"manque au stock (`{model}.csv`) : {len(todo)} enregistrements, ≈ {hours:.1f} h ; "
            f"à {rate:g} fenêtres/s ({source})"
        )
        print(lines[-1])
        if len(stale):
            print(stale.groupby(["dataset", "site", "month"])["windows"].agg(["size", "sum"]))
    pd.DataFrame(summary).to_csv(out / "resume.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))

    # Extraits annotés : la piste écoutée contre la piste encodée, avant et après les règles.
    spans = pd.read_sql_query("SELECT span_id, recording_id, conditions FROM spans", con)
    spans = spans.merge(
        recs[["recording_id", "dataset", "site", "mic_id", "start_utc", "channels", "qc_flags"]],
        on="recording_id",
    )
    by_id = spans.drop_duplicates("recording_id")
    before = expected_channels(by_id, channel, [])
    after = expected_channels(by_id, channel, rules, listened=listened)
    spans["encodee_avant"] = spans["recording_id"].map(before)
    spans["encodee_apres"] = spans["recording_id"].map(after)
    sent = [(json.loads(c) if c else {}).get("channel_listened") for c in spans["conditions"]]
    spans["ecoutee"] = [
        None if e is None else channel_key(int(e) if str(e) in ("0", "1") else e, n)
        for e, n in zip(sent, spans["channels"], strict=True)
    ]
    spans["positif"] = spans["recording_id"].isin(positive_recordings(con))
    test_file = args.repo / TEST_QUEUE
    test_ids = set()
    if test_file.exists():
        test_ids = set(pd.read_csv(test_file, dtype=str)["recording_id"])
    spans["jeu_de_test"] = spans["recording_id"].isin(test_ids)
    heard = spans[spans["ecoutee"].notna()]
    off_before = heard[heard["ecoutee"] != heard["encodee_avant"]]
    off_after = heard[heard["ecoutee"] != heard["encodee_apres"]]
    moved = spans[spans["encodee_avant"] != spans["encodee_apres"]]
    spans.drop(columns=["conditions", "qc_flags"]).to_csv(out / "extraits_annotes.csv", index=False)
    # Les trois groupes : la piste de la règle n'est pas le canal par défaut.
    other = expected_channels(recs, channel, rules) != pd.Series(
        [channel_key(channel, n) for n in recs["channels"]], index=recs["recording_id"]
    )
    touched = set(other[other].index)
    test_touched = recs[recs["recording_id"].isin(test_ids & touched)]
    notes = [
        f"- Extraits annotés : {len(spans)}, dont {len(heard)} avec la piste écoutée.",
        f"- Écoutés sur une piste, encodés sur l'autre : {len(off_before)} avant les règles "
        f"(dont {int(off_before['positif'].sum())} dans un enregistrement à A. blanci), "
        f"{len(off_after)} après (dont {int(off_after['positif'].sum())}).",
        f"- Extraits dont la piste encodée change avec les règles : {len(moved)} "
        f"({moved['recording_id'].nunique()} enregistrements, "
        f"{int(moved['positif'].sum())} extraits à A. blanci, "
        f"{int(moved['jeu_de_test'].sum())} dans le jeu de test).",
        f"- Jeu de test (test_v1) dans la file : {len(test_touched)} enregistrements "
        + ", ".join(
            f"{s} {m} : {n}"
            for (s, m), n in test_touched.groupby(["site", "mic_id"]).size().items()
        ),
    ]
    print("\n".join(notes))
    (out / "LISEZMOI.md").write_text(
        f"# File d'encodage « bonnes pistes »\n\nÉcrite le {datetime.now():%d/%m/%Y à %H:%M} "
        f"par `scripts/encodage/file_pistes.py`, base `{db}`.\n\n## Par modèle\n\n"
        + "\n".join(lines)
        + "\n\n## Extraits annotés et jeu de test\n\n"
        + "\n".join(notes)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
