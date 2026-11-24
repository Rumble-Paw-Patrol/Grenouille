"""File d'écoute « commentaires_ambigus » : les extraits dont le commentaire de Léonard peut
prêter à confusion, plus les deux enregistrements du relevé 7 de Mataroni écartés pour durée.

Le tri des commentaires est fait à la main (lecture des 115 commentaires des extraits écoutés,
dernière version de chaque extrait) : la liste `EXTRAITS` dit, pour chaque `span_id` de
`annotations/extraits_annotes.csv`, pourquoi il est là. Les `span_id` sont ceux de l'export du
24/11/2026 ; ils changent si la base est reconstruite : relire alors la liste par recording_id et
début d'extrait (colonnes écrites dans `candidats.csv`).

Motifs (colonne `reason`, du plus net au plus léger) :

- `1_label_contredit` : le label dit l'inverse du commentaire, ou le commentaire doute alors que
  le label est tranché (« pas convaincu », « je crois que… », « sans conviction », fond avec du
  chant) ;
- `2_hesitation_non_codee` : label « A. blanci ? » dont le commentaire dit vers quoi ça penche
  (congénère, blanci saturé, spectro contre oreille) ou entre quelles hypothèses on hésite
  (chœur de blanci ou congénère) : le label ne le garde pas ;
- `3_commentaire_obscur` : commentaire illisible seul (« idem », « le même », « non, faux ami »),
  ou mélange de blanci et de faux ami que les intervalles ne montrent pas ;
- `4_reference_citee` : extrait qu'un « idem » ou un « le même » cite, directement ou par une
  chaîne d'« idem » (`RENVOIS`) : il accompagne les extraits qui y renvoient, juste avant eux.

Un « idem » renvoie au dernier extrait précédent (dans l'ordre des écoutes) qui porte un
commentaire, et un « idem » peut renvoyer à un autre « idem » : la colonne `groupe` donne l'extrait
racine de la chaîne, la file garde la chaîne d'un bloc, racine d'abord, et la colonne `role` marque
les extraits cités (« voisin, référence ») et ceux qui renvoient (« candidat, idem »). « FA » : faux
ami (extrait 1185 : rien d'obscur).

Lancer : `uv run python scripts/annotation/file_commentaires_ambigus.py` (lecture seule de la
base ; écrit `annotations/files_ecoute/commentaires_ambigus/`). Relancer `blanci export-labels`
avant, pour que les commentaires soient ceux de la base.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from blanci.annotation.selection import _queue_readme  # noqa: E402
from blanci.core.config import config_path, load_config  # noqa: E402

NAME = "commentaires_ambigus"
CONTREDIT = "1_label_contredit"
HESITATION = "2_hesitation_non_codee"
OBSCUR = "3_commentaire_obscur"
CITE = "4_reference_citee"

# « idem » / « le même » -> le ou les extraits cités (le dernier extrait précédent qui porte un
# commentaire ; 8 : « comme les autres » = les trois extraits corrigés le 28/10, anciens n° 4 à 6,
# commentés « pas sûr, ça pourrait être un congénère » / « idem » / « idem »).
RENVOIS: dict[int, list[int]] = {
    8: [455, 456, 457],
    9: [8],
    17: [9],
    23: [17],
    36: [23],
    37: [36],
    144: [141],
    531: [530],
    1101: [1100],
    1103: [1101],
    1323: [1322],
    1325: [1323],
    1415: [1414],
    1416: [1415],
    1417: [1416],
}

# span_id -> (motif, ce qui prête à confusion)
EXTRAITS: dict[int, tuple[str, str]] = {
    # Faux amis dits « je pense » ; les suivants renvoient à eux (« idem », « le même »)
    8: (
        CONTREDIT,
        "faux ami, mais « je pense que c'est un congénère » ; « les autres » : les 3 du dessus",
    ),
    9: (CONTREDIT, "faux ami, « idem » : même doute que l'extrait cité"),
    17: (OBSCUR, "faux ami, « le même » : le faux ami cité, supposé congénère"),
    23: (OBSCUR, "faux ami, « toujours le même » : le faux ami cité, supposé congénère"),
    36: (OBSCUR, "faux ami, « toujours le même faux ami » : cité, supposé congénère"),
    37: (OBSCUR, "faux ami, « idem » : le faux ami cité, supposé congénère"),
    # Chœurs et notes de hauteurs différentes : chœur de blanci ou faux ami ?
    63: (HESITATION, "« A. blanci ? » : un congénère à deux notes, ou blanci et un congénère ?"),
    82: (CONTREDIT, "blanci seul tracé, mais « à la fois blanci et des congénères »"),
    103: (HESITATION, "« A. blanci ? », mais « quasi sûr que c'est un congénère »"),
    123: (CONTREDIT, "faux ami, mais « pas certain »"),
    130: (CONTREDIT, "faux ami et blanci tracés, « pas convaincu de mes annotations »"),
    141: (HESITATION, "« A. blanci ? » : chœur de blanci ou congénère ?"),
    144: (HESITATION, "« A. blanci ? » : deux blanci ou un congénère ?"),
    145: (
        HESITATION,
        "« A. blanci ? », « je ne sais pas » (même série que les deux extraits cités)",
    ),
    157: (CONTREDIT, "faux ami, « quasi sûr »"),
    160: (OBSCUR, "blanci tracé, « un congénère chante en même temps » (pas tracé)"),
    174: (OBSCUR, "faux ami, mais « fait la note de blanci » et d'autres harmoniques"),
    278: (OBSCUR, "blanci tracé, « un faux ami chante par dessus » (pas tracé)"),
    344: (CONTREDIT, "chœur, mais « les mâles ne chantent pas tous la même note ?? »"),
    371: (CONTREDIT, "blanci, « pas convaincu »"),
    375: (CONTREDIT, "blanci et faux ami tracés, « pas convaincu »"),
    405: (CONTREDIT, "blanci et « A. blanci ? » tracés, « aucune idée »"),
    410: (CONTREDIT, "chœur, mais « ça me perturbe » : pas tous la même note"),
    416: (CONTREDIT, "chœur, « même problème » : différentes notes"),
    435: (CONTREDIT, "faux ami « selon moi » : double note, empreinte pas de blanci"),
    437: (CONTREDIT, "faux ami « donc je pars du principe » : spectro de blanci, oreille non"),
    459: (HESITATION, "« A. blanci ? » : chœur de blanci ou chœur de congénères ?"),
    481: (CONTREDIT, "chœur, mais « ou congénère qui fait des doubles notes ? »"),
    530: (CONTREDIT, "fond, mais « ça chante » (quasi inaudible, « inclassable »)"),
    531: (CONTREDIT, "fond, « idem » : ça chante, inclassable"),
    631: (HESITATION, "« A. blanci ? » : le spectro dit blanci, l'oreille l'inverse"),
    633: (CONTREDIT, "faux ami et blanci tracés, « douteux »"),
    635: (CONTREDIT, "chœur, « sans conviction »"),
    636: (CONTREDIT, "faux ami, « sans conviction »"),
    637: (CONTREDIT, "faux ami, « sans conviction »"),
    666: (CONTREDIT, "faux ami, « difficile à déterminer »"),
    667: (CONTREDIT, "faux ami, « difficile »"),
    671: (OBSCUR, "faux ami, « non, faux ami » (non à quoi ?)"),
    672: (CONTREDIT, "faux ami tracé, mais « rien à signaler »"),
    673: (HESITATION, "« A. blanci ? », « je pense que ce ne sont pas blanci »"),
    734: (OBSCUR, "chœur : le micro 1 entend une grenouille, le micro 2 en entend deux"),
    802: (CONTREDIT, "faux amis tracés au spectro, « on entend pas les faux amis ici »"),
    993: (HESITATION, "« A. blanci ? » : des notes de blanci, toutes à des fréquences différentes"),
    994: (CONTREDIT, "blanci, « A. blanci ? » et chœur tracés ; « plus bas en fréquence »"),
    1100: (HESITATION, "« A. blanci ? » : blanci très près du micro (saturé) ou congénère ?"),
    1101: (HESITATION, "« A. blanci ? », « idem » : blanci saturé ?"),
    1103: (HESITATION, "« A. blanci ? », « idem » : blanci saturé ?"),
    1322: (HESITATION, "« A. blanci ? » : faux ami ou blanci saturé, penche pour le faux ami"),
    1323: (HESITATION, "« A. blanci ? », « idem » : faux ami ou blanci saturé"),
    1325: (HESITATION, "« A. blanci ? », « idem » : faux ami ou blanci saturé"),
    1414: (HESITATION, "« A. blanci ? » : deux mâles coordonnés ou un congénère ?"),
    1415: (HESITATION, "« A. blanci ? », « idem » : deux mâles coordonnés ou un congénère"),
    1416: (HESITATION, "« A. blanci ? », « idem » : deux mâles coordonnés ou un congénère"),
    1417: (HESITATION, "« A. blanci ? », « idem » : deux mâles coordonnés ou un congénère"),
    1419: (HESITATION, "« A. blanci ? », « quasi sûr » congénère, ou blanci saturé"),
    1431: (CONTREDIT, "blanci, « 99 % sûr » pour la première note, mais à 4,1 kHz"),
    1551: (CONTREDIT, "faux ami, « reste douteux »"),
    1552: (OBSCUR, "« A. blanci ? », « effectivement, avec l'intervalle c'est mieux » (de quoi ?)"),
    # Cités par « comme les autres » (extrait 8), donc par la chaîne des « idem » / « le même »
    455: (CITE, "cité par « comme les autres » de l'extrait qui suit : faux ami corrigé le 28/10"),
    456: (CITE, "cité par « comme les autres » de l'extrait qui suit : faux ami corrigé le 28/10"),
    457: (CITE, "cité par « comme les autres » de l'extrait qui suit : faux ami corrigé le 28/10"),
}

# Enregistrements écartés pour durée anormale (< 120 s), jamais écoutés : relevé 7 de Mataroni.
ENREGISTREMENTS = {
    "ccb103842f82e105": "2LA04111 du 19/05/2026, 103,0 s",
    "9dbbae27daac245b": "2LA04168 du 28/05/2026, 107,25 s",
}
REASON_MATARONI = "commentaires_ambigus/R7_Mataroni_duree_anormale"


def _groupes() -> dict[int, int]:
    """span_id -> extrait racine de sa chaîne de renvois (le plus petit span_id de la chaîne)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for span, cited in RENVOIS.items():
        for other in cited:
            a, b = find(span), find(other)
            parent[max(a, b)] = min(a, b)
    return {x: find(x) for x in list(parent)}


def _explicite(span: int, comments: dict[int, str]) -> str:
    """Premier commentaire non « idem » en remontant les renvois depuis `span`."""
    seen = set()
    while span in RENVOIS and span not in seen:
        seen.add(span)
        if not re.match(r"(idem|toujours le m|le m)", str(comments.get(span, "")), re.I):
            break
        span = RENVOIS[span][0]
    return str(comments.get(span, ""))


def _extraits(annotations: Path) -> pd.DataFrame:
    spans = pd.read_csv(annotations / "extraits_annotes.csv")
    current = spans[spans["remplace_par"].isna() & spans["span_id"].isin(EXTRAITS)]
    wanted = set(EXTRAITS) | {i for cited in RENVOIS.values() for i in cited}
    missing = sorted(wanted - set(current["span_id"]))
    if missing:
        raise SystemExit(f"span_id absents ou remplacés : {missing} (ré-exporter les labels ?)")
    # Versions précédentes d'un extrait réécouté : leur commentaire éclaire le nouveau
    # (« effectivement… », « non, faux ami » répondent à l'ancienne version).
    previous: dict[int, list[int]] = {}
    for old, new in spans.dropna(subset=["remplace_par"])[["span_id", "remplace_par"]].to_numpy():
        previous.setdefault(int(new), []).append(int(old))
    by_id = spans.set_index("span_id")

    def versions(span: int) -> list[int]:
        out = []
        for old in previous.get(span, []):
            out += versions(old) + [old]
        return out

    comments = by_id["comment"].to_dict()
    groups = _groupes()
    rank = {k: int(v[0][0]) for k, v in EXTRAITS.items()}
    group_rank: dict[int, int] = {}
    for span in EXTRAITS:
        root = groups.get(span, span)
        group_rank[root] = min(group_rank.get(root, 9), rank[span])
    cited = {i for ids in RENVOIS.values() for i in ids}

    def depth(span: int) -> int:
        return 1 + max(depth(i) for i in RENVOIS[span]) if span in RENVOIS else 0

    rows = []
    for span in current["span_id"]:
        r = by_id.loc[span]
        cause = EXTRAITS[span][1]
        if span in RENVOIS and re.match(r"(idem|toujours le m|le m)", str(r["comment"]), re.I):
            cause += f" ; renvoie à « {_explicite(RENVOIS[span][0], comments)} »"
        for old in versions(span):
            o = by_id.loc[old]
            if isinstance(o["comment"], str):
                cause += f" ; version précédente ({o['classes']}) : « {o['comment']} »"
        group = groups.get(span, span)
        rows.append(
            {
                "recording_id": r["recording_id"],
                "path": r["path"],
                "site": r["site"],
                "mic_id": r["mic_id"],
                "start_utc": r["start_utc"],
                "offset_s": r["start_s"],
                "dur_s": round(r["end_s"] - r["start_s"], 2),
                "score": float("nan"),
                "reason": f"{NAME}/{EXTRAITS[span][0]}",
                "source": r["source"],  # « plan » : l'extrait de 30 s, comme à la première écoute
                "span_id": span,
                "groupe": group if group != span or span in cited or span in RENVOIS else pd.NA,
                "role": "candidat, renvoi"
                if span in RENVOIS
                else ("voisin, référence" if span in cited else pd.NA),
                "classes": r["classes"],
                "commentaire": r["comment"],
                "cause": cause,
                "_tri": (group_rank[group], group, depth(span), r["created_at"], span),
            }
        )
    out = pd.DataFrame(rows)
    out = out.iloc[sorted(range(len(out)), key=lambda i: out["_tri"].iloc[i])]
    out = out.drop(columns="_tri").reset_index(drop=True)
    out["span_id"] = out["span_id"].astype("Int64")
    out["groupe"] = out["groupe"].astype("Int64")
    return out


def _enregistrements(con: sqlite3.Connection) -> pd.DataFrame:
    marks = ",".join("?" * len(ENREGISTREMENTS))
    rec = pd.read_sql_query(
        "SELECT recording_id, path, site, mic_id, start_utc, duration_s FROM recordings "
        f"WHERE recording_id IN ({marks})",
        con,
        params=tuple(ENREGISTREMENTS),
    )
    if len(rec) != len(ENREGISTREMENTS):
        raise SystemExit("enregistrements de Mataroni introuvables dans la base")
    return pd.DataFrame(
        {
            "recording_id": rec["recording_id"],
            "path": rec["path"],
            "site": rec["site"],
            "mic_id": rec["mic_id"],
            "start_utc": rec["start_utc"],
            "offset_s": 0.0,
            "dur_s": rec["duration_s"].astype(float).round(2),  # enregistrement entier
            "score": float("nan"),
            "reason": REASON_MATARONI,
            "source": "flag",  # comme les files de durée anormale (autres_drapeaux)
            "cause": [ENREGISTREMENTS[r] + " : écarté, durée < 120 s" for r in rec["recording_id"]],
        }
    )


def main() -> None:
    cfg = load_config(ROOT / "config" / "local.yaml")  # default.yaml + local.yaml
    extraits = _extraits(ROOT / "annotations")
    con = sqlite3.connect(f"file:{config_path(cfg, 'db').as_posix()}?mode=ro", uri=True)
    enreg = _enregistrements(con)
    queue = pd.concat([extraits, enreg], ignore_index=True)
    for col in ("span_id", "groupe"):
        queue[col] = queue[col].astype("Int64")
    folder = ROOT / "annotations" / "files_ecoute" / NAME
    folder.mkdir(parents=True, exist_ok=True)
    queue.to_csv(folder / "candidats.csv", index=False)
    about = {
        "origine": f"commentaires de Léonard relus le {datetime.now():%d/%m/%Y} "
        "(scripts/annotation/file_commentaires_ambigus.py)",
        "contenu": f"{len(extraits)} extraits de 30 s dont le commentaire prête à confusion avec "
        f"le label, puis {len(enreg)} enregistrements entiers de Mataroni (relevé 7) écartés "
        "pour durée < 120 s, jamais écoutés",
        "à faire après l'écoute": "réannoter l'extrait (la nouvelle version remplace l'ancienne) ; "
        "la colonne `cause` dit ce qui prête à confusion",
    }
    (folder / "LISEZMOI.md").write_text(_queue_readme(queue, NAME, about), encoding="utf-8")
    print(f"{len(queue)} candidats : {folder}")


if __name__ == "__main__":
    main()
