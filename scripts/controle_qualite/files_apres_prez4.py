"""Trois files d'écoute tirées du récap de la présentation n° 4 (`recap_qc_presentation_4.md`).

- `kaw_d_micro_sourd` : la panne de la piste 0 de Kaw D (SMA14079, mai-août 2024), qui a passé
  les contrôles `silent` et `muffled`. Un enregistrement entier par étape, au même créneau
  (`--creneau`, 6 h par défaut, heure locale) : avant, les deux fichiers qui encadrent la chute,
  pendant (un par semaine), les deux qui encadrent la reprise, après. Basculer micro 1 / micro 2
  dans le poste pour comparer. Colonnes `rms_0_dbfs` et `hf_0` : les indices de la piste 0 en base.
- `blancinet_ecartes` : les écartés que BlanciNet note à `--seuil` (0,5) ou plus, sans écoute en
  base ; enregistrements entiers, du plus haut score au plus bas.
- `coupes_programme` : les fichiers de 30 à 119 s qui commencent sur un créneau du programme,
  écartés et jamais écoutés (décision n° 216 : les garder s'ils durent au moins 30 s ?) ;
  enregistrements entiers.

Lecture seule de la base ; écrit les files dans `paths.queues`.

    uv run python scripts/controle_qualite/files_apres_prez4.py
"""

from __future__ import annotations

import argparse

import pandas as pd

from blanci.annotation.selection import write_queue
from blanci.core.config import default_user_config, load_config
from blanci.exploration import inventaire as inv

KAW_D = "SMA14079"
# Bornes lues sur la série de la piste 0 en base (rms_dbfs, médiane par jour, 19/11/2026) :
# −31 dBFS jusqu'au 22/05, −56 à −67 du 23/05 au 22/07, vers −50 à partir du 23/07.
CHUTE, REPRISE = pd.Timestamp("2024-05-23"), pd.Timestamp("2024-07-23")
FENETRE = (pd.Timestamp("2024-05-13"), pd.Timestamp("2024-08-29"))
# Seuil qui sépare les deux états à chaque bascule : la chute va de −31 à −56 dBFS, la reprise
# n'est que partielle, de −63 à −52 (la piste 0 reste ~20 dB sous son niveau de mai).
SEUILS_DBFS = {CHUTE: -45.0, REPRISE: -58.0}
COLONNES = ["recording_id", "path", "site", "mic_id", "start_utc", "offset_s", "dur_s", "score",
            "reason", "source", "dataset"]


def entiers(rows: pd.DataFrame, reason: pd.Series | str, score=None) -> pd.DataFrame:
    """Lignes de file pour des enregistrements entiers (début 0, durée du fichier)."""
    return pd.DataFrame({
        "recording_id": rows["recording_id"].to_numpy(), "path": rows["path"].to_numpy(),
        "site": rows["site"].to_numpy(), "mic_id": rows["mic_id"].to_numpy(),
        "start_utc": rows["start_utc"].to_numpy(), "offset_s": 0.0,
        "dur_s": rows["duration_s"].round(3).to_numpy(),
        "score": score if score is not None else None,
        "reason": reason if isinstance(reason, str) else reason.to_numpy(),
        "source": "flag", "dataset": rows["dataset"].to_numpy()})


def file_kaw_d(rec: pd.DataFrame, creneau: str) -> pd.DataFrame:
    k = rec[(rec["mic_id"] == KAW_D) & (rec["local"] >= FENETRE[0])
            & (rec["local"] < FENETRE[1]) & rec["normal"]].sort_values("local").copy()
    k["rms_0"] = k["fl"].map(lambda f: (f.get("indices") or {}).get("rms_dbfs"))
    k["hf_0"] = k["fl"].map(lambda f: (f.get("indices") or {}).get("hf_ratio"))
    # Les deux fichiers qui encadrent chaque changement d'état autour des deux dates : la
    # première descente sous le seuil pour la chute, la première remontée pour la reprise.
    bords = {}
    for (date, seuil), etiquette in zip(SEUILS_DBFS.items(), ("2_chute", "4_reprise"),
                                       strict=True):
        sourd = k["rms_0"] < seuil
        vers = etiquette == "2_chute"  # état atteint après la bascule
        bascules = k.index[sourd.ne(sourd.shift()) & sourd.shift().notna() & (sourd == vers)]
        proche = [i for i in bascules if abs(k.at[i, "day"] - date) <= pd.Timedelta(days=1)]
        for i in proche[:1]:
            j = k.index.get_loc(i)
            bords |= {k.index[j - 1]: etiquette, i: etiquette}
    slot = k[k["slot"] == creneau]
    etapes = []
    for _, r in slot.iterrows():
        d = r["day"]
        if d < CHUTE and (CHUTE - d).days <= 10 and (CHUTE - d).days % 3 == 1:
            etapes.append((r.name, "1_avant"))
        elif CHUTE <= d < REPRISE and (d - CHUTE).days % 7 == 3:
            etapes.append((r.name, "3_pendant"))
        elif d >= REPRISE and (d - REPRISE).days % 10 == 4:
            etapes.append((r.name, "5_apres"))
    lignes = dict(etapes)
    lignes |= bords
    choix = k.loc[sorted(lignes, key=lambda i: k.at[i, "local"])]
    out = entiers(choix, pd.Series([f"kaw_d_micro_sourd/{lignes[i]}" for i in choix.index]))
    out["rms_0_dbfs"] = choix["rms_0"].round(1).to_numpy()
    out["hf_0"] = choix["hf_0"].round(3).to_numpy()
    out["heure_locale"] = choix["local"].dt.strftime("%d/%m/%Y %H:%M").to_numpy()
    return out


def file_blancinet(rec: pd.DataFrame, con, seuil: float, ecoutes: set[str]) -> pd.DataFrame:
    det = inv.excluded_with_detections(rec, inv.recording_detections(con, "blancinet"))
    det = det[(det["score_max"] >= seuil) & ~det["recording_id"].isin(ecoutes)]
    det = det.sort_values("score_max", ascending=False)
    out = entiers(det, "blancinet_ecartes/" + det["cause"].str.replace(" ", "_"),
                  det["score_max"].round(3).to_numpy())
    out["detections"] = det["detections"].to_numpy()
    out["cause"] = det["cause"].to_numpy()
    out["heure_locale"] = det["local"].dt.strftime("%d/%m/%Y %H:%M").to_numpy()
    return out


def file_coupes(rec: pd.DataFrame, con, cfg, ecoutes: set[str]) -> pd.DataFrame:
    d = inv.duration_classes(rec, cfg["qc"])
    c = d[(d["duration_class"] == "coupé") & d["on_slot"] & d["excluded"]
          & ~d["recording_id"].isin(ecoutes)].sort_values("local")
    det = inv.recording_detections(con, "blancinet").set_index("recording_id")["score_max"]
    out = entiers(c, "coupes_programme/" + c["site"], c["recording_id"].map(det).to_numpy())
    out["heure_locale"] = c["local"].dt.strftime("%d/%m/%Y %H:%M").to_numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--creneau", default="06:00", help="créneau local des étapes de Kaw D")
    ap.add_argument("--seuil", type=float, default=0.5, help="score BlanciNet minimal")
    args = ap.parse_args()
    cfg = load_config(default_user_config())
    con = inv.open_database(cfg)
    rec = inv.load_recordings(con, cfg)
    ecoutes = set(inv.listened_recordings(con)["recording_id"])
    origine = "scripts/controle_qualite/files_apres_prez4.py (récap de la présentation n° 4)"

    kaw = file_kaw_d(rec, args.creneau)
    write_queue(cfg, kaw, "kaw_d_micro_sourd", {
        "origine": origine,
        "contenu": f"Kaw D (SMA14079), piste 0 sourde du 23/05 au 22/07/2024 (−56 à −67 dBFS au "
                   f"lieu de −31), passée sous `silent` (−70 dBFS) et `muffled` (son indice "
                   f"d'aigus monte : il ne reste que le souffle de l'électronique) ; "
                   f"enregistrements entiers au créneau de {args.creneau}, plus les fichiers "
                   f"qui encadrent la chute et la reprise ; basculer micro 1 / micro 2 ; "
                   f"dossier de terrain « SM D_SMA14079 disfonction »"})
    bn = file_blancinet(rec, con, args.seuil, ecoutes)
    write_queue(cfg, bn, "blancinet_ecartes", {
        "origine": origine,
        "contenu": f"écartés que BlanciNet note à {args.seuil} ou plus, sans écoute en base ; "
                   f"enregistrements entiers, par score décroissant ; un chant entendu les "
                   f"protège (`positive_recordings`), sinon noter poubelle"})
    cp = file_coupes(rec, con, cfg, ecoutes)
    write_queue(cfg, cp, "coupes_programme", {
        "origine": origine,
        "contenu": "fichiers de 30 à 119 s qui commencent sur un créneau du programme, écartés "
                   "pour leur durée et jamais écoutés ; enregistrements entiers ; décision n° 216 "
                   "à prendre (garder un coupé d'au moins 30 s)"})
    for nom, f in (("kaw_d_micro_sourd", kaw), ("blancinet_ecartes", bn),
                   ("coupes_programme", cp)):
        print(f"{nom} : {len(f)} enregistrements, {f['dur_s'].sum() / 60:.1f} min")
    print(kaw[["heure_locale", "reason", "rms_0_dbfs", "hf_0"]].to_string(index=False))


if __name__ == "__main__":
    main()
