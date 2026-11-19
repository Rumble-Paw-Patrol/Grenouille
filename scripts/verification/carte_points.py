"""Carte interactive des points d'écoute, d'après la position GPS des en-têtes GUANO.

Lit l'échantillon des pistes (`paths.reports/pistes/echantillon_pistes.csv` : un enregistrement
par micro et par jour, avec `lat` et `lon`) et la base (nom de terrain des micros, lu dans le
dossier : « 2LA04185_GI19 » → GI19). `inventaire.gps_places` range chaque position en lieu de
pose ou lieu de réglage (lieu partagé par des points de plusieurs sites).

Écrit `paths.reports/carte_points_ecoute.html` (hors dépôt : les positions d'une espèce
menacée ne se publient pas). Fonds OpenStreetMap et Esri, Leaflet chargé depuis cdnjs : il faut
une connexion pour afficher le fond, pas pour les points.

    uv run python scripts/verification/carte_points.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from blanci.core.config import config_path, default_user_config, load_config
from blanci.exploration import inventaire as inv

COULEURS = {"2023": "#2A7F72", "2026": "#C8901E"}


def noms_de_terrain(rec: pd.DataFrame) -> pd.Series:
    """« GI19 » pour le dossier « 2LA04185_GI19 » ; vide si le dossier n'a que le numéro."""
    dossier = rec["path"].str.split("/").str[-3]
    nom = dossier.str.split("_", n=1).str[1].fillna("")
    return nom.groupby([rec[c] for c in inv.POINT]).agg(
        lambda s: ", ".join(sorted({x for x in s if x})))


def main() -> None:
    cfg = load_config(default_user_config())
    reports = config_path(cfg, "reports")
    ech = pd.read_csv(reports / "pistes" / "echantillon_pistes.csv",
                      dtype={"dataset": str, "mic_id": str})
    places = inv.gps_places(ech)
    con = inv.open_database(cfg)
    rec = inv.load_recordings(con, cfg)
    noms = noms_de_terrain(rec)
    points = rec.groupby(inv.POINT).agg(n=("path", "size"), debut=("local", "min"),
                                        fin=("local", "max")).reset_index()
    points["nom"] = [noms.get(tuple(k), "") for k in points[inv.POINT].itertuples(index=False)]

    terrain = places[~places["setup"]]
    reglage = places[places["setup"]]
    avec = set(map(tuple, terrain[inv.POINT].itertuples(index=False)))
    points["terrain"] = [tuple(k) in avec for k in points[inv.POINT].itertuples(index=False)]

    marques = []
    for _, p in terrain.iterrows():
        info = points.set_index(inv.POINT).loc[(p["dataset"], p["site"], p["mic_id"])]
        marques.append({"lat": p["lat"], "lon": p["lon"], "jeu": p["dataset"],
                        "site": p["site"], "micro": p["mic_id"], "nom": info["nom"],
                        "fichiers": int(p["files"]), "n": int(info["n"]),
                        "debut": f"{info['debut']:%d/%m/%Y}", "fin": f"{info['fin']:%d/%m/%Y}",
                        "premier": str(p["first"])[:10], "dernier": str(p["last"])[:10]})
    lieux = []
    for _, g in reglage.groupby(["place_lat", "place_lon"]):
        par_site = g.groupby(["dataset", "site"])["mic_id"].nunique()
        lieux.append({"lat": float(g["lat"].median()), "lon": float(g["lon"].median()),
                      "points": int(len(g)), "fichiers": int(g["files"].sum()),
                      "sites": [f"{s} {j} : {n}" for (j, s), n in par_site.items()],
                      "premier": str(g["first"].min())[:10],
                      "dernier": str(g["last"].max())[:10]})
    bilan = points.groupby(["dataset", "site"]).agg(
        points=("mic_id", "size"), terrain=("terrain", "sum")).reset_index()
    sans = points[~points["terrain"]].sort_values(inv.POINT)
    donnees = {"marques": marques, "lieux": lieux,
               "bilan": bilan.to_dict("records"),
               "sans": sans[["dataset", "site", "mic_id", "nom"]].to_dict("records")}

    gabarit = (Path(__file__).with_name("carte_points.html")).read_text(encoding="utf-8")
    sortie = reports / "carte_points_ecoute.html"
    sortie.write_text(gabarit.replace("/*DONNEES*/null", json.dumps(donnees, default=int)),
                      encoding="utf-8")
    print(f"{len(points)} points, {int(points['terrain'].sum())} avec une position de terrain, "
          f"{len(lieux)} lieux de réglage ({int(reglage['mic_id'].nunique())} micros) → {sortie}")
    print(bilan.to_string(index=False))


if __name__ == "__main__":
    main()
