"""Construit le tableau de bord du projet (artifact claude.ai) depuis les fichiers du dépôt.

    uv run python documentation/tableau-de-bord/construire.py

Lit les rapports et les CSV de documentation/benchmarks/, les tableaux PNG, la bibliographie,
le glossaire, l'inventaire de documentation/commandes.md, les tests, l'historique git,
structure.yaml (objectif, chaîne, débit), en_cours.yaml (le travail en cours, tenu à la main)
et, si la base locale existe, l'avancement des annotations. Écrit :

- index.html : modele.html avec toutes les données intégrées ;
- fichiers.json : les images à publier à côté de la page (chemin publié → chemin du dépôt) ;
- annotations.json : l'avancement des annotations, relu tel quel quand la base est absente
  (session sans les données) ; versionné.

    uv run python documentation/tableau-de-bord/construire.py [--config config/local.yaml]

Rien n'est calculé de neuf : les chiffres sont ceux des CSV, regroupés. Pour publier, voir
LISEZMOI.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ICI = Path(__file__).resolve().parent
RACINE = ICI.parents[1]
DOC = RACINE / "documentation"
BENCH = DOC / "benchmarks"

# Benchmarks « têtes et régularisations, un site à la fois » : un encodeur par dossier.
TETES = {
    "2026-09-28_anuraset_perch_v2": "perch_v2",
    "2026-09-29_anuraset_protoclr": "protoclr",
    "2026-09-29_anuraset_perch_bird": "perch_bird",
    "2026-09-29_anuraset_birdmae_base": "birdmae_base",
    "2026-09-29_anuraset_birdnet": "birdnet",
    "2026-09-29_anuraset_birdmae_huge": "birdmae_huge",
}
ENCODEURS = "2026-09-30_anuraset_encodeurs"


def arrondi(x, n=4):
    try:
        if pd.isna(x):
            return None
    except TypeError:
        pass
    return round(float(x), n)


def lignes(df: pd.DataFrame, colonnes: list[str]) -> list[list]:
    """Tableau compact : liste de lignes, dans l'ordre des colonnes."""
    out = []
    for row in df[colonnes].itertuples(index=False):
        out.append([arrondi(v) if isinstance(v, float) else v for v in row])
    return out


def compter_tests() -> dict[str, int]:
    compte: dict[str, int] = {}
    for f in (RACINE / "tests").rglob("test_*.py"):
        n = len(re.findall(r"^\s*def test_", f.read_text(encoding="utf-8"), flags=re.M))
        cle = str(f.parent.relative_to(RACINE))
        compte[cle] = compte.get(cle, 0) + n
    return compte


def historique(n: int = 20) -> list[list[str]]:
    try:
        sortie = subprocess.run(
            ["git", "log", f"-{n}", "--date=short", "--pretty=format:%h\t%ad\t%s"],
            cwd=RACINE,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    out = []
    for ligne in sortie.splitlines():
        h, d, s = ligne.split("\t", 2)
        s = re.sub(r"\s*\((DECISIONS )?n° [^)]*\)", "", s)  # renvois au journal retirés
        out.append([h, d, s.split(" - ")[0].strip()])
    return out


def commit_courant() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=RACINE,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def inventaire() -> dict:
    """Tableau « Inventaire » de documentation/commandes.md."""
    texte = (DOC / "commandes.md").read_text(encoding="utf-8")
    bloc = texte.split("### Inventaire", 1)[1]
    sites, jeu = [], ""
    for ligne in bloc.splitlines():
        if not ligne.startswith("|") or "---" in ligne or "Enregistrements" in ligne:
            if sites and not ligne.startswith("|"):
                break
            continue
        cellules = [c.strip().replace("*", "") for c in ligne.strip("|").split("|")]
        j, site, n, n120, h, go = cellules
        if j and not j.startswith("Total"):
            jeu = "2023" if j.startswith("2023") else "2026"
        if site.startswith("total") or j.startswith("Total"):
            continue

        def nombre(s: str) -> float:
            return float(s.replace(" ", "").replace(" ", "").replace(",", "."))

        sites.append(
            {
                "jeu": jeu,
                "site": site,
                "enregistrements": int(nombre(n)),
                "heures": nombre(h),
                "go": nombre(go),
            }
        )
    fenetres = re.findall(r"\| (\d) s \| ([\d,]+) s \| (\d+) \| ([\d ]+) \|", bloc)
    drapeaux = re.search(r"Écartés par les drapeaux.*?encodables", bloc, flags=re.S)
    return {
        "sites": sites,
        "fenetres": [
            [f"{a} s", f"{b} s", int(c), int(d.replace(" ", ""))] for a, b, c, d in fenetres
        ],
        "drapeaux": re.sub(r"\s+", " ", drapeaux.group(0)) if drapeaux else "",
        "jeux": {
            "2023": "Phénologie : 3 sites, 2 micros par site, décembre 2023 → novembre 2024",
            "2026": "Campagne 2026 : 5 sites, un relevé d'environ une semaine par site",
        },
    }


def index_benchmarks() -> dict[str, dict]:
    """Tableau de documentation/benchmarks/LISEZMOI.md : numéro, objet, statut."""
    out = {}
    for ligne in (BENCH / "LISEZMOI.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (\d+) \| `([^`]+)/` \| (.+?) \| (\w+) \|", ligne)
        if m:
            out[m.group(2)] = {"numero": m.group(1), "objet": m.group(3), "statut": m.group(4)}
    return out


def rapports(fichiers: dict[str, str]) -> list[dict]:
    index = index_benchmarks()
    out = []
    for dossier in sorted(p for p in BENCH.iterdir() if (p / "RAPPORT.md").exists()):
        texte = (dossier / "RAPPORT.md").read_text(encoding="utf-8")
        figures = []
        for png in sorted((dossier / "figures").glob("*.png")):
            publie = f"figures/{dossier.name}/{png.name}"
            fichiers[publie] = str(png.relative_to(RACINE))
            figures.append(publie)
        texte = re.sub(r"\]\(figures/", f"](figures/{dossier.name}/", texte)
        titre = texte.splitlines()[0].lstrip("# ").strip()
        meta = texte.splitlines()[2] if len(texte.splitlines()) > 2 else ""
        info = index.get(dossier.name, {})
        out.append(
            {
                "id": dossier.name,
                "titre": titre,
                "meta": meta,
                "numero": info.get("numero", ""),
                "objet": info.get("objet", ""),
                "statut": info.get("statut", ""),
                "date": dossier.name[:10],
                "texte": texte,
                "figures": figures,
                "donnees": sorted(p.name for p in (dossier / "donnees").glob("*.csv")),
            }
        )
    return sorted(out, key=lambda r: r["numero"])


def benchmark_encodeurs() -> dict:
    d = BENCH / ENCODEURS / "donnees"
    enc = pd.read_csv(d / "encodeurs.csv")
    best = pd.read_csv(d / "meilleures.csv")
    transfert = pd.read_csv(d / "transfert.csv")
    comp = pd.read_csv(d / "comparaisons.csv")
    comp_libre = pd.read_csv(d / "comparaisons_meilleur_libre.csv")
    courbe = pd.read_csv(d / "courbe.csv")

    courbe_moy = (
        courbe.groupby(["encoder", "head", "k"])[["ap_window", "ap_minute"]].mean().reset_index()
    )
    enc = enc.drop(columns=["source"]).merge(best, on="encoder", how="left")
    return {
        "encodeurs": json.loads(enc.to_json(orient="records", force_ascii=False)),
        "transfert": lignes(transfert, ["encoder", "species", "head", "level", "ap", "ap_site"]),
        "comparaisons": lignes(
            comp,
            [
                "encoder",
                "head",
                "species",
                "level",
                "diff",
                "lo",
                "hi",
                "p_holm",
                "significant_holm",
            ],
        ),
        "comparaisons_libre": lignes(
            comp_libre,
            ["encoder", "head", "species", "diff", "lo", "hi", "p_holm", "significant_holm"],
        ),
        "courbe": lignes(courbe_moy, ["encoder", "head", "k", "ap_window", "ap_minute"]),
    }


def benchmark_tetes() -> list[list]:
    """AP poolée (fenêtre, enregistrement) et AP moyenne par site (fenêtre), benchmarks 01–06."""
    out = []
    for dossier, encodeur in TETES.items():
        d = BENCH / dossier / "donnees"
        tetes = pd.read_csv(d / "tetes.csv")
        sites = pd.read_csv(d / "sites.csv")
        par_site = sites.dropna(subset=["ap"]).groupby(["species", "head"])["ap"].mean()
        comp = pd.read_csv(d / "comparaisons.csv")
        sig = {
            (r.species, r.head, r.level): (r.diff, bool(r.significant_holm))
            for r in comp.itertuples()
            if r.reference == "logistic"
        }
        for r in tetes.itertuples():
            level = {"window": "fenetre", "recording": "minute"}.get(r.level, r.level)
            ap_site = par_site.get((r.species, r.head)) if level == "fenetre" else None
            diff, signif = sig.get((r.species, r.head, r.level), (None, None))
            out.append(
                [
                    encodeur,
                    r.species,
                    r.head,
                    level,
                    arrondi(r.ap),
                    arrondi(ap_site),
                    arrondi(diff),
                    signif,
                ]
            )
    return out


def tableaux(fichiers: dict[str, str]) -> list[dict]:
    dossier = DOC / "tableaux"
    desc = {}
    for ligne in (dossier / "LISEZMOI.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"- `([\w-]+\.png)` : (.+)", ligne)
        if m:
            desc[m.group(1)] = m.group(2)
    out = []
    for png in sorted(dossier.glob("*.png")):
        publie = f"tableaux/{png.name}"
        fichiers[publie] = str(png.relative_to(RACINE))
        out.append(
            {
                "image": publie,
                "nom": png.stem.replace("-", " "),
                "description": desc.get(png.name, ""),
            }
        )
    return out


# Sources de labels qui ne relèvent pas du plan d'annotation v1 : l'import Blancinet (choisi
# par un détecteur, hors entraînement et évaluation) et l'écoute des enregistrements écartés.
HORS_PLAN = ("import", "flag")


def annotations(config: Path | None) -> dict:
    """Avancement du plan d'annotation v1, compté dans la base locale (lecture seule).

    Entraînement : enregistrements hors jeu gelé ayant au moins un label du plan. Évaluation :
    enregistrements du jeu gelé (toutes versions) ayant au moins un label ; positifs : ceux qui
    ont au moins une fenêtre positive. Dernier label de chaque fenêtre. Sans base, relit
    annotations.json (dernier comptage connu).
    """
    sauvegarde = ICI / "annotations.json"
    try:
        from blanci.core.config import config_path, load_config
        from blanci.inputs.dataset import current_labels, recordings_table
        from blanci.inputs.frozen import frozen_recordings
        from blanci.inputs.labels import POSITIVE_LABELS

        if config is None and (RACINE / "config" / "local.yaml").exists():
            config = RACINE / "config" / "local.yaml"
        cfg = load_config(config)
        base = config_path(cfg, "db")
        if not base.exists():
            raise FileNotFoundError(base)
        con = sqlite3.connect(f"file:{base.as_posix()}?mode=ro", uri=True)
        try:
            labels = current_labels(con)
            recs = recordings_table(con)[["recording_id", "site"]]
        finally:
            con.close()
        try:
            geles = frozen_recordings(cfg)
        except (OSError, ValueError):
            geles = set()
    except (ImportError, OSError, sqlite3.Error) as err:
        if sauvegarde.exists():
            ancien = json.loads(sauvegarde.read_text(encoding="utf-8"))
            ancien["source"] = f"dernier comptage connu ({type(err).__name__} : base absente)"
            return ancien
        return {"compte": None, "source": "base locale absente, aucun comptage enregistré"}

    labels = labels[~labels["source"].isin(HORS_PLAN)]
    par_rec = labels.groupby("recording_id")["label"].apply(
        lambda s: bool(s.isin(POSITIVE_LABELS).any())
    )
    par_rec = par_rec.rename("positif").reset_index().merge(recs, on="recording_id", how="left")
    par_rec["gele"] = par_rec["recording_id"].isin(geles)
    entr, ev = par_rec[~par_rec["gele"]], par_rec[par_rec["gele"]]
    sites = (
        par_rec.assign(jeu=par_rec["gele"].map({True: "evaluation", False: "entrainement"}))
        .groupby(["site", "jeu"])["positif"]
        .agg(["size", "sum"])
        .reset_index()
    )
    resultat = {
        "compte": {
            "entrainement": int(len(entr)),
            "positifs_entrainement": int(entr["positif"].sum()),
            "evaluation": int(len(ev)),
            "positifs_evaluation": int(ev["positif"].sum()),
        },
        "sites": [
            [r.site or "?", r.jeu, int(r.size), int(r.sum)] for r in sites.itertuples(index=False)
        ],
        "compte_le": date.today().isoformat(),
        "source": "base locale",
    }
    sauvegarde.write_text(json.dumps(resultat, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultat


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--config", type=Path, default=None, help="config/local.yaml par défaut")
    config = args.parse_args().config
    contenu = yaml.safe_load((ICI / "structure.yaml").read_text(encoding="utf-8"))
    contenu.update(yaml.safe_load((ICI / "en_cours.yaml").read_text(encoding="utf-8")))
    fichiers: dict[str, str] = {}
    tests = compter_tests()
    for etape in contenu["chaine"]:
        etape["n_tests"] = tests.get(etape.get("tests", ""), 0)
    donnees = {
        "genere": {"commit": commit_courant(), "date": date.today().isoformat()},
        "contenu": contenu,
        "annotations": annotations(config),
        "tests": {"total": sum(tests.values()), "par_dossier": tests},
        "historique": historique(),
        "inventaire": inventaire(),
        "rapports": rapports(fichiers),
        "encodeurs": benchmark_encodeurs(),
        "tetes": benchmark_tetes(),
        "tableaux": tableaux(fichiers),
        "biblio": (DOC / "biblio" / "biblio.md").read_text(encoding="utf-8"),
        "glossaire": (DOC / "glossaire-bioacoustique.md").read_text(encoding="utf-8"),
    }
    brut = json.dumps(donnees, ensure_ascii=False, separators=(",", ":"), default=str)
    brut = brut.replace("</", "<\\/")  # pas de </script> dans les données
    modele = (ICI / "modele.html").read_text(encoding="utf-8")
    page = modele.replace("/*DONNEES*/null", brut)
    (ICI / "index.html").write_text(page, encoding="utf-8")
    (ICI / "fichiers.json").write_text(
        json.dumps(fichiers, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"index.html : {len(page) / 1e6:.2f} Mo · {len(fichiers)} images")


if __name__ == "__main__":
    main()
