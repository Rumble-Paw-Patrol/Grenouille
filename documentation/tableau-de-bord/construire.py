"""Construit le tableau de bord du projet (artifact claude.ai) depuis les fichiers du dépôt.

    uv run python documentation/tableau-de-bord/construire.py

Lit les rapports et les CSV de documentation/benchmarks/, les tableaux PNG, la bibliographie,
l'inventaire de documentation/commandes.md, les tests, l'historique git,
structure.yaml (chaîne : modules, états), textes.yaml (tous les textes de la page) et
en_cours.yaml (le travail en cours), ces deux derniers tenus à la main par Léonard
et, si la base locale existe, l'avancement des annotations. Écrit :

- index.html : modele.html avec toutes les données intégrées ;
- fichiers.json : les images à publier à côté de la page (chemin publié → chemin du dépôt),
  dont le spectrogramme du bandeau (spectrogramme.jpg, fait par spectrogramme.py) et, s'il
  existe, la capture du poste d'annotation (poste-annotation.png) ;
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


def git(*args: str) -> str:
    """Sortie d'une commande git à la racine du dépôt ; chaîne vide si git échoue."""
    try:
        return subprocess.run(
            ["git", *args], cwd=RACINE, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def ref_main() -> str:
    """La branche main telle qu'elle est sur GitHub (origin/main, rafraîchie si le réseau le
    permet), sinon main locale, sinon HEAD. Les numéros de commit se comptent sur elle."""
    try:
        subprocess.run(
            ["git", "fetch", "-q", "origin", "main"], cwd=RACINE, capture_output=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    for ref in ("origin/main", "main"):
        if git("rev-parse", "--verify", "--quiet", ref).strip():
            return ref
    return "HEAD"


def numeros(ref: str) -> dict[str, int]:
    """Numéro de chaque commit poussé sur main (premier parent) : n° 1 = le premier commit du
    dépôt. Ce sont les numéros qu'accepte le skill audit-n-from ; un commit d'une autre branche
    n'en a pas."""
    hashes = git("rev-list", "--reverse", "--first-parent", "--abbrev-commit", ref).split()
    return {h: i for i, h in enumerate(hashes, start=1)}


def historique(ref: str, n: int = 20) -> list[list]:
    """Derniers commits de main, avec leur numéro."""
    sortie = git(
        "log", "--first-parent", f"-{n}", "--date=short", "--pretty=format:%h\t%ad\t%s", ref
    )
    num = numeros(ref)
    out = []
    for ligne in sortie.splitlines():
        h, d, s = ligne.split("\t", 2)
        if s.startswith("Merge "):
            # Une fusion ne dit que le nom d'une branche de travail (claude/…) : on montre à la
            # place le dernier commit qu'elle apporte, ou rien si elle n'apporte rien de neuf.
            apports = git("log", "--no-merges", "--format=%s", f"{h}^1..{h}^2").splitlines()
            if not apports:
                continue
            s = apports[0] + (f" (+{len(apports) - 1} autres)" if len(apports) > 1 else "")
        s = re.sub(r"\s*\((DECISIONS )?n° [^)]*\)", "", s)  # renvois au journal retirés
        out.append([h, d, s.split(" - ")[0].strip(), num.get(h)])
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
    drapeaux = re.sub(r"\s+", " ", drapeaux.group(0)) if drapeaux else ""
    drapeaux = drapeaux.replace(
        "Écartés par les drapeaux (jamais encodés)", "Enregistrements écartés du projet"
    ).replace("micros dans le sac", "micros allumés dans le sac")
    # Le renvoi aux DECISIONS et les 8 notés à l'écoute n'ont pas leur place sur la page.
    drapeaux = re.sub(r" \(calculés[^)]*\) et \d+ notés à l'écoute", "", drapeaux)
    drapeaux = drapeaux.replace(", 88 hors relevé, ", ", 88 hors relevé et ")
    return {
        "sites": sites,
        "fenetres": [
            [f"{a} s", f"{b} s", int(c), int(d.replace(" ", ""))] for a, b, c, d in fenetres
        ],
        "drapeaux": drapeaux,
        "jeux": {
            "2023": "Phénologie 2023 : 3 sites, 2 points d'écoute par site, 1 an",
            "2026": "Campagne 2026 : 3 sites, ~150 points d'écoute, 1 semaine "
            "(au pic d'activité annuel)",
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


def anuraset_jeu() -> dict:
    """Le jeu AnuraSet réduit : positifs par espèce et par site (fenêtre de 5 s et minute),
    taille de chaque site, et signature des chants des cinq espèces (benchmarks 01 et 07)."""
    ts = pd.read_csv(BENCH / "2026-09-29_anuraset_global" / "donnees" / "transfert_sites.csv")
    ts = ts[(ts["encoder"] == "perch_v2") & (ts["head"] == "logistic")].copy()
    ts["total"] = ts["n_pos"] + ts["n_neg"]
    # Taille d'un site : toutes ses fenêtres (ou minutes) ; une espèce en écarte parfois quelques
    # fichiers (signalée sans chant daté, n° 138), d'où le maximum sur les espèces.
    sites = ts.groupby(["site", "level"])["total"].max().unstack("level")
    especes = pd.read_csv(BENCH / "2026-09-28_anuraset_perch_v2" / "donnees" / "especes.csv")
    # Quantiles des durées de chant (durees_chants.py) : médiane et 90e centile identiques.
    durees = pd.read_csv(ICI / "durees_chants.csv").rename(columns={"espece": "species"})
    especes = especes.merge(durees, on="species", how="left")
    return {
        "sites": [
            {"site": s, "fenetres": int(r["fenetre"]), "minutes": int(r["minute"])}
            for s, r in sites.iterrows()
        ],
        "positifs": lignes(ts, ["species", "site", "level", "n_pos", "total"]),
        "especes": lignes(
            especes,
            [
                "species",
                "n_calls",
                "n_recordings",
                "n_sites",
                "dominant_hz",
                "p10_s",
                "q1_s",
                "mediane_s",
                "q3_s",
                "p90_s",
            ],
        ),
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


# Source de labels qui ne relève pas du plan d'annotation v1 : l'écoute des enregistrements
# écartés par un drapeau.
# Hors plan : drapeaux posés à l'écoute, et labels du détecteur externe, sortis de
# l'entraînement et de l'évaluation (DECISIONS n° 157).
HORS_PLAN = ("flag", "import")


def annotations(config: Path | None) -> dict:
    """Avancement du plan d'annotation v1, compté dans la base locale (lecture seule).

    Entraînement : enregistrements hors jeu gelé ayant au moins un label du plan. Évaluation :
    enregistrements du jeu gelé (toutes versions) ou du jeu de test v1 tiré par plan
    (`files/test_v1/`) ayant au moins un label ; positifs : ceux qui
    ont au moins une fenêtre positive. Dernier label de chaque fenêtre. Sans base, relit
    annotations.json (dernier comptage connu).
    """
    sauvegarde = ICI / "annotations.json"
    try:
        import pandas as pd

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
            labels = current_labels(con)[["recording_id", "label", "source"]]
            # Annotation par intervalles (n° 182) : un extrait écouté, positif s'il porte un
            # intervalle d'A. blanci ; « background » sinon (seul le caractère positif compte).
            spans = pd.read_sql_query(
                "SELECT s.recording_id, COALESCE(i.label, 'background') AS label, s.source "
                "FROM spans s LEFT JOIN intervals i USING (span_id)",
                con,
            )
            labels = pd.concat([labels, spans], ignore_index=True)
            recs = recordings_table(con)[["recording_id", "site", "mic_id"]]
        finally:
            con.close()
        try:
            geles = frozen_recordings(cfg)
        except (OSError, ValueError):
            geles = set()
        # Jeu de test v1 tiré par plan (n° 196), avant son gel : compté en évaluation.
        test_v1 = config_path(cfg, "reports") / "files" / "test_v1" / "candidats.csv"
        if test_v1.exists():
            geles = set(geles) | set(pd.read_csv(test_v1, usecols=["recording_id"])["recording_id"])
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
    micros = (
        par_rec.assign(mic_id=par_rec["mic_id"].fillna("?"))
        .groupby(["site", "mic_id"])["positif"]
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
        "micros": [
            [r.site or "?", r.mic_id, int(r.size), int(r.sum)]
            for r in micros.itertuples(index=False)
        ],
        "compte_le": date.today().isoformat(),
        "source": "base locale",
    }
    sauvegarde.write_text(json.dumps(resultat, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultat


def encodage(config: Path | None, structure: dict) -> dict:
    """Temps d'encodage du corpus ONF par encodeur (`temps_encodage` de structure.yaml).

    Lu dans la base locale : `models.params_json.totals` (cumul de tous les passages d'`embed`,
    durée réelle) et le nombre d'enregistrements encodables (`select_recordings`). Un encodeur
    est « fini » quand son stock couvre tous les encodables. Sans base, ou pour un stock encodé
    avant le cumul, relit encodage.json, puis les valeurs connues de structure.yaml."""
    cfg_t = structure.get("temps_encodage", {})
    noms = [n for n, _ in cfg_t.get("encodeurs", [])]
    sortie = {
        n: dict(cfg_t.get("connus", {}).get(n, {}), nom=nom)
        for n, nom in cfg_t.get("encodeurs", [])
    }
    sauvegarde = ICI / "encodage.json"
    if sauvegarde.exists():
        for n, v in json.loads(sauvegarde.read_text(encoding="utf-8")).items():
            if n in sortie and v.get("statut"):
                sortie[n].update(v)
    try:
        from blanci.core.config import config_path, load_config
        from blanci.embedding.embed import select_recordings

        if config is None and (RACINE / "config" / "local.yaml").exists():
            config = RACINE / "config" / "local.yaml"
        cfg = load_config(config)
        base = config_path(cfg, "db")
        if not base.exists():
            raise FileNotFoundError(base)
        con = sqlite3.connect(f"file:{base.as_posix()}?mode=ro", uri=True)
        try:
            encodables = len(select_recordings(con))
            lignes = con.execute(
                "SELECT name, params_json FROM models WHERE kind = 'encoder'"
            ).fetchall()
        finally:
            con.close()
    except (ImportError, OSError, sqlite3.Error):
        return {"encodeurs": sortie, "source": "dernier relevé connu"}
    mesures = {}
    for nom, params in lignes:
        if nom not in noms or not params:
            continue
        tot = json.loads(params).get("totals")
        if not tot or tot.get("recordings", 0) <= (mesures.get(nom, {}).get("enregistrements", 0)):
            continue
        fini = tot["recordings"] >= encodables
        mesures[nom] = {
            "statut": "fini" if fini else "en_cours",
            "enregistrements": tot["recordings"],
            "encodables": encodables,
            "heures": round(tot["wall_s"] / 3600, 1),
            "s_par_enregistrement": round(tot["wall_s"] / tot["recordings"], 2),
            "source": f"base locale, {date.today().isoformat()}",
        }
    for nom, m in mesures.items():
        if not (sortie[nom].get("statut") == "fini" and m["statut"] != "fini"):
            sortie[nom].update(m)
    sauvegarde.write_text(
        json.dumps(
            {n: v for n, v in sortie.items() if v.get("statut")}, ensure_ascii=False, indent=1
        ),
        encoding="utf-8",
    )
    return {"encodeurs": sortie, "source": "base locale"}


PUBLICATION = ICI / "publication.json"
# Ce que construire.py lit tout seul : rien à faire à la main quand ces chemins changent.
AUTOMATIQUE = (
    "documentation/benchmarks/",
    "documentation/tableaux/",
    "documentation/biblio/biblio.md",
    "documentation/commandes.md",
    "tests/",
    "documentation/tableau-de-bord/",
)
# Ce que le tableau de bord ne montre pas, volontairement.
# resultats/ : sorties brutes des calculs, déjà rassemblées dans les CSV des benchmarks.
IGNORE = (
    "resultats/",
    "DECISIONS.md",
    "uv.lock",
    ".gitignore",
    ".python-version",
    ".claude/",
    "config/",
)


def commandes_cli() -> set[str]:
    """Commandes de `blanci` lues dans cli.py (décorateurs @app.command et sous-commandes)."""
    texte = (RACINE / "blanci" / "cli.py").read_text(encoding="utf-8")
    noms = set()
    for m in re.finditer(r"@(\w+)\.command\((?:\"([\w-]+)\")?[^)]*\)\s*\ndef (\w+)", texte):
        nom = m.group(2) or m.group(3).replace("_", "-")
        noms.add(f"anuraset {nom}" if m.group(1) == "anuraset_app" else nom)
    return noms


def charger_contenu() -> dict:
    """structure.yaml (tenu par Claude), textes.yaml et en_cours.yaml (tenus par Léonard), réunis.
    Les textes d'une étape de la chaîne (nom, détail, repère) viennent de textes.yaml."""
    lire = lambda nom: yaml.safe_load((ICI / nom).read_text(encoding="utf-8"))  # noqa: E731
    contenu, textes = lire("structure.yaml"), lire("textes.yaml")
    textes_chaine = textes.pop("chaine")
    for etape in contenu["chaine"]:
        if etape["id"] not in textes_chaine:
            raise SystemExit(
                f"textes.yaml : pas de textes pour l'étape « {etape['id']} » (chaine:)"
            )
        etape.update(textes_chaine[etape["id"]])
    contenu.update(textes)
    contenu.update(lire("en_cours.yaml"))
    return contenu


def changements() -> str:
    """Rapport en Markdown : ce qui a changé dans le dépôt depuis la dernière publication du
    tableau de bord, rangé selon ce qu'il faut en faire. Lu par le skill tableau-de-bord."""
    structure = charger_contenu()
    base = ""
    if PUBLICATION.exists():
        base = json.loads(PUBLICATION.read_text(encoding="utf-8")).get("commit", "")
    if not base or not git("rev-parse", "--verify", "--quiet", base + "^{commit}").strip():
        base = git("log", "-1", "--format=%h", "--", "documentation/tableau-de-bord/").strip()
    num = numeros(ref_main())
    lignes = [f"# Changements depuis la dernière publication ({base}, n° {num.get(base, '?')})", ""]
    commits = git("log", "--first-parent", "--format=%h\t%ad\t%s", "--date=short", f"{base}..HEAD")
    lignes += ["## Commits", ""]
    lignes += [
        f"- n° {num.get(c.split(chr(9))[0], '?')} · {c.replace(chr(9), ' · ')}"
        for c in commits.splitlines()
    ] or ["- aucun"]

    auto, structure_l, discuter = [], [], []
    for ligne in git("diff", "--name-status", "--find-renames", f"{base}..HEAD").splitlines():
        etat, *chemins = ligne.split("\t")
        chemin = chemins[-1]
        texte = f"{etat[0]} {' → '.join(chemins)}"
        if chemin.startswith(IGNORE):
            continue
        if chemin.startswith(AUTOMATIQUE):
            auto.append(texte)
        elif chemin.startswith("blanci/"):
            if etat[0] in "ADR" or chemin.endswith("cli.py"):
                structure_l.append(texte)
        else:
            discuter.append(texte)

    cli = commandes_cli()
    declarees = {c for e in structure["chaine"] for c in e.get("commandes", [])}
    declarees |= set(structure.get("commandes_hors_chaine", []))
    for e in structure["chaine"]:
        if not (RACINE / e["module"]).exists():
            structure_l.append(f"module disparu : {e['module']} (étape {e['etape']})")
        for c in e.get("commandes", []):
            if c not in cli:
                structure_l.append(f"commande disparue : blanci {c} (étape {e['etape']})")
    nouvelles = sorted(cli - declarees)
    if nouvelles:
        structure_l.append(
            "commandes absentes de la chaîne (à placer dans une étape, ou à laisser si ce sont "
            "des outils de recherche) : " + ", ".join(nouvelles)
        )

    def bloc(titre: str, items: list[str], aide: str) -> list[str]:
        return [f"## {titre}", "", aide, ""] + ([f"- {i}" for i in items] or ["- rien"]) + [""]

    lignes.append("")
    lignes += bloc("Pris en compte tout seul", auto, "Relu par construire.py, rien à faire.")
    lignes += bloc(
        "À reporter dans structure.yaml",
        structure_l,
        "Fichiers de blanci/ ajoutés, supprimés ou renommés, et écarts entre la chaîne et la CLI.",
    )
    lignes += bloc(
        "Sans place dans le tableau de bord",
        discuter,
        "Ajout ou suppression que la page ne montre pas encore : proposer une carte ou une "
        "section, ou demander à Léonard.",
    )
    return "\n".join(lignes)


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--config", type=Path, default=None, help="config/local.yaml par défaut")
    args.add_argument(
        "--changements", action="store_true", help="rapport des changements, sans construire"
    )
    args.add_argument("--publie", action="store_true", help="noter HEAD comme dernière publication")
    opts = args.parse_args()
    if opts.changements:
        print(changements())
        return
    if opts.publie:
        PUBLICATION.write_text(
            json.dumps(
                {
                    "commit": git("rev-parse", "--short", "HEAD").strip(),
                    "date": date.today().isoformat(),
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        return
    config = opts.config
    contenu = charger_contenu()
    fichiers: dict[str, str] = {
        nom: str((ICI / nom).relative_to(RACINE))
        for nom in ("spectrogramme.jpg", "poste-annotation.png")
        if (ICI / nom).exists()
    }
    tests = compter_tests()
    for etape in contenu["chaine"]:
        etape["n_tests"] = tests.get(etape.get("tests", ""), 0)
    donnees = {
        "genere": {"commit": commit_courant(), "date": date.today().isoformat()},
        "contenu": contenu,
        "annotations": annotations(config),
        "encodage": encodage(config, contenu),
        "tests": {"total": sum(tests.values()), "par_dossier": tests},
        "historique": historique(ref_main()),
        "inventaire": inventaire(),
        "rapports": rapports(fichiers),
        "encodeurs": benchmark_encodeurs(),
        "tetes": benchmark_tetes(),
        "anuraset": anuraset_jeu(),
        "tableaux": tableaux(fichiers),
        "biblio": (DOC / "biblio" / "biblio.md").read_text(encoding="utf-8"),
        "poste": "poste-annotation.png" if (ICI / "poste-annotation.png").exists() else None,
        "grenouille": json.loads((ICI / "grenouille.json").read_text(encoding="utf-8"))
        if (ICI / "grenouille.json").exists()
        else None,
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
