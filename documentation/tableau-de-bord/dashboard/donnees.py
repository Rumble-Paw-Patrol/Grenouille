"""Écrit les fichiers de données du dashboard claude.ai dans dashboard/donnees/ (CSV et JSON),
depuis le dépôt, avec les fonctions de construire.py. Voir LISEZMOI.md.

    python documentation/tableau-de-bord/dashboard/donnees.py
"""
import csv, json, re, sys
from pathlib import Path

ICI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ICI))
import construire as c  # noqa: E402
import yaml  # noqa: E402

OUT = Path(__file__).resolve().parent / "donnees"
OUT.mkdir(exist_ok=True)


def ecrire_csv(nom, rows, cols):
    with open(OUT / f"{nom}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(k) is None else r.get(k) for k in cols])


def ecrire_json(nom, rows):
    (OUT / f"{nom}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")


contenu = c.charger_contenu()
lieux = contenu["lieux"]
ETATS = {"fait": "fait", "termine": "terminé", "en_cours": "en cours", "a_venir": "à venir", "bloque": "bloqué"}
TETES = {
    "logistic": "logistique", "proto_probe": "sonde à prototypes (jetons)", "attentive": "sonde attentive (jetons)",
    "logistic:max": "logistique sur le max des jetons", "simple_prototype": "prototype simple", "lda_shrunk": "LDA rétrécie",
    "logistic+R37=glmm": "logistique + biais par site (GLMM)", "logistic+R37": "logistique + biais par site", "prototype": "prototype différentiel",
    "knn:k=5": "5 plus proches voisins", "dann": "DANN (adversaire du site)", "loss:focal": "logistique, perte focale",
    "logistic+R13": "logistique + poids par micro", "logistic+R18=64": "logistique + ACP à 64 dimensions", "logistic+R19": "logistique + centrage par point",
    "logistic+R19+R37": "logistique + centrage par point + biais par site", "logistic+R20": "logistique + AdaBN par site",
    "logistic+R21": "logistique + retrait des directions du site", "classifieur_origine": "classifieur d'origine (sans entraînement)",
}
nom_enc = lambda e: re.sub(r"^esp_aves2_", "aves2 ", e)

# 1. Inventaire
inv = c.inventaire()
ecrire_csv("inventaire", [dict(x, lieu=lieux.get(x["site"], x["site"])) for x in inv["sites"]],
           ["jeu", "site", "lieu", "enregistrements", "heures", "go"])
ENC_FEN = {"3 s": "BirdNET", "5 s": "Perch", "6 s": "autres"}
ecrire_csv("fenetres", [{"fenetre": a, "pas": b, "encodeurs": ENC_FEN.get(a, ""), "par_enregistrement": n, "total": t}
                        for a, b, n, t in inv["fenetres"]], ["fenetre", "pas", "encodeurs", "par_enregistrement", "total"])
d = inv["drapeaux"]
nb = lambda s: int(s.replace(" ", "").replace(" ", "").replace("\xa0", ""))
m = re.search(r"(\d[\d  ]*) de durée anormale, (\d[\d  ]*) hors relevé et (\d[\d  ]*) micros allumés dans le sac ou étouffés ; (\d[\d  ]*) enregistrements restent encodables", d)
ecrire_csv("controle_qualite", [
    {"motif": "Durée anormale", "decision": "écarté", "enregistrements": nb(m[1])},
    {"motif": "Hors relevé", "decision": "écarté", "enregistrements": nb(m[2])},
    {"motif": "Micro allumé dans le sac ou étouffé", "decision": "écarté", "enregistrements": nb(m[3])},
    {"motif": "Encodable", "decision": "gardé", "enregistrements": nb(m[4])},
], ["motif", "decision", "enregistrements"])

# 2. Chaîne et tests
tests = c.compter_tests()
chaine = []
for e in contenu["chaine"]:
    chaine.append({
        "id": e["id"], "etape": e["etape"], "groupe": e["groupe"], "etat": ETATS[e["etat"]],
        "optionnel": bool(e.get("optionnel", False)), "module": e["module"],
        "commandes": " · ".join("blanci " + x for x in e.get("commandes", [])),
        "dossier_tests": e["tests"], "tests": tests.get(e["tests"], 0),
        "suite": ",".join(e.get("suite", [])), "repere": e.get("chiffre", ""), "resume": e["resume"].strip(),
    })
ecrire_json("chaine", chaine)
ecrire_csv("tests", [{"dossier": k, "tests": v} for k, v in sorted(tests.items(), key=lambda kv: -kv[1])], ["dossier", "tests"])

# 3. Annotation (dernier comptage) et encodage
ann = json.loads((ICI / "annotations.json").read_text(encoding="utf-8"))
JEU = {"entrainement": "entraînement", "evaluation": "évaluation"}
ecrire_csv("annotation_sites", [{"site": s, "jeu": JEU.get(j, j), "annotes": n, "positifs": p, "compte_le": ann["compte_le"]}
                                for s, j, n, p in ann["sites"]], ["site", "jeu", "annotes", "positifs", "compte_le"])
ecrire_csv("annotation_micros", [{"site": s, "micro": mic, "annotes": n, "positifs": p} for s, mic, n, p in ann["micros"]],
           ["site", "micro", "annotes", "positifs"])
enc = c.encodage(None, contenu)["encodeurs"]
ETAT_ENC = {"fini": "fini", "en_cours": "en cours", None: "pas encore encodé"}
ecrire_csv("encodage", [{"encodeur": k, "nom": v["nom"], "statut": ETAT_ENC[v.get("statut")], "heures": v.get("heures"),
                         "enregistrements": v.get("enregistrements"), "encodables": v.get("encodables"),
                         "s_par_enregistrement": v.get("s_par_enregistrement"), "source": v.get("source", "")}
                        for k, v in enc.items()],
           ["encodeur", "nom", "statut", "heures", "enregistrements", "encodables", "s_par_enregistrement", "source"])

# 4. Chantiers et commits
ecrire_json("chantiers", [{"rang": i + 1, "titre": t["titre"].strip(), "etat": ETATS.get(t["etat"], t["etat"]),
                           "detail": re.sub(r"\s+", " ", t.get("detail", "")).strip(),
                           "suite": re.sub(r"\s+", " ", t.get("suite", "") or "").strip()}
                          for i, t in enumerate(contenu["en_cours"])])
ecrire_csv("commits", [{"numero": n, "commit": h, "date": dt, "message": msg} for h, dt, msg, n in c.historique(c.ref_main())],
           ["numero", "commit", "date", "message"])

# 5. Repères (cibles d'annotation, AnuraSet complet, A. blanci)
cib = contenu["annotation_cibles"]
A = contenu["anuraset"]
ecrire_csv("reperes", [
    {"cle": "cible_entrainement", "valeur": cib["entrainement"], "unite": "enregistrements", "libelle": "Extraits de 30 s à annoter pour l'entraînement (premier lot)", "source": "structure.yaml"},
    {"cle": "cible_evaluation", "valeur": cib["evaluation"], "unite": "enregistrements", "libelle": "Enregistrements de 2 min à écouter en entier pour le test", "source": "structure.yaml"},
    {"cle": "cible_positifs_evaluation", "valeur": cib["positifs_evaluation"], "unite": "enregistrements", "libelle": "Positifs visés dans le jeu de test", "source": "structure.yaml"},
    {"cle": "anuraset_enregistrements", "valeur": A["total_enregistrements"], "unite": "enregistrements", "libelle": "Enregistrements d'une minute du jeu AnuraSet complet", "source": "textes.yaml"},
    {"cle": "anuraset_especes", "valeur": A["total_especes"], "unite": "espèces", "libelle": "Espèces du jeu AnuraSet complet", "source": "textes.yaml"},
    {"cle": "blanci_note_s", "valeur": 0.09, "unite": "s", "libelle": "Durée d'une note d'A. blanci", "source": "textes.yaml (chaîne, traitement du signal)"},
    {"cle": "blanci_bande_basse_khz", "valeur": 4.4, "unite": "kHz", "libelle": "Bas de la bande de fréquence du chant d'A. blanci", "source": "textes.yaml (chaîne, traitement du signal)"},
    {"cle": "blanci_bande_haute_khz", "valeur": 5.5, "unite": "kHz", "libelle": "Haut de la bande de fréquence du chant d'A. blanci", "source": "textes.yaml (chaîne, traitement du signal)"},
], ["cle", "valeur", "unite", "libelle", "source"])

# 6. AnuraSet : jeu réduit
J = c.anuraset_jeu()
NOMS_ESP = {"DENMIN": "Dendropsophus minutus", "PITAZU": "Pithecopus azureus", "PHYCUV": "Physalaemus cuvieri",
            "LEPLAT": "Leptodactylus latrans", "BOAFAB": "Boana faber"}
ecrire_csv("anuraset_sites", J["sites"], ["site", "fenetres", "minutes"])
ecrire_csv("anuraset_positifs", [{"espece": sp, "site": s, "unite": {"fenetre": "fenêtre"}.get(u, u), "positifs": n, "total": t} for sp, s, u, n, t in J["positifs"]],
           ["espece", "site", "unite", "positifs", "total"])
ecrire_csv("anuraset_especes", [{"espece": sp, "nom": NOMS_ESP[sp], "chants": n, "enregistrements": r, "sites": s,
                                 "frequence_hz": round(hz, 1), "p10_s": p10, "q1_s": q1, "mediane_s": med, "q3_s": q3, "p90_s": p90}
                                for sp, n, r, s, hz, p10, q1, med, q3, p90 in J["especes"]],
           ["espece", "nom", "chants", "enregistrements", "sites", "frequence_hz", "p10_s", "q1_s", "mediane_s", "q3_s", "p90_s"])

# 7. Benchmarks
B = c.benchmark_encodeurs()
rows = []
for e in B["encodeurs"]:
    rows.append({"encodeur": e["encoder"], "nom": nom_enc(e["encoder"]), "famille": e["famille"], "fenetre_s": e["fenetre_s"],
                 "dimension": None if e["dim"] is None else int(e["dim"]), "fenetres_par_s": e["fenetres_par_s"],
                 "licence": e["licence"], "libre": e["libre"], "meilleure_tete_id": e.get("best_head"),
                 "meilleure_tete": TETES.get(e.get("best_head"), e.get("best_head")),
                 "ap_site_meilleure": None if e.get("ap_site_best") is None else round(e["ap_site_best"], 4),
                 "ap_site_logistique": None if e.get("ap_site_logistic") is None else round(e["ap_site_logistic"], 4),
                 "tetes_essayees": None if e.get("heads_tried") is None else int(e["heads_tried"])})
ecrire_csv("encodeurs", rows, list(rows[0].keys()))
UNITE = {"fenetre": "fenêtre", "minute": "minute"}
ecrire_csv("transfert", [{"encodeur": a, "espece": b, "tete": h, "unite": UNITE.get(l, l), "ap_poolee": ap, "ap_site": aps}
                         for a, b, h, l, ap, aps in B["transfert"]], ["encodeur", "espece", "tete", "unite", "ap_poolee", "ap_site"])
ecrire_csv("comparaisons", [{"encodeur": a, "tete": h, "espece": sp, "unite": UNITE.get(l, l), "ecart": df, "ic_bas": lo, "ic_haut": hi,
                             "p_holm": p, "significatif": bool(s)} for a, h, sp, l, df, lo, hi, p, s in B["comparaisons"]],
           ["encodeur", "tete", "espece", "unite", "ecart", "ic_bas", "ic_haut", "p_holm", "significatif"])
ecrire_csv("amorcage", [{"encodeur": a, "tete": h, "k": k, "annotations": "tout" if k == -1 else str(k), "ap_fenetre": w, "ap_minute": mn}
                        for a, h, k, w, mn in B["courbe"]], ["encodeur", "tete", "k", "annotations", "ap_fenetre", "ap_minute"])
UNITE2 = {"fenetre": "fenêtre", "minute": "enregistrement"}
ecrire_csv("tetes_01_06", [{"encodeur": a, "espece": sp, "tete": h, "unite": UNITE2.get(l, l), "ap_poolee": ap, "ap_site": aps,
                            "ecart_logistique": df, "significatif": "" if s is None else bool(s)}
                           for a, sp, h, l, ap, aps, df, s in c.benchmark_tetes()],
           ["encodeur", "espece", "tete", "unite", "ap_poolee", "ap_site", "ecart_logistique", "significatif"])

# 8. Rapports, tableaux, biblio
DEPOT = contenu["projet"]["depot"]
ecrire_csv("rapports", [{"numero": r["numero"], "titre": re.sub(r"^Benchmark \d+ — ", "", r["titre"]), "date": r["date"],
                         "statut": r["statut"], "figures": len(r["figures"]), "donnees": len(r["donnees"]), "dossier": r["id"],
                         "lien": f"{DEPOT}/tree/main/documentation/benchmarks/{r['id']}"} for r in c.rapports({})],
           ["numero", "titre", "date", "statut", "figures", "donnees", "dossier", "lien"])
tab = c.tableaux({})
ecrire_csv("tableaux", [{"tableau": t["nom"], "fichier": t["image"].split("/")[-1], "description": t["description"],
                         "lien": f"{DEPOT}/blob/main/documentation/{t['image']}"} for t in tab],
           ["tableau", "fichier", "description", "lien"])

bib = (c.DOC / "biblio" / "biblio.md").read_text(encoding="utf-8")
refs, section, cur = [], "", None
for ligne in bib.splitlines():
    if ligne.startswith("## "):
        section = ligne[3:].strip()
        continue
    if ligne.startswith("- "):
        if cur:
            refs.append(cur)
        cur = {"section": section, "brut": ligne[2:].strip()}
    elif cur and ligne.startswith("  ") and ligne.strip():
        cur["brut"] += " " + ligne.strip()
    elif cur and not ligne.strip():
        refs.append(cur)
        cur = None
if cur:
    refs.append(cur)
out = []
for i, r in enumerate(refs, 1):
    brut = r["brut"]
    liens = re.findall(r"<(https?://[^>]+)>", brut)
    titre, reste = "", brut
    i = brut.find("** — ")
    if brut.startswith("**") and i > 0:
        titre, reste = brut[: i + 2].replace("**", ""), brut[i + 5 :]
    texte = re.sub(r"\s*·?\s*<https?://[^>]+>\s*·?", " ", reste)
    texte = texte.replace("**", "").replace("*", "").replace("`", "").lstrip(". ")
    texte = re.sub(r"\s+([.,])", r"\1", re.sub(r"\s+", " ", texte)).strip(" —-")
    out.append({"id": i, "section": r["section"], "titre": titre.strip(), "texte": texte.strip(), "liens": " ".join(liens)})
ecrire_json("biblio", out)

for p in sorted(OUT.iterdir()):
    print(f"{p.name:28} {p.stat().st_size:>8}")
