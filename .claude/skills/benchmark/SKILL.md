---
name: benchmark
description: Mène un benchmark du projet Grenouille de bout en bout (encodeurs, têtes, régularisations, fusion… sur AnuraSet ou sur les données ONF) et le documente dans documentation/benchmarks/ selon MODELE_RAPPORT.md (RAPPORT.md, donnees/, figures/, generer.py), puis met à jour l'index, le tableau PNG et le tableau de bord. À utiliser quand Léonard demande de lancer, refaire, compléter ou rédiger un benchmark, ou de comparer des encodeurs ou des têtes.
---

# Benchmark

Références à lire avant de commencer :
- `documentation/benchmarks/MODELE_RAPPORT.md` (plan et règles du rapport) ;
- le rapport le plus proche de la demande, comme exemple
  (`2026-09-30_anuraset_encodeurs/` pour des encodeurs, `2026-09-28_anuraset_perch_v2/` pour
  des têtes et régularisations) ;
- `documentation/commandes.md`, section « Benchmarks », pour les commandes.

## 1. Cadrer

Écrire la question en une phrase et la faire valider par Léonard avant de lancer quoi que ce
soit de long : jeu (AnuraSet ou ONF), encodeurs, têtes ou régularisations, unité (fenêtre,
minute ou enregistrement), référence **fixée d'avance**.

Garde-fous du projet, à vérifier avant de lancer :
- **Données ONF : aucun benchmark sans le go d'Élodie et Benoît** sur les annotations. Sans
  go, s'arrêter et le dire.
- Après le choix de l'encodeur (20/11/2026), plus de benchmark d'encodeur.
- Un encodeur ne s'écarte qu'avec la tête qui correspond à sa sortie (jetons → sonde à
  prototypes ou attentive) et après le témoin BOAFAB ; sinon « en retrait en sondage
  linéaire ».
- Encodeur non libre : comparé au meilleur libre, jamais livré.

## 2. Lancer

Commandes `blanci` (ou `uv run blanci --config anuraset/anuraset.yaml anuraset …` pour
AnuraSet). Les calculs longs tournent en arrière-plan ; noter la commande exacte, la durée et
le commit pour l'annexe « Reproduire ».

## 3. Dossier du rapport

`documentation/benchmarks/AAAA-MM-JJ_<jeu>_<objet>/` :

| Fichier | Contenu |
|---|---|
| `donnees/*.csv` | tout ce dont le rapport tire un chiffre ; colonnes en anglais comme les CSV existants (`encoder`, `species`, `head`, `level`, `ap`, `ap_site`…) |
| `generer.py` | figures et tableaux recalculés depuis `donnees/` seul, sans rien relancer |
| `figures/N_nom.png` | numérotées dans l'ordre du rapport ; tableau lourd en PNG |
| `RAPPORT.md` | plan de `MODELE_RAPPORT.md` |

Figures : charger le skill `dataviz` avant d'écrire `generer.py`. Garder les couleurs des
rapports précédents pour les mêmes encodeurs.

## 4. Rédiger `RAPPORT.md`

Suivre le plan du modèle (En bref, Question, Données, Pipeline et choix, Résultats,
Hypothèses, Ce qu'on en retient pour A. blanci, Limites, Suites, Annexes) et ses règles :
- en-tête : date · commit · statut (**indicateur** ou **décision**) ;
- toujours les deux lectures de l'AP (poolée, et moyenne par groupe tenu à l'écart) ; écarts
  appariés, correction de Holm, et dire si le bootstrap est optimiste ;
- Résultats (mesuré) séparés des Hypothèses (supposé, chacune avec son chiffre ou son test) ;
- colonne « Pourquoi » du tableau Pipeline : la raison en une phrase, pas de renvoi à
  `DECISIONS.md` (les lecteurs ne lisent pas ce fichier) ;
- français, phrases courtes, virgule décimale, un chiffre seulement s'il porte une conclusion.

Relire chaque chiffre du texte contre `donnees/` avant de rendre.

## 5. Mettre à jour ce qui en dépend

1. `documentation/benchmarks/LISEZMOI.md` : une ligne dans le tableau (N°, dossier, objet,
   statut).
2. `documentation/tableaux/generer.py` : si le benchmark introduit des têtes, encodeurs ou
   régularisations nouveaux, compléter la liste `TABLEAUX`, puis
   `uv run --group notebook python documentation/tableaux/generer.py`.
3. Tableau de bord : `uv run python documentation/tableau-de-bord/construire.py`, puis
   republier l'artifact (URL dans `documentation/tableau-de-bord/LISEZMOI.md`) avec l'outil
   Artifact, `url` = cette URL, et les nouvelles figures dans `files`.
4. `README.md` « Où en est le projet » si la conclusion change l'état du projet (meilleur
   libre, encodeur retenu).

## 6. Rendre

Résumer à Léonard en 3 à 5 lignes : la question, la réponse avec son chiffre, le statut, ce
qui reste à faire. Ne pas committer sans qu'il le demande.
