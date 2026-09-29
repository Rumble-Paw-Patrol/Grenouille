# Modèle de rapport de benchmark

Un dossier par benchmark : `documentation/benchmarks/AAAA-MM-JJ_<jeu>_<objet>/`, avec
`RAPPORT.md`, `figures/`, `donnees/` (les CSV dont tout est tiré) et `generer.py` (figures et
tableaux recalculés depuis `donnees/`, sans rien relancer d'autre). Exemple :
`2026-09-28_anuraset_perch_v2/`.

## Règles

- Clair et court : une idée par phrase, pas de chiffre qui ne sert pas une conclusion. Le
  détail va dans `donnees/`, pas dans le texte.
- Tableaux lourds en PNG (Xcode rend mal les tableaux Markdown) ; un tableau Markdown seulement
  s'il tient en 3–5 lignes.
- Toujours deux lectures de l'AP : poolée (tous les groupes ensemble) et par groupe tenu à
  l'écart ; écarts appariés avec correction de Holm, et dire si le bootstrap est optimiste.
- Séparer ce qu'on mesure (Résultats) de ce qu'on suppose (Hypothèses), et chaque hypothèse
  s'appuie sur un chiffre ou propose un test.
- Statut en tête : indicateur ou décision.
- Un encodeur ne s'écarte qu'avec la tête qui correspond à sa sortie (jetons et sondage
  attentif ou par prototypes pour les transformers auto-supervisés) et après le témoin BOAFAB ;
  sinon il est « en retrait en sondage linéaire », pas « écarté » (`encodeurs-bacpipe.md`,
  « Règle », n° 151).

## Plan

```
# Benchmark NN — <objet> (<jeu>, <encodeur>)
date · commit · statut

## En bref            3 à 5 puces : ce qu'on retient, avec le chiffre qui le porte
## 1. Question        une phrase
## 2. Données         jeu, sélection, volumes ; figure de la composition (où sont les positifs)
## 3. Pipeline et choix   tableau Étape | Choix | Pourquoi (renvoi aux DECISIONS)
## 4. Résultats       tableau principal (PNG), 2 à 3 figures, légendes d'une ligne
## 5. Hypothèses      numérotées, chacune étayée ou avec son test
## 6. Ce qu'on en retient pour A. blanci
## 7. Limites
## 8. Suites          3 au plus, par ordre d'intérêt
## Annexes            corrections, variantes écartées, reproduire (commandes, durées)
```
