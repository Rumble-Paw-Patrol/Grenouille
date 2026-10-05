---
name: audit
description: Audit complet du dépôt Grenouille (code blanci/, tests, benchmarks, documentation, notebooks) en lecture seule, chaque constat vérifié dans le code, puis rapport documentation/audit-<date>.md au format de l'audit du 28/09. À utiliser quand Léonard demande un audit, une relecture d'ensemble, « cherche les bugs partout », ou avant une réunion de suivi ou un jalon (choix de l'encodeur, livraison).
---

# Audit du dépôt

Modèle : `documentation/audit-28-septembre.md`. Le relire avant de commencer : même ton,
mêmes sections, mêmes tableaux.

## 1. Préparer

- `git log --oneline` depuis le dernier audit (`documentation/audit-*.md`, le plus récent) :
  l'audit porte d'abord sur ce qui a changé depuis, puis sur le reste.
- État de départ, noté tel quel dans le rapport :
  `uv run ruff check .` et `uv run pytest -m "not slow" -q` (≈ 3 min 30). Si un test échoue
  avant l'audit, c'est un constat, pas une raison de s'arrêter.

## 2. Lire, par axes

Un sous-agent `Explore` par axe, lancés ensemble, **en lecture seule** (aucun ne modifie de
fichier). Chacun rend une liste de constats : fichier:ligne, ce qui ne va pas, scénario
concret qui le déclenche, gravité.

| Axe | Où regarder | Ce qu'on cherche |
|---|---|---|
| Entrées | `blanci/inputs/`, `blanci/core/` | inventaire, drapeaux, import des labels, chemins, config |
| Encodage | `blanci/embedding/` | stocks mélangés, reprise, rééchantillonnage, ONNX / OpenVINO |
| Têtes | `blanci/heads/` | fuites entre plis, régularisations mal branchées, numérique (float32, NaN, inf) |
| Évaluation | `blanci/evaluation/` | métriques optimistes : plis, bootstrap, seuil choisi sur les scores évalués, Holm |
| Fusion, résultats | `blanci/combination/`, `blanci/results/` | descripteurs manquants, empilement non imbriqué, agrégation |
| Annotation, interfaces | `blanci/annotation/`, `blanci/cli.py`, `blanci/service.py` | labels écrasés, commandes incohérentes entre elles |
| Benchmarks | `anuraset/`, `documentation/benchmarks/` | chiffres du rapport ≠ `donnees/`, `generer.py` qui ne tourne plus, conclusions plus fortes que les données |
| Documentation | `README.md`, `documentation/*.md`, `notebooks/`, `documentation/tableaux/` | doc qui contredit le code, commandes qui n'existent plus, chiffres périmés, liens morts |

## 3. Vérifier chaque constat

Un constat n'entre dans le rapport qu'après vérification dans le code (lecture, ou petit test
qui le reproduit). Les faux positifs sont écartés sans bruit. Ne rien corriger à cette étape.

## 4. Proposer, puis corriger sur accord

Présenter à Léonard la liste vérifiée, par gravité, et demander ce qu'il veut corriger
(tout, une partie, rien). Pour chaque correction acceptée : correctif minimal, test qui
échouait avant et passe après. À la fin : `ruff` et suite rapide de nouveau, chiffres notés.

## 5. Rapport

`documentation/audit-<jour>-<mois en lettres>.md` (ex. `audit-12-octobre.md`) :

```
# Audit du dépôt — JJ/MM/AAAA

Comment l'audit a été fait (axes, nombre d'agents), commit de départ, commit des corrections,
suite de tests (réussis / échoués / sautés), ruff.

## 1. Erreurs trouvées et corrigées      tableaux Problème | Effet | Correction, groupés par thème
## 2. Erreurs trouvées, non corrigées    tableau Point | Pourquoi
## 3. À savoir                           conséquences pour les résultats déjà publiés : chiffres à
                                         relire, modèles à réapprendre, stocks à réencoder
```

Règles d'écriture : français, phrases courtes, un chiffre seulement s'il porte une
conclusion, virgule décimale, nombres à espaces fines (1 813). Les rapports s'adressent à
Léonard et à ses encadrants : pas de renvoi à `DECISIONS.md`, la raison est écrite dans la
phrase elle-même.

Dire à la fin, en une ligne, si un rapport de benchmark ou le tableau de bord
(`documentation/tableau-de-bord/`) doit être régénéré à cause d'une correction.
