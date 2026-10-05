---
name: audit-n-from
description: Audit du dépôt Grenouille en lecture seule sur une plage de commits choisie, avec un nombre de sous-agents choisi — /audit-n-from <N> <de> [<à>] [sur <chemin>], ex. « /audit-n-from 2 30 34 », « /audit-n-from 0 dernier-audit », « /audit-n-from 8 debut ». Chaque constat vérifié dans le code, corrections seulement sur accord, rapport documentation/audit-<date>.md. Lancé uniquement par Léonard.
disable-model-invocation: true
---

# Audit sur une plage de commits

Modèle du rapport : `documentation/audit-28-septembre.md` (même ton, mêmes sections).

## 0. Arguments : jamais de valeur par défaut

`/audit-n-from <N> <de> [<à>] [sur <chemin>…]`

| Argument | Valeurs |
|---|---|
| `N` | nombre de sous-agents, 0 à 8 (0 = je lis moi-même, sans sous-agent) |
| `de` | numéro de commit (`30`), hash (`a5ca347`), date (`2026-10-01`), `dernier-audit`, `debut` |
| `à` | mêmes formes ; absent = le dernier commit (HEAD) |
| `sur` | un ou plusieurs dossiers ou fichiers pour restreindre encore (`sur blanci/heads tests/heads`) |

**Si `N` ou `de` manque, ne rien lancer** : demander les deux en une question, en rappelant
les derniers numéros de commit (`git log` ci-dessous). Ne jamais auditer tout le dépôt sans
que Léonard ait écrit `debut`.

Résoudre les bornes :
- numéros de commit = position le long de la branche courante, premier parent, n° 1 = premier
  commit ; ce sont ceux du tableau de bord et de
  `git rev-list --reverse --first-parent HEAD | nl`. Le n° k est la k-ième ligne ;
- `de` est inclus : la plage est `<parent du commit de>..<commit à>` ;
- date `de` = premier commit de ce jour ou après ; date `à` = dernier commit de ce jour ou avant ;
- `dernier-audit` = le commit noté en tête du dernier `documentation/audit-*.md` ;
- `debut` = tout le dépôt tel qu'il est à la borne `à` (pas de diff).

Annoncer avant de commencer, en une ligne : « Audit à N agents, commits n° a → n° b (k
commits, f fichiers), sur … ». Si f > 60 avec N ≤ 2, prévenir que ce sera long et proposer de
restreindre avec `sur`.

## 1. Ce qu'on audite

Les fichiers modifiés dans la plage (`git diff --name-only <de>^..<à>`, filtrés par `sur`),
lus dans leur état à la borne `à`, plus ce qui les appelle directement quand un changement
de signature ou de comportement peut casser l'appelant. Le diff lui-même dit ce qui a changé :
lire d'abord les messages de commit de la plage pour savoir ce que chaque changement voulait
faire, puis chercher où il ne le fait pas.

État de départ, noté dans le rapport : `uv run ruff check .` et
`uv run pytest -m "not slow" -q`, ou seulement les tests des dossiers touchés si N ≤ 2.

## 2. Répartir entre N agents

Ranger les fichiers par axe, puis répartir les axes non vides en N lots de taille voisine,
axes voisins ensemble. Agents `Explore`, lancés ensemble, en lecture seule. Chaque constat :
fichier:ligne, ce qui ne va pas, scénario concret qui le déclenche, gravité, commit en cause.

| N° | Axe | Où | Ce qu'on cherche |
|---|---|---|---|
| 1 | Entrées | `blanci/inputs/`, `blanci/core/` | inventaire, drapeaux, import des labels, chemins, config |
| 2 | Encodage | `blanci/embedding/` | stocks mélangés, reprise, rééchantillonnage, ONNX / OpenVINO |
| 3 | Têtes | `blanci/heads/` | fuites entre plis, régularisations mal branchées, numérique |
| 4 | Évaluation | `blanci/evaluation/` | métriques optimistes : plis, bootstrap, seuil choisi sur les scores évalués, Holm |
| 5 | Fusion, résultats | `blanci/combination/`, `blanci/results/` | descripteurs manquants, empilement non imbriqué, agrégation |
| 6 | Annotation, interfaces | `blanci/annotation/`, `blanci/cli.py`, `blanci/service.py` | labels écrasés, commandes incohérentes |
| 7 | Benchmarks | `anuraset/`, `documentation/benchmarks/` | chiffres du rapport ≠ `donnees/`, conclusions plus fortes que les données |
| 8 | Documentation | `README.md`, `documentation/`, `notebooks/` | doc qui contredit le code, commandes disparues, chiffres périmés |

Les tests (`tests/`) vont avec l'axe du code qu'ils testent.

## 3. Vérifier, proposer, corriger sur accord

- Un constat n'entre dans la liste qu'après vérification dans le code (lecture ou petit test
  qui le reproduit). Faux positifs écartés sans bruit.
- Présenter la liste vérifiée, par gravité, et demander ce qu'il faut corriger.
- Pour chaque correction acceptée : correctif minimal et test qui échouait avant. À la fin,
  `ruff` et tests de nouveau.

## 4. Rapport

`documentation/audit-<jour>-<mois en lettres>.md` :

```
# Audit du dépôt — JJ/MM/AAAA

Plage : commits n° a → n° b (hash a → hash b), sur …, N agents. Commit des corrections.
Tests (réussis / échoués / sautés), ruff.

## 1. Erreurs trouvées et corrigées      Problème | Effet | Correction, par thème
## 2. Erreurs trouvées, non corrigées    Point | Pourquoi
## 3. À savoir                           résultats publiés à relire, modèles à réapprendre…
```

Français, phrases courtes, virgule décimale. Pas de renvoi à `DECISIONS.md`. Dire en une
ligne si un rapport de benchmark ou le tableau de bord (`/tableau-de-bord`) est à régénérer.
