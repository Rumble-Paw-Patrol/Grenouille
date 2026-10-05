---
name: audit
description: Audit du dépôt Grenouille (code blanci/, tests, benchmarks, documentation, notebooks) en lecture seule, chaque constat vérifié dans le code, puis rapport documentation/audit-<date>.md au format de l'audit du 28/09. Argument facultatif — nombre de sous-agents (0 à 8, défaut 8) et portée (« tout » ou « depuis » le dernier audit), ex. « /audit 2 depuis ». À utiliser quand Léonard demande un audit, une relecture d'ensemble, « cherche les bugs partout », ou avant une réunion de suivi ou un jalon.
---

# Audit du dépôt

Modèle : `documentation/audit-28-septembre.md`. Le relire avant de commencer : même ton,
mêmes sections, mêmes tableaux.

## 0. Lire les arguments

`/audit [N] [tout|depuis]` :

- **N** = nombre de sous-agents, de 0 à 8. Sans N : 8. Hors bornes : ramener dans 0–8.
- **portée** : `tout` (défaut avec N ≥ 4) ou `depuis` (défaut avec N ≤ 3) = seulement les
  fichiers modifiés depuis le dernier audit
  (`git diff --name-only <commit du dernier audit>..HEAD`, le commit est noté en tête du
  dernier `documentation/audit-*.md` ; à défaut, sa date de création dans `git log`).

Annoncer en une ligne avant de commencer : « Audit à N agents, portée X ». Ce qui coûte le
plus, c'est la quantité de code relu, pas le nombre d'agents : moins d'agents relisent autant
de code, en série plutôt qu'en parallèle. Pour vraiment économiser, la portée `depuis` compte
plus que N ; le dire à Léonard s'il demande peu d'agents sur tout le dépôt.

## 1. Préparer

- État de départ, noté tel quel dans le rapport :
  `uv run ruff check .` et `uv run pytest -m "not slow" -q` (≈ 3 min 30). Si un test échoue
  avant l'audit, c'est un constat, pas une raison de s'arrêter.
- Portée `depuis` : lister les fichiers touchés et ne garder que les axes qui en contiennent.

## 2. Lire, par axes

Les huit axes ci-dessous. Avec N agents, les répartir en N lots de taille voisine, en gardant
ensemble les axes voisins dans cet ordre (ex. N = 2 : axes 1–4 et 5–8 ; N = 3 : 1–3, 4–5, 6–8).
Les agents sont des `Explore`, lancés ensemble, **en lecture seule**. Avec N = 0, lire
soi-même, axe par axe, sans sous-agent.

Chaque agent (ou chaque axe lu soi-même) rend une liste de constats : fichier:ligne, ce qui ne
va pas, scénario concret qui le déclenche, gravité.

| N° | Axe | Où regarder | Ce qu'on cherche |
|---|---|---|---|
| 1 | Entrées | `blanci/inputs/`, `blanci/core/` | inventaire, drapeaux, import des labels, chemins, config |
| 2 | Encodage | `blanci/embedding/` | stocks mélangés, reprise, rééchantillonnage, ONNX / OpenVINO |
| 3 | Têtes | `blanci/heads/` | fuites entre plis, régularisations mal branchées, numérique (float32, NaN, inf) |
| 4 | Évaluation | `blanci/evaluation/` | métriques optimistes : plis, bootstrap, seuil choisi sur les scores évalués, Holm |
| 5 | Fusion, résultats | `blanci/combination/`, `blanci/results/` | descripteurs manquants, empilement non imbriqué, agrégation |
| 6 | Annotation, interfaces | `blanci/annotation/`, `blanci/cli.py`, `blanci/service.py` | labels écrasés, commandes incohérentes entre elles |
| 7 | Benchmarks | `anuraset/`, `documentation/benchmarks/` | chiffres du rapport ≠ `donnees/`, `generer.py` qui ne tourne plus, conclusions plus fortes que les données |
| 8 | Documentation | `README.md`, `documentation/*.md`, `notebooks/`, `documentation/tableaux/`, `documentation/tableau-de-bord/` | doc qui contredit le code, commandes qui n'existent plus, chiffres périmés, liens morts |

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

Comment l'audit a été fait (N agents, portée, axes), commit de départ, commit des
corrections, suite de tests (réussis / échoués / sautés), ruff.

## 1. Erreurs trouvées et corrigées      tableaux Problème | Effet | Correction, groupés par thème
## 2. Erreurs trouvées, non corrigées    tableau Point | Pourquoi
## 3. À savoir                           conséquences pour les résultats déjà publiés : chiffres à
                                         relire, modèles à réapprendre, stocks à réencoder
```

Règles d'écriture : français, phrases courtes, un chiffre seulement s'il porte une
conclusion, virgule décimale, nombres à espaces fines (1 813). Les rapports s'adressent à
Léonard et à ses encadrants : pas de renvoi à `DECISIONS.md`, la raison est écrite dans la
phrase elle-même.

Dire à la fin, en une ligne, si un rapport de benchmark ou le tableau de bord doit être
régénéré à cause d'une correction (skill `tableau-de-bord`).
