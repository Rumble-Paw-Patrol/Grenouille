# Audit du dépôt — 28/09/2026

Audit réalisé par 8 sous-agents en lecture seule. Chaque constat a été vérifié dans le code avant
d'être corrigé. Corrections sur `main` (fusion `daac868`), décisions n° 138 à 140 et 142 à 146
de `DECISIONS.md` (le n° 141 est la campagne AnuraSet refaite). Suite de tests : 842 réussis,
ruff propre.

## 1. Erreurs trouvées et corrigées

### AnuraSet (touchait la campagne du n° 137)

| Problème | Effet | Correction |
|---|---|---|
| Fenêtres jointives : un chant à cheval sur la jonction rendait les deux fenêtres NaN | Chants perdus, biais vers les chants centrés | La fenêtre qui porte la plus grande part du chant devient positive. Positives : DENMIN 1 813 → 2 001, PHYCUV 905 → 984, PITAZU 1 415 → 1 475, BOAFAB 1 814 → 1 857 (n° 138) |
| Fichier gardé pour une espèce, qui en signale une autre sans chant daté : compté négatif pour l'autre | Positifs cachés parmi les négatifs : PITAZU 25 fichiers, PHYCUV 20, LEPLAT 12, DENMIN 9, BOAFAB 6 | Fenêtres écartées pour cette espèce (`weak_only_files`) |
| `weak_labels.csv` absent : aucun vrai négatif, sans avertissement | Campagne faussée en silence | Erreur explicite |
| En-tête de `config/anuraset.yaml` : nom de stock sans `@o0` | « Aucun embedding » | Corrigé |

### Méthode d'évaluation (métriques trop optimistes)

| Problème | Correction |
|---|---|
| Bootstrap par enregistrement alors que les plis sont par micro : intervalles trop étroits | Bootstrap par micro sur les données ONF (n° 139). AnuraSet garde l'enregistrement : 2 à 4 sites ne se tirent pas |
| Aucune correction pour comparaisons multiples (13 têtes × 4 espèces à 5 %) | p-valeurs et `significant_holm` dans tous les tableaux de comparaisons |
| Référence du benchmark complet = la meilleure AP sur ces mêmes données | Référence fixée d'avance (`benchmark.reference`), sinon signalée « choisie après coup » |
| Seuil de `recall@p` choisi sur les scores mêmes qu'on évalue | Seuil choisi sur les autres plis ; l'oracle reste en `recall@p…_oracle` (n° 140) |

### Bugs

| Problème | Correction |
|---|---|
| Tête multi-classes : score +inf (float32) dès qu'une fenêtre est très sûre, et l'AP plante | Logit calculé en float64 (n° 145) |
| `blanci score --site B` effaçait les décisions du site A | Ne remplace que les enregistrements scorés |
| `queue` et `fusion` prenaient la tête la plus récente, `score` l'adoptée : échec après un `retrain` non adopté | Toutes les commandes prennent l'adoptée, sinon la plus récente |
| Fusion : un descripteur manquant valait 0 (« rythme parfaitement régulier ») | Vaut la moyenne de l'entraînement. **Réapprendre les fusions existantes** |
| Import des labels : un décalage vide faisait échouer tout l'import | Ligne signalée « illisible » (n° 142) |
| « blanci ? » devenait un positif ferme, « blanci sans doute » un négatif | Verdict non tranché : ligne signalée, l'import attend |
| `import-detections` : aucun contrôle de bornes ; « 0,87 » arrêtait l'import | Bornes contrôlées, virgule décimale lue |
| Reprendre `embed` avec un autre canal ou checkpoint mélangeait les embeddings dans le même stock | Refusé à la reprise ; ordre du filtre et plancher du débruitage dans le nom hors défaut (n° 143) |
| Changement de disque : une copie de relevé laissée sur une carte SD était « déplacée » vers un autre site | Doublon signalé, le site d'origine est gardé (n° 144) |
| « RELEVE 10 » inventorié avant « RELEVE 2 » | Tri naturel |
| Chemins relatifs résolus depuis le dossier de lancement : lancée depuis `D:\`, une commande écrivait sur le disque externe | Résolus depuis la racine du projet |
| Section vide dans `local.yaml` → `None` → plantage ailleurs | Section par défaut gardée |

### Hygiène et tests (n° 146)

- `torch` et `onnx` déclarés dans le groupe `research`. Sans `onnx`, l'export ONNX du livrable ne pouvait pas tourner. `onnx` est limité à < 1.18 à cause de TensorFlow 2.15.
- Export ONNX et `OnnxEncoder` enfin testés : équivalence torch (cosinus > 0,999), paquet corrompu refusé, int8.
- Socle torch commun (attentive, R85, R66) au lieu de trois copies.
- Marqueur `slow` : `uv run pytest -m "not slow"` donne une suite rapide (~3 min 30 contre ~6 min). `-rs` affiche les tests sautés.
- Test attentive rendu robuste : marge de 0,13 → ≥ 0,27.

## 2. Erreurs trouvées, non corrigées

| Point | Pourquoi |
|---|---|
| Score des négatifs présumés = max des seules fenêtres tirées, pas de l'enregistrement entier | Changement de structure de l'évaluation : l'AP par enregistrement reste optimiste |
| Stacking non imbriqué (le niveau 1 a vu le pli du niveau 2) | Coûteux ; effet faible mais optimiste pour la fusion |
| AnuraSet : bootstrap par enregistrement | Trop peu de sites pour tirer des sites ; lire l'AP par site tenu à l'écart |
| Révision Hugging Face du checkpoint `birdmae_base` non figée | À faire quand le checkpoint sera choisi |
| `pheno-blanci.pdf` (5,2 Mo) et le `.pptx` dans l'historique git | Réécrire l'historique obligerait chaque clone à repartir de zéro |
| Points mineurs non repris : `evaluate_frozen` charge tout le stock en mémoire, migrations SQLite non atomiques, CSV sans BOM (accents mal affichés dans Excel), tri v9/v10 des têtes | Hors du périmètre des corrections annoncées |

## 3. À savoir

- **Rappels publiés avant le n° 140 : optimistes.** Les « significatif » d'avant le n° 139 se relisent avec `significant_holm`.
- **Conclusions du n° 137 provisoires**, en particulier le « R37 significativement meilleur sur PITAZU » et les niveaux d'AP de PITAZU et PHYCUV. Le n° 141 les refait.
- **Fusions enregistrées avant le n° 145 : à réapprendre.**
- Le venv local a reçu `onnx` 1.17.0 (`uv pip install`). La 1.19 plante avec `ml-dtypes` 0.3.2.
- Les noms des stocks d'embeddings existants n'ont pas changé : aucun réencodage n'est nécessaire.
- La suite complète se termine (~6 min). Le blocage vu pendant l'audit venait de huit agents qui tournaient en parallèle.
