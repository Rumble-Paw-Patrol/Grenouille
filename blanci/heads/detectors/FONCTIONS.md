# `blanci/heads/detectors/` : le code fonction par fonction

Fiche vérifiée le 10/11/2026 sur le commit `9473508` du dépôt public (`git log -1 --format=%h`), arbre de travail sans modification dans ce dossier. Pour savoir si elle est périmée : `git diff --stat 9473508 -- blanci/heads/detectors` ; une sortie vide veut dire que le code de ce dossier n'a pas bougé.

`detectors/` : 4 fichiers, 257 lignes. Un détecteur prend des fenêtres de son brut et rend un score par fenêtre, sans encodeur ni tête (DECISIONS n° 97). Le dossier contient le contrat, le banc d'essai (`blanci detector-bench`), un détecteur de référence sans apprentissage (`band_contrast`) et deux emplacements réservés, non programmés (`distilled`, `homemade`). Le reste de l'étape 3 est décrit dans la fiche du dossier parent : [`../FONCTIONS.md`](../FONCTIONS.md).

| Fichier | Lignes | Rôle |
|---|---|---|
| [`base.py`](base.py) | 110 | le contrat (`Detector`, `TrainableDetector`), la lecture de l'audio d'évaluation, le banc d'essai |
| [`__init__.py`](__init__.py) | 48 | `get_detector` (nom → détecteur) et le détecteur de référence `BandContrastDetector` |
| [`distilled.py`](distilled.py) | 54 | emplacement du modèle distillé : lève `NotImplementedError` |
| [`homemade.py`](homemade.py) | 45 | emplacement du modèle fait maison : lève `NotImplementedError` |

## La chaîne

Lecture : `a → b` veut dire « `a` appelle `b` » ; un point-virgule sépare les appels successifs. Les noms sans préfixe sont dans `detectors/`.

- **`blanci detector-bench --detector band_contrast`** : `cli.detector_bench` → `get_detector` → `BandContrastDetector` ; `evaluate_detector` → `read_evaluation_audio` → `baselines.evaluation_windows` ; `baselines.read_windows` ; `core/audio.resample` ; puis `inputs/dataset.folds_for` ; `BandContrastDetector.score` → `baselines.window_features` ; `evaluation/oof.labels_fingerprint` ; `oof_frame` ; `save_oof` ; `evaluation/evaluate.evaluate` (deux fois : fenêtres, enregistrements).
- **`blanci detector-bench --detector distilled`** (ou `homemade`) : `cli.detector_bench` → `get_detector` → `DistilledDetector` ; `evaluate_detector` → `read_evaluation_audio` (tout l'audio d'évaluation est lu) ; `folds_for` ; `evaluation/evaluate.grouped_folds` ; `DistilledDetector.fit` → `NotImplementedError`. Rien n'est écrit.
- **Après le banc d'essai** : le fichier de scores rejoint le stock commun `paths.reports/oof/`, que relisent `blanci sources` (`evaluation/oof.list_sources`), `blanci benchmark-all` (`full_benchmark.run_full_benchmark` → `load_oof`) et `blanci ensemble --sources` (`ensemble.recording_table` → `load_oof`). Aucune de ces commandes n'appelle le code de `detectors/`.
- **Aucun autre appelant** : ni `service.py`, ni `anuraset/`, ni `scripts/`, ni le tableau de bord n'importent ce dossier.

## `base.py` : le contrat et le banc d'essai

Lit : la base (par `baselines.evaluation_windows` et `inputs/dataset.folds_for`) et les fichiers audio sous `paths.raw`, en lecture partielle. Écrit : un fichier de scores hors-fold dans `paths.reports/oof/` (par `evaluation/oof.save_oof`) ; jamais d'audio, rien dans la base. Utilisé par la seule commande `blanci detector-bench`. Tests : dans [`tests/heads/test_baselines.py`](../../../tests/heads/test_baselines.py) ; il n'y a pas de fichier de tests propre aux détecteurs.

#### `Detector` et `TrainableDetector` — [base.py:27](base.py#L27)

- Deux protocoles (des contrats, pas des classes à hériter), vérifiables à l'exécution (`runtime_checkable`). `Detector` ([base.py:27](base.py#L27)) : un `name`, une `version`, et `score(fenêtres, f_e)` qui rend un score par fenêtre (plus haut = plus probablement A. blanci). `TrainableDetector` ([base.py:36](base.py#L36)) : en plus `fit(fenêtres, f_e, labels)`, qui rend un détecteur appris.
- Quand : `evaluate_detector` teste `isinstance(detector, TrainableDetector)` pour savoir s'il doit entraîner fold par fold ; un détecteur est donc « entraînable » dès qu'il a une méthode `fit`.
- Pourquoi (docstring du fichier) : c'est la forme du modèle de distillation et du modèle fait maison, un petit réseau qui tourne seul, sans encodeur de fondation.

#### `evaluate_detector(con, cfg, detector, audio=None)` — [base.py:66](base.py#L66)

- Quand : appelée par [`cli.detector_bench`](../../cli.py#L1228) (`blanci detector-bench --detector <nom>`), seul appelant hors tests.
- Reçoit : la connexion, la configuration, un détecteur, et au besoin l'audio déjà lu (`audio`, pour les tests ou pour juger plusieurs détecteurs sans relire). Renvoie : un dictionnaire `source` (`detector/<nom>`), `table` (une ligne de métriques par niveau, fenêtre puis enregistrement), `scores`.
- Fait : 1. lit l'audio avec `read_evaluation_audio`, sauf s'il est fourni ; 2. prend les folds communs à tous les modèles (`inputs/dataset.folds_for`) ; 3. si le détecteur a une méthode `fit` : pour chaque fold (`evaluation/evaluate.grouped_folds`, groupes = `point`), l'entraîne sur les fenêtres d'entraînement et note les fenêtres de test ; sinon note toutes les fenêtres d'un coup ; 4. range les scores sous la source `detector/<nom>`, de sorte `detector` (`oof_frame`, `save_oof`, avec l'empreinte des labels `labels_fingerprint`) ; 5. calcule les métriques aux deux niveaux avec `evaluation/evaluate.evaluate` (`benchmark.precisions`, `benchmark.n_boot`, `head.seed`).
- Détails : le fichier écrit est `paths.reports/oof/detector_<nom>-<6 caractères>.parquet` (`evaluation/oof.source_path` remplace le `/` et ajoute une empreinte du nom) ; une source déjà présente est remplacée. `evaluate` est appelée sans `folds` ni `clusters` : ni AP moyenne par fold, ni rappel à seuil choisi sur les autres folds, et le bootstrap tire des enregistrements, pas des micros (écart n° 5). La commande n'affiche que l'AP et son intervalle par niveau, puis le nom de la source.
- Pourquoi (docstring du fichier) : entraîné fold par fold sur les folds communs, un détecteur a des scores hors-fold, comparables à ceux des encodeurs, des baselines et des ensembles.

#### `read_evaluation_audio(con, cfg)` — [base.py:40](base.py#L40)

- Quand : dans `evaluate_detector`, quand `audio` n'est pas fourni ; et directement dans `tests/heads/test_baselines.py`.
- Reçoit : la connexion et la configuration. Renvoie : `(tableau des fenêtres, forme d'onde de chacune, fréquence d'échantillonnage commune)`.
- Fait : 1. `baselines.evaluation_windows` ([baselines.py:75](../baselines.py#L75)) donne les fenêtres des baselines : annotations et négatifs appariés présumés, sur la grille `w3`, jeu gelé exclu ; 2. choisit les canaux : `audio.channel` vaut `mean` → canaux 0 et 1, sinon le canal indiqué (0 dans `config/default.yaml`) ; 3. `baselines.read_windows` ([baselines.py:108](../baselines.py#L108)) lit, fichier par fichier, seulement les secondes utiles ; 4. chaque fenêtre est la moyenne des canaux lus, en `float32` ; 5. la fréquence cible est la plus fréquente parmi les fenêtres ; les autres sont rééchantillonnées (`core/audio.resample`).
- Détails : avec `mean`, un fichier mono donne deux fois son unique canal (`read_windows`). Sans aucune fenêtre, la fréquence rendue vaut 0.
- Pourquoi (docstring) : toutes les fenêtres doivent avoir la même fréquence (Song Meter à 48 kHz).

## `__init__.py` : choisir un détecteur

Réexporte `Detector`, `TrainableDetector`, `evaluate_detector` de `base.py` et définit `get_detector` et `BandContrastDetector`. `__all__` ([\_\_init\_\_.py:17](__init__.py#L17)) liste `Detector`, `TrainableDetector`, `evaluate_detector`, `get_detector`. Ne lit et n'écrit rien. Tests : `tests/heads/test_baselines.py`.

#### `get_detector(name, cfg)` — [\_\_init\_\_.py:37](__init__.py#L37)

- Quand : appelée par [`cli.detector_bench`](../../cli.py#L1228), juste avant `evaluate_detector` ; et dans les tests.
- Reçoit : `distilled`, `homemade` ou `band_contrast`, et la configuration. Renvoie : le détecteur correspondant.
- Fait : importe `distilled.py` ou `homemade.py` seulement si on les demande ; un autre nom lève `ValueError` « détecteur inconnu : … (distilled, homemade, band_contrast) ».

#### `BandContrastDetector(cfg)` — [\_\_init\_\_.py:20](__init__.py#L20)

- Quand : construit par `get_detector("band_contrast", cfg)`, seul endroit.
- Le détecteur de référence, sans apprentissage : `name = "band_contrast"`, `version = "1"`. À la construction, garde la section `signal` de la configuration.
- `score(windows, sr)` ([\_\_init\_\_.py:29](__init__.py#L29)) : pour chaque fenêtre, appelle `baselines.window_features` ([baselines.py:132](../baselines.py#L132)) et rend son score fixe `band_contrast` : le contraste en dB entre la bande de la note et les bandes voisines, lissé, maximum moins médiane sur la fenêtre.
- Détail : il n'a pas de méthode `fit` ; `evaluate_detector` le note donc en une fois, sans folds.
- Pourquoi (docstring du fichier) : il montre que le banc d'essai, le stock hors-fold et le benchmark complet sont prêts avant que les vrais détecteurs soient écrits.

## `distilled.py` : emplacement du modèle distillé

Aucun calcul : une classe dont les deux méthodes lèvent `NotImplementedError`. La docstring du fichier (32 lignes) porte le plan : un petit réseau « élève » qui imite la chaîne « professeur » gelée (encodeur, tête, fusion), ses cinq étapes, et un avertissement. Le code la marque « EN ATTENTE — emplacement gardé pour plus tard, volontairement non branché ; ne pas signaler comme code mort à l'audit ». Test : `tests/heads/test_baselines.py`.

#### `DistilledDetector(cfg=None)` — [distilled.py:41](distilled.py#L41)

- Quand : construit par `get_detector("distilled", cfg)`, seul endroit.
- `name = "distilled"`, `version = "0"`. À la construction, garde `cfg["detectors"]["distilled"]` (dans `config/default.yaml` : `band_hz: [3000, 7000]`), ou un dictionnaire vide.
- `fit(windows, sr, y)` ([distilled.py:50](distilled.py#L50)) et `score(windows, sr)` ([distilled.py:53](distilled.py#L53)) : lèvent `NotImplementedError` avec le message `PLANNED` ([distilled.py:38](distilled.py#L38)) : « distillation : emplacement réservé, pas encore programmé (§3, après le jalon M4) ».
- Pourquoi l'avertissement (docstring) : l'élève devra être distillé fold par fold à partir d'un professeur lui-même appris sans le micro testé, sinon il hérite de scores en-fold et paraît meilleur qu'il n'est.

## `homemade.py` : emplacement du modèle fait maison

Même forme que `distilled.py`, même mention « EN ATTENTE ». La docstring décrit l'idée (un réseau entraîné directement sur les labels, sans professeur), le contrat à respecter et les contraintes. Test : `tests/heads/test_baselines.py`.

#### `HomemadeDetector(cfg=None)` — [homemade.py:32](homemade.py#L32)

- Quand : construit par `get_detector("homemade", cfg)`, seul endroit.
- `name = "homemade"`, `version = "0"`. À la construction, garde `cfg["detectors"]["homemade"]` (`band_hz: [3000, 7000]`), ou un dictionnaire vide.
- `fit(windows, sr, y)` ([homemade.py:41](homemade.py#L41)) et `score(windows, sr)` ([homemade.py:44](homemade.py#L44)) : lèvent `NotImplementedError` avec `PLANNED` ([homemade.py:29](homemade.py#L29)) : « modèle fait maison : emplacement réservé, pas encore programmé (§3) ».

## Écarts relevés

Signalés, pas corrigés.

1. **`detector-bench` lit tout l'audio avant d'échouer sur un emplacement réservé.** `DistilledDetector` et `HomemadeDetector` ont une méthode `fit` : `evaluate_detector` les traite comme entraînables, lit d'abord toutes les fenêtres d'évaluation, puis reçoit `NotImplementedError` au premier fold. L'aide de la commande dit seulement « emplacements réservés ».
2. **L'audio est cherché sous `paths.raw` seulement.** `read_evaluation_audio` passe par `baselines.read_windows`, qui ouvre `paths.raw/<chemin d'origine>` : ni `core/locate.variants` (dossiers renommés), ni recherche sur les autres disques. Même constat que l'écart n° 4 de la fiche de `core/` et que l'écart n° 15 de la fiche de `heads/`.
3. **`__all__` ne liste pas tout ce qui sert.** `BandContrastDetector` est définie dans `__init__.py` sans y figurer, et `read_evaluation_audio` n'est pas réexportée : le test l'importe de `detectors.base`.
4. **Renvois à d'anciens numéros de section.** « §3 », « §13.1 », « §13.6 », « §13.7 », « jalon M4 » dans les docstrings de `base.py`, `__init__.py`, `distilled.py`, `homemade.py` : des sections d'un document de cadrage, pas des numéros de décision (voir la fiche de `core/`, n° 7, et celle de `embedding/`, n° 8). Seul « DECISIONS n° 97 » (et « n° 121 » pour les régularisations prévues) est un numéro de décision.
5. **Le banc d'essai des détecteurs calcule moins que celui des têtes.** `evaluate_detector` appelle `evaluate` sans `folds` ni `clusters` ; `head_benchmark.level_rows` passe les deux (AP moyenne par fold, rappel jugé à part, bootstrap par micro). Les lignes d'un détecteur et celles d'une tête ne portent donc pas les mêmes colonnes ni le même intervalle. À rapprocher de la docstring de `base.py`, qui dit ces scores « comparables ».
6. **Chiffres datés dans une docstring.** `homemade.py` écrit « 51 enregistrements, 13 micros, tous à Mataroni » : rien dans le code ne le calcule ; à vérifier contre la base avant de le citer.
7. **`blanci/heads/__init__.py` ne cite pas le détecteur de référence.** Il décrit `detectors/` comme « distillation, modèle maison : à venir » ; `band_contrast` est pourtant branché et testé.
8. **Deux régularisations prévues ici n'ont pas d'appelant.** Les docstrings de `distilled.py` et `homemade.py` renvoient à `regularization.distillation_loss` (R63), `optimise`, `fit_with_options`, `dropout` ; `distillation_loss` n'est appelée par aucun code hors tests (voir la fiche de [`regularization/`](../regularization/FONCTIONS.md)). `homemade.py` cite aussi R65 (pré-entraînement sur AnuraSet), que le catalogue donne « en attente ».
