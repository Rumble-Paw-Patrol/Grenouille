# `blanci/combination/` : le code fonction par fonction

Fiche vérifiée le 10/11/2026 sur le commit `9473508` du dépôt public (`git log -1 --format=%h`), arbre de travail sans modification dans ce dossier. Pour savoir si elle est périmée : `git diff --stat 9473508 -- blanci/combination` ; une sortie vide veut dire que le code de ce dossier n'a pas bougé.

`combination/` est l'étape 4, facultative : 4 fichiers, 1 287 lignes. Il combine le score d'une tête avec d'autres informations (le rythme du chant, la persistance des détections dans l'enregistrement, la tête d'un autre encodeur) pour en tirer un seul score. Vocabulaire du code : le « niveau 1 » est l'ensemble des entrées à combiner, toutes calculées hors-fold (sans le micro jugé) ; le « niveau 2 » est la fusion qui les combine ; l'ensemble s'appelle le stacking.

| Fichier | Lignes | Rôle |
|---|---|---|
| [`stacking.py`](stacking.py) | 513 | préparer les entrées de niveau 1, choisir où entre le traitement du signal, benchmark de la fusion |
| [`fusion.py`](fusion.py) | 546 | les méthodes de fusion (niveau 2), la fusion enregistrée, le passage d'une grille à une autre |
| [`ensemble.py`](ensemble.py) | 222 | combiner des modèles déjà évalués, par enregistrement, ou concaténer des embeddings |
| [`__init__.py`](__init__.py) | 6 | une docstring, aucun code |

## La chaîne

Lecture : `a → b` veut dire « `a` appelle `b` » ; un point-virgule sépare les appels successifs d'une même fonction. Les noms sans préfixe sont dans `combination/`, les autres portent leur fichier. Chaque ligne a été remontée depuis `cli.py` et `service.py` en cherchant les appelants dans le code.

- **`blanci fusion`** : `cli.fusion` → `service.train_fusion` → `service.load_head` ; `positions_from` ; `build_level1` ; `fused_oof` → `_fused` ; `evaluation/evaluate.recall_at_precision`, `evaluate`, `to_recordings`, `paired_bootstrap` ; `final_model` → `design_matrix`, `fusion_options`, `fit_fusion_model` ; `signal_processing.onset_gates` ; `core/db.register_model` (sorte `fusion`).
- **Dans `build_level1`** : `core/db.encoder_params` ; `inputs/dataset.embedded_training_set` ; `signal_processing.load_onsets` ; `store_rows` → `EmbeddingStore.fragments`, `read` ; `inputs/dataset.folds_for` ; `evaluation/evaluate.grouped_folds` ; puis par fold : `heads/head._choose_C`, `fit_logistic`, `signal_processing.apply_gate`, `recording_persistence` ; ensuite `signal_features` → `signal_processing.window_rhythm` ; `signal_processing.onset_counts` ; et par source supplémentaire `_source_kind`, `_secondary_head_oof` → `embedded_training_set`, `store_rows`, `_choose_C`, `fit_logistic`, `project_scores`.
- **Dans `_fused`** : `design_matrix` → `seq_columns` → `signal_columns` ; `fusion_model_oof` → `grouped_folds` ; par fold `fit_fusion_model` (→ `regularization.fusion_settings`, `choose_fusion_C`, `_orientation`, `fit_fusion_logistic`, `_simplex`), `FusionModel.decision`, `_fold_calibration` (→ `regularization.usable_folds`, `fit_fusion_model`, `fold_platt`) ; puis `upstream_pass` → `onset_gates`, `gate_mask` ; `apply_gate`.
- **`blanci score --fusion`** : `cli.score` → `service.score_and_decide` → `service.fused_scores` → `signal_processing.recording_persistence`, `load_onsets` ; `signal_features` ; pour une fusion de la première version : `FusionWeights.from_dict`, `FusionWeights.decision` ; sinon `FusionModel.from_dict` ; pour une entrée `head:<encodeur>` : `service.load_head`, `store_rows`, `project_scores` ; `FusionModel.decision` ; `onset_counts`, `gate_mask`, `apply_gate`.
- **`blanci fusion-bench`** : `cli.fusion_bench` → `positions_from` ; `fusion_benchmark` → `build_level1` ; `evaluation/oof.labels_fingerprint` ; pour chaque emplacement et chaque méthode : `design_matrix`, `_fused`, `position_tag`, `evaluation/oof.oof_frame`, `save_oof`, `evaluate`, `to_recordings`, `paired_bootstrap`, `final_model` → `FusionModel.weights`.
- **`blanci ensemble --sources`** : `cli.ensemble` → `run_ensemble` → `regularization.fusion_variant` ; `recording_table` → `evaluation/oof.load_oof`, `recording_scores` ; `evaluate` ; par méthode `fusion_model_oof`, `paired_bootstrap`, `oof_frame`, `save_oof`.
- **`blanci ensemble --concat`** : `cli.ensemble` → `concat_benchmark` → `concat_training_set` → `core/db.encoder_params`, `evaluation/head_benchmark.benchmark_data`, `embedding/index.l2_normalize`, `EmbeddingStore.load` ; `folds_for` ; par tête `heads/head.oof_scores`, `oof_frame`, `save_oof`, `evaluate`.
- **Ailleurs** : `heads/regularization/fusion_logistic.choose_fusion_C` rappelle `fit_fusion_model` pour essayer chaque valeur de C ; `heads/head.py` fournit `OOFScores`, le type des scores hors-fold que la fusion reçoit et rend.

## `stacking.py` : les entrées de niveau 1 et le benchmark de la fusion

Fabrique, pour un encodeur, tout ce qu'une fusion peut combiner, puis évalue les combinaisons. Lit : les tables `labels`, `windows`, `spans`, `intervals`, `recordings` (par `embedded_training_set`), `onsets`, `models` ; le stock d'embeddings de l'encodeur principal et des autres sources. Écrit : les scores hors-fold de chaque combinaison dans `paths.reports/oof/` (`fusion_benchmark`, par `evaluation/oof.save_oof`) ; rien dans la base. Utilisé par `blanci fusion`, `blanci fusion-bench`, `blanci score --fusion`. Tests : [`tests/combination/test_stacking.py`](../../tests/combination/test_stacking.py).

L'emplacement du traitement du signal est une liste parmi `SEQ_POSITIONS` ([stacking.py:68](stacking.py#L68)) : `upstream` (amont : le rythme sert de porte, une fenêtre sans assez de notes prend le score le plus bas), `parallel` (les descripteurs de rythme entrent dans la fusion à côté de la tête), `downstream` (aval : les descripteurs de persistance, calculés sur les scores de la tête, entrent dans la fusion). Une liste vide veut dire la tête seule. Le réglage est `signal_processing.position` (`[parallel, downstream]` dans `config/default.yaml`).

#### `build_level1(con, cfg, encoder_id, sources=(), require_onsets=False)` — [stacking.py:172](stacking.py#L172)

- Quand : au début de [`service.train_fusion`](../service.py#L1428) et de `fusion_benchmark`.
- Reçoit : la connexion, la configuration, le stock de l'encodeur principal, les autres sources (`fusion.sources`, de la forme `head:<encodeur>`). Renvoie : un `Level1`.
- Fait : 1. charge les fenêtres étiquetées et leurs négatifs appariés avec leurs embeddings (`embedded_training_set`, jeu gelé exclu) ; 2. relit les débuts de notes de leurs enregistrements dans la table `onsets` ; avec `require_onsets`, s'il n'y en a aucun, `ValueError` ; 3. charge toutes les fenêtres encodées de ces enregistrements (`store_rows`), pas seulement les étiquetées ; 4. découpe en folds par point d'écoute (`folds_for`, `grouped_folds`) ; 5. dans chaque fold : choisit C et entraîne une tête logistique sans le fold jugé ni les fenêtres arrêtées par une porte, score les fenêtres jugées, puis score toutes les fenêtres de leurs enregistrements pour en tirer les descripteurs de persistance (`recording_persistence`, seuil 0) ; 6. calcule le rythme de chaque fenêtre (`signal_features`) et les comptes de notes pour la porte amont (`onset_counts`) ; 7. pour chaque autre source, `_secondary_head_oof`.
- Détail qui ne se voit pas : la persistance d'un enregistrement est calculée avec la tête du fold qui ne l'a pas vu ; elle est donc hors-fold elle aussi. `PERSISTENCE_THRESHOLD` ([stacking.py:69](stacking.py#L69)) vaut 0, la frontière de décision de la tête logistique (commentaire).
- `_secondary_head_oof(con, cfg, encoder_id, target, assignment)` ([stacking.py:247](stacking.py#L247)) : pour un autre encodeur, et pour chaque fold commun : une tête apprise sur ses propres fenêtres des autres points d'écoute score toutes ses fenêtres des enregistrements jugés, et ces scores sont ramenés sur la grille de l'encodeur principal (`project_scores`). Un fold sans les deux classes à l'entraînement est sauté (scores laissés à `NaN`).
- `_source_kind(source)` ([stacking.py:144](stacking.py#L144)) : `head:<id>` → `("head", id)` ; un identifiant sans préfixe est pris pour une tête ; un autre préfixe lève `ValueError`.

#### `Level1` — [stacking.py:158](stacking.py#L158)

- Ce que rend `build_level1` : le stock, le tableau des fenêtres évaluées (avec `dur_s`), les labels `y`, les points d'écoute (`groups`), les enregistrements, l'affectation des points aux folds, le score hors-fold de la tête (`head`, un `OOFScores`), les descripteurs (`features` : rythme et persistance), les comptes de notes (`counts`), une colonne par source supplémentaire (`sources`), et le nombre de fenêtres dont l'enregistrement a des débuts de notes.

#### `design_matrix(level1, position, cfg, sources=())` — [stacking.py:304](stacking.py#L304)

- Quand : chaque fois qu'il faut le tableau d'entrées d'une fusion : `_fused`, `fusion_benchmark`, `final_model`.
- Renvoie : `(tableau, noms des colonnes)` : la tête (`head`), puis les descripteurs que l'emplacement fait entrer, puis les autres sources.
- Détail : le score de tête d'une fenêtre arrêtée par une porte (`GATED_SCORE`) est remplacé par `NaN` ; la fusion le remplacera par la moyenne, et la porte est réappliquée à la sortie (docstring).
- `seq_columns(position, cfg)` ([stacking.py:133](stacking.py#L133)) : parmi les descripteurs de la configuration, ceux de rythme si `parallel` est dans l'emplacement, ceux de persistance si `downstream` y est.
- `signal_columns(cfg)` ([stacking.py:118](stacking.py#L118)) : la liste `signal_processing.columns`, à défaut l'ancien `fusion.columns` (4 descripteurs dans `config/default.yaml` : `frac_ioi_blanci`, `onset_rate_hz`, `frac_windows`, `longest_run`).

#### `fused_oof(level1, method, position, cfg, sources=())` — [stacking.py:343](stacking.py#L343)

- Quand : dans [`service.train_fusion`](../service.py#L1428), pour évaluer la fusion avant de l'enregistrer.
- Renvoie : `(scores hors-fold, noms des entrées)`. S'il n'y a qu'une entrée (la tête), ce sont les scores de la tête.
- `_fused(level1, method, position, cfg, sources=())` ([stacking.py:356](stacking.py#L356)) : fait le calcul, et rend en plus les scores d'avant la recalibration par fold. Appelle `fusion_model_oof` avec les réglages de `calibration_options` et `fusion_options`, puis réapplique les portes : celle du rythme si `upstream` est dans l'emplacement (`upstream_pass`), et toujours celle du seuillage à l'encodage (colonne `gated`).
- `upstream_pass(counts, cfg)` ([stacking.py:111](stacking.py#L111)) : les fenêtres qui passent les portes de rythme de la configuration (`onset_gates`) ; une valeur manquante passe.

#### `final_model(level1, method, position, cfg, sources=())` — [stacking.py:493](stacking.py#L493)

- Quand : dans [`service.train_fusion`](../service.py#L1428), pour la fusion qui sera enregistrée ; dans `fusion_benchmark`, pour lire la part de chaque entrée.
- Renvoie : un `FusionModel` appris sur toutes les fenêtres du jeu de développement (plus de fold tenu à l'écart).

#### `fusion_options(cfg, columns)` — [stacking.py:323](stacking.py#L323)

- Quand : dans `_fused` et `final_model`.
- Renvoie : les réglages de `fit_fusion_model` lus dans la section `fusion` : `C`, `C_grid`, `cap` (R56), `bins` (R54), `segments` et `smoothness` (R55), `grid_step`, et `weights`.
- Détail : pour les poids fixés à la main (`fusion.weights`), les entrées sans poids se partagent à parts égales ce qui reste jusqu'à 1.

#### `fusion_benchmark(con, cfg, encoder_id, methods=None, positions=None, sources=None)` — [stacking.py:393](stacking.py#L393)

- Quand : c'est la commande `blanci fusion-bench` ([`cli.fusion_bench`](../cli.py#L1100)).
- Reçoit : l'encodeur, et au besoin les méthodes (défaut `fusion.benchmark_methods`, sinon `FUSION_METHODS`), les emplacements (défaut `fusion.benchmark_positions`, sinon `DEFAULT_POSITIONS`, [stacking.py:70](stacking.py#L70)), les autres sources. Renvoie : un dictionnaire : `table` (AP par emplacement, méthode et niveau), `comparisons` (écart apparié à la tête seule), `weights` (part de chaque entrée), `n_windows`, `n_with_onsets`.
- Fait : 1. `build_level1` une fois (les débuts de notes sont exigés si un emplacement contient `upstream` ou `parallel`) ; 2. pour chaque emplacement, et chaque méthode s'il y a quelque chose à fusionner (sinon une seule ligne `none`) : scores hors-fold (`_fused`), rangés dans le stock commun sous `<encodeur>/fusion:<emplacement>/<méthode>` ; 3. métriques aux niveaux fenêtre et enregistrement (`evaluate`, avec l'AP d'avant recalibration si elle existe) ; 4. comparaison à la tête seule par bootstrap apparié par enregistrement ; 5. part de chaque entrée dans la fusion finale.
- Détail : la méthode `weighted` est sautée si `fusion.weights` est vide.

#### `positions_from(value)` et `position_tag(position)` — [stacking.py:80](stacking.py#L80)

- `positions_from` : appelée par [`cli.fusion_bench`](../cli.py#L1100), `service.train_fusion`, `fusion_benchmark`. Accepte une liste, un texte (`"parallel,downstream"`), `"none"`, `"aucun"`, vide ou `None` ; rend la liste dans l'ordre amont, parallèle, aval ; un nom inconnu lève `ValueError`.
- `position_tag` ([stacking.py:93](stacking.py#L93)) : `parallel+downstream`, ou `aucun` pour une liste vide ; sert de nom dans les tableaux et le stock hors-fold.

#### `store_rows(store, recording_ids)` — [stacking.py:97](stacking.py#L97)

- Quand : `build_level1`, `_secondary_head_oof`, [`service.fused_scores`](../service.py#L1540).
- Renvoie : toutes les fenêtres encodées de ces enregistrements et leurs embeddings en `float32`, en parcourant tout le stock fichier par fichier.

#### `signal_features(windows, persistence, onsets, cfg)` — [stacking.py:124](stacking.py#L124)

- Quand : `build_level1` et [`service.fused_scores`](../service.py#L1540).
- Renvoie : un tableau aligné sur les fenêtres : le rythme dans chaque fenêtre (`signal_processing.window_rhythm`, avec l'intervalle entre notes attendu `signal.ioi_range_s`) et, recopiée sur chaque fenêtre, la persistance de son enregistrement.

## `fusion.py` : les méthodes de fusion

Des calculs sur des tableaux : ne lit ni la base ni les fichiers. La fusion apprise se range en JSON (dans la table `models`, par `service.train_fusion`) et s'applique en numpy, sans pickle. Tests : [`tests/combination/test_fusion.py`](../../tests/combination/test_fusion.py) pour la première version, [`tests/combination/test_stacking.py`](../../tests/combination/test_stacking.py) pour la fusion à N entrées.

#### `fit_fusion_model(method, X, y, columns, C=1.0, weights=None, grid_step=0.1, max_grid=5000, seed=0, groups=None, C_grid=None, n_splits=5, cap=2.0, bins=5, segments=8, smoothness=1.0)` — [fusion.py:313](fusion.py#L313)

- Quand : pour apprendre une fusion : `fusion_model_oof` (une par fold), `_fold_calibration`, [`stacking.final_model`](stacking.py#L493), et `heads/regularization/fusion_logistic.choose_fusion_C` (une par valeur de C essayée).
- Reçoit : le nom de la méthode, le tableau des entrées de niveau 1, les labels, le nom de chaque colonne, et les réglages. Renvoie : un `FusionModel`.
- Fait : 1. reconnaît les variantes de la logistique (`logistic+R50+R52…`, par `regularization.fusion_settings`) ; une méthode inconnue lève `ValueError` ; 2. avec `+R50` et des groupes, choisit C par validation groupée (`choose_fusion_C`) ; 3. standardise chaque entrée sur l'entraînement ; une valeur manquante devient 0, c'est-à-dire la moyenne ; 4. oriente chaque entrée (`_orientation`) ; 5. apprend selon la méthode.
- Les méthodes (`FUSION_METHODS`, [fusion.py:178](fusion.py#L178)) : `logistic` (régression logistique, classes équilibrées, C fixé) ; `logistic+R50` (C choisi) ; `weighted` (somme pondérée, poids de `fusion.weights` ; sans poids, `ValueError`) ; `weight_grid` (poids positifs de somme 1 cherchés sur une grille de pas `grid_step`, on garde ceux qui donnent la meilleure AP sur l'entraînement ; au-delà de 5 000 combinaisons, tirage au hasard) ; `mean` (moyenne) ; `rank_mean` (moyenne des rangs, d'après 101 quantiles de l'entraînement) ; `max` (une entrée forte suffit) ; `min` (toutes doivent être fortes). Les autres variantes de la logistique (`+R52` poids positifs, `+R53` sélection L1, `+R54` paliers, `+R55` courbes lisses, `+R56` plafond) sont apprises par `regularization.fit_fusion_logistic`.
- Pourquoi (docstring du fichier) : hors logistique, chaque entrée est d'abord orientée pour qu'un descripteur « plus c'est bas, plus c'est A. blanci » compte dans le bon sens.
- `_orientation(z, y)` ([fusion.py:305](fusion.py#L305)) : +1 si l'entrée monte avec le label sur l'entraînement (signe de la corrélation), −1 sinon ; +1 partout s'il n'y a qu'une classe.
- `_simplex(n, step)` ([fusion.py:290](fusion.py#L290)) : toutes les combinaisons de `n` poids positifs de somme 1 au pas `step`.

#### `FusionModel` — [fusion.py:193](fusion.py#L193)

- Quand : rendu par `fit_fusion_model` ; relu d'un dictionnaire par [`service.fused_scores`](../service.py#L1540) au moment de `blanci score --fusion`.
- Contient : la méthode, les noms des entrées, leur moyenne et leur écart-type à l'entraînement, leur orientation, et selon la méthode les coefficients, les quantiles, le C retenu, le plafond R56, la base des descripteurs R54 ou R55.
- `decision(X)` ([fusion.py:213](fusion.py#L213)) : standardise (une entrée manquante vaut la moyenne), puis applique la méthode. Avec R56, la part des descripteurs est écrasée par une tangente hyperbolique à ±`cap` : ils déplacent le score d'au plus `cap`, sans pouvoir annuler la tête.
- `weights()` ([fusion.py:238](fusion.py#L238)) : la part de chaque entrée dans la décision : valeur absolue des coefficients ramenée à une somme de 1 (pour R54 et R55, l'amplitude de la courbe de chaque descripteur) ; parts égales pour les règles sans coefficients (`rank_mean`, `max`, `min`).
- `to_dict()` ([fusion.py:251](fusion.py#L251)) et `from_dict(d)` ([fusion.py:273](fusion.py#L273)) : l'aller-retour vers le JSON rangé dans `models`.

#### `fusion_model_oof(method, X, y, groups, columns, n_splits=5, seed=0, assignment=None, calibration=None, calibration_splits=3, **options)` — [fusion.py:420](fusion.py#L420)

- Quand : pour évaluer une fusion sans tricher : [`stacking._fused`](stacking.py#L356) et [`ensemble.run_ensemble`](ensemble.py#L68).
- Renvoie : un `OOFScores` nommé `fusion:<méthode>` : pour chaque fold, une fusion apprise sans le groupe jugé, appliquée à ce groupe.
- Détail : avec `calibration="platt"`, les scores de chaque fold sont recalibrés (`a × score + b`) avant d'être mis bout à bout, et les scores d'avant sont gardés dans `raw` (docstring, DECISIONS n° 135).
- `_fold_calibration(method, model, X, y, groups, columns, seed, calibration, splits, options)` ([fusion.py:474](fusion.py#L474)) : apprend `(a, b)` sur des scores hors-fold internes à l'entraînement du fold, avec le C que le fold a retenu ; rend `(1, 0)` pour les règles fixes (`weighted`, `mean`, `rank_mean`, `max`, `min`) et quand aucun fold interne n'est utilisable.

#### `project_scores(target, source)` — [fusion.py:519](fusion.py#L519)

- Quand : pour ramener les scores d'un autre encodeur sur les fenêtres de l'encodeur principal : [`stacking._secondary_head_oof`](stacking.py#L247) et [`service.fused_scores`](../service.py#L1540).
- Reçoit : les fenêtres cibles (`recording_id`, `offset_s`, `dur_s`) et les fenêtres sources (les mêmes colonnes et `score`). Renvoie : un score par fenêtre cible.
- Fait : pour chaque fenêtre cible, le maximum des scores des fenêtres sources qui la recouvrent sur au moins la moitié de la plus courte des deux ; s'il n'y en a pas, le score de la fenêtre source dont le centre est le plus proche, dans le même enregistrement ; `NaN` si l'enregistrement n'a aucune fenêtre source.

#### `FusionWeights` — [fusion.py:132](fusion.py#L132)

- Quand : dans [`service.fused_scores`](../service.py#L1540), seulement pour relire une fusion enregistrée dans l'ancien format (paramètres sans clé `model`, avec une clé `weights`).
- Contient : les colonnes, la moyenne, l'écart-type, les coefficients et l'ordonnée à l'origine d'une fusion logistique. `decision(head_score, features)` ([fusion.py:171](fusion.py#L171)) l'applique en numpy ; `from_dict(d)` ([fusion.py:162](fusion.py#L162)) et `to_dict()` ([fusion.py:152](fusion.py#L152)) font l'aller-retour JSON ; `from_fusion(fusion)` ([fusion.py:142](fusion.py#L142)) la tire d'un objet `Fusion`.

#### `fit_fusion(head_oof, features, y, columns, C=1.0)` et `fusion_oof(head_oof, features, y, groups, columns, n_splits=5, seed=0, assignment=None)` — [fusion.py:92](fusion.py#L92)

- La première version de la fusion : une régression logistique sur le score de la tête et les descripteurs. `fit_fusion` rend un objet `Fusion` ([fusion.py:57](fusion.py#L57)) ; `fusion_oof` ([fusion.py:107](fusion.py#L107)) en donne les scores hors-fold.
- Quand : dans les tests seulement (voir les écarts).
- `_check(head_oof, y, columns)` ([fusion.py:80](fusion.py#L80)) : refuse tout score de tête qui n'est pas un `OOFScores` (`TypeError`), et avertit s'il y a moins de 10 positifs par coefficient (`POSITIVES_PER_COEF`, [fusion.py:53](fusion.py#L53)). Pourquoi (docstring du fichier) : un score calculé sur les exemples que la tête a vus ferait croire à la fusion que la tête est parfaite.
- `_design(head_score, features, columns)` ([fusion.py:67](fusion.py#L67)) et `_fill(X, mean)` ([fusion.py:74](fusion.py#L74)) : assemblent la tête et les descripteurs, puis remplacent une valeur manquante par la moyenne de l'entraînement. Pourquoi (docstring, DECISIONS n° 145) : un `ioi_cv` manquant (moins de deux notes) remplacé par 0 voudrait dire « rythme parfaitement régulier ».

## `ensemble.py` : ensembles de modèles

Combine des modèles déjà évalués. Lit : le stock de scores hors-fold `paths.reports/oof/` (`run_ensemble`) ; la base et les stocks d'embeddings (`concat_benchmark`). Écrit : les scores hors-fold de chaque ensemble dans ce même stock. Utilisé par `blanci ensemble`. Tests : dans [`tests/combination/test_stacking.py`](../../tests/combination/test_stacking.py).

#### `run_ensemble(con, cfg, sources, methods=None, allow_mixed=False)` — [ensemble.py:68](ensemble.py#L68)

- Quand : `blanci ensemble --sources a,b,…` ([`cli.ensemble`](../cli.py#L1155)).
- Reçoit : au moins deux noms de sources du stock hors-fold (encodeur et tête, baseline, fusion, détecteur ; `blanci sources` les liste), et les méthodes (défaut : `mean`, `rank_mean`, `max`, `logistic`, `weight_grid`). Renvoie : `table` (AP de chaque source et de chaque ensemble, par enregistrement), `comparisons` (chaque ensemble contre la meilleure source seule, bootstrap apparié), `n_recordings`.
- Fait : 1. refuse une méthode inconnue ; 2. `recording_table` ; 3. évalue chaque source seule et retient la meilleure ; 4. pour chaque méthode, scores hors-fold de la combinaison (`fusion_model_oof`, sur les folds déjà attribués aux points d'écoute, avec des poids égaux pour `weighted`) ; 5. range ces scores sous `ensemble:<méthode>(<a>+<b>)`.
- Pourquoi (docstring du fichier) : un ensemble n'est jugé que contre sa meilleure source seule ; s'il n'y gagne pas, il ne vaut pas le double calcul. Travailler par enregistrement permet de combiner des grilles différentes (3 s, 5 s).
- `recording_table(cfg, sources, allow_mixed=False)` ([ensemble.py:47](ensemble.py#L47)) : une ligne par enregistrement évalué par toutes les sources, avec son point d'écoute, son fold, son label et le score maximal de ses fenêtres pour chaque source. Refuse moins de deux sources, et des sources calculées sur des labels différents (empreintes différentes), sauf `allow_mixed`.

#### `concat_benchmark(con, cfg, encoder_ids, methods=None)` — [ensemble.py:168](ensemble.py#L168)

- Quand : `blanci ensemble --concat a,b` ([`cli.ensemble`](../cli.py#L1155)).
- Renvoie : un tableau des métriques, aux niveaux fenêtre et enregistrement, de chaque tête (défaut : `logistic`, `prototype`) apprise sur les embeddings mis bout à bout ; scores hors-fold rangés sous `<a>+<b>/<tête>`.
- `concat_training_set(con, cfg, encoder_ids)` ([ensemble.py:136](ensemble.py#L136)) : les fenêtres évaluées du premier encodeur, et les embeddings de tous, chaque bloc normalisé à la norme 1 (aucun encodeur ne pèse par sa seule échelle, docstring). Refuse moins de deux encodeurs, des grilles différentes (fenêtre ou pas), et un encodeur qui n'a pas encodé toutes les fenêtres du premier.

## `__init__.py`

Une docstring qui présente `stacking`, `fusion`, `ensemble`. Aucun code, aucun import.

## Écarts relevés

Signalés, pas corrigés.

1. **La première version de la fusion n'a plus d'appelant hors tests.** La docstring de `fusion.py` dit que `Fusion`, `fit_fusion`, `fusion_oof` et `FusionWeights` sont gardées « pour `blanci fusion` et `score --fusion` ». `blanci fusion` (`service.train_fusion`) passe par `stacking.fused_oof` et `stacking.final_model` ; `fit_fusion`, `fusion_oof` et `Fusion` ne sont appelées que par `tests/combination/test_fusion.py`. Seule `FusionWeights` sert encore, dans `service.fused_scores`, pour relire une fusion enregistrée dans l'ancien format.
2. **Message d'erreur périmé dans `build_level1`.** Sans débuts de notes, il dit qu'« ils se calculent pendant `blanci embed`, ou avec `blanci onsets` ». Depuis DECISIONS n° 178, `embed` ne les calcule plus : seule `blanci onsets` les range.
3. **`PERSISTENCE_THRESHOLD` défini deux fois.** `stacking.py` (ligne 69) et `service.py` (ligne 1421) ont chacun leur constante, toutes deux à 0. `build_level1` lit la première, `train_fusion` range la seconde dans `models`, et `fused_scores` relit la valeur rangée.
4. **`FUSION_METHODS` ne contient pas les variantes R52 à R56.** Le tableau de la docstring de `fusion.py` les présente comme des méthodes ; la constante n'en a que huit (`logistic`, `logistic+R50`, `weighted`, `weight_grid`, `mean`, `rank_mean`, `max`, `min`). Les variantes sont acceptées parce que `fit_fusion_model` et `run_ensemble` les reconnaissent par `regularization.fusion_settings` et `fusion_variant`. Le défaut de `fusion-bench` vient de `fusion.benchmark_methods`, qui en liste treize.
5. **`cli.py` appelle ce dossier sans passer par `service.py`.** `cli.fusion_bench` appelle `stacking.fusion_benchmark` et `cli.ensemble` appelle `ensemble.run_ensemble` et `concat_benchmark` directement ; `service.py` se présente pourtant comme le seul point d'entrée de la CLI. Déjà relevé dans la fiche de `core/` (n° 10).
6. **Docstring de `seq_columns`.** Elle parle des « descripteurs de `fusion.columns` » ; la liste est lue d'abord dans `signal_processing.columns`, `fusion.columns` n'étant que l'ancien nom encore accepté (`signal_columns`).
7. **Fonctions publiques à un seul usage.** `upstream_pass`, `signal_columns`, `seq_columns`, `position_tag` ne sont appelées que depuis `stacking.py` ; `recording_table` et `concat_training_set` que depuis `ensemble.py`. Aucune fonction du dossier n'est sans appelant si l'on compte les tests.
