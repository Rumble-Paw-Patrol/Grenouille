# `blanci/evaluation/` : le code fonction par fonction

Fiche vérifiée le 10/11/2026 sur le commit `9473508` du dépôt public (`git log -1 --format=%h`), arbre de travail sans modification dans ce dossier. Pour savoir si elle est périmée : `git diff --stat 9473508 -- blanci/evaluation` ; une sortie vide veut dire que le code de ce dossier n'a pas bougé.

`evaluation/` est l'étape 5, la mesure : 9 fichiers, 3 154 lignes. Il découpe les données en folds sans mélanger les micros, calcule les métriques et leurs intervalles, range les scores de chaque benchmark dans un format commun, et contient les benchmarks eux-mêmes (encodeurs, têtes, tout ensemble, AnuraSet), la calibration du contrôle qualité et la mesure du débit des encodeurs. Vocabulaire : un score « hors-fold » (`oof`) est le score d'une fenêtre donné par un modèle entraîné sans le micro de cette fenêtre ; l'AP (précision moyenne) résume un classement, 1 étant parfait ; le « rappel à précision plancher » est la part des positifs retrouvés quand on exige qu'au moins 10 % (ou 50 %) des fenêtres signalées soient de vrais positifs.

| Fichier | Lignes | Rôle |
|---|---|---|
| [`evaluate.py`](evaluate.py) | 439 | folds groupés, AP, rappel à précision plancher, bootstrap, Wilson, Holm |
| [`oof.py`](oof.py) | 185 | le stock des scores hors-fold : format commun de tous les benchmarks |
| [`benchmark.py`](benchmark.py) | 338 | benchmark des encodeurs (`blanci benchmark`), tableaux markdown |
| [`head_benchmark.py`](head_benchmark.py) | 644 | benchmark des têtes (`blanci heads`), courbe selon le nombre d'annotations (`blanci heads-curve`) |
| [`full_benchmark.py`](full_benchmark.py) | 182 | benchmark complet de toutes les sources (`blanci benchmark-all`) |
| [`anuraset.py`](anuraset.py) | 823 | pré-benchmark sur le jeu public AnuraSet (`blanci anuraset …`) |
| [`qc_calibration.py`](qc_calibration.py) | 309 | calibration des seuils du contrôle qualité (`blanci qc-calibrate`) |
| [`throughput.py`](throughput.py) | 223 | débit, mémoire et dimension des encodeurs (`blanci throughput`) |
| [`__init__.py`](__init__.py) | 11 | une docstring, aucun code |

## La chaîne

Lecture : `a → b` veut dire « `a` appelle `b` » ; un point-virgule sépare les appels successifs d'une même fonction. Les noms sans préfixe sont dans `evaluation/`, les autres portent leur fichier. Chaque ligne a été remontée depuis `cli.py` et `service.py` en cherchant les appelants dans le code.

- **Les folds communs** : toute commande qui évalue → `inputs/dataset.folds_for` → `benchmark_folds` → `fold_assignment` (→ `lomo_assignment` si `head.n_splits: lomo`) ; puis, dans chaque calcul hors-fold (`heads/head.oof_scores`, `combination/fusion.fusion_model_oof`, `combination/stacking.build_level1`, `heads/baselines.oof_template_scores`, `heads/detectors/base.evaluate_detector`), `grouped_folds` avec cette affectation.
- **`blanci train`** : `cli.train` → `service.train_and_register` → `heads/head.oof_scores` → `grouped_folds` ; `recall_at_precision` ; `regularization.cross_fitted_threshold` → `cross_fitted_recall` ; `evaluate` → `to_recordings`, `average_precision`, `bootstrap_ci` → `_unit_index`, `recall_at_precision`, `wilson_interval`.
- **`blanci evaluate`** : `cli.evaluate` → `service.evaluate_holdout` → `heads/head.oof_scores` ou `train_head` ; `evaluate` ; `recall_at_precision` ; `recall_by_group` → `wilson_interval`. Avec `--frozen` : `service.evaluate_frozen` → `evaluate` (deux niveaux) ; `false_alarms_per_hour` ; `recall_by_group`.
- **`blanci benchmark`** : `cli.benchmark` → `run_benchmark` → par encodeur `benchmark_encoder` → `inputs/dataset.embedded_training_set`, `folds_for` ; `probe_table` → `heads/head.oof_scores`, `evaluate` (avec `regularization.fold_ids`) ; `labels_fingerprint` ; `oof_frame`, `save_oof` → `source_path` ; `knn_top1`. Puis `compare_encoders` → `to_recordings`, `paired_bootstrap` → `bootstrap_p` ; `with_holm` → `holm`. Enfin `cli.benchmark` → `write_report` → `to_markdown`.
- **`blanci heads`** : `cli.heads` → `run_head_benchmark` → `benchmark_data` (→ `embedded_training_set`, `load_tokens` → `heads/attentive.TokenStore.load`, `heads/pooling.as_grid`) ; `expand_methods` ou `head_methods` ; `regularization_context` ; `folds_for` ; `labels_fingerprint` ; par tête : `regularization.regularizer_for`, `_inputs` (→ `heads/pooling.pool`), `heads/head.oof_scores`, `oof_frame`, `save_oof`, `level_rows` → `evaluate` ; puis `compare_to_reference` → `_pair`, `cluster_units`, `paired_bootstrap`, `with_holm` ; `regularization.selection_estimate` ; `by_positive_type` ; `background_diagnostic`.
- **`blanci heads-curve`** : `cli.heads_curve` → `annotation_curve` → `benchmark_data` ; `regularization_context` ; `regularizer_for` ; par cible `heads/head.choose_C` ; par tirage `_half`, `heads/head.fit_and_score`, `_ap` ; `_curve_summary` ; `_curve_gaps`. Puis `plot_curve`.
- **`blanci benchmark-all`** : `cli.benchmark_all` → `run_full_benchmark` → `load_oof` ; `recording_scores` ; `labels_fingerprint` ; `_encoder_costs` ; par source : `bootstrap_ci`, `average_precision`, `fold_mean_ap`, `recall_at_precision`, `cross_fitted_recall`, `wilson_interval` ; par source comparée : `paired_bootstrap` ; `with_holm`.
- **`blanci sources`** : `cli.sources` → `list_sources` → `load_oof`.
- **`blanci fusion`, `fusion-bench`, `ensemble`, `baselines`, `detector-bench`, `upstream-bench`** : leurs fonctions (`service.train_fusion`, `combination/stacking.fusion_benchmark`, `combination/ensemble.run_ensemble` et `concat_benchmark`, `heads/baselines.run_baselines`, `heads/detectors/base.evaluate_detector`, `service.upstream_bench`) appellent `evaluate`, `paired_bootstrap`, `to_recordings`, et rangent leurs scores par `oof_frame` et `save_oof`.
- **`blanci qc-calibrate`** : `cli.qc_calibrate` → `labelled_windows` (→ `_group`, `inputs/dataset.load_spans`, `_span_grid` → `embedding/grid.window_grid`, `inputs/dataset.interval_labels`, `_stored_indices`) ; `calibration_indices` → `inputs/qc.qc_indices` ; `suggest_thresholds` → `bagged_recordings` (→ `inputs/qc.muffled_runs`), `_suggest`, `_flagged_now`, `_flagged_suggested` ; `current_flags` → `bagged_recordings`, `inputs/qc.qc_flags`.
- **`blanci throughput`** : `cli.throughput` → `measure_in_subprocess` → lance `python -m blanci.evaluation.throughput <encodeur>` → `main` → `core/config.default_user_config`, `load_config` ; `measure_by_name` → `encoders.get_encoder`, `embedding/grid.overlap_from_cfg`, `measure_encoder` → `peak_memory_mb`. Puis `machine_description` ; `write_throughput_report` → `to_markdown`.
- **`blanci anuraset prepare`** : `cli.anuraset_prepare` → `prepare` → `extract_raw` ; `inputs/ingest.ingest`.
- **`blanci anuraset profile`** : `cli.anuraset_profile` → `read_strong_labels` ; `species_profile` ; avec `--audio` : `dominant_frequencies`, `suggest_species`.
- **`blanci anuraset benchmark`** : `cli.anuraset_benchmark` → `read_strong_labels` ; `run_anuraset_benchmark` → `read_weak_labels` ; par espèce et encodeur : `_encoder_windows`, `window_labels`, `weak_only_files`, `species_rows`, `probe_table` ; `compare_encoders` ; `with_holm`. Puis `write_anuraset_report`.
- **`blanci anuraset heads`** : `cli.anuraset_heads` → `read_strong_labels` ; `run_anuraset_heads` → `expand_methods` ; `read_weak_labels` ; `_encoder_windows` ; `regularization.domain_statistics` ; par espèce : `window_labels`, `species_rows`, puis par tête `regularizer_for`, `_inputs`, `heads/head.oof_scores`, `level_rows` ; `compare_to_reference` ; `window_comparisons` ; `selection_estimate` ; `_holm_by_level`. Puis `write_anuraset_heads_report`.
- **`blanci anuraset campaign`** : `cli.anuraset_campaign` → `prepare` ; `read_strong_labels` ; `species_profile` ; `dominant_frequencies` ; `choose_species` ; `embedding/embed.select_recordings` ; `campaign_recordings` ; par encodeur `embed.embed_recordings` ; `run_anuraset_campaign` → `run_anuraset_benchmark`, `rank_encoders`, `run_anuraset_heads`, `rank_heads` ; `write_campaign_report` → `write_anuraset_report`, `write_anuraset_heads_report`.
- **Scripts d'`anuraset/`** : `anuraset/encodage/encoder.py` (`read_strong_labels`, `campaign_recordings`), `anuraset/calcul/bench.py` (`read_strong_labels`, `run_anuraset_heads`, `CAMPAIGN_HEADS`), `anuraset/calcul/global_bench.py`, `rassembler*.py`, `apparie.py` et `anuraset/benchmarks/2026-09-29_anuraset_birdmae_base/amorcage.py` (`average_precision`, `recall_at_precision`, `holm`, `bootstrap_p`, `to_recordings`, `window_labels`, `read_weak_labels`, `weak_only_files`, `_encoder_windows`, `species_rows`, `_inputs`, `_holm_by_level`) ; `documentation/tableau-de-bord/durees_chants.py` (`read_strong_labels`).

## `evaluate.py` : folds, métriques, intervalles

Des calculs sur des tableaux : ne lit ni la base ni les fichiers, n'écrit rien. C'est le fichier le plus appelé du dossier : presque tout ce qui évalue y passe. Tests : [`tests/evaluation/test_evaluate.py`](../../tests/evaluation/test_evaluate.py).

#### `grouped_folds(y, groups, n_splits=5, seed=0, assignment=None)` — [evaluate.py:21](evaluate.py#L21)

- Quand : au début de tout calcul hors-fold : 8 fonctions, `heads/head.oof_scores`, `combination/fusion.fusion_oof` et `fusion_model_oof`, `combination/stacking.build_level1` et `fusion_benchmark`, `heads/baselines.oof_template_scores`, `heads/detectors/base.evaluate_detector`, `heads/regularization/selection.usable_folds`.
- Reçoit : les labels, le groupe de chaque exemple (le point d'écoute, ou le site), le nombre de folds, et au besoin une affectation `{groupe: fold}`. Renvoie : une liste de couples `(indices d'entraînement, indices de test)`.
- Fait : 1. refuse l'absence de groupes ou un seul groupe (`ValueError`) ; 2. avec `n_splits="lomo"` et sans affectation, en calcule une par `lomo_assignment` ; 3. avec une affectation, les folds sont exactement ceux-là ; un groupe absent de l'affectation lève `ValueError` ; 4. sans affectation, `StratifiedGroupKFold` de scikit-learn, avec au plus autant de folds que de groupes.
- Pourquoi (docstring, DECISIONS n° 91) : un groupe n'est jamais des deux côtés d'un fold, et avec l'affectation tous les modèles d'un benchmark sont jugés sur exactement les mêmes folds.

#### `fold_assignment(groups, has_positive, n_splits=5, seed=0)` — [evaluate.py:76](evaluate.py#L76)

- Quand : une fois par commande d'évaluation, dans `inputs/dataset.benchmark_folds` (appelée par `folds_for`).
- Reçoit : une ligne par enregistrement annoté : son groupe (micro) et s'il contient A. blanci. Renvoie : `{groupe: numéro de fold}`.
- Pourquoi (docstring, DECISIONS n° 91) : calculée sur les enregistrements, elle ne dépend ni de la grille ni des négatifs tirés pour chaque encodeur ; tous les modèles partagent donc ces folds.
- `lomo_assignment(groups, y, seed=0)` ([evaluate.py:59](evaluate.py#L59)) : R77, un fold par micro qui a des positifs ; les micros sans positif sont répartis au hasard entre ces folds, à parts égales. Moins de deux micros positifs : `ValueError`. `LOMO` ([evaluate.py:56](evaluate.py#L56)) est le texte `"lomo"`, la valeur de `head.n_splits` qui le demande.

#### `average_precision(y, scores)` — [evaluate.py:102](evaluate.py#L102)

- Quand : partout où une AP est calculée : dans ce fichier (`evaluate`, `fold_mean_ap`, `bootstrap_ci`, `paired_bootstrap`), dans les benchmarks, dans le choix de C des têtes et des fusions (`heads/head.select_C`, `regularization.choose_fusion_C`, `selection_estimate`, `torch_training`), dans `combination/fusion.fit_fusion_model`, dans les scripts d'`anuraset/calcul/` ; 35 fonctions de tests.
- Renvoie : l'AP de scikit-learn, ou `nan` s'il n'y a qu'une classe.

#### `evaluate(scores, labels, groups, level, precisions=(0.1, 0.5), n_boot=1000, seed=0, how="max", folds=None, clusters=None)` — [evaluate.py:328](evaluate.py#L328)

- Quand : pour produire la ligne de métriques d'un modèle : 12 fonctions, dans `service.py` (5 : `train_and_register`, `evaluate_holdout`, `evaluate_frozen`, `train_fusion`, `upstream_bench`), `combination/` (3), `benchmark.probe_table`, `head_benchmark.level_rows`, `heads/baselines.run_baselines`, `heads/detectors/base.evaluate_detector`.
- Reçoit : les scores, les labels, l'enregistrement de chaque fenêtre (`groups`), le niveau (`"window"` ou `"recording"`), les précisions plancher, le nombre de tirages du bootstrap, et au besoin le fold de test de chaque fenêtre (`folds`) et son micro (`clusters`). Renvoie : un dictionnaire : `level`, `n_pos`, `n_neg`, `ap`, `ap_lo`, `ap_hi`, et pour chaque précision `recall@p…`, ses bornes de Wilson et `threshold@p…`.
- Fait : 1. au niveau enregistrement, ramène d'abord les scores à un score par enregistrement (`to_recordings`) ; 2. AP et son intervalle par bootstrap ; l'unité tirée est l'enregistrement, ou le micro si `clusters` est donné ; 3. avec `folds`, ajoute `ap_fold_mean` et `n_folds_ap` ; 4. pour chaque précision : sans `folds`, le rappel au seuil choisi sur ces mêmes scores ; avec `folds`, le rappel au seuil choisi sur les autres folds (`cross_fitted_recall`), l'autre étant gardé sous `recall@p…_oracle` ; 5. intervalle de Wilson du rappel, compté en positifs retrouvés.
- Pourquoi (docstring, DECISIONS n° 139, 140) : les enregistrements d'un même micro partagent fond et faune, les tirer un à un rend l'intervalle trop étroit ; un seuil choisi et jugé sur les mêmes scores connaît les labels qu'il mesure, son rappel est optimiste.
- `to_recordings(scores, labels, recordings, how="max")` ([evaluate.py:314](evaluate.py#L314)) : un score par enregistrement (le maximum de ses fenêtres, ou avec `how="top3"` la moyenne des trois meilleures) et un label (positif si une fenêtre l'est). Appelée aussi par `service.train_fusion`, `service._recording_pair`, `combination/stacking.fusion_benchmark`, `benchmark.compare_encoders`, `head_benchmark._pair` et `_ap`, `regularization.selection_estimate`, `anuraset/calcul/global_bench.py`.
- `cluster_units(units, recordings, clusters)` ([evaluate.py:302](evaluate.py#L302)) : le micro de chaque enregistrement quand `clusters` est donné, sinon les enregistrements eux-mêmes ; c'est l'unité que le bootstrap tire.

#### `recall_at_precision(y, scores, min_precision)` — [evaluate.py:126](evaluate.py#L126)

- Quand : pour fixer le seuil d'une tête ou d'une fusion ([`service.train_and_register`](../service.py#L135), [`service.train_fusion`](../service.py#L1428), [`service.evaluate_holdout`](../service.py#L1065)), et dans `evaluate`, `cross_fitted_recall`, `full_benchmark.run_full_benchmark`, des scripts d'`anuraset/`.
- Renvoie : `(rappel, seuil)` : le plus grand rappel parmi les seuils qui gardent la précision au-dessus du plancher. `(nan, nan)` sans positif ; `(0, +inf)` si aucun seuil n'atteint le plancher.
- Détail (docstring) : à rappel égal, le seuil le plus élevé est retenu : même rappel, moins de candidats à vérifier.

#### `cross_fitted_recall(y, scores, folds, min_precision)` — [evaluate.py:147](evaluate.py#L147)

- Quand : dans `evaluate` (avec `folds`), `full_benchmark.run_full_benchmark`, et `heads/regularization/selection.cross_fitted_threshold` (appelée par `service.train_and_register`).
- Fait : pour chaque fold, choisit le seuil sur les scores des autres folds, puis l'applique à ce fold. Les fenêtres sans fold (`-1`) sont ignorées.
- Renvoie : `recall`, `k` (positifs retrouvés), `n_pos`, `precision` obtenue, `thresholds` (un seuil par fold) et `decided` (chaque exemple au-dessus du seuil de son fold).
- Pourquoi (docstring, R74, DECISIONS n° 140) : c'est ce que le seuil fera sur des micros qu'il n'a pas vus.

#### `fold_mean_ap(y, scores, folds)` — [evaluate.py:109](evaluate.py#L109)

- Quand : dans `evaluate` et `full_benchmark.run_full_benchmark`.
- Renvoie : `(moyenne des AP de chaque fold, nombre de folds comptés)` ; un fold sans les deux classes n'est pas compté.
- Pourquoi (docstring, DECISIONS n° 133, 135) : l'AP « poolée » met bout à bout les scores de modèles différents, un par fold ; s'ils n'ont pas retenu les mêmes réglages, leurs scores ne sont pas sur la même échelle et le classement commun se dégrade. L'AP par fold ne compare jamais deux modèles ; plus bruitée, elle se lit à côté de l'AP poolée, pas à sa place.

#### `bootstrap_ci(y, scores, units, metric=average_precision, n_boot=1000, seed=0, alpha=0.05)` — [evaluate.py:198](evaluate.py#L198)

- Quand : dans `evaluate` et `full_benchmark.run_full_benchmark`.
- Fait : tire `n_boot` fois, avec remise, autant d'unités entières qu'il y en a (enregistrements ou micros), recalcule la métrique, et rend les quantiles 2,5 % et 97,5 % ; les tirages où la métrique est indéfinie sont écartés.
- `_unit_index(units)` ([evaluate.py:191](evaluate.py#L191)) : la liste des lignes de chaque unité, calculée une fois.

#### `paired_bootstrap(y, scores_a, scores_b, units, metric=average_precision, n_boot=1000, seed=0, alpha=0.05)` — [evaluate.py:223](evaluate.py#L223)

- Quand : pour dire si un modèle A fait mieux qu'un modèle B sur les mêmes exemples : `service.train_fusion`, `service.upstream_bench`, `combination/ensemble.run_ensemble`, `combination/stacking.fusion_benchmark`, `benchmark.compare_encoders`, `full_benchmark.run_full_benchmark`, `head_benchmark.compare_to_reference` et `background_diagnostic`, `anuraset.window_comparisons`.
- Renvoie : `diff` (métrique de A moins métrique de B), `lo`, `hi` (intervalle de la différence sur les mêmes tirages), `significant` (l'intervalle exclut zéro), `p`.
- Détail (docstring) : la p-valeur n'est jamais nulle, sa plus petite valeur est 2/(n + 1) ; avec la correction de Holm sur M comparaisons, il faut donc environ 2M/α tirages pour qu'une différence puisse rester significative.
- `bootstrap_p(diffs)` ([evaluate.py:270](evaluate.py#L270)) : la p-valeur bilatérale, `2 × (min(nombre de Δ ≤ 0, nombre de Δ ≥ 0) + 1) / (n + 1)`, plafonnée à 1. Appelée aussi par `anuraset/calcul/apparie.py`.

#### `with_holm(comparisons, alpha=0.05)` — [evaluate.py:292](evaluate.py#L292)

- Quand : à la fin de tout tableau de comparaisons : `benchmark.compare_encoders`, `full_benchmark.run_full_benchmark`, `head_benchmark.compare_to_reference`, `anuraset._holm_by_level` et `run_anuraset_benchmark`.
- Fait : ajoute `p_holm` et `significant_holm` au tableau (colonne `p`) ; la colonne `significant`, calculée comparaison par comparaison, reste pour mémoire.
- Pourquoi (docstring, DECISIONS n° 139) : vingt têtes comparées à 5 % donnent une « victoire » par hasard.
- `holm(p_values)` ([evaluate.py:278](evaluate.py#L278)) : les p-valeurs ajustées de Holm ; les valeurs manquantes restent manquantes et ne comptent pas dans la famille. Appelée aussi par deux scripts d'`anuraset/calcul/`.

#### `wilson_interval(k, n, z=1.96)` — [evaluate.py:180](evaluate.py#L180)

- Quand : pour l'intervalle d'une proportion : `evaluate`, `recall_by_group`, `full_benchmark.run_full_benchmark`, [`service.estimate_prevalence`](../service.py#L309), `heads/signal_processing.gate_sweep`, `results/activity.py`.
- Renvoie : les bornes de l'intervalle de Wilson de `k/n` ; `(nan, nan)` si `n` vaut 0.

#### `recall_by_group(scores, labels, strata, threshold)` — [evaluate.py:401](evaluate.py#L401)

- Quand : dans [`service.evaluate_holdout`](../service.py#L1065) et [`service.evaluate_frozen`](../service.py#L1339), par qualité du chant (A, B, C) et par site.
- Renvoie : un tableau par strate : nombre de positifs, rappel au seuil, intervalle de Wilson. Les négatifs sont ignorés ; une strate non renseignée apparaît comme `?`.

#### `false_alarms_per_hour(scores, labels, threshold, audio_hours)` — [evaluate.py:428](evaluate.py#L428)

- Quand : dans [`service.evaluate_frozen`](../service.py#L1339) seulement.
- Renvoie : le nombre de négatifs au-dessus du seuil par heure d'audio ; `nan` si la durée est nulle.
- Pourquoi (docstring) : n'a de sens que sur des enregistrements écoutés en entier (jeu gelé) ; sur des négatifs tirés, le dénominateur ne représente pas l'audio réel.

## `oof.py` : le stock des scores hors-fold

Chaque benchmark range ses scores dans `paths.reports/oof/<source>-<empreinte>.parquet`, une ligne par fenêtre évaluée. Lit et écrit ces fichiers ; lit la base pour l'empreinte des labels. Utilisé par tous les benchmarks, par `blanci sources`, `blanci ensemble` et `blanci benchmark-all`. Tests : dans [`tests/evaluation/test_benchmark.py`](../../tests/evaluation/test_benchmark.py).

Colonnes (`OOF_COLUMNS`, [oof.py:34](oof.py#L34)) : `source`, `kind`, `window_id`, `recording_id`, `offset_s`, `dur_s`, `point`, `site`, `y`, `presumed`, `score`, `fold`, `fingerprint`. Sortes de sources (`KINDS`, [oof.py:49](oof.py#L49)) : `encoder_head`, `baseline`, `fusion`, `ensemble`, `detector`.

#### `labels_fingerprint(con, cfg)` — [oof.py:52](oof.py#L52)

- Quand : une fois par benchmark, avant d'enregistrer ses scores : `benchmark.benchmark_encoder`, `head_benchmark.run_head_benchmark`, `full_benchmark.run_full_benchmark`, `combination/ensemble.py` (2), `combination/stacking.fusion_benchmark`, `heads/baselines.run_baselines`, `heads/detectors/base.evaluate_detector`.
- Renvoie : 12 caractères hexadécimaux, le début du SHA-256 de ce qui définit un benchmark : les labels courants, les jeux gelés, les réglages des négatifs appariés, `negatives_per_positive`, `head.n_splits`, `head.seed`.
- Pourquoi (docstring du fichier) : deux sources d'empreintes différentes n'ont pas vu les mêmes données ; le benchmark complet le signale au lieu de les comparer en silence. L'empreinte change dès qu'un label est ajouté.

#### `oof_frame(source, kind, data, scores, assignment, fingerprint, dur_s=None)` — [oof.py:79](oof.py#L79)

- Quand : juste avant `save_oof`, par les mêmes appelants.
- Reçoit : le nom de la source (`<stock>/<tête>`, `baseline/…`, `<stock>/fusion:…`, `ensemble:…`), sa sorte, le tableau des fenêtres évaluées, leurs scores, l'affectation des points aux folds, l'empreinte. Renvoie : le tableau au format du stock.
- Détail : une sorte hors de `KINDS` ou un nombre de scores différent du nombre de fenêtres lève `ValueError` ; sans affectation, le fold vaut −1.

#### `save_oof(cfg, frame)` — [oof.py:116](oof.py#L116)

- Écrit, en le remplaçant, le fichier d'une seule source ; plusieurs sources dans le tableau lèvent `ValueError`.
- `source_path(cfg, source)` ([oof.py:71](oof.py#L71)) : le nom du fichier : le nom de la source, les caractères interdits remplacés par `_`, suivi de 6 caractères de l'empreinte du nom exact, pour que deux sources ne tombent jamais sur le même fichier.

#### `load_oof(cfg, sources=None)` — [oof.py:130](oof.py#L130)

- Quand : [`combination/ensemble.recording_table`](../combination/ensemble.py#L47), `full_benchmark.run_full_benchmark`, `list_sources`.
- Renvoie : le tableau de toutes les sources enregistrées, ou de celles demandées ; une source demandée sans fichier lève `ValueError`.
- Détail : sans liste, les fichiers d'une sorte qui n'est plus dans `KINDS` sont ignorés, avec un message une seule fois par processus (`_IGNORED`, [oof.py:127](oof.py#L127)) ; d'après la docstring, ce sont d'anciennes sources `external/…` qu'aucune commande ne sait relancer.

#### `list_sources(cfg)` — [oof.py:155](oof.py#L155)

- Quand : c'est la commande `blanci sources` ([`cli.sources`](../cli.py#L1211)).
- Renvoie : une ligne par source : sorte, nombre de fenêtres, d'enregistrements, de positifs, empreinte.

#### `recording_scores(table)` — [oof.py:175](oof.py#L175)

- Quand : `combination/ensemble.recording_table` et `full_benchmark.run_full_benchmark`.
- Renvoie : par source et par enregistrement : le score maximal de ses fenêtres, son label, son point d'écoute et son fold (−1 s'il n'en a pas).
- Pourquoi (docstring du fichier) : une fenêtre de 3 s et une de 5 s ne se comparent pas, deux scores maximaux par enregistrement si.

## `benchmark.py` : le benchmark des encodeurs

Compare des encodeurs avec les mêmes labels, les mêmes folds et quatre têtes légères appelées sondes (`PROBES`, [benchmark.py:47](benchmark.py#L47)) : `knn`, `simple_prototype`, `prototype`, `logistic` ; plus `attentive` si des jetons existent. Lit : la base, les stocks d'embeddings, le stock de jetons. Écrit : les scores hors-fold de chaque sonde (`oof/`), et les rapports `benchmark.csv`, `benchmark_comparaisons.csv`, `benchmark.md` sous `paths.reports`. Tests : [`tests/evaluation/test_benchmark.py`](../../tests/evaluation/test_benchmark.py).

#### `run_benchmark(con, encoder_ids, store_root, cfg, filters=None)` — [benchmark.py:200](benchmark.py#L200)

- Quand : c'est la commande `blanci benchmark` ([`cli.benchmark`](../cli.py#L728)).
- Renvoie : `(tableau par encodeur, sonde et niveau, trié par AP décroissante ; comparaisons appariées entre encodeurs)`.

#### `benchmark_encoder(con, encoder_id, store_root, cfg, filters=None)` — [benchmark.py:129](benchmark.py#L129)

- Quand : une fois par encodeur dans `run_benchmark`.
- Renvoie : `(tableau, scores hors-fold de la sonde logistique, labels, enregistrements)` ; les trois derniers servent aux comparaisons entre encodeurs.
- Fait : 1. charge les fenêtres étiquetées, leurs négatifs appariés et leurs embeddings, jeu gelé exclu ; une seule classe lève `ValueError` ; 2. `probe_table` avec les folds communs, les jetons s'il y en a, et le micro comme unité du bootstrap ; 3. range les scores hors-fold de chaque sonde sous `<stock>/<sonde>` ; 4. ajoute au tableau la dimension, la durée de fenêtre, `knn_top1`, le débit du dernier passage d'`embed` (`last_run`) et le nombre de micros.

#### `probe_table(X, y, groups, recordings, n_splits=5, C_grid=None, precisions=(0.1, 0.5), n_boot=1000, seed=0, tokens=None, gated=None, assignment=None, calibration=None, clusters=None)` — [benchmark.py:72](benchmark.py#L72)

- Quand : `benchmark_encoder` et [`anuraset.run_anuraset_benchmark`](anuraset.py#L541).
- Renvoie : `(une ligne par sonde et par niveau, les scores hors-fold de chaque sonde)`.
- Détail : avec une recalibration par fold (`benchmark.fold_calibration`), ajoute `ap_raw`, l'AP poolée des scores d'avant recalibration (DECISIONS n° 133, 135).

#### `compare_encoders(per_encoder, cfg, clusters=None)` — [benchmark.py:219](benchmark.py#L219)

- Quand : `run_benchmark` et `anuraset.run_anuraset_benchmark`.
- Fait : pour chaque paire d'encodeurs, bootstrap apparié de la sonde logistique sur leurs enregistrements communs, en tirant des micros entiers si `clusters` est donné ; p-valeurs corrigées de Holm sur toutes les paires.
- Pourquoi (docstring) : le niveau enregistrement est le seul comparable, deux encodeurs de fenêtres différentes n'ayant pas la même grille.

#### `knn_top1(X, y, recordings, chunk_rows=4096)` — [benchmark.py:51](benchmark.py#L51)

- Renvoie : la part des positifs dont le plus proche voisin (cosinus) est lui aussi positif ; `nan` sans positif.
- Détail (docstring) : les fenêtres du même enregistrement sont exclues du voisinage ; voisines dans le temps, elles partagent le fond sonore et donneraient un score de fuite.

#### `write_report(results, comparisons, reports_dir, stem="benchmark")` — [benchmark.py:307](benchmark.py#L307)

- Quand : à la fin de `blanci benchmark`.
- Écrit : `<stem>.csv` (complet), `<stem>_comparaisons.csv`, `<stem>.md` (colonnes de `REPORT_COLUMNS`, [benchmark.py:269](benchmark.py#L269)). Renvoie les trois chemins.

#### `to_markdown(df, floatfmt="{:.3f}")` — [benchmark.py:290](benchmark.py#L290)

- Quand : pour tous les rapports markdown du projet : 6 commandes de `cli.py` (`heads`, `heads_curve`, `upstream_bench_command`, `fusion_bench`, `ensemble`, `benchmark_all`), `anuraset.py` (3), `write_report`, `throughput.write_throughput_report`, `heads/baselines.write_baseline_report`.
- Renvoie : un tableau markdown ; les nombres à virgule sont mis au format donné, une valeur manquante devient `—`, un tableau vide `_(vide)_`.
- Pourquoi (docstring) : `pandas.to_markdown` exige le paquet `tabulate`.

## `head_benchmark.py` : le benchmark des têtes

Compare toutes les têtes sur les embeddings d'un même encodeur. Lit : la base, le stock d'embeddings, le stock de jetons. Écrit : les scores hors-fold de chaque tête (`oof/`) ; les rapports sont écrits par `cli.heads` et `cli.heads_curve`. Tests : [`tests/evaluation/test_head_benchmark.py`](../../tests/evaluation/test_head_benchmark.py).

#### `run_head_benchmark(con, cfg, encoder_id, methods=None, filters=None)` — [head_benchmark.py:228](head_benchmark.py#L228)

- Quand : c'est la commande `blanci heads` ([`cli.heads`](../cli.py#L538)).
- Renvoie : un dictionnaire : `table` (une ligne par tête et par niveau), `comparisons` (chaque tête contre la référence `head.reference`, `logistic` par défaut), `selection`, `by_positive`, `background`, `scores`, `n_mics`.
- Fait : 1. `benchmark_data` ; 2. liste des têtes : celles demandées (`expand_methods`), sinon toutes (`head_methods`) ; 3. `regularization_context` ; 4. pour chaque tête : `regularizer_for` sépare le nom de la tête et ses régularisations, `_inputs` choisit l'entrée (embedding ou jetons), `oof_scores` donne les scores hors-fold sur les folds communs, rangés sous `<stock>/<tête>`, puis `level_rows` ; 5. comparaisons à la référence ; 6. `selection_estimate` (R74, R80) : la tête gagnante doit-elle sa place à la chance ; 7. AP par type de positif ; 8. diagnostic du fond capté.

#### `benchmark_data(con, cfg, encoder_id, filters=None)` — [head_benchmark.py:203](head_benchmark.py#L203)

- Quand : `run_head_benchmark`, `annotation_curve`, [`combination/ensemble.concat_training_set`](../combination/ensemble.py#L136).
- Renvoie : `(fenêtres évaluées, embeddings, jetons ou None)` : les mêmes fenêtres que le benchmark des encodeurs ; une seule classe lève `ValueError`.
- `load_tokens(con, cfg, encoder_id, window_ids)` ([head_benchmark.py:187](head_benchmark.py#L187)) : les jetons de ces fenêtres, relus du stock de jetons, et remis en grille temps × fréquence si l'entrée de l'encodeur dans la configuration a un réglage `token_grid`.

#### `head_methods(tokens, variants=None)` — [head_benchmark.py:115](head_benchmark.py#L115)

- Renvoie : la liste des têtes par défaut : celles de `heads/head.METHODS` qui ne demandent pas de jetons, sans `gated` ni `dann` si torch n'est pas installé, avec `knn:k=3` et `knn:k=5` juste après `knn` (`NEIGHBOR_DEFAULTS`, [head_benchmark.py:97](head_benchmark.py#L97)) ; avec des jetons, `logistic:<pooling>` pour chaque pooling possible et les têtes de `TOKEN_METHODS` ([head_benchmark.py:93](head_benchmark.py#L93)) : `attentive`, `proto_probe`, `cascade` ; puis les variantes de `regularization.variants`.

#### `expand_methods(methods)` — [head_benchmark.py:129](head_benchmark.py#L129)

- Quand : `run_head_benchmark`, `annotation_curve`, `anuraset.run_anuraset_heads`.
- Fait : remplace le groupe `losses` par la logistique et toutes les têtes `loss:<nom>`, et le groupe `neighbors` par la liste `NEIGHBORS` ([head_benchmark.py:98](head_benchmark.py#L98)) ; `None` reste `None`.

#### `regularization_context(con, cfg, encoder_id, data, specs, filters=None)` — [head_benchmark.py:143](head_benchmark.py#L143)

- Quand : `run_head_benchmark` et `annotation_curve`.
- Renvoie : un `regularization.Context` : le groupe de chaque fenêtre (`regularization.by`, le point d'écoute par défaut), les négatifs réellement annotés, la classe de chaque fenêtre ; et seulement si une tête de la liste le demande, les statistiques du stock par groupe (`store_domain_statistics`) et un réservoir de fenêtres non annotées (`sample_pool`, 20 000 par défaut, réglage `regularization.R81.pool`).

#### `level_rows(oof, y, recordings, cfg, labels, clusters=None)` — [head_benchmark.py:299](head_benchmark.py#L299)

- Quand : `run_head_benchmark` et `anuraset.run_anuraset_heads`.
- Renvoie : deux lignes de métriques (fenêtres, enregistrements) pour des scores hors-fold, par `evaluate` avec les folds ; ajoute `ap_raw` si les folds ont été recalibrés.

#### `compare_to_reference(scores, reference, y, recordings, cfg, clusters=None)` — [head_benchmark.py:362](head_benchmark.py#L362)

- Renvoie : chaque tête contre la référence, au niveau enregistrement, par bootstrap apparié (micros entiers avec `clusters`), p-valeurs corrigées de Holm ; un tableau vide si la référence n'est pas dans la liste.
- `_pair(a, b, y, recordings)` ([head_benchmark.py:424](head_benchmark.py#L424)) : les scores par enregistrement de deux têtes, alignés.

#### `background_diagnostic(scores, y, recordings, cfg, clusters=None)` — [head_benchmark.py:392](head_benchmark.py#L392)

- Renvoie : la comparaison appariée du prototype différentiel (`prototype`) au prototype simple, avec un verdict en une phrase : si le différentiel fait nettement mieux, « le fond sonore pollue l'embedding » ; si le simple fait mieux, « le différentiel retire aussi du chant » ; sinon « pas d'écart net ». Un dictionnaire vide si l'une des deux têtes manque.

#### `by_positive_type(scores, data, min_windows=5)` — [head_benchmark.py:333](head_benchmark.py#L333)

- Renvoie : l'AP de chaque tête par type de positif (`blanci_solo`, `blanci_chorus`, `blanci`) : les positifs de ce type contre tous les négatifs, au niveau fenêtre ; les types de moins de 5 fenêtres sont omis.

#### `annotation_curve(con, cfg, encoder_id, methods=None, k_grid=None, repeats=None, by=None, level="recording")` — [head_benchmark.py:451](head_benchmark.py#L451)

- Quand : c'est la commande `blanci heads-curve` ([`cli.heads_curve`](../cli.py#L611)).
- Renvoie : `runs` (une ligne par tête, cible, tirage et k), `summary` (moyenne par tête et k), `gaps` (écart de chaque tête à la référence, par k, avec intervalle).
- Fait : 1. réglages par défaut dans `head.curve` (têtes `prototype` et `logistic`, k = 0, 1, 2, 5, 10, 20, 5 tirages, cible = point d'écoute) ; 2. écarte les fenêtres arrêtées par une porte ; 3. pour chaque cible ayant au moins deux enregistrements positifs : choisit le C de chaque tête une fois, sur les autres cibles (`choose_C`) ; 4. pour chaque tirage : coupe en deux moitiés au hasard les enregistrements positifs, les négatifs annotés et les négatifs présumés de la cible ; une moitié sert de test, fixe pour tous les k, l'autre de réserve ; 5. pour chaque k : entraînement = toutes les autres cibles + les négatifs présumés de la réserve + k enregistrements positifs de la réserve et une part proportionnelle de ses négatifs annotés ; chaque tête est entraînée et notée sur le test (`fit_and_score`) ; 6. résumé et écarts.
- Pourquoi (docstring du fichier, DECISIONS n° 93) : tester si le prototype différentiel fait mieux que la régression logistique sur un site peu annoté. k = 0 est le transfert pur ; les k sont emboîtés (k = 2 contient k = 1).
- `_half(items, rng)` ([head_benchmark.py:443](head_benchmark.py#L443)) : `(test, réserve)`, moitié au hasard, le test prenant l'élément impair. `_stable_id(text)` ([head_benchmark.py:438](head_benchmark.py#L438)) : un entier stable tiré du nom de la cible, pour la graine (le `hash()` de Python change à chaque lancement). `_ap(scores, y, recordings, level)` ([head_benchmark.py:569](head_benchmark.py#L569)) : l'AP au niveau demandé. `_curve_summary(runs)` ([head_benchmark.py:576](head_benchmark.py#L576)) : moyenne, écart-type et effectifs par tête et k. `_curve_gaps(runs, reference, n_boot, seed)` ([head_benchmark.py:589](head_benchmark.py#L589)) : écart moyen à la référence par k, apparié par cible et tirage, intervalle par bootstrap sur les cibles.

#### `plot_curve(summary, path)` — [head_benchmark.py:623](head_benchmark.py#L623)

- Écrit : un PNG de l'AP moyenne selon k, une ligne par tête ; rend `False` si matplotlib n'est pas installé.

#### Fonctions privées du fichier

- `_inputs(method, X, tokens)` ([head_benchmark.py:174](head_benchmark.py#L174)) : `logistic:<pooling>` devient `logistic` sur les jetons résumés par ce pooling ; `attentive` et `proto_probe` reçoivent les jetons ; les autres l'embedding. Sans jetons, les deux premiers cas lèvent `ValueError` (« demande les jetons de l'encodeur (blanci tokens) »). Appelée aussi par `anuraset.run_anuraset_heads` et deux scripts d'`anuraset/`.

## `full_benchmark.py` : le benchmark complet

Classe tout ce que le stock hors-fold contient. Lit : `paths.reports/oof/`, la base (empreinte des labels, table `models`). N'écrit rien : `cli.benchmark_all` écrit `benchmark_complet.csv`, `benchmark_complet_comparaisons.csv`, `benchmark_complet.md`. Test : dans `tests/combination/test_stacking.py`.

#### `run_full_benchmark(con, cfg, sources=None, reference=None, common=True)` — [full_benchmark.py:69](full_benchmark.py#L69)

- Quand : c'est la commande `blanci benchmark-all` ([`cli.benchmark_all`](../cli.py#L1247)).
- Renvoie : `table` (une ligne par source), `comparisons` (chaque source contre la référence), `by_site` (rappel par site), `reference`, `reference_post_hoc`, `n_common_recordings`.
- Fait : 1. charge les sources, les ramène à un score par enregistrement ; 2. par défaut (`common`), ne garde que les enregistrements évalués par toutes les sources ; s'il n'y en a aucun, `ValueError` ; 3. pour chaque source : AP et intervalle (bootstrap par micro), AP moyenne par fold, rappel aux précisions plancher avec un seuil choisi sur les autres folds, rappel par site à la précision la plus basse, coût de l'encodeur (`_encoder_costs`), et `up_to_date` (son empreinte est celle des labels d'aujourd'hui) ; 4. référence : celle donnée, sinon `benchmark.reference`, sinon la meilleure AP, alors marquée `reference_post_hoc` ; 5. comparaison appariée de chaque source à la référence, p-valeurs corrigées de Holm.
- Pourquoi (docstring du fichier) : une référence choisie après coup doit une part de sa place à la chance ; « meilleur » seulement si la p-valeur corrigée de Holm est sous 5 %.
- `_encoder_costs(con)` ([full_benchmark.py:50](full_benchmark.py#L50)) : dimension, fenêtre et débit du dernier passage de chaque stock, relus de la table `models`.

## `anuraset.py` : le pré-benchmark AnuraSet

AnuraSet est un jeu public de 1 612 enregistrements d'une minute, 4 sites du Brésil, 42 espèces, chaque chant daté. Le fichier y juge les encodeurs et les têtes comme sur les données de l'ONF, sur des espèces à note brève. Tout vit à part : la configuration `anuraset/anuraset.yaml` donne une base, des stocks et des rapports séparés. Lit : `raw_data.zip`, `strong_labels.zip`, `weak_labels.csv`, l'audio extrait, la base et les stocks d'AnuraSet. Écrit : l'audio extrait sous `paths.raw/anuraset/<site>/`, la table `recordings` (par `ingest`), les rapports `anuraset_*.csv` et `.md`. Tests : [`tests/evaluation/test_anuraset.py`](../../tests/evaluation/test_anuraset.py).

#### `prepare(con, cfg)` — [anuraset.py:79](anuraset.py#L79)

- Quand : `blanci anuraset prepare` ([`cli.anuraset_prepare`](../cli.py#L1548)) et au début de `blanci anuraset campaign`.
- Fait : extrait l'audio de l'archive (`anuraset.archive`) puis lance l'inventaire sans contrôle qualité ni empreinte de fichier. Renvoie : fichiers extraits, enregistrements ajoutés, erreurs.
- `extract_raw(archive, root)` ([anuraset.py:52](anuraset.py#L52)) : copie les `.wav` et `.flac` de l'archive sous `<racine>/anuraset/<site>/` ; un fichier déjà extrait et de même taille est sauté. `file_key(name)` ([anuraset.py:45](anuraset.py#L45)) : le nom du fichier sans dossier ni extension, en minuscules ; c'est la clé qui relie audio, chants datés et labels faibles.

#### `read_strong_labels(source)` — [anuraset.py:91](anuraset.py#L91)

- Quand : au début de toutes les commandes `blanci anuraset …` sauf `prepare`, dans quatre scripts d'`anuraset/` et dans `documentation/tableau-de-bord/durees_chants.py`.
- Reçoit : `strong_labels.zip` ou son dossier extrait. Renvoie : un tableau, un chant par ligne : `file_key`, `site`, `start_s`, `end_s`, `species`, `quality` (`low`, `medium`, `high`).

#### `species_profile(calls, max_call_s=5.0)` — [anuraset.py:132](anuraset.py#L132)

- Renvoie : par espèce : nombre de chants, d'enregistrements, de sites, durée médiane et 90ᵉ centile. Les annotations de plus de `max_call_s` sont des chœurs continus : comptées à part (`n_long`), exclues des durées.

#### `dominant_frequencies(calls, raw_root, recordings, per_species=30, max_call_s=1.0, fmin_hz=500.0, seed=0)` — [anuraset.py:151](anuraset.py#L151)

- Fait : pour chaque espèce, lit jusqu'à 30 chants brefs de qualité moyenne ou haute dans l'audio, prend le pic du spectre au-dessus de 500 Hz, et rend la médiane de ces fréquences.

#### `suggest_species(profile, band_hz=(3000.0, 6000.0), max_duration_s=0.3, min_calls=300, min_sites=2)` et `choose_species(profile, n=3)` — [anuraset.py:189](anuraset.py#L189)

- `suggest_species` (pour `blanci anuraset profile --audio`) : les espèces proches d'A. blanci : note brève, dominante entre 3 et 6 kHz, au moins 300 chants, sur au moins deux sites.
- `choose_species` ([anuraset.py:669](anuraset.py#L669), pour `campaign`) : jusqu'à `n` espèces, par niveaux du plus strict au plus large (`SPECIES_LEVELS`, [anuraset.py:662](anuraset.py#L662)) : proche d'A. blanci, puis note ≤ 0,5 s en 2–7 kHz et ≥ 150 chants, puis note ≤ 1 s et ≥ 100 chants sans condition de fréquence ; toujours au moins deux sites. La colonne `criterion` dit quel niveau a retenu l'espèce.

#### `window_labels(windows, calls, species, eps=1e-6, unsure_files=frozenset())` — [anuraset.py:207](anuraset.py#L207)

- Quand : `run_anuraset_heads`, `run_anuraset_benchmark`, et deux scripts d'`anuraset/`.
- Renvoie : pour chaque fenêtre, 1 si elle contient un chant entier de l'espèce (ou tient tout entière dans un chœur annoté), 0 si aucun chant de l'espèce ne la touche, `NaN` si elle coupe un chant (fenêtre écartée).
- Détails : un chant qu'aucune fenêtre ne contient entier rend positive la fenêtre qui en porte la plus grande part (DECISIONS n° 138) ; toutes les fenêtres des fichiers de `unsure_files` sont écartées.
- `weak_only_files(weak, calls, species)` ([anuraset.py:268](anuraset.py#L268)) : les fichiers où les labels faibles signalent l'espèce sans aucun chant daté : ses chants y sont peut-être, non datés. `read_weak_labels(path)` ([anuraset.py:251](anuraset.py#L251)) : lit `weak_labels.csv` ; `None` sans chemin ; un chemin donné mais absent lève `FileNotFoundError`.

#### `run_anuraset_benchmark(con, cfg, encoder_ids, species, calls)` — [anuraset.py:541](anuraset.py#L541)

- Quand : `blanci anuraset benchmark` ([`cli.anuraset_benchmark`](../cli.py#L1743)) et `run_anuraset_campaign`.
- Renvoie : `(tableau encodeur × espèce × sonde × niveau, comparaisons appariées par espèce)`.
- Fait : pour chaque espèce et chaque encodeur : labels des fenêtres, tirage des fenêtres jugées (`species_rows`), puis `probe_table` avec le site comme groupe ; ensuite `compare_encoders` par espèce et correction de Holm sur toutes les paires de toutes les espèces.
- `_encoder_windows(con, cfg, encoder_id)` ([anuraset.py:283](anuraset.py#L283)) : tout le stock de l'encodeur, avec la clé de fichier, le site et la durée de fenêtre. `species_rows(meta, y, species, negatives_per_positive, rng, encoder_id="")` ([anuraset.py:300](anuraset.py#L300)) : tous les positifs de l'espèce, et dans chaque site `negatives_per_positive` négatifs par positif du site ; moins de 10 fenêtres positives lève `ValueError`.
- `write_anuraset_report(results, comparisons, reports_dir)` ([anuraset.py:591](anuraset.py#L591)) : `anuraset_benchmark.csv`, `anuraset_comparaisons.csv`, `anuraset_benchmark.md`.

#### `run_anuraset_heads(con, cfg, encoder_id, species, calls, methods=None)` — [anuraset.py:341](anuraset.py#L341)

- Quand : `blanci anuraset heads` ([`cli.anuraset_heads`](../cli.py#L1586)), `run_anuraset_campaign`, `anuraset/calcul/bench.py`.
- Renvoie : `table` (espèce × tête × niveau), `sites` (AP de chaque tête sur chaque site tenu à l'écart), `comparisons` (contre la référence, aux niveaux enregistrement et fenêtre), `selection`.
- Fait : têtes par défaut `anuraset.heads`, sinon `HEADS` ([anuraset.py:325](anuraset.py#L325), 12 têtes), sans `dann` ni `gated` si torch manque ; pour chaque espèce, chaque tête est évaluée avec un fold par site (`n_splits` = nombre de sites) ; le groupe des régularisations est le site.
- Pourquoi (docstring, R78) : en attendant des positifs hors de Mataroni, c'est là qu'on juge ce qui généralise d'un site à l'autre.
- `window_comparisons(scores, reference, y, recordings, cfg)` ([anuraset.py:470](anuraset.py#L470)) : chaque tête contre la référence au niveau fenêtre, en tirant des enregistrements entiers. Pourquoi (docstring, DECISIONS n° 136) : une espèce commune chante dans presque chaque enregistrement d'une minute, l'AP par enregistrement n'y est plus définie.
- `_holm_by_level(comparisons)` ([anuraset.py:460](anuraset.py#L460)) : la correction de Holm sur toutes les têtes et toutes les espèces d'un même niveau.
- `write_anuraset_heads_report(out, encoder_id, reports_dir)` ([anuraset.py:501](anuraset.py#L501)) : `anuraset_tetes_<encodeur>_<clé>.csv` pour chaque tableau, et `anuraset_tetes_<encodeur>.md`.

#### `run_anuraset_campaign(con, cfg, encoder_ids, calls, species, methods=None)` — [anuraset.py:740](anuraset.py#L740)

- Quand : `blanci anuraset campaign` ([`cli.anuraset_campaign`](../cli.py#L1657)).
- Fait : benchmark des encodeurs sur les espèces choisies, classement (`rank_encoders`), puis toutes les têtes de `CAMPAIGN_HEADS` ([anuraset.py:642](anuraset.py#L642), 14 têtes) sur le meilleur encodeur, et leur classement (`rank_heads`).
- `rank_encoders(results, level="window")` ([anuraset.py:696](anuraset.py#L696)) : encodeurs classés par l'AP moyenne de la sonde logistique sur les espèces, au niveau fenêtre. `rank_heads(table, level="window")` ([anuraset.py:707](anuraset.py#L707)) : têtes classées par leur rang moyen sur les espèces, puis par leur AP moyenne.
- `campaign_recordings(recordings, calls, weak_labels=None)` ([anuraset.py:719](anuraset.py#L719)) : les enregistrements à encoder : ceux qui ont au moins un chant daté, et ceux que les labels faibles disent sans aucune espèce (vrais négatifs) ; d'après la docstring, 13 enregistrements sur 1 612 sont écartés. Appelée par `cli.anuraset_campaign` et `anuraset/encodage/encoder.py`.
- `write_campaign_report(out, reports_dir)` ([anuraset.py:767](anuraset.py#L767)) : `anuraset_campagne.md`, et les deux rapports détaillés.

## `qc_calibration.py` : calibrer les seuils du contrôle qualité

Regarde ce que les seuils du contrôle audio (`qc.muffled_hf_ratio`, `qc.silent_dbfs`) feraient aux enregistrements déjà étiquetés, et propose un seuil. Lit : les tables `labels`, `windows`, `recordings`, `spans`, `intervals` ; l'audio des enregistrements étiquetés, sous `paths.raw`. N'écrit rien, ni dans la base ni dans la configuration : `cli.qc_calibrate` écrit `qc_indices.csv`, `qc_calibration.csv` et `qc_drapeaux_actuels.csv` sous `paths.reports`. Tests : [`tests/evaluation/test_qc_calibration.py`](../../tests/evaluation/test_qc_calibration.py).

#### `labelled_windows(con)` — [qc_calibration.py:52](qc_calibration.py#L52)

- Quand : première étape de `blanci qc-calibrate` ([`cli.qc_calibrate`](../cli.py#L1494)).
- Renvoie : une ligne par fenêtre étiquetée (son dernier label), avec le chemin, le jeu, le site, le micro, l'heure, et un groupe : `muffled` (label `artefact_muffled` ou l'ancien `artefact_in_bag`), `rain` (label `rain`, ou mention de pluie dans les conditions), `blanci` (label positif), `other`.
- Fait : ajoute les fenêtres de la grille de 3 s couvertes par un extrait écouté (annotation par intervalles), étiquetées par `inputs/dataset.interval_labels` ; une fenêtre que son label et un intervalle étiquettent en désaccord est écartée ; d'accord, le label de fenêtre prime. Attache au tableau (`attrs["series"]`) les indices déjà rangés par le contrôle audio pour tous les enregistrements.
- `_group(label, tags)` ([qc_calibration.py:40](qc_calibration.py#L40)) : le groupe ci-dessus. `_span_grid(con, window_s=3.0, hop_s=1.5)` ([qc_calibration.py:97](qc_calibration.py#L97)) : la grille des enregistrements qui ont un extrait écouté, calculée sans la table `windows`, qu'un extrait ne remplit pas. `_stored_indices(con)` ([qc_calibration.py:120](qc_calibration.py#L120)) : `hf_ratio` et `rms_dbfs` rangés dans `recordings.qc_flags`.

#### `calibration_indices(windows, raw_root, channel=0)` — [qc_calibration.py:143](qc_calibration.py#L143)

- Fait : lit chaque enregistrement une fois, en entier, sur le canal demandé ; calcule les indices du contrôle audio sur l'enregistrement (colonnes `r_…`) et sur chaque fenêtre étiquetée (colonnes `w_…`). Un fichier illisible est sauté sans message.

#### `suggest_thresholds(indices, thresholds)` — [qc_calibration.py:252](qc_calibration.py#L252)

- Renvoie : une ligne par drapeau de `FLAGS` ([qc_calibration.py:34](qc_calibration.py#L34) : `muffled` sur `hf_ratio`, `silent` sur `rms_dbfs`, tous deux levés quand l'indice est sous le seuil) : le seuil actuel, le nombre de cibles et d'enregistrements à A. blanci, combien sont signalés au seuil actuel, le seuil proposé, combien de cibles il signalerait, et la raison.
- Fait : tout se juge au niveau de l'enregistrement. Un enregistrement est « à protéger » s'il contient une fenêtre A. blanci ; « cible » de `muffled` s'il contient une fenêtre de ce groupe et aucun chant. Le seuil proposé est celui qui signale le plus de cibles sans toucher aucun enregistrement protégé.
- `_suggest(targets, protected, direction)` ([qc_calibration.py:171](qc_calibration.py#L171)) : si cibles et protégés sont séparés, le milieu entre les deux ; s'ils se chevauchent, juste sous le protégé le plus bas ; sans cible ou sans protégé, `nan` et la raison.
- `bagged_recordings(indices, thresholds)` ([qc_calibration.py:195](qc_calibration.py#L195)) : les enregistrements que `muffled` retient vraiment : ratio sous le seuil, pas silencieux, et dans une suite d'au moins `muffled_min_run` enregistrements consécutifs du même micro (`inputs/qc.muffled_runs`, DECISIONS n° 194), voisins non étiquetés compris.
- `_flagged_now` ([qc_calibration.py:224](qc_calibration.py#L224)), `_flagged_suggested` ([qc_calibration.py:233](qc_calibration.py#L233)), `_raised` ([qc_calibration.py:188](qc_calibration.py#L188)) : comptent les enregistrements signalés au seuil actuel et au seuil proposé, par la règle des suites pour `muffled`.

#### `current_flags(indices, thresholds)` — [qc_calibration.py:294](qc_calibration.py#L294)

- Renvoie : par enregistrement étiqueté : ses groupes, et les drapeaux `silent` et `muffled` que lèveraient les seuils actuels.

## `throughput.py` : le débit des encodeurs

Mesure la vitesse d'un encodeur sur du bruit synthétique à 48 kHz : aucun enregistrement n'est lu. Écrit : `debit.csv` et `debit.md` sous `paths.reports`. Tests : [`tests/evaluation/test_throughput.py`](../../tests/evaluation/test_throughput.py).

#### `measure_in_subprocess(name, config, n_windows=64, timeout_s=3600)` — [throughput.py:127](throughput.py#L127)

- Quand : une fois par encodeur dans `blanci throughput` ([`cli.throughput`](../cli.py#L1865)).
- Fait : lance `python -m blanci.evaluation.throughput <nom>` dans un processus neuf et relit la dernière ligne JSON de sa sortie ; en cas d'échec ou au-delà d'une heure, rend une ligne avec l'erreur.
- Pourquoi (docstring du fichier) : le pic de mémoire est alors celui de cet encodeur seul, et un modèle qui échoue n'arrête pas les autres.
- `main(argv=None)` ([throughput.py:207](throughput.py#L207)) : le point d'entrée du sous-processus ; charge la configuration (`--config`, sinon `config/local.yaml`) et imprime le résultat de `measure_by_name`. `measure_by_name(name, cfg, n_windows=64)` ([throughput.py:112](throughput.py#L112)) : charge l'encodeur par `get_encoder`, chronomètre le chargement, puis `measure_encoder` avec le pas de `encoders.overlap`.

#### `measure_encoder(encoder, n_windows=64, hop_ratio=0.5, seed=0, load_s=nan)` — [throughput.py:73](throughput.py#L73)

- Fait : fabrique `n_windows` fenêtres de bruit à 48 kHz, fait un tour à vide sur deux fenêtres, puis chronomètre `encoder.embed` (rééchantillonnage compris).
- Renvoie : le nom, la version, la fréquence, la fenêtre, la dimension, `has_tokens`, le temps de chargement, les fenêtres par seconde, le facteur temps réel (secondes d'audio couvertes par seconde de calcul), les minutes par heure d'audio, la durée projetée d'une campagne de 575 h (`CAMPAIGN_HOURS`, [throughput.py:27](throughput.py#L27)) et de ses seules heures de pic (÷ 3,6), le pic de mémoire, et si toutes les sorties sont finies.
- `peak_memory_mb()` ([throughput.py:32](throughput.py#L32)) : le pic de mémoire résidente du processus, en Mo, par l'API de Windows ou par `resource` ailleurs.

#### `write_throughput_report(rows, reports_dir, machine)` — [throughput.py:165](throughput.py#L165)

- Écrit : `debit.csv` et `debit.md`. Si `debit.csv` existe, la mesure d'un encodeur remplace sa ligne et les autres encodeurs restent.
- `machine_description()` ([throughput.py:198](throughput.py#L198)) : processeur, nombre de fils, système, pour l'en-tête du rapport.

## `__init__.py`

Une docstring qui liste les huit fichiers et dit du pré-benchmark AnuraSet qu'il est « terminé (outils et config dans `anuraset/`) ». Aucun code, aucun import.

## Écarts relevés

Signalés, pas corrigés.

1. **`README.md` range `evaluate` et `freeze` sous `evaluation/`.** Le code de `blanci evaluate` est dans `service.evaluate_holdout` et `service.evaluate_frozen` ; celui de `blanci freeze` dans `inputs/frozen.py`. De ce dossier, seule `blanci benchmark` y a son code, avec `heads`, `heads-curve`, `benchmark-all`, `sources`, `qc-calibrate`, `throughput` et les commandes `anuraset`, que le README ne cite pas à cette ligne.
2. **La docstring d'`evaluate.py` dit que « toute évaluation passe par ici ».** C'est vrai des métriques. Mais `full_benchmark.run_full_benchmark` refait à la main l'assemblage que fait `evaluate` (AP, intervalle, AP par fold, rappels, Wilson) au lieu de l'appeler.
3. **`how="top3"` n'est jamais demandé.** `to_recordings` et `evaluate` acceptent la moyenne des trois meilleures fenêtres ; aucun appelant ne passe autre chose que le défaut `"max"`.
4. **`head_benchmark.py` renvoie à `notes.md`.** Deux docstrings (« Hypothèse à tester (notes.md) », « §3, notes.md ») citent un fichier qui n'existe ni à la racine ni dans `documentation/`.
5. **Le tableau des têtes de la docstring de `head_benchmark.py` est incomplet.** Il ne cite ni `dann` ni `proto_probe`, que `head_methods` et `TOKEN_METHODS` connaissent. À recouper avec la fiche de `heads/`.
6. **`qc_calibration.py` : le groupe `rain` ne sert à aucun seuil.** La docstring annonce des cibles « `artefact_muffled`, `rain` ou mention de pluie » ; `FLAGS` n'a que `muffled` et `silent`, et pour `silent` la liste des cibles est toujours vide : sa ligne dit toujours « aucune cible étiquetée ». Le nom `bagged_recordings` garde l'ancien nom du drapeau (« micro dans sac »).
7. **`calibration_indices` lit l'audio sous `paths.raw/<chemin d'origine>`.** Comme `compute_onsets` et `compute_tokens` (fiche de `core/`, n° 4), sans `variants` ni recherche sur les disques : un enregistrement sous un dossier renommé est sauté sans message, comme un fichier illisible.
8. **Texte figé dans le rapport de débit.** `write_throughput_report` écrit « CPU seul » et « grille au pas d'une demi-fenêtre » ; le pas mesuré vient de `encoders.overlap`, et un encodeur réglé avec `openvino` calcule sur le processeur graphique.
9. **`anuraset.py` : « un fold par site » dépend de `head.n_splits`.** `run_anuraset_heads` impose un fold par site ; `run_anuraset_benchmark` passe `head.n_splits` (5 par défaut), qui ne donne un fold par site que parce qu'AnuraSet a 4 sites, moins que 5. Les rapports de campagne écrivent « un fold par site » pour les deux.
10. **`anuraset.py` : chiffres et titres.** La docstring de `_holm_by_level` parle de « 13 têtes × 4 espèces » ; `HEADS` en a 12 et `CAMPAIGN_HEADS` 14. Le rapport des têtes titre « Contre la référence (enregistrements) » un tableau qui contient aussi les comparaisons au niveau fenêtre.
11. **Commentaire sur des sources « externes ».** `run_full_benchmark` garde un cas « source sans folds (externe) » ; `load_oof` ignore pourtant les sources d'une sorte hors de `KINDS` quand aucune liste n'est donnée. Le cas ne sert plus qu'à une source dont le fold vaut −1.
12. **`cli.py` appelle ce dossier sans passer par `service.py`.** `run_benchmark`, `run_head_benchmark`, `annotation_curve`, `run_full_benchmark`, `list_sources`, les fonctions de `qc_calibration.py`, de `throughput.py` et d'`anuraset.py` sont appelées directement par `cli.py`. Déjà relevé dans la fiche de `core/` (n° 10).
13. **Fonctions publiques à un seul usage.** `lomo_assignment`, `cluster_units`, `bootstrap_ci`, `fold_mean_ap`, `source_path`, `knn_top1`, `head_methods`, `load_tokens`, `by_positive_type`, `background_diagnostic`, `suggest_species`, `rank_encoders`, `rank_heads`, `window_comparisons`, `bagged_recordings`, `measure_by_name`, `measure_encoder` n'ont, hors tests, qu'un ou deux appelants, dans leur propre fichier ou dans `full_benchmark.py`. Aucune fonction du dossier n'est sans appelant.
