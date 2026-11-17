# `blanci/heads/` : le code fonction par fonction

Fiche vérifiée le 10/11/2026 sur le commit `9473508` du dépôt public (`git log -1 --format=%h`), arbre de travail sans modification dans ce dossier. Pour savoir si elle est périmée : `git diff --stat 9473508 -- blanci/heads` ; une sortie vide veut dire que le code de ce dossier n'a pas bougé.

`heads/` est l'étape 3 : 12 fichiers, 3 146 lignes, plus deux sous-dossiers qui ont chacun leur fiche : [`regularization/FONCTIONS.md`](regularization/FONCTIONS.md) (11 fichiers, 2 463 lignes) et [`detectors/FONCTIONS.md`](detectors/FONCTIONS.md) (4 fichiers, 257 lignes). Une « tête » reçoit le vecteur d'une fenêtre (son embedding) et rend un nombre, le score ; plus il est haut, plus la fenêtre ressemble à un chant d'A. blanci. Le dossier contient aussi ce qui se calcule sans encodeur : le traitement du signal (débuts de notes, rythme, portes), les baselines et le clustering.

Vocabulaire commun : `X` est le tableau des embeddings (une ligne par fenêtre), `y` vaut 1 pour un positif et 0 pour un négatif, `groups` donne le micro (ou le site) de chaque fenêtre. Un « fold » est un paquet de micros mis de côté pour être noté par une tête apprise sur les autres. Les « jetons » sont les morceaux temps × fréquence de la fenêtre que certains encodeurs rendent en plus de l'embedding. C est le réglage de la régression logistique : plus il est petit, plus la tête est bridée.

| Fichier | Lignes | Rôle |
|---|---|---|
| [`head.py`](head.py) | 900 | les têtes sur embeddings (prototypes, kNN, logistique, LDA, multi-classes, cascade), `fit_and_score`, la validation croisée par micro `oof_scores` |
| [`pooling.py`](pooling.py) | 106 | résumer les jetons d'une fenêtre en un vecteur |
| [`losses.py`](losses.py) | 146 | la même tête linéaire avec d'autres fonctions de perte |
| [`attentive.py`](attentive.py) | 275 | sonde attentive sur les jetons d'un encodeur, et le stock de jetons `TokenStore` |
| [`gated.py`](gated.py) | 156 | sonde à portes sur l'embedding (R85) |
| [`proto_probe.py`](proto_probe.py) | 101 | sonde à prototypes sur les jetons |
| [`dann.py`](dann.py) | 186 | tête R66 : petit réseau entraîné à ne pas reconnaître le micro |
| [`signal_processing.py`](signal_processing.py) | 628 | débuts de notes, rythme, persistance, et en amont de l'encodeur : transformations du son et portes |
| [`baselines.py`](baselines.py) | 356 | détecteurs sans encodeur (énergie, contraste, notes, rythme, gabarits), jugés comme les encodeurs |
| [`cluster.py`](cluster.py) | 212 | voie non supervisée : groupes HDBSCAN sur ACP, tests C0 et C1, variance par site et par micro |
| [`finetune.py`](finetune.py) | 63 | emplacement réservé du fine-tuning et du LoRA, pas encore programmé |
| [`__init__.py`](__init__.py) | 17 | une docstring, aucun code |

## La chaîne

Lecture : `a → b` veut dire « `a` appelle `b` » ; un point-virgule sépare les appels successifs d'une même fonction. Les noms sans préfixe sont dans `heads/` ; dans le premier groupe, ceux sans nom de fichier sont dans `head.py`. Chaque ligne a été remontée depuis `cli.py` et `service.py` en cherchant les appelants dans le code.

Premier groupe, les têtes (`head.py`, `pooling.py`, `losses.py`, `gated.py`, `proto_probe.py`) :

- **Toute commande `blanci …`** : `cli.main` → `regularization.configure` (pose la règle `head.selection_rule` dont se sert `select_C` par `grouped_search` → `pick`).
- **Dans `oof_scores`, pour la tête `logistic`** : `evaluate.grouped_folds` ; pour chaque fold `fit_and_score` → `_prepared` ; `_fitter` ; `_choose_C` → `select_C` → `regularization.grouped_search` → `usable_folds` → `grouped_folds` ; (pour chaque C et chaque fold interne) `fit_logistic` → `regularization.bias_sigmas`, `group_bias_scale` ; `standardize` ; `_penalty` ; `Head.decision` ; `evaluate.average_precision` ; puis `fit_rows` → `fit_logistic` ; `Head.decision`.
- **`blanci train`** : `cli.train` → `service.train_and_register` → `oof_scores` (tête `logistic`) ; `train_head` → `select_C` ; `fit_logistic` ; `Head.save` ; `service.write_regularization_path` (relit `head.meta`).
- **`blanci retrain`** : `cli.retrain` → `service.retrain` → `train_and_register` (comme `train`) ; `evaluate_frozen` → `load_head` → `Head.load` ; `Head.decision`.
- **`blanci score`** : `cli.score` → `service.score_and_decide` → `load_head` → `Head.load` ; `Head.decision` (un appel par fichier du stock) ; avec `--fusion` : `fused_scores` → pour une entrée `head:<autre encodeur>` : `load_head` → `Head.load` ; `Head.decision`.
- **`blanci evaluate`** : `cli.evaluate` → sans `--holdout` : `service.evaluate_holdout` → `oof_scores` (tête `logistic`) ; avec `--holdout` : `evaluate_holdout` → `train_head` → `select_C` ; `fit_logistic` ; puis `Head.decision` ; avec `--frozen` : `service.evaluate_frozen` → `load_head` → `Head.load` ; `Head.decision`.
- **`blanci heads`** : `cli.heads` → `head_benchmark.run_head_benchmark` → `benchmark_data` → `load_tokens` → `pooling.as_grid` (si `token_grid`) ; `expand_methods` (lit `losses.LOSSES`) ou `head_methods` (lit `METHODS`, `pooling.available_poolings`) ; pour chaque tête : `regularization.regularizer_for` ; `_inputs` → `pooling.pool` → `pooling.gem` ; `calibration_options` ; `oof_scores` → `fit_and_score`, ou `calibrated_fold_scores` → `choose_C` ; `fit_and_score` ; `regularization.usable_folds` ; `fit_and_score` ; `regularization.fold_platt`.
- **Dans `fit_and_score`, selon la tête** : `loss:<nom>` : `_fitter` → `_loss_fitter` → `losses.fit_loss` → `standardize` ; `losses._balanced` ; `losses.loss_and_grad`. `gated` : `regularization.fit_with_options` → `gated.fit_gated` → `import_torch`, `feature_scaling`, `balanced_bce`, `validation_criterion`, `optimise`, `dropout` ; `GatedHead.decision`. `proto_probe` : `pooling.flat_tokens` ; `proto_probe.fit_proto_probe` → `import_torch`, `feature_scaling`, `_normalise`, `balanced_bce`, `optimise` ; `ProtoProbeHead.decision` → `activations`. `attentive` : `pooling.flat_tokens` ; `fit_with_options` → `attentive.fit_attentive`. `cascade` : `cascade_scores` → `fit_logistic` ; `pooling.flat_tokens` ; `attentive.fit_attentive`. `multiclass` : `_choose_multiclass_C` → `grouped_search` → `fit_multiclass` ; `fit_multiclass` → `regularization.merge_rare_classes` ; `MulticlassHead.decision`. `dann` : `fit_with_options` → `dann.fit_dann`. `prototype` : `differential_prototype` ; `prototype_scores`. `knn…`, `exemplar…` : `neighbor_options` ; `knn_scores` ou `exemplar_scores` → `regularization.nearest_similarity`.
- **`blanci heads-curve`** : `cli.heads_curve` → `head_benchmark.annotation_curve` → `benchmark_data` (→ `load_tokens`) ; `expand_methods` ; `regularizer_for` ; `_inputs` (→ `pooling.pool`) ; `choose_C` → `_fitter` ; `_prepared` ; `_choose_C` ; puis `fit_and_score` avec ce C.
- **`blanci benchmark`** : `cli.benchmark` → `evaluation/benchmark.run_benchmark` → `benchmark_encoder` → `calibration_options` ; `probe_table` → `oof_scores` pour `knn`, `simple_prototype`, `prototype`, `logistic`, et `attentive` si le stock a des jetons.
- **`blanci upstream-bench --encoder …`** : `cli.upstream_bench_command` → `service.upstream_bench` → `oof_scores` (tête `logistic`), une fois sans porte, une fois avec les portes de la combinaison si elle en a.
- **`blanci fusion`** : `cli.fusion` → `service.train_fusion` → `load_head` → `Head.load` (seule la ligne du registre sert) ; `stacking.build_level1` → `_choose_C` ; `fit_logistic` ; `Head.decision` ; `OOFScores` ; pour chaque source `head:<encodeur>` : `stacking._secondary_head_oof` → `_choose_C` ; `fit_logistic` ; `Head.decision` ; puis `stacking.fused_oof` → `_fused` → `calibration_options` ; `fusion.fusion_model_oof` → `fusion._fold_calibration` (lit `FOLD_CALIBRATIONS`).
- **`blanci fusion-bench`** : `cli.fusion_bench` → `stacking.fusion_benchmark` → `build_level1` (comme ci-dessus) ; `_fused` → `calibration_options` ; `fusion_model_oof`.
- **`blanci ensemble --concat`** : `cli.ensemble` → `ensemble.concat_benchmark` → `oof_scores` (têtes `logistic` et `prototype` par défaut).
- **`blanci anuraset heads`** : `cli.anuraset_heads` → `evaluation/anuraset.run_anuraset_heads` → `head_benchmark.expand_methods` ; `regularizer_for` ; `head_benchmark._inputs` ; `calibration_options` ; `oof_scores` (un fold par site).
- **`blanci anuraset benchmark`** : `cli.anuraset_benchmark` → `run_anuraset_benchmark` → `calibration_options` ; `benchmark.probe_table` → `oof_scores`.
- **`blanci anuraset campaign`** : `cli.anuraset_campaign` → `run_anuraset_campaign` → `run_anuraset_benchmark` ; `run_anuraset_heads`.
- **Programmes d'`anuraset/`** : `anuraset/calcul/global_bench.py` → `choose_C` ; `fit_and_score` (transfert par site ; avec `--tokens`, têtes `attentive`, `logistic:max`, `proto_probe` ; courbe d'amorçage avec `--curve`). `anuraset/calcul/bench.py` remplace `head.oof_scores` par une fonction qui garde les scores, puis appelle `run_anuraset_heads`. `anuraset/benchmarks/2026-09-29_anuraset_birdmae_base/amorcage.py` → `calibration_options` ; `oof_scores`.
- **Notebooks, têtes** : `notebooks/01_explorer_une_fenetre.ipynb` → `explore.corpus_prototypes` → `differential_prototype`. `notebooks/04_attention_jetons.ipynb` → `pooling.flat_tokens` ; `attention_map.attention_grid` → `pooling.as_grid`.

Second groupe, le traitement du signal, les baselines, le clustering, les jetons et la tête DANN :

- **`blanci onsets`** : `cli.onsets` → `embedding/embed.select_recordings` ; avec `--subset benchmark` : `inputs/dataset.benchmark_subset` ; `signal_processing.compute_onsets` → `core/ahead.ahead` → `prepare` (dans les fils) → `core/audio.load_audio` ; `recording_onsets` → `detect_onsets` → `band_envelope_db`. Puis, dans le fil principal : `store_onsets` → `core/db.utc_now`.
- **`blanci embed`, avant la boucle** : `cli.embed` → `cli.upstream_chain` → `signal_processing.upstream_from_cfg` → `_upstream_section` ; `gate_threshold` (ou un `Upstream()` vide si `signal_processing.position` ne contient pas `upstream` et que `--upstream` n'est pas donné) ; `encoders.get_encoder` → `_with_transforms` → `upstream_from_cfg(only=…)` ; `UpstreamEncoder` → `Upstream.transform_tag` ; `embed.embed_recordings` → `Upstream.gate_tag` ; `stock_identity` (qui range `Upstream.signal_cfg`).
- **`blanci embed`, pour chaque enregistrement, avec des portes** : `embed_recordings` → `signal_processing.load_onsets` ; à défaut `recording_onsets` → `detect_onsets` ; `gate_values` → `baselines.window_features` → `core/audio.resample`, `detect_onsets` ; `onset_counts` ; `Upstream.passes` → `gate_mask`. Avec des transformations : `UpstreamEncoder.embed` → `Upstream.transform` → `bandpass`, `denoise`.
- **`blanci baselines`** : `cli.baselines` → `baselines.run_baselines` → `evaluation_windows` → `inputs/frozen.frozen_recordings` ; `inputs/dataset.benchmark_recordings` ; `embedding/grid.window_grid` ; `core/db.window_id_for` ; `inputs/dataset.training_set` ; `recordings_table`. Puis `read_windows` ; `window_features` → `core/audio.resample`, `_band_rows`, `signal_processing.detect_onsets` ; `inputs/dataset.folds_for` ; `evaluation/oof.labels_fingerprint` ; `oof_template_scores` → `evaluation/evaluate.grouped_folds` ; `note_patch` → `_band_profile` ; `_znorm` ; `template_scores` ; `evaluation/oof.oof_frame`, `save_oof` ; `evaluation/evaluate.evaluate`. Enfin `cli.baselines` → `write_baseline_report` → `evaluation/benchmark.to_markdown`.
- **`blanci upstream-bench`, les portes** : `cli.upstream_bench_command` → `signal_processing.upstream_from_cfg` ; `service.upstream_bench` → sans encodeur `baselines.evaluation_windows`, avec `inputs/dataset.embedded_training_set` ; `service.window_gate_values` → `baselines.read_windows` ; `signal_processing.gate_values` ; `load_onsets` ; `onset_counts`. Puis `gate_sweep` → `gate_mask`, `_sweep_row`, `evaluation/evaluate.wilson_interval`. Avec un encodeur : `head.oof_scores` ; `apply_gate` ; `gate_mask` ; et si des portes sont actives : `Upstream.passes` ; `head.oof_scores(gated=…)` ; `Upstream.gate_tag`.
- **`blanci detector-bench`** : `cli.detector_bench` → `detectors.get_detector` ; `detectors/base.evaluate_detector` → `read_evaluation_audio` → `baselines.evaluation_windows`, `read_windows` ; pour `band_contrast` : `BandContrastDetector.score` → `baselines.window_features`.
- **`blanci fusion`, côté traitement du signal** : `cli.fusion` → `service.train_fusion` → `combination/stacking.build_level1` → `signal_processing.load_onsets` ; `apply_gate` ; `recording_persistence` → `persistence_features` → `surrounded_by_positives` ; `stacking.signal_features` → `window_rhythm` → `rhythm_features` ; `onset_counts`. Puis `stacking.fused_oof` → `_fused` → `design_matrix` (`GATED_SCORE`) ; si l'emplacement contient `upstream` : `upstream_pass` → `onset_gates`, `gate_mask` ; `apply_gate`. Enfin `train_fusion` → `onset_gates` (rangé avec la fusion).
- **`blanci fusion-bench`, côté traitement du signal** : `cli.fusion_bench` → `combination/stacking.fusion_benchmark` → `build_level1` (comme ci-dessus) ; `_fused` pour chaque emplacement et chaque méthode.
- **`blanci score`, les portes** : `cli.score` → `service.score_and_decide` → `signal_processing.apply_gate` ; avec `--fusion` : `service.fused_scores` → `recording_persistence` ; `load_onsets` ; `stacking.signal_features` → `window_rhythm` ; si la fusion est placée en amont : `onset_counts` ; `gate_mask` ; `apply_gate`.
- **`blanci evaluate`, `blanci retrain`, les portes** : `service.evaluate_holdout` et `service.evaluate_frozen` → `signal_processing.apply_gate` (`retrain` passe par `evaluate_frozen`).
- **`blanci cluster`** : `cli.cluster` → `service.run_clustering` → `cluster.cluster_embeddings` → `embedding/index.l2_normalize` ; `cluster_table` ; en mode `c1` : `c1_verdict` → `cluster_table`, `c0_summary` → `ami` ; en mode `c0` : `c0_summary` → `ami` ; `variance_partition`.
- **`blanci pca`** : `cli.pca` → `service.run_pca_curve` → `cluster.pca_information_curve` → `lost` ; `components_for` ; puis `service.write_pca_curve`.
- **`blanci select --method cluster`** (et le poste d'annotation, `annotation/app.py`) : `selection.select_candidates` → `cluster_candidates` → `cluster.cluster_embeddings` ; `NOISE`. **`blanci cluster-label`** : `cli.cluster_label_command` → `selection.label_cluster` (`NOISE`).
- **`blanci select --method gaps`** : `selection.select_candidates` → `gap_candidates` → `signal_processing.surrounded_by_positives` (rayon `signal_processing.gap_radius_s`, repli `GAP_RADIUS_S`).
- **Tout ce qui tire des négatifs appariés** (`train`, `benchmark`, `heads`, `tokens`, `fusion`, `baselines`…) : `inputs/dataset.training_set` → `suspect_false_negatives` → `signal_processing.surrounded_by_positives` ; `paired_negatives` → `draw_same_recording` → `surrounded_by_positives`.
- **`blanci tokens`** : `cli.tokens` → `service.compute_tokens` → `attentive.TokenStore` ; `TokenStore.read` (fenêtres déjà faites) ; pour chaque enregistrement : `encoder.embed_tokens` ; `TokenStore.write` → `read`.
- **`blanci benchmark`, la sonde attentive** : `cli.benchmark` → `evaluation/benchmark.run_benchmark` → `benchmark_encoder` → `attentive.TokenStore.load` → `read` ; `probe_table` → `head.oof_scores(method="attentive")` (seulement si les jetons existent) → `fit_and_score` → `regularization.fit_with_options` → `attentive.fit_attentive` → `regularization.optimise` ; `AttentiveHead.decision` → `_softmax`.
- **`blanci heads`, têtes `attentive`, `cascade` et `dann`** : `cli.heads` → `evaluation/head_benchmark.run_head_benchmark` → `benchmark_data` → `load_tokens` → `attentive.TokenStore.load` ; `head.oof_scores` → `fit_and_score` ; tête `attentive` : `fit_with_options` → `fit_attentive` ; tête `cascade` : `head.cascade_scores` → `fit_attentive` ; tête `dann` : `fit_with_options` → `dann.fit_dann` → `regularization.grad_reverse`, `dann_strength`, `optimise` ; `DannHead.decision` → `representation`. **`blanci heads-curve`** : `head_benchmark.annotation_curve` → `benchmark_data` ; `head.fit_and_score` (mêmes branches).
- **Notebooks, traitement du signal et jetons** (repérés par une recherche dans le texte des `.ipynb`, non exécutés) : `01_explorer_une_fenetre.ipynb` (`rhythm_features`, `upstream_from_cfg`, et par `exploration/explore.py` : `gate_values`, `Upstream`, `gate_threshold`, `detect_onsets`) ; `02_negatifs_apparies.ipynb` (`evaluation_windows`, `window_features`) ; `03_traitement_du_signal.ipynb` (`evaluation_windows`, `GATES`, `gate_mask`, `gate_sweep`, `gate_threshold`, `service.window_gate_values`) ; `04_attention_jetons.ipynb` (`AttentiveHead.load`, `attention`, `decision`, `fit_attentive`).
- **Rien dans `anuraset/`, `scripts/` ni `documentation/`** n'importe ces sept fichiers (Grep sur `blanci.heads` et `signal_processing` dans leurs `.py`).

## `head.py` : les têtes sur embeddings et leur validation croisée

Rassemble les têtes courtes (recherche par l'exemple, prototypes, kNN, logistique, LDA, multi-classes, cascade), la fonction qui apprend n'importe quelle tête sur des fenêtres d'entraînement puis note des fenêtres de test (`fit_and_score`), et la validation croisée par micro bâtie dessus (`oof_scores`). Lit : rien dans la base ni dans le stock ; tout arrive en tableaux numpy. Seule `Head.load` lit deux fichiers. Écrit : seulement `Head.save`, soit `weights.npz` et `manifest.json` dans le dossier qu'on lui donne (pour `blanci train` : `paths.models/head/<encodeur>-<version>`, d'après `service.head_directory`). Utilisé par `blanci train`, `retrain`, `score`, `evaluate`, `heads`, `heads-curve`, `benchmark`, `upstream-bench`, `fusion`, `fusion-bench`, `ensemble --concat`, `anuraset heads`, `anuraset benchmark`, `anuraset campaign`, et par trois programmes d'`anuraset/`. Tests : [`tests/heads/test_head.py`](../../tests/heads/test_head.py) ; d'autres fichiers passent aussi par ce code : [`test_regularization.py`](../../tests/heads/test_regularization.py), [`test_new_heads.py`](../../tests/heads/test_new_heads.py), [`test_attentive.py`](../../tests/heads/test_attentive.py), [`test_gated.py`](../../tests/heads/test_gated.py), [`test_losses.py`](../../tests/heads/test_losses.py), [`test_proto_probe.py`](../../tests/heads/test_proto_probe.py), et hors de `tests/heads/` : `tests/evaluation/test_head_benchmark.py` (exemplar, médoïde, cascade, tête inconnue), `tests/embedding/test_upstream.py`.

#### `oof_scores(X, y, groups, n_splits=5, method="logistic", C_grid=None, seed=0, gated=None, assignment=None, tokens=None, cascade_fraction=0.2, regularizer=None, calibration=None, calibration_splits=3)` — [head.py:768](head.py#L768)

- Quand : chaque fois qu'il faut noter une tête sur des fenêtres qu'elle n'a pas apprises. 7 fonctions de `blanci/` : [`service.train_and_register`](../service.py#L135) (les scores qui fixent le seuil), [`service.evaluate_holdout`](../service.py#L1065) (sans site tenu à l'écart), [`service.upstream_bench`](../service.py#L1858) (deux appels : sans porte, puis avec les portes), [`head_benchmark.run_head_benchmark`](../evaluation/head_benchmark.py#L228), [`benchmark.probe_table`](../evaluation/benchmark.py#L72), [`anuraset.run_anuraset_heads`](../evaluation/anuraset.py#L341), [`ensemble.concat_benchmark`](../combination/ensemble.py#L168). Hors du paquet : `anuraset/benchmarks/2026-09-29_anuraset_birdmae_base/amorcage.py` et `anuraset/calcul/bench.py`. Dans les tests : 24 fonctions de 8 fichiers.
- Reçoit : les embeddings (ou les jetons pour `attentive` et `proto_probe`), les labels, le micro de chaque fenêtre, le nombre de folds, le nom de la tête, la grille de C, les fenêtres arrêtées par une porte (`gated`), l'affectation commune des micros aux folds (`assignment`), les jetons (pour `cascade`), un régularisateur, le mode de recalibration. Renvoie : un `OOFScores`.
- Fait : 1. met `X` en `float32`, `y` en entiers ; sans `gated`, aucune fenêtre n'est arrêtée ; 2. découpe en folds par micro avec [`evaluate.grouped_folds`](../evaluation/evaluate.py#L21) : avec `assignment`, les folds sont ceux de l'affectation ; sans, un tirage stratifié qui ne coupe jamais un micro en deux ; 3. pour chaque fold, retire de l'entraînement les fenêtres arrêtées, puis appelle `fit_and_score` (ou `calibrated_fold_scores` si `calibration` est donné) et range les scores des fenêtres de test ; 4. à la fin, toute fenêtre arrêtée reçoit `GATED_SCORE` (−1 000 000, défini dans `signal_processing.py`), dans les scores et dans `raw`.
- Détails qui ne se voient pas : une fenêtre arrêtée du fold de test est quand même notée par la tête, puis son score est écrasé à l'étape 4. `n_splits` sert deux fois : pour les folds et pour la validation interne qui choisit C. Il peut valoir `"lomo"` (un fold par micro positif, géré par `grouped_folds`), bien que la signature annonce un entier. Sans `C_grid`, les têtes à C prennent C = 1. Moins de deux micros : `grouped_folds` lève « au moins deux groupes sont nécessaires pour une validation groupée » ; un micro absent de `assignment` : « groupes sans fold dans l'affectation ».
- Pourquoi (docstring) : pour `logistic`, le C est choisi dans chaque fold sur les seules données d'entraînement ; une fenêtre arrêtée n'est jamais apprise.

#### `OOFScores` — [head.py:35](head.py#L35)

- Ce que rend `oof_scores` : `values` (un score par fenêtre), `folds` (les couples entraînement / test), `method` (le nom de la tête), `raw` (les scores d'avant la recalibration par fold, `None` sans recalibration). Objet figé (`frozen=True`).
- Quand : construit par `oof_scores`, par [`stacking.build_level1`](../combination/stacking.py#L172) (scores de la tête logistique, champ `Level1.head`), et dans `combination/fusion.py` par `fusion_oof` et [`fusion_model_oof`](../combination/fusion.py#L420). Ses `folds` sont relus par `regularization.fold_ids` pour savoir dans quel fold chaque fenêtre a été notée.
- Pourquoi (docstring) : chaque score vient d'un modèle entraîné sans le micro de la fenêtre ; « seul type accepté en entrée de la fusion » (voir l'écart n° 5).

#### `fit_and_score(method, X, y, groups, train, test, C=None, C_grid=None, n_splits=5, seed=0, tokens=None, cascade_fraction=0.2, regularizer=None)` — [head.py:619](head.py#L619)

- Quand : c'est la brique commune. Appelée par `oof_scores` (une fois par fold), par `calibrated_fold_scores` (le fold, puis chaque fold interne), par [`head_benchmark.annotation_curve`](../evaluation/head_benchmark.py#L451) (une fois par tête, cible, tirage et k) et par `anuraset/calcul/global_bench.py` (transfert d'un site à l'autre et courbe d'amorçage).
- Reçoit : le nom de la tête, toutes les fenêtres (`X`, `y`, `groups`), les numéros de ligne d'entraînement (`train`) et de test (`test`), un C imposé ou `None`. Renvoie : un score par fenêtre de `test`.
- Fait : 1. `_prepared` applique les régularisations à `X` et rend les poids des fenêtres d'entraînement et les options de la logistique (rien de tout cela sans régularisateur) ; 2. avec un régularisateur, la grille de C passe par `regularizer.grid` (R76 : grille plus fine) ; 3. pour une tête à C (`_fitter` en rend l'ajustement) et sans C imposé, `_choose_C` choisit C sur `train` ; 4. la tête demandée est apprise sur `train` et note `test`, selon le tableau ci-dessous.

| `method` | Ce qui est fait |
|---|---|
| `logistic`, `logistic_to_prototype`, `loss:<nom>` | La tête est ajustée au C retenu, avec les poids et les options du régularisateur (`logistic_to_prototype` ne reçoit que les poids). Si le régularisateur porte R81 : `regularization.with_pseudo_labels` (des fenêtres non annotées très bien notées rejoignent l'entraînement) ; sinon, s'il porte R79 : `regularization.bagged` (moyenne de têtes apprises sur des tirages de micros). Score : `decision`. |
| `multiclass` | Demande les classes des fenêtres dans `regularizer.context.classes`, sinon `ValueError` « multiclass : classes des fenêtres manquantes (Context.classes) ». Effectif minimal d'une classe : réglage `min_count` de R67 (10). C choisi par `_choose_multiclass_C` s'il n'est pas imposé, puis `fit_multiclass`. |
| `lda_shrunk` | `lda_shrunk_scores`, sans C ni poids. |
| `gated` | [`fit_with_options`](regularization/torch_training.py#L361) autour de [`gated.fit_gated`](gated.py#L49), avec `regularizer.torch_options()`. |
| `dann` | `fit_with_options` autour de `dann.fit_dann` ; le « domaine » à effacer est le groupe du contexte (`regularizer.context.groups`). |
| `prototype` | `differential_prototype` sur les positifs et les négatifs de `train`, puis `prototype_scores`. |
| `attentive` | `X` contient les jetons ; `pooling.flat_tokens` les met à plat ; avec R47 (`shrink`), le C de la logistique de départ est choisi par `_choose_C` sur la moyenne des jetons ; puis `fit_with_options` autour de `attentive.fit_attentive`. |
| `proto_probe` | `X` contient les jetons ; [`proto_probe.fit_proto_probe`](proto_probe.py#L49) avec la seule graine, tous ses autres réglages par défaut. |
| `cascade` | Demande `tokens`, sinon `ValueError` « la cascade a besoin des jetons (tokens=…) » ; `cascade_scores`. |
| `exemplar`, `exemplar_medoid`, `simple_prototype` | `exemplar_scores`, `medoid_scores`, `simple_prototype_scores`, avec les seuls positifs de `train`. |
| `knn` | `knn_scores` avec k = 1. |
| `knn:k=5`, `exemplar:k=3`, `knn:k=5:w` | `neighbor_options` lit le nom, puis `knn_scores` ou `exemplar_scores` avec k voisins (`:w` : moyenne pondérée). |
| autre | `ValueError` « méthode inconnue : … (connues : … ; variantes knn:k=5, exemplar:k=3, knn:k=5:w) ». |

- Cas limite : `dann` sans régularisateur échoue sur `regularizer.context` (`AttributeError` de Python, sans message du projet), alors que `multiclass` a son message. Par le benchmark, les deux reçoivent toujours un régularisateur (`CONTEXT_HEADS` dans `regularization/names.py`).
- Pourquoi (docstring) : brique commune de la validation croisée et de la courbe selon le nombre d'annotations ; les transformations du régularisateur sont ajustées sur `train` seulement.
- `_prepared(X, y, train, regularizer, seed)` ([head.py:580](head.py#L580)) : sans régularisateur rend `(X, None, {})` ; sinon [`regularizer.prepare`](regularization/assembly.py#L256).
- `_fitter(method)` ([head.py:551](head.py#L551)) : la fonction d'ajustement d'une tête à C : celle de `FITTERS`, ou pour `loss:<nom>` celle que fabrique `_loss_fitter` ; `None` pour une tête sans C.
- `_loss_fitter(name)` ([head.py:542](head.py#L542)) : rend une fonction qui appelle [`losses.fit_loss`](losses.py#L69) avec `loss=name`.
- `_choose_C(X, y, groups, C_grid, n_splits, seed, fitter=None, **fit_kw)` ([head.py:572](head.py#L572)) : si la grille a plusieurs valeurs et l'entraînement au moins deux groupes, le C de `select_C` ; sinon la première valeur de la grille, ou 1 sans grille. Aussi importée par `combination/stacking.py` (`build_level1`, `_secondary_head_oof`).
- `_choose_multiclass_C(X, y, classes, groups, C_grid, n_splits, seed, sw, min_count)` ([head.py:558](head.py#L558)) : même logique pour la tête multi-classes, jugée sur la précision moyenne (AP) d'A. blanci avec `regularization.grouped_search`.
- `fit_rows(rows, weights)` ([head.py:662](head.py#L662)), fonction interne : ajuste la tête sur les lignes données ; c'est elle que `bagged` rappelle à chaque tirage.

#### `METHODS` et `FITTERS` — [head.py:518](head.py#L518)

- `METHODS` : les 14 noms de têtes de base, dans cet ordre : `exemplar_medoid`, `exemplar`, `knn`, `simple_prototype`, `prototype`, `logistic`, `logistic_to_prototype`, `lda_shrunk`, `gated`, `dann`, `multiclass`, `attentive`, `proto_probe`, `cascade`. Lu par [`head_benchmark.head_methods`](../evaluation/head_benchmark.py#L115) pour dresser la liste par défaut de `blanci heads`, et cité dans le message d'erreur de `fit_and_score`. Les variantes `knn:k=…`, `exemplar:k=…`, `loss:<nom>` et `logistic:<pooling>` n'y figurent pas.
- `FITTERS` ([head.py:535](head.py#L535)) : les têtes à C et leur ajustement : `logistic` et `cascade` → `fit_logistic`, `logistic_to_prototype` → `fit_logistic_to_prototype`. Lu par `_fitter` seulement.

#### `choose_C(method, X, y, groups, rows, C_grid, n_splits=5, seed=0, regularizer=None)` — [head.py:587](head.py#L587)

- Quand : pour choisir C une fois, avant plusieurs appels à `fit_and_score` avec ce C imposé : `calibrated_fold_scores`, [`head_benchmark.annotation_curve`](../evaluation/head_benchmark.py#L451) (une fois par cible, sur les autres cibles), `anuraset/calcul/global_bench.py`.
- Reçoit : le nom de la tête, toutes les fenêtres et les lignes `rows` sur lesquelles choisir. Renvoie : le C retenu, ou `None` pour une tête sans ajustement à C (prototypes, kNN, `multiclass`, têtes torch…).
- Fait : `_fitter` ; grille de R76 s'il y a un régularisateur ; `_prepared` sur `rows` ; `_choose_C` sur ces lignes, avec les poids et les options du régularisateur.

#### `select_C(X, y, groups, C_grid, n_splits=5, seed=0, fitter=None, sample_weight=None, with_se=False, **fit_kw)` — [head.py:459](head.py#L459)

- Quand : appelée par `train_head` et par `_choose_C`.
- Reçoit : les fenêtres d'entraînement, la grille de C, l'ajustement à utiliser (`fit_logistic` par défaut). Renvoie : `(C retenu, {C: AP moyenne})`, et avec `with_se` aussi `{C: erreur type de cette moyenne}`.
- Fait : [`regularization.grouped_search`](regularization/selection.py#L98) découpe l'entraînement en folds internes par micro (ceux dont l'entraînement et le test ont les deux classes) ; pour chaque C et chaque fold interne, une tête est ajustée sur l'un et sa précision moyenne (AP) mesurée sur l'autre ; `pick` retient ensuite une valeur selon la règle `head.selection_rule`.
- Détails : avec la règle par défaut `one_se`, la valeur retenue est le plus petit C dont l'AP moyenne est à moins d'une erreur type de la meilleure (`regularization/selection.py`). La règle est posée au démarrage de chaque commande par [`cli.main`](../cli.py#L117), qui appelle `regularization.configure`. Si rien n'a pu être jugé (un seul micro, aucun fold interne avec les deux classes), le C rendu est la première valeur de la grille, soit 0,001 avec `config/default.yaml`, et les AP rendues sont `nan`.

#### `fit_logistic(X, y, C, seed=0, sample_weight=None, l1_ratio=0.0, bias_columns=0, bias_scale=1.0, bias_levels=None, glmm_grid=None)` — [head.py:251](head.py#L251)

- Quand : c'est l'ajustement de la tête retenue. Appelée par `train_head` (la tête que `blanci train` enregistre), par `select_C` (ajustement par défaut), par `cascade_scores` (premier étage), par `fit_and_score` à travers `FITTERS`, par [`stacking.build_level1`](../combination/stacking.py#L172) et [`stacking._secondary_head_oof`](../combination/stacking.py#L247) (une tête par fold pour la fusion), et par [`torch_training.logistic_start`](regularization/torch_training.py#L326) (point de départ de l'attentive, R47).
- Reçoit : les fenêtres d'entraînement et C (plus C est petit, plus la tête est bridée). Renvoie : un `Head`.
- Fait, cas ordinaire : 1. `standardize` centre et réduit chaque dimension de l'embedding ; 2. une régression logistique de scikit-learn à classes équilibrées (`class_weight="balanced"` : positifs et négatifs pèsent autant malgré leur nombre) est ajustée, avec `sample_weight` s'il est donné ; 3. les poids, le biais, la moyenne et l'échelle sont rangés en `float32` dans un `Head`, avec `meta = {"C": C}`, plus `l1_ratio` s'il n'est pas nul, plus `group_biases` et `bias_scale` s'il y a des colonnes de biais.
- Fait, avec des colonnes de biais (R37) : les `bias_columns` dernières colonnes de `X` sont des indicatrices de micro, ajoutées par le régularisateur ; leur échelle vaut σ / √C (`regularization.bias_sigmas`, `group_bias_scale`). Si un σ vaut `"glmm"`, [`regularization.glmm_scales`](regularization/glmm.py#L63) essaie les σ de `glmm_grid` (à défaut `GLMM_GRID`) en rappelant `fit_logistic` (fonction interne `fit_with`, [head.py:272](head.py#L272)), garde le meilleur, et la tête rendue porte `meta["glmm_evidence"]`.
- `_penalty(l1_ratio)` ([head.py:212](head.py#L212)) : les arguments de pénalité pour scikit-learn. `l1_ratio` nul : la pénalité L2 par défaut, 2 000 itérations au plus. Sinon : solveur `saga`, 5 000 itérations, et `penalty="elasticnet"` seulement avant scikit-learn 1.8, où cet argument devient déprécié (docstring). `_SKLEARN` ([head.py:209](head.py#L209)) : la version de scikit-learn, lue une fois à l'import.

#### `standardize(X, bias_columns=0, bias_scale=1.0)` — [head.py:223](head.py#L223)

- Quand : appelée par `fit_logistic` et par [`losses.fit_loss`](losses.py#L69).
- Reçoit : les fenêtres. Renvoie : `(X standardisé, moyenne, échelle)`, tels que `(X − moyenne) / échelle` redonne la même chose : c'est ce que refait `Head.decision`.
- Fait : calcule la moyenne et l'écart-type de chaque dimension (`StandardScaler`) ; une dimension constante garde l'échelle 1. Les `bias_columns` dernières colonnes ne sont pas standardisées : moyenne 0, échelle `1 / bias_scale`, donc elles sont multipliées par `bias_scale` (une valeur, ou une par colonne).
- Pourquoi (docstring) : avec `bias_scale = σ / √C`, le biais d'un micro est pénalisé comme s'il suivait une loi normale d'écart-type σ, quel que soit C ; σ petit : biais proches de 0 ; σ grand : un biais libre par micro.

#### `Head` — [head.py:174](head.py#L174)

- La tête logistique telle qu'elle sert après l'entraînement : `mean`, `scale` (la standardisation), `coef` (un poids par dimension), `intercept` (le biais), `meta` (un dictionnaire libre). Construite par `fit_logistic`, `fit_logistic_to_prototype`, `losses.fit_loss` et `Head.load`.
- `decision(X)` ([head.py:183](head.py#L183)) : le score, `((X − mean) / scale) · coef + intercept`, en numpy. Appelée sur une tête relue du disque par [`service.score_and_decide`](../service.py#L438), [`service.evaluate_frozen`](../service.py#L1339) et [`service.fused_scores`](../service.py#L1540) ; sur une tête juste apprise par `service.evaluate_holdout`, `stacking.build_level1`, `stacking._secondary_head_oof`, et dans ce fichier.
- `predict_proba(X)` ([head.py:188](head.py#L188)) : le score ramené entre 0 et 1 par la fonction logistique (`scipy.special.expit`, qui ne déborde pas sur −1 000 000 d'après le commentaire). Aucun appelant hors de `tests/heads/test_head.py`.
- Pourquoi (docstring du fichier) : la tête retenue est rangée sans pickle (JSON et npz) et calculée en numpy : elle se recharge sur n'importe quelle machine, quelle que soit la version de scikit-learn.

#### `Head.save(directory)` — [head.py:193](head.py#L193)

- Quand : une fois par entraînement, dans [`service.train_and_register`](../service.py#L135) ; aucun autre appelant hors tests.
- Écrit : crée le dossier au besoin ; `weights.npz` (`mean`, `scale`, `coef`) ; `manifest.json` (`intercept` puis tout `meta`, indenté, en UTF-8).
- Détail : pour `blanci train`, `meta` contient à ce moment `C` (posé par `fit_logistic`), `cv_ap`, `cv_se`, `n_pos`, `n_neg`, `seed` (posés par `train_head`), puis ce qu'ajoute `train_and_register` : `encoder_id`, `version`, `threshold`, `threshold_id`, `min_precision`, `recall_at_threshold`, `crossfit_recall`, `crossfit_precision`, `threshold_benchmark`, `benchmark_prevalence`, `prevalence`, `expected_precision`, `platt`, `trained_at`, `frozen_excluded`.

#### `Head.load(directory)` — [head.py:201](head.py#L201)

- Quand : un seul appelant, [`service.load_head`](../service.py#L398), qui retrouve le dossier dans la table `models`. `load_head` est elle-même appelée par `score_and_decide`, `make_queue`, `evaluate_frozen`, `train_fusion`, `fused_scores`, `adopt_head`, `activity_curves` (toutes dans `service.py`) et par `annotation/selection._head_model`.
- Fait : relit `weights.npz` et `manifest.json` ; `intercept` est retiré du manifeste, tout le reste devient `meta`.

#### `train_head(X, y, groups, C_grid, seed=0, gated=None)` — [head.py:489](head.py#L489)

- Quand : pour la tête qui sera gardée : [`service.train_and_register`](../service.py#L135) (commandes `train` et `retrain`) et [`service.evaluate_holdout`](../service.py#L1065) (tête apprise hors des sites tenus à l'écart).
- Reçoit : toutes les fenêtres étiquetées, leur micro, la grille de C, les fenêtres arrêtées. Renvoie : un `Head`.
- Fait : 1. écarte les fenêtres arrêtées par une porte (embedding nul) ; 2. `select_C` avec `with_se=True` et 5 folds internes (valeur par défaut de `select_C`, voir l'écart n° 4) ; 3. `fit_logistic` sur toutes les fenêtres au C retenu ; 4. ajoute à `meta` : `cv_ap` et `cv_se` (AP moyenne et erreur type pour chaque C, clés en texte), `n_pos`, `n_neg`, `seed`.
- Détail : `cv_ap`, `cv_se` et `C` sont relus par [`service.write_regularization_path`](../service.py#L355), qui en fait le CSV et la figure `chemin_C_<encodeur>_<version>`.

#### `differential_prototype(E_pos, E_neg_paired)` — [head.py:50](head.py#L50)

- Quand : tête `prototype` de `fit_and_score` ; [`explore.corpus_prototypes`](../exploration/explore.py#L502), que seul le notebook `notebooks/01_explorer_une_fenetre.ipynb` appelle.
- Reçoit : les embeddings des positifs et ceux des négatifs appariés. Renvoie : `(w, b)`.
- Fait : ramène chaque embedding à la norme 1, calcule le vecteur moyen des positifs et celui des négatifs ; `w` est leur différence ; `b` place le score 0 à mi-chemin des deux moyennes.
- Pourquoi (docstring) : avec des négatifs appariés (mêmes micros, mêmes heures), la différence retire le fond sonore commun aux deux.

#### `prototype_scores(X, w, b)` — [head.py:129](head.py#L129)

- Quand : dans `fit_and_score`, tête `prototype`, seul appelant hors tests.
- Renvoie : `X` ramené à la norme 1, multiplié par `w`, plus `b`.
- À ne pas confondre : `exploration/explore.py` a sa propre fonction du même nom (`prototype_scores(X, prototypes)`, ligne 531), qui ne passe pas par celle-ci.

#### `simple_prototype_scores(E_pos, X)` — [head.py:61](head.py#L61)

- Quand : tête `simple_prototype` de `fit_and_score`, seul appelant hors tests.
- Renvoie : la similarité cosinus de chaque fenêtre au vecteur moyen des positifs ; aucun négatif n'intervient.
- Pourquoi (docstring) : l'écart avec le prototype différentiel mesure le fond sonore capté par l'embedding : si retrancher la moyenne des négatifs aide beaucoup, les positifs se ressemblent surtout par leur ambiance.

#### `exemplar_scores(E_pos, X, k=1, weighted=False)` — [head.py:71](head.py#L71)

- Quand : têtes `exemplar` et `exemplar:k=…` de `fit_and_score`.
- Renvoie : pour chaque fenêtre, sa similarité cosinus au positif d'entraînement le plus proche (k = 1), ou la moyenne sur les k plus proches ([`regularization.nearest_similarity`](regularization/neighbors.py#L10)), pondérée par l'inverse de la distance avec `weighted`.
- Pourquoi (docstring) : aucun apprentissage, aucun négatif ; c'est ce que donne une requête « trouve-moi des fenêtres comme celles-ci ».

#### `medoid_scores(E_pos, X)` — [head.py:82](head.py#L82)

- Quand : tête `exemplar_medoid` de `fit_and_score`, seul appelant hors tests.
- Fait : choisit le positif le plus central (celui dont la similarité cosinus moyenne à tous les positifs, lui compris, est la plus grande), puis rend la similarité de chaque fenêtre à ce seul exemple.
- Pourquoi (docstring) : ce qu'on a en arrivant sur un nouveau site avec un seul chant validé.

#### `knn_scores(X_train, y_train, X, k=1, weighted=False)` — [head.py:133](head.py#L133)

- Quand : têtes `knn` et `knn:k=…` de `fit_and_score`, seul appelant hors tests.
- Renvoie : pour chaque fenêtre, similarité à ses k positifs les plus proches moins similarité à ses k négatifs les plus proches.
- Fait : `ValueError` « la marge kNN exige des positifs et des négatifs dans l'entraînement » s'il manque une classe ; similarités cosinus ; `nearest_similarity` dans chaque classe.
- Pourquoi (docstring) : les voisins sont pris dans chaque classe séparément, car un vote parmi les k plus proches toutes classes confondues serait écrasé par la vingtaine de négatifs par positif.

#### `neighbor_options(method)` — [head.py:159](head.py#L159)

- Quand : à la fin de `fit_and_score`, pour un nom qui n'est aucune des têtes connues.
- Renvoie : `("knn", 5, True)` pour `knn:k=5:w` ; `None` si le nom n'a pas la forme `knn:k=<n>` ou `exemplar:k=<n>`, suivie ou non de `:w` (expression `_NEIGHBORS`, [head.py:156](head.py#L156)). k = 0 lève « k doit valoir au moins 1 ».

#### `fit_logistic_to_prototype(X, y, C, seed=0, sample_weight=None)` — [head.py:314](head.py#L314)

- Quand : tête `logistic_to_prototype` (R30), par `FITTERS` ; aucun appel direct hors tests.
- Renvoie : un `Head`, avec `meta = {"C", "shrink_to": "differential_prototype", "prototype_scale"}`.
- Fait : 1. standardise `X` ; 2. calcule la direction `u` qui va de la moyenne des négatifs à celle des positifs, ramenée à la norme 1 ; 3. ajuste une logistique à une seule variable (la projection de chaque fenêtre sur `u`) pour obtenir l'ampleur α et le biais de départ ; le point de départ est `w₀ = α·u` ; 4. minimise avec L-BFGS (scipy, 2 000 itérations au plus) la perte logistique à classes équilibrées multipliée par C, plus la moitié du carré de l'écart entre `w` et `w₀`.
- Détails : `seed` n'est pas utilisé dans le corps de la fonction. Le code ne regarde pas si l'optimiseur a convergé (`result.success` n'est pas lu).
- Pourquoi (docstring) : la pénalité ordinaire tire les poids vers 0 ; ici elle les tire vers le prototype différentiel. C petit : la tête est le prototype mis à l'échelle ; C grand : la logistique libre.

#### `lda_shrunk_scores(X_train, y_train, X)` — [head.py:375](head.py#L375)

- Quand : tête `lda_shrunk` (R31) de `fit_and_score`, seul appelant hors tests.
- Fait : analyse discriminante linéaire de scikit-learn (`solver="lsqr"`, `shrinkage="auto"`, rétrécissement de Ledoit-Wolf) ajustée sur l'entraînement ; rend sa `decision_function` sur `X`. Pas de standardisation, pas de C, et les poids des fenêtres du régularisateur ne lui sont pas passés.
- Pourquoi (docstring) : aucun réglage à choisir ; le degré de rétrécissement est calculé sur les données, entre le prototype différentiel et la LDA complète.

#### `fit_multiclass(X, classes, C, seed=0, sample_weight=None, min_count=10)` — [head.py:428](head.py#L428)

- Quand : tête `multiclass` (R67) de `fit_and_score`, et dans `_choose_multiclass_C`.
- Reçoit : les fenêtres et la classe de son de chacune (`blanci`, `amphibien`, `oiseau`, `fond`…). Renvoie : un `MulticlassHead`, avec dans `meta` le C et l'effectif de chaque classe.
- Fait : 1. [`regularization.merge_rare_classes`](regularization/classes.py#L44) verse dans `autre` les classes de moins de `min_count` fenêtres (hors `blanci`), et `autre` dans `fond` si elle reste sous ce seuil ; 2. standardise ; 3. ajuste une régression logistique à plusieurs classes, équilibrées, 2 000 itérations au plus.
- Pourquoi (docstring) : la tête apprend à distinguer toutes les classes de sons, pas seulement « A. blanci ou non », pour que la frontière se place sur ce qui est propre à A. blanci.

#### `MulticlassHead` — [head.py:390](head.py#L390)

- Champs : `mean`, `scale`, `coef` (une ligne par classe, ou une seule ligne à deux classes), `intercept`, `classes`, `meta`. Pas de `save` ni de `load` : cette tête ne vit que le temps d'un benchmark.
- `decision(X)` ([head.py:415](head.py#L415)) : le score d'A. blanci contre toutes les autres classes réunies, calculé sur les logits sans passer par la probabilité. La classe cherchée est le texte `"blanci"` ; son absence de `classes` lève l'erreur `ValueError` de `list.index`.
- `proba(X)` ([head.py:405](head.py#L405)) : la probabilité de chaque classe. Aucun appelant hors de `tests/heads/test_new_heads.py`.
- `_logits(X)` ([head.py:401](head.py#L401)) : embedding standardisé multiplié par les poids, plus les biais, en `float64`.
- Pourquoi (docstring de `decision`) : la probabilité, arrondie à 1 en `float32` dès un logit d'environ 17, donnait un score infini que l'AP refusait.
- À savoir : pour AnuraSet, `run_anuraset_heads` nomme `blanci` les positifs de l'espèce étudiée, quelle qu'elle soit, et `fond` le reste (`evaluation/anuraset.py`, ligne 400).

#### `cascade_scores(X_train, y_train, T_train, X_test, T_test, C=1.0, fraction=0.2, seed=0, **fit_kw)` — [head.py:91](head.py#L91)

- Quand : tête `cascade` de `fit_and_score`, seul appelant hors tests ; `fraction` vient de `head.cascade_fraction` (0,2) par `run_head_benchmark` et `annotation_curve`.
- Reçoit : embeddings et jetons des fenêtres d'entraînement et de test. Renvoie : les scores de test.
- Fait : 1. `fit_logistic` sur les embeddings d'entraînement note entraînement et test ; 2. le seuil des candidats est le score au-dessus duquel se trouve la part `fraction` (20 % par défaut) des fenêtres d'entraînement les mieux notées ; 3. `attentive.fit_attentive` est apprise sur les jetons des candidats d'entraînement, ou sur tout l'entraînement si ces candidats n'ont qu'une classe ; 4. les fenêtres de test au-dessus du seuil sont renotées par l'attentive : leur score devient le plus haut score des non-candidats, plus 1, plus le score de l'attentive ramené entre 0 et 1 ; les autres gardent leur score logistique.
- Détail : l'attentive du second étage ne reçoit que la graine : ni les poids des fenêtres ni les options torch du régularisateur.
- Pourquoi (docstring) : seules les fenêtres candidates ont besoin de leurs jetons (mémoire) ; un candidat reste toujours devant un non-candidat : le second étage reclasse, il ne repêche pas.

#### `calibration_options(cfg)` — [head.py:831](head.py#L831)

- Quand : pour passer le réglage de recalibration à `oof_scores` ou à `fusion_model_oof` : [`head_benchmark.run_head_benchmark`](../evaluation/head_benchmark.py#L228), [`benchmark.benchmark_encoder`](../evaluation/benchmark.py#L129), [`anuraset.run_anuraset_heads`](../evaluation/anuraset.py#L341), [`anuraset.run_anuraset_benchmark`](../evaluation/anuraset.py#L541), [`stacking._fused`](../combination/stacking.py#L356), `anuraset/benchmarks/2026-09-29_anuraset_birdmae_base/amorcage.py`.
- Renvoie : `{}` si `benchmark.fold_calibration` vaut `none`, est absent ou faux ; sinon `{"calibration": <valeur>, "calibration_splits": <benchmark.calibration_splits, 3 par défaut>}`.
- À savoir : `config/default.yaml` porte `fold_calibration: none`. Les appels à `oof_scores` de `service.py` (`train_and_register`, `evaluate_holdout`, `upstream_bench`) et de `ensemble.concat_benchmark` ne passent pas par cette fonction : `blanci train`, `evaluate`, `upstream-bench` et `ensemble --concat` ne recalibrent jamais.

#### `calibrated_fold_scores(method, X, y, groups, train, test, calibration="platt", splits=3, **options)` — [head.py:842](head.py#L842)

- Quand : dans `oof_scores`, à la place de `fit_and_score`, quand `calibration` est donné ; seul appelant.
- Renvoie : `(scores recalibrés, scores bruts)` des fenêtres de test du fold.
- Fait : 1. refuse un mode absent de `FOLD_CALIBRATIONS` (« recalibration par fold inconnue ») ; 2. pour une tête sans ajustement à C et absente de `TRAINED_METHODS`, rend deux fois les scores de `fit_and_score`, sans rien recalibrer ; 3. sinon `choose_C` choisit C une fois sur l'entraînement du fold ; 4. `fit_and_score` note le test avec ce C : ce sont les scores bruts ; 5. [`regularization.usable_folds`](regularization/selection.py#L42) découpe l'entraînement en `splits` folds internes par micro ; s'il n'y en a pas, les scores restent bruts ; 6. chaque fold interne est noté par une tête apprise sur les autres, au même C ; 7. [`regularization.fold_platt`](regularization/decision.py#L82) ajuste sur ces scores internes une droite `a·score + b` ; 8. rend `a × bruts + b` et les bruts.
- `FOLD_CALIBRATIONS` ([head.py:826](head.py#L826)) : `("platt",)` ; aussi lu par [`fusion._fold_calibration`](../combination/fusion.py#L474). `TRAINED_METHODS` ([head.py:828](head.py#L828)) : `gated`, `dann`, `multiclass`, `attentive`, les têtes sans ajustement à C qui sont quand même recalibrées.
- Pourquoi (docstring, DECISIONS n° 133 et 135) : chaque fold choisit son C, et un C petit resserre les scores ; mis bout à bout, des scores d'échelles différentes faussent le classement commun. La droite ramène chaque fold sur une échelle commune sans changer le classement à l'intérieur du fold (a > 0). Les têtes par similarité ne choisissent rien fold par fold : les recalibrer n'ajouterait que le bruit de l'estimation.

## `pooling.py` : résumer les jetons en un vecteur

Transforme les jetons d'une fenêtre (un tableau fenêtres × jetons × dimensions, ou fenêtres × temps × fréquence × dimensions) en un seul vecteur par fenêtre, de plusieurs façons, pour que la tête logistique puisse s'en servir ; sait aussi remettre des jetons à plat en grille, et l'inverse. Des calculs seulement : ne lit et n'écrit rien. Utilisé par `blanci heads` et `blanci heads-curve` (têtes `logistic:<pooling>`, `attentive`, `proto_probe`, `cascade`), par `anuraset/calcul/global_bench.py --tokens`, par `exploration/attention_map.py` et par le notebook `notebooks/04_attention_jetons.ipynb`. Tests : il n'y a pas de `tests/heads/test_pooling.py` ; `pool`, `as_grid`, `available_poolings` et `flat_tokens` sont testées dans `tests/evaluation/test_head_benchmark.py`, `gem` dans [`tests/heads/test_regularization.py`](../../tests/heads/test_regularization.py).

#### `POOLINGS`, `GRID_POOLINGS`, `GEM_P` — [pooling.py:30](pooling.py#L30)

- `POOLINGS` : les 8 résumés connus : `mean`, `max`, `meanmax`, `meanstd`, `topk`, `gem`, `mean_f_max_t`, `max_f_mean_t`. `GRID_POOLINGS` ([pooling.py:32](pooling.py#L32)) : les deux derniers, qui demandent une grille temps × fréquence. `GEM_P` ([pooling.py:31](pooling.py#L31)) : 3, l'exposant par défaut de `gem`.
- Pourquoi (docstring du fichier) : l'embedding par défaut est le plus souvent la moyenne des jetons ; une note de 0,09 s dans une fenêtre de 5 s ne tient que dans un ou deux jetons : la moyenne la dilue, le maximum la garde.

#### `pool(tokens, how, k=2)` — [pooling.py:61](pooling.py#L61)

- Quand : un seul appelant hors tests, [`head_benchmark._inputs`](../evaluation/head_benchmark.py#L174), pour une tête `logistic:<pooling>` : les jetons sont résumés, puis la tête `logistic` ordinaire est apprise sur ce résumé.
- Reçoit : les jetons et le nom du résumé. Renvoie : un vecteur par fenêtre, en `float32`.
- Fait : sur une grille (4 dimensions), `mean_f_max_t` fait la moyenne sur les fréquences puis le maximum sur le temps, `max_f_mean_t` l'inverse ; pour les autres résumés la grille est mise à plat. Puis : `mean` et `max` sur tous les jetons ; `meanmax` et `meanstd` collent bout à bout moyenne et maximum, ou moyenne et écart-type (dimension doublée) ; `topk` fait, dimension par dimension, la moyenne des `k` plus fortes valeurs (`k` ramené entre 1 et le nombre de jetons) ; un nom qui commence par `gem` appelle `gem`, avec l'exposant écrit après (`gem5` : p = 5), ou `GEM_P`.
- Erreurs : jetons ni en 3 ni en 4 dimensions (« jetons attendus en 3 ou 4 dimensions ») ; résumé de grille sur des jetons à plat (« demande une grille temps × fréquence (jetons 4-D) ») ; nom inconnu (« pooling inconnu »). Un nom comme `gemx` échoue avant, sur la conversion du texte en nombre, avec l'erreur de Python.
- Détail : `_inputs` appelle `pool(tokens, how)` sans `k` : par le benchmark, `topk` vaut toujours k = 2.

#### `gem(tokens, p=GEM_P)` — [pooling.py:96](pooling.py#L96)

- Quand : appelée par `pool` seulement (hors tests).
- Reçoit : des jetons à plat et l'exposant `p`. Renvoie : un vecteur par fenêtre.
- Fait : `p ≤ 0` lève « GeM : p > 0 attendu » ; pour chaque fenêtre et chaque dimension : retranche la plus petite valeur parmi les jetons, divise par la plus grande (1 si tous les jetons sont égaux), élève à la puissance `p`, fait la moyenne, prend la racine `p`-ième, puis remet l'échelle et la valeur retranchée. Calcul en `float64`, résultat en `float32`.
- Pourquoi (docstring du fichier, R22, DECISIONS n° 108) : moyenne généralisée entre la moyenne (p = 1) et le maximum (p très grand) ; comme les jetons ne sont pas positifs, elle est calculée sur l'écart au minimum puis recentrée, de sorte que p = 1 redonne exactement la moyenne. `p` n'est pas appris : `gem2`, `gem5`… se comparent au benchmark.

#### `available_poolings(tokens)` — [pooling.py:52](pooling.py#L52)

- Quand : dans [`head_benchmark.head_methods`](../evaluation/head_benchmark.py#L115), pour ajouter les têtes `logistic:<pooling>` à la liste par défaut de `blanci heads`.
- Renvoie : une liste vide sans jetons ; les 8 noms pour des jetons en grille ; les 6 premiers sinon.
- Détail : `gem2`, `gem5`… ne sont pas dans `POOLINGS` : ils n'entrent dans un benchmark que demandés par `--methods logistic:gem5`.

#### `as_grid(tokens, time, freq, order="time_major")` — [pooling.py:35](pooling.py#L35)

- Quand : [`head_benchmark.load_tokens`](../evaluation/head_benchmark.py#L187), si la configuration de l'encodeur porte `token_grid` ; [`attention_map.attention_grid`](../exploration/attention_map.py#L69), que seul le notebook 04 appelle.
- Reçoit : des jetons à plat et la taille de la grille. Renvoie : les mêmes en (fenêtres, temps, fréquence, dimensions).
- Fait : refuse un nombre de jetons différent de `time × freq` ; `time_major` : les jetons d'un même instant se suivent ; `freq_major` : ceux d'une même bande se suivent (remise en forme puis échange des deux axes) ; un autre ordre lève « ordre inconnu ».

#### `flat_tokens(tokens)` — [pooling.py:90](pooling.py#L90)

- Quand : avant les têtes qui veulent des jetons à plat : `head.cascade_scores`, `head.fit_and_score` (têtes `attentive` et `proto_probe`), et le notebook 04.
- Renvoie : une grille mise à plat en (fenêtres, temps × fréquence, dimensions) ; des jetons déjà à plat sont rendus tels quels.

## `losses.py` : une même tête linéaire, plusieurs fonctions de perte

Ajuste la même tête que la logistique (standardisation, un poids par dimension, classes équilibrées, pénalité L2 réglée par C) en changeant seulement la fonction de perte, c'est-à-dire la façon de compter une erreur. Ne lit et n'écrit rien. Utilisé par les têtes `loss:<nom>` de `head.fit_and_score` : `blanci heads --methods losses` les prend toutes, avec `logistic` en référence (`head_benchmark.expand_methods`) ; elles ne sont pas dans la liste par défaut de `blanci heads` ; `loss:focal` et `loss:sigmoid` sont dans la liste par défaut de `blanci anuraset heads` (`HEADS`), `loss:focal` dans celle de `blanci anuraset campaign` (`CAMPAIGN_HEADS`). Tests : [`tests/heads/test_losses.py`](../../tests/heads/test_losses.py).

#### `LOSSES` et `PARAMS` — [losses.py:32](losses.py#L32)

- `LOSSES` : les 7 pertes : `hinge`, `squared_hinge`, `least_squares`, `focal`, `gce`, `sce`, `sigmoid`. Lu par [`head_benchmark.expand_methods`](../evaluation/head_benchmark.py#L129) (le groupe `losses`) et par `fit_loss`.
- `PARAMS` ([losses.py:33](losses.py#L33)) : les réglages par défaut : `focal` γ = 2 ; `gce` q = 0,7 ; `sce` α = 0,1, β = 1, A = −4.
- Ce que chaque perte change, d'après le tableau de la docstring : `hinge` et `squared_hinge` (R34, machine à vecteurs de support) ignorent les exemples bien classés au-delà de la marge ; `least_squares` : moindres carrés sur ±1 ; `focal` (R35) écrase les exemples faciles et insiste sur les durs ; `gce`, `sce` et `sigmoid` (R35) sont robustes à un label faux (`sigmoid` : un label faux ne coûte jamais plus que 1).

#### `fit_loss(X, y, C, seed=0, sample_weight=None, loss="hinge", bias_columns=0, bias_scale=1.0, bias_levels=None, glmm_grid=None, **params)` — [losses.py:69](losses.py#L69)

- Quand : un seul appelant hors tests, la fonction que fabrique [`head._loss_fitter`](head.py#L542) pour une tête `loss:<nom>` ; elle sert à `select_C` (choix de C) puis à `fit_and_score` (ajustement final).
- Reçoit : comme `fit_logistic`, plus le nom de la perte et ses réglages. Renvoie : un `head.Head`, avec `meta = {"C", "loss"}`, plus `group_biases` et `bias_scale` s'il y a des colonnes de biais, plus les réglages de la perte pour les quatre dernières.
- Fait : 1. `head.standardize`, avec les colonnes de biais de R37 si le régularisateur en a ajouté ; 2. selon la perte : `hinge` et `squared_hinge` → `LinearSVC` de scikit-learn (classes équilibrées, 20 000 itérations au plus, graine `seed`) ; `least_squares` → `RidgeClassifier` avec `alpha = 1 / (2C)` ; `focal`, `gce`, `sce`, `sigmoid` → une logistique de même C sert de point de départ, puis L-BFGS (scipy, 2 000 itérations au plus) minimise C × somme des poids × perte + moitié du carré des poids, avec `loss_and_grad` ; 3. un autre nom lève « perte inconnue : … (connues : …) ».
- Détails : `glmm_grid` est accepté mais ignoré (commentaire : le σ estimé de R37 ne vaut que pour la logistique) ; un σ `"glmm"` serait refusé par `regularization.bias_sigmas`. `seed` ne sert qu'à `LinearSVC`. `loss="logistic"` est refusée : `logistic` n'est pas dans `LOSSES`, la référence est la tête `logistic` elle-même. Le code ne regarde pas si L-BFGS a convergé.
- Pourquoi (docstring du fichier, DECISIONS n° 112) : même convention que scikit-learn pour C, choisi par validation groupée comme pour la logistique ; `gce`, `sce` et `sigmoid` ne sont pas convexes, le point de départ compte, d'où le départ depuis la logistique.
- `_balanced(y, sample_weight)` ([losses.py:63](losses.py#L63)) : le poids de chaque fenêtre pour des classes équilibrées (effectif total ÷ deux fois l'effectif de sa classe), multiplié par `sample_weight` s'il est donné.
- `objective(theta)` ([losses.py:123](losses.py#L123)), fonction interne : la valeur à minimiser et son gradient, pour L-BFGS.

#### `loss_and_grad(name, m, **params)` — [losses.py:40](losses.py#L40)

- Quand : dans `objective` de `fit_loss`, à chaque pas de L-BFGS, pour `focal`, `gce`, `sce` et `sigmoid`.
- Reçoit : le nom de la perte et la marge `m` de chaque fenêtre (positive quand la fenêtre est du bon côté, grande quand la tête en est sûre). Renvoie : `(perte, dérivée de la perte par rapport à la marge)`, fenêtre par fenêtre.
- Fait : calcule `p`, la probabilité donnée à la vraie classe, puis : `logistic` → −log p ; `focal` → −(1 − p)^γ · log p ; `gce` → (1 − p^q) / q ; `sce` → α · (−log p) + β · |A| · (1 − p) ; `sigmoid` → 1 − p ; un autre nom lève « perte inconnue ».
- Détail : la branche `logistic` n'est atteinte par aucun appel de `fit_loss` ; hors tests elle ne sert pas.

## `attentive.py` : sonde attentive sur les jetons, et stock de jetons

Deux choses : une tête qui apprend à quels jetons d'une fenêtre (morceaux de temps × fréquence) prêter attention avant de les résumer, et `TokenStore`, le fichier où sont rangés les jetons des fenêtres du benchmark. Lit et écrit : `paths.tokens/<stock>.npz` (`TokenStore`, `data/tokens` dans `config/default.yaml`) ; `weights.npz` et `manifest.json` dans le dossier qu'on donne à `AttentiveHead.save` et `load`. L'entraînement se fait avec torch (groupe `research`), l'application en numpy. Utilisé par `blanci tokens` (écriture du stock), `blanci benchmark` (sonde « attentive » ajoutée quand les jetons existent), `blanci heads` (têtes `attentive` et `cascade`, dans la liste par défaut quand les jetons existent), `blanci heads-curve` (si on les demande avec `--methods`), et le notebook `04_attention_jetons.ipynb`. Tests : [`tests/heads/test_attentive.py`](../../tests/heads/test_attentive.py) ; `TokenStore` sert aussi dans `tests/evaluation/test_head_benchmark.py`.

#### `TokenStore` — [attentive.py:237](attentive.py#L237)

- Quand : fabriqué par [`service.compute_tokens`](../service.py#L1595) (`blanci tokens`), [`benchmark.benchmark_encoder`](../evaluation/benchmark.py#L129) (`blanci benchmark`) et [`head_benchmark.load_tokens`](../evaluation/head_benchmark.py#L187) (appelée par `benchmark_data`, donc `blanci heads`, `heads-curve`).
- Contient : `root` (le dossier `paths.tokens`) et `encoder_id` (le nom du stock). `path` ([attentive.py:245](attentive.py#L245)) = `<root>/<encoder_id>.npz` : un seul fichier par stock.
- Pourquoi (docstring) : on ne garde que les jetons des fenêtres du benchmark (étiquetées et négatifs appariés), un stock complet serait trop lourd (perch_v2 : 16 × 1 536 valeurs par fenêtre).

#### `TokenStore.write(window_ids, tokens)` — [attentive.py:248](attentive.py#L248)

- Quand : appelée par [`service.compute_tokens`](../service.py#L1595), une fois par enregistrement.
- Reçoit : des identifiants de fenêtres et leurs jetons. Renvoie : le chemin du fichier.
- Fait : 1. convertit les jetons en `float16` ; 2. si le fichier existe, le relit en entier, retire les fenêtres qui sont dans le nouvel envoi et met les nouvelles à la suite (une fenêtre déjà présente est remplacée) ; 3. écrit dans `<nom>.tmp.npz` puis renomme : le fichier n'est jamais à moitié écrit.
- `read()` ([attentive.py:263](attentive.py#L263)) : rend `(liste des identifiants, tableau des jetons)` tels que rangés ; `compute_tokens` s'en sert pour savoir quelles fenêtres sont déjà faites.

#### `TokenStore.load(window_ids)` — [attentive.py:267](attentive.py#L267)

- Quand : appelée par `benchmark.benchmark_encoder` et `head_benchmark.load_tokens`.
- Renvoie : les jetons en `float32`, dans l'ordre de `window_ids` ; `None` si le fichier n'existe pas ou s'il manque une seule des fenêtres demandées. C'est ce `None` qui fait que le benchmark n'ajoute pas la sonde attentive.

#### `fit_attentive(tokens, y, weight_decay=1e-3, epochs=300, lr=5e-2, seed=0, *, optimizer="adam", token_dropout=0.0, entropy=0.0, dim_dropout=0.0, shrink=0.0, shrink_C=1.0, validation=None, patience=20, monitor="loss", warmup=0, clip_norm=None, average=None, ema_decay=0.99, swa_start=0.75)` — [attentive.py:102](attentive.py#L102)

- Quand : appelée par [`head.fit_and_score`](head.py#L619) pour la tête `attentive`, à travers `regularization.fit_with_options` (qui ajoute R40 et R42 autour) ; par [`head.cascade_scores`](head.py#L91) pour le deuxième étage de la cascade, avec les réglages par défaut ; par le notebook `04_attention_jetons.ipynb`. `fit_and_score` est elle-même appelée par `head.oof_scores` (`blanci benchmark`, `blanci heads`) et `head_benchmark.annotation_curve` (`blanci heads-curve`).
- Reçoit : les jetons en `(fenêtres, jetons, dimension)`, les labels 0/1, et des options toutes coupées par défaut. Renvoie : un `AttentiveHead`.
- Fait : 1. charge torch, ou lève `RuntimeError` qui dit quoi installer ; 2. refuse des jetons qui ne sont pas à trois dimensions (`ValueError` « jetons attendus en (fenêtres, jetons, dim) ») ; 3. standardise chaque dimension sur l'ensemble des jetons de toutes les fenêtres (`feature_scaling`) ; 4. part d'une requête q nulle (l'attention est alors la moyenne simple des jetons), d'un vecteur w nul et d'un biais nul ; 5. à chaque époque, sur le lot entier : poids d'attention = softmax sur les jetons de `x·q / √d`, résumé z = somme des jetons pondérée, score = `z·w + b`, perte = entropie croisée où les deux classes pèsent autant (`balanced_bce`) ; 6. optimise avec `regularization.optimise` (Adam par défaut, 300 époques, pas 0,05, weight decay 0,001) ; 7. range dans `meta` le weight decay, les époques, le pas, le nombre de jetons et les options qui ne sont pas à leur valeur par défaut.
- Options (docstring) : `optimizer="adamw"` (R41) ; `entropy` (R43 : pénalité d'entropie de l'attention, > 0 piquée, < 0 diffuse) ; `token_dropout` (R45 : jetons masqués avant le softmax, à l'entraînement) ; `dim_dropout` (R46 : dimensions de z éteintes) ; `shrink` λ > 0 (R47 : w part de la logistique apprise sur la moyenne des jetons avec C = `shrink_C`, et ½λ‖w − w₀‖² remplace le weight decay sur w) ; `validation`, `patience`, `monitor` (R42 : arrêt sur des micros tenus à l'écart, critère « loss » ou « ap », courbe rangée dans `meta["validation_curve"]`) ; `warmup`, `clip_norm` (R59) ; `average`, `ema_decay`, `swa_start` (R64).
- Détail : `patience` ne joue que si `validation` est donnée.
- Pourquoi (docstring du fichier, DECISIONS n° 117 pour le tri des régularisations) : une note de 0,1 s dans une fenêtre de 5 s ne tient que dans un ou deux jetons, la moyenne la dilue dans le fond ; la tête n'a que 2·d + 1 paramètres, peu pour 150 à 200 positifs.
- `logits(inputs, train)` ([attentive.py:161](attentive.py#L161)) et `loss_of()` ([attentive.py:171](attentive.py#L171)), fonctions internes : le calcul du score (avec les dropouts à l'entraînement) et la perte d'une époque (avec les pénalités R43 et R47).

#### `AttentiveHead` — [attentive.py:61](attentive.py#L61)

- Quand : rendu par `fit_attentive`. Contient la moyenne et l'écart-type de standardisation, la requête `query`, le vecteur `weight`, le biais, et `meta`.
- `decision(tokens)` ([attentive.py:74](attentive.py#L74)) : le score de chaque fenêtre, en numpy : standardise, calcule les poids d'attention, résume, applique `weight` et `bias`. Appelée par `head.fit_and_score`, `head.cascade_scores` et `regularization.fit_with_options`.
- `attention(tokens)` ([attentive.py:69](attentive.py#L69)) : les poids d'attention `(fenêtres, jetons)`, pour voir quels morceaux de la fenêtre comptent. Appelée par le notebook 04 et par des tests ; aucun appelant dans `blanci/`.
- `save(directory)` ([attentive.py:79](attentive.py#L79)) et `load(directory)` ([attentive.py:93](attentive.py#L93)) : `weights.npz` (quatre tableaux) et `manifest.json` (`kind: "attentive"`, biais, `meta`), sans pickle (docstring du fichier, §7). `load` est appelée par le notebook 04 ; `save` par un test seulement.
- `_softmax(x, axis)` ([attentive.py:54](attentive.py#L54)) : softmax numpy, après avoir retranché le maximum pour éviter les débordements.

## `gated.py` : la sonde à portes (R85)

Une tête sur l'embedding où le poids de chaque dimension dépend de la fenêtre : une « porte » entre 0 et 1, calculée à partir de la fenêtre elle-même, laisse passer ou éteint chaque dimension avant le classement. Ne lit et n'écrit rien ; entraînement avec torch, application en numpy. Utilisé par la tête `gated` de `head.fit_and_score` : elle est dans la liste par défaut de `blanci heads` quand torch est installé (`head_methods` la retire sinon) ; ailleurs (`heads-curve`, `anuraset heads`, `anuraset campaign`) seulement si on la demande par `--methods` ou par la configuration, car elle n'est ni dans `head.curve.methods` de `config/default.yaml`, ni dans `HEADS`, ni dans `CAMPAIGN_HEADS` (`evaluation/anuraset.py`). Tests : [`tests/heads/test_gated.py`](../../tests/heads/test_gated.py), et un test d'arrêt précoce dans [`test_attentive.py`](../../tests/heads/test_attentive.py).

#### `fit_gated(X, y, rank=8, weight_decay=1e-2, epochs=300, lr=5e-2, seed=0, *, dim_dropout=0.0, validation=None, patience=20, monitor="loss", warmup=0, clip_norm=None, average=None, ema_decay=0.99, swa_start=0.75)` — [gated.py:49](gated.py#L49)

- Quand : un seul appelant hors tests : [`head.fit_and_score`](head.py#L619), qui la donne à [`fit_with_options`](regularization/torch_training.py#L361) ; c'est `fit_with_options` qui l'appelle, une ou plusieurs fois.
- Reçoit : les embeddings d'entraînement et leurs labels ; le rang ; les réglages de l'entraînement. Renvoie : un `GatedHead`.
- Fait : 1. charge torch ; sans torch : `RuntimeError` « R85 s'entraîne avec torch (uv sync --group research) » ; 2. standardise chaque dimension (`feature_scaling`) ; 3. ramène le rang entre 1 et la dimension ; 4. fixe la graine de torch ; départ : `down` au hasard (divisé par la racine de la dimension), `up`, `gate_bias`, `weight` et `bias` à 0 ; 5. le score d'une fenêtre est : embedding standardisé, multiplié dimension par dimension par sa porte, puis par `weight`, plus `bias` ; la porte est la fonction logistique de `(x · down) · up + gate_bias` ; 6. perte : entropie croisée à classes équilibrées (`balanced_bce`) ; 7. si `validation` est donné, ses fenêtres sont standardisées avec la moyenne et l'échelle de l'entraînement et servent de critère ([`validation_criterion`](regularization/torch_training.py#L284) : la perte, ou 1 − AP) ; 8. [`optimise`](regularization/torch_training.py#L70) : tout le lot à chaque pas, optimiseur AdamW, pénalité `weight_decay` sur les cinq paramètres ; 9. rend la tête en `float32`.
- Détails : `meta` contient toujours `rank`, `weight_decay`, `epochs`, `lr` ; `dim_dropout`, `warmup`, `clip_norm`, `average` seulement s'ils sont actifs ; `best_epoch` et `validation_curve` seulement avec `validation`. Sans `validation`, `patience` ne fait rien. Hors tests, `fit_with_options` passe `validation` avec `patience=None` (elle relève la courbe entière de chaque fold interne), puis réentraîne sans validation sur le nombre d'époques retenu.
- Options reçues du régularisateur (`Regularizer.torch_options`, pour `gated` : R40, R42, R46, R59, R64 d'après `regularization/names.py`) : `dim_dropout` (R46 : une part des dimensions éteinte au hasard à l'entraînement), `warmup` et `clip_norm` (R59), `average`, `ema_decay`, `swa_start` (R64), `epochs` et `weight_decay`.
- Pourquoi (docstring du fichier, DECISIONS n° 110) : la logistique donne un poids fixe à chaque dimension ; ici le poids dépend du son entendu. Les deux matrices sont de rang faible (8), sinon il y aurait autant de paramètres que le carré de la dimension pour une cinquantaine d'enregistrements positifs. Au départ `up` et `gate_bias` sont nuls : toutes les portes valent ½, la tête part d'une logistique et la porte ne s'en écarte que si les données le demandent.
- `logits(inputs, train)` ([gated.py:100](gated.py#L100)), fonction interne : le score de l'étape 5, avec le dropout de R46 quand `train` est vrai.
- `numpy(t)` ([gated.py:131](gated.py#L131)), fonction interne : un paramètre torch rendu en tableau `float32`.

#### `GatedHead` — [gated.py:29](gated.py#L29)

- Champs : `mean`, `scale`, `down` (dimension × rang), `up` (rang × dimension), `gate_bias`, `weight`, `bias`, `meta`. Pas de `save` ni de `load`.
- `decision(X)` ([gated.py:44](gated.py#L44)) : le score, en numpy. Appelée par `head.fit_and_score` et par `fit_with_options` (choix du weight decay).
- `gates(X)` ([gated.py:39](gated.py#L39)) : la porte de chaque dimension pour chaque fenêtre, « ce que la tête écoute » (docstring). Aucun appelant hors de `tests/heads/test_gated.py`.
- `_sigmoid(x)` ([gated.py:24](gated.py#L24)) : la fonction logistique, écrite `0,5 × (1 + tanh(x / 2))`.

## `proto_probe.py` : la sonde à prototypes sur les jetons

Une tête apprise sur les jetons : 8 « prototypes » de la taille d'un jeton ; chaque jeton de la fenêtre est comparé à chaque prototype, on garde pour chaque prototype sa meilleure ressemblance dans la fenêtre, et une couche linéaire transforme ces 8 nombres en score. Ne lit et n'écrit rien ; s'entraîne avec torch, s'applique en numpy. Utilisé par la tête `proto_probe` de `head.fit_and_score`, donc par `blanci heads` quand l'encodeur a des jetons, et par `anuraset/calcul/global_bench.py --tokens`. Tests : [`tests/heads/test_proto_probe.py`](../../tests/heads/test_proto_probe.py).

#### `fit_proto_probe(tokens, y, n_prototypes=8, weight_decay=1e-3, epochs=200, lr=5e-2, seed=0, patience=20)` — [proto_probe.py:49](proto_probe.py#L49)

- Quand : un seul appelant hors tests, [`head.fit_and_score`](head.py#L619), avec `seed` pour tout réglage.
- Reçoit : les jetons à plat des fenêtres d'entraînement et leurs labels. Renvoie : un `ProtoProbeHead`.
- Fait : 1. charge torch (`import_torch`) ; sans torch : `RuntimeError` « la sonde à prototypes s'entraîne avec torch (uv sync --group research) » ; 2. refuse des jetons qui ne sont pas en 3 dimensions ; 3. calcule moyenne et écart-type de chaque dimension sur tous les jetons de toutes les fenêtres ([`feature_scaling`](regularization/torch_training.py#L24)), standardise, puis ramène chaque jeton à la norme 1 ; 4. tire les prototypes de départ : 8 fenêtres positives au hasard (avec remise ; toutes les fenêtres s'il n'y a aucun positif) et, dans chacune, un jeton au hasard ; 5. poids de la couche linéaire et biais à 0 ; 6. perte : entropie croisée où les positifs pèsent `négatifs / positifs` ([`balanced_bce`](regularization/torch_training.py#L32)) ; 7. [`optimise`](regularization/torch_training.py#L70) : descente sur tout le lot à la fois, optimiseur Adam, 200 pas, pénalité `weight_decay` sur le poids et le biais mais pas sur les prototypes ; 8. rend les prototypes ramenés à la norme 1, le poids, le biais, et `meta` (`n_prototypes`, `epochs`, `lr`, `n_tokens`).
- Détail : `optimise` est appelée sans critère de validation ; `patience` ne fait donc rien, les 200 époques sont toujours faites.
- Pourquoi (docstring du fichier, n° 151) : c'est la tête que proposent les auteurs de Bird-MAE ; une note brève dans un seul jeton n'est pas diluée par la moyenne de la fenêtre. Version simplifiée : pas de perte d'orthogonalité entre prototypes.
- `_normalise(x)` ([proto_probe.py:27](proto_probe.py#L27)) : divise chaque vecteur par sa norme (au moins 1e-8).

#### `ProtoProbeHead` — [proto_probe.py:32](proto_probe.py#L32)

- Champs : `mean`, `scale`, `prototypes` (K lignes de norme 1), `weight` (K valeurs), `bias`, `meta`. Pas de `save` ni de `load`.
- `decision(tokens)` ([proto_probe.py:45](proto_probe.py#L45)) : `activations` multipliées par `weight`, plus `bias`. Appelée par `head.fit_and_score`.
- `activations(tokens)` ([proto_probe.py:40](proto_probe.py#L40)) : un tableau (fenêtres, K) : pour chaque prototype, la plus grande similarité cosinus trouvée parmi les jetons de la fenêtre.

## `dann.py` : tête R66, apprentissage adverse contre le micro

Une tête à petit réseau qui apprend à reconnaître le chant tout en rendant le micro méconnaissable dans sa représentation. Ne lit et n'écrit aucun fichier ni aucune table. Entraînement avec torch (groupe `research`), application en numpy. Utilisé par la tête `dann` de [`head.fit_and_score`](head.py#L619), donc par `blanci heads` (la tête est dans la liste par défaut quand torch est installé, `head_benchmark.head_methods`). Tests : [`tests/heads/test_new_heads.py`](../../tests/heads/test_new_heads.py).

#### `fit_dann(X, y, seed=0, *, groups, hidden=32, strength=1.0, domain_on="negatives", weight_decay=1e-2, epochs=300, lr=1e-2, dim_dropout=0.0, validation=None, patience=20, monitor="loss", warmup=0, clip_norm=None, average=None, ema_decay=0.99, swa_start=0.75)` — [dann.py:53](dann.py#L53)

- Quand : un seul appelant hors tests, [`head.fit_and_score`](head.py#L619) pour la méthode `dann`, à travers [`regularization.fit_with_options`](regularization/torch_training.py#L361). Les micros viennent de `regularizer.context.groups`, les réglages de `Regularizer.dann_options()` (section `regularization.R66` : `hidden` 32, `strength` 1,0, `domain_on` negatives, `weight_decay` 0,01, `epochs` 300, `lr` 0,01), complétés par les options torch du nom de la tête (`dann+R42`…).
- Reçoit : les embeddings, les labels 0/1, et `groups`, le micro de chaque fenêtre (le « domaine »). Renvoie : un `DannHead`.
- Fait : 1. charge torch ; refuse un `domain_on` autre que `negatives` ou `all` (`ValueError`) ; 2. standardise les embeddings ; 3. numérote les micros ; 4. choisit les fenêtres sur lesquelles le micro est appris : les négatifs seuls (défaut) ou toutes ; 5. crée la représentation h = tanh(A·x̃ + a) à `hidden` dimensions, la sortie A. blanci `w·h + b`, et un classifieur de micro branché sur h ; 6. à chaque époque : perte = entropie croisée A. blanci à classes équilibrées + entropie croisée du micro, calculée derrière une inversion du gradient (`grad_reverse`) dont la force monte de 0 à `strength` au fil des époques (`dann_strength`) ; 7. optimise avec `regularization.optimise`, toujours en AdamW ; 8. rend la tête sans le classifieur de micro, avec dans `meta` les réglages, le nombre de micros et `adversary`.
- Détails : l'adversaire n'est branché que s'il y a au moins deux micros parmi les fenêtres choisies et si `strength` > 0 ; sinon c'est « un petit réseau ordinaire » (docstring), et `meta["adversary"]` vaut faux. Le weight decay s'applique à tous les paramètres, classifieur de micro compris.
- Pourquoi (docstring du fichier, DECISIONS n° 124 et 109) : c'est la version apprise de R21 ; le micro n'est appris que sur les négatifs pour que la représentation rende les fonds indiscernables sans être poussée à cacher le chant : si A. blanci chante surtout sur certains micros, effacer le micro partout effacerait une partie du chant.
- `fit_dann.needs_groups = True` ([dann.py:186](dann.py#L186)) : un attribut posé sur la fonction ; `fit_with_options` le lit pour lui passer les micros (`groups=`), ce qu'elle ne fait pas pour les autres têtes torch.
- `represent(inputs)` ([dann.py:116](dann.py#L116)), `logits(inputs, train)` ([dann.py:119](dann.py#L119)), `loss_of()` ([dann.py:124](dann.py#L124)), `numpy(t)` ([dann.py:160](dann.py#L160)), fonctions internes : la représentation, le score, la perte d'une époque (elle compte les époques pour la force de l'inversion), la conversion des poids en numpy `float32`.

#### `DannHead` — [dann.py:35](dann.py#L35)

- Quand : rendu par `fit_dann`. Contient la standardisation, la matrice `down` (A) et son biais, le vecteur `weight`, le biais, `meta`.
- `representation(X)` ([dann.py:44](dann.py#L44)) : h, « ce que la tête garde de l'embedding, le micro en moins » (docstring) ; appelée par `decision` et par un test.
- `decision(X)` ([dann.py:49](dann.py#L49)) : le score, `representation(X) · weight + bias` ; appelée par `head.fit_and_score` et `fit_with_options`.
- Détail : pas de `save` ni de `load`, à la différence de `AttentiveHead`.

## `signal_processing.py` : traitement du signal sans apprentissage

Calcule, sans encodeur ni apprentissage, ce qui se mesure directement sur le son : les débuts de notes dans la bande du chant, le rythme (intervalles entre notes), la persistance (suites de fenêtres positives), et en amont de l'encodeur deux transformations du son (passe-bande, débruitage) et quatre portes qui arrêtent une fenêtre avant l'encodage. Lit : les fichiers audio sous `paths.raw` (`compute_onsets`), la table `onsets` (`load_onsets`), les sections `signal` et `signal_processing.upstream` de la configuration (ou l'ancienne section `prefilter`). Écrit : la table `onsets` seulement (`store_onsets`, en `INSERT OR REPLACE`) ; jamais sur les disques des enregistrements. Utilisé par `blanci onsets`, `blanci embed` (transformations et portes), `blanci upstream-bench`, `blanci fusion`, `blanci fusion-bench`, `blanci score --fusion`, `blanci select --method gaps`, par `inputs/dataset.py` (faux négatifs suspects), par `heads/head.py` (`GATED_SCORE`), par `exploration/` et les notebooks `01_explorer_une_fenetre.ipynb` et `03_traitement_du_signal.ipynb`. Tests : [`tests/heads/test_signal_processing.py`](../../tests/heads/test_signal_processing.py) (enveloppe, débuts de notes, rythme, persistance, table `onsets`) et [`tests/embedding/test_upstream.py`](../../tests/embedding/test_upstream.py) (transformations, portes, `Upstream`) ; aussi `tests/combination/test_stacking.py` (`store_onsets`, `onset_counts`), `tests/exploration/test_explore.py`, `tests/embedding/test_embed.py` (`Upstream.transform_tag`), et `tests/interfaces/test_cli_pipeline.py`, seul endroit où `compute_onsets` est exercée (par la commande `onsets`).

#### `compute_onsets(con, recordings, raw_root, signal_cfg, channel=0, commit_every=50, workers=1)` — [signal_processing.py:281](signal_processing.py#L281)

- Quand : c'est le cœur de `blanci onsets`, seul appelant ([`cli.onsets`](../cli.py#L1024)). La commande lui passe la liste de `embed.select_recordings(con, site=site)` (réduite à `dataset.benchmark_subset` avec `--subset benchmark`), `paths.raw`, `cfg["signal"]`, `cfg["audio"]["channel"]` (0 dans `config/default.yaml`) et `workers` (4 par défaut dans la commande, 1 dans la fonction).
- Reçoit : la connexion, le tableau des enregistrements, la racine de l'audio, les réglages `signal`, le canal. Renvoie : un dictionnaire `computed`, `skipped`, `errors`, `onsets` (nombre total de débuts de notes trouvés).
- Fait : 1. lit les `recording_id` déjà présents dans la table `onsets` ; 2. garde les enregistrements qui n'y sont pas (`skipped` = les autres) ; 3. les lit et calcule dans `workers` fils avec `core.ahead.ahead` (avance de `2 * workers`) ; 4. dans le fil principal, pour chaque résultat : `None` compte dans `errors`, sinon `store_onsets` et les compteurs ; 5. valide la base tous les `commit_every` enregistrements calculés, puis une dernière fois à la fin.
- Détails qui ne se voient pas : la reprise est guidée par la table seule. Un enregistrement qui y a une ligne est sauté, quels que soient le canal et les réglages `signal` avec lesquels elle a été calculée. Le fichier est lu à `raw_root/<chemin de la base>`, sans `core.locate.variants` ni `locate` : un fichier renommé à la copie ou rangé sur un autre disque compte dans `errors` (« illisibles » dans le message de la commande).
- Pourquoi : la docstring dit « lecture seule de l'audio » ; la docstring de `cli.onsets` dit que c'est la seule commande qui calcule les débuts de notes et qu'il faut la lancer avant `fusion`.
- `prepare(rec)` ([signal_processing.py:301](signal_processing.py#L301)), fonction interne, tourne dans les fils : `core.audio.load_audio(Path(raw_root) / rec.path, channel)`, puis `recording_onsets` ; toute exception à la lecture rend `None` (commentaire : « fichier illisible : déjà signalé à l'inventaire »).

#### `recording_onsets(wav, sr, signal_cfg)` — [signal_processing.py:197](signal_processing.py#L197)

- Quand : dans `prepare` de `compute_onsets` ; et dans [`embed.embed_recordings`](../embedding/embed.py#L198), quand des portes sont actives et que l'enregistrement n'a pas de ligne dans `onsets` (les débuts de notes sont alors calculés pour la porte, sans être rangés).
- Reçoit : le signal d'un enregistrement entier, sa fréquence, les réglages `signal`. Renvoie : les instants de début de note, en secondes.
- Fait : appelle `detect_onsets` avec `signal.band_hz`, `signal.note_dur_s` et `signal.onset_k_mad` (4 400–5 500 Hz, 0,07–0,13 s et 4,0 dans `config/default.yaml`) ; le lissage reste celui par défaut (0,01 s).

#### `detect_onsets(wav, sr, band=(4400, 5500), note_dur_s=(0.07, 0.13), k_mad=4.0, smooth_s=0.01)` — [signal_processing.py:51](signal_processing.py#L51)

- Quand : appelée par `recording_onsets` (enregistrement entier), `note_snr_db`, [`baselines.window_features`](baselines.py#L132) (une fenêtre, à 24 kHz), [`explore.onset_sweep`](../exploration/explore.py#L277) et [`explore_plots.plot_window`](../exploration/explore_plots.py#L155).
- Reçoit : un signal mono, sa fréquence, la bande, les durées de note acceptées, le facteur du seuil, le lissage. Renvoie : un tableau des instants (s) où commence un événement de la durée d'une note.
- Fait : 1. calcule l'enveloppe en dB dans la bande (`band_envelope_db`) ; 2. pose le seuil à médiane + `k_mad` × MAD de cette enveloppe (MAD = médiane des écarts absolus à la médiane, plus 1e-6) ; 3. repère chaque passage continu au-dessus du seuil (un passage qui touche le début ou la fin du signal est fermé au bord) ; 4. garde ceux dont la durée est entre `note_dur_s[0]` et `note_dur_s[1]`, bornes comprises ; 5. rend leurs instants de début.
- Pourquoi (docstring) : le seuil est adaptatif, « robuste au fond de chaque enregistrement ». La docstring ajoute « jamais de filtre amont (bande saturée, §3) », ce que les portes `notes` et `rhythm` contredisent (voir l'écart n° 18).

#### `band_envelope_db(wav, sr, band, smooth_s=0.01)` — [signal_processing.py:36](signal_processing.py#L36)

- Quand : appelée par `detect_onsets`, `note_snr_db` et [`explore_plots.plot_window`](../exploration/explore_plots.py#L155) (pour tracer l'enveloppe et le seuil).
- Reçoit : un signal, sa fréquence, la bande `(bas, haut)` en Hz, la durée du lissage. Renvoie : l'enveloppe en dB, un point par échantillon.
- Fait : 1. ramène la borne haute sous 0,99 × la fréquence de Nyquist ; 2. filtre passe-bande Butterworth d'ordre 4, appliqué dans les deux sens (`sosfiltfilt`, pas de décalage dans le temps) ; 3. prend le module du signal analytique (`hilbert`) : l'amplitude instantanée ; 4. la lisse par une moyenne glissante de `max(1, round(smooth_s × sr))` échantillons ; 5. rend `20 × log10(enveloppe + 1e-10)`.
- Détail : la moyenne glissante passe par `uniform_filter1d(mode="constant")` quand le signal est au moins aussi long que la fenêtre de lissage, sinon par `np.convolve(..., "same")` ; le commentaire dit que c'est le même calcul, sept fois plus vite.

#### `store_onsets(con, recording_id, onsets, channel)` — [signal_processing.py:208](signal_processing.py#L208)

- Quand : appelée par `compute_onsets` seulement (et par des tests de `test_signal_processing.py` et `test_stacking.py`).
- Écrit : une ligne de la table `onsets` en `INSERT OR REPLACE` : l'enregistrement, le canal en texte, les instants en liste JSON arrondis à 4 décimales, la date (`core.db.utc_now`). Ne valide pas : c'est l'appelant qui fait `commit`.
- Détail : la clé de la table est `recording_id` seul (migration 2 de `core/db.py`) : une ligne par enregistrement, quel que soit le canal.

#### `load_onsets(con, recording_ids=None)` — [signal_processing.py:220](signal_processing.py#L220)

- Quand : appelée par [`embed.embed_recordings`](../embedding/embed.py#L198) (un enregistrement à la fois, si des portes sont actives), [`service.fused_scores`](../service.py#L1540), [`service.window_gate_values`](../service.py#L1819) et [`stacking.build_level1`](../combination/stacking.py#L172).
- Reçoit : la connexion et, au besoin, des identifiants d'enregistrements. Renvoie : un dictionnaire `{recording_id: tableau des débuts de notes (s)}` ; un enregistrement jamais traité n'y figure pas.
- Fait : sans identifiants, lit toute la table ; avec, retire les doublons et interroge par paquets de 500 (commentaire : « sous la limite de paramètres de SQLite »). La colonne `channel` n'est pas lue.

#### `rhythm_features(onsets, duration_s, ioi_range_s=(1.2, 1.906))` — [signal_processing.py:76](signal_processing.py#L76)

- Quand : appelée par `window_rhythm` (une fenêtre) et `signal_features` ; importée par le notebook `01_explorer_une_fenetre.ipynb`.
- Reçoit : des débuts de notes, la durée du segment, la plage d'intervalles d'A. blanci. Renvoie : un dictionnaire de cinq descripteurs.
- Fait : trie les débuts de notes, calcule les intervalles entre notes consécutives (IOI), puis : `onset_rate_hz` (notes par seconde, 0 si la durée est nulle), `ioi_median_s` (`nan` sans intervalle), `ioi_cv` (écart-type ÷ moyenne des intervalles, `nan` s'il y a moins de deux intervalles), `frac_ioi_blanci` (part des intervalles dans la plage, 0 sans intervalle), `frac_ioi_short` (part des intervalles sous 0,6 s, 0 sans intervalle).
- Pourquoi : pas de docstring ; la docstring du fichier donne la plage (IOI 1,200–1,906 s, moyenne 1,414 s) et dit que les intervalles courts signalent plusieurs chanteurs (hypothèse H21, rappelée en commentaire sur `frac_ioi_short`).

#### `window_rhythm(windows, onsets, ioi_range_s=(1.2, 1.906))` — [signal_processing.py:250](signal_processing.py#L250)

- Quand : un seul appelant, [`stacking.signal_features`](../combination/stacking.py#L124), elle-même appelée par `stacking.build_level1` (entraînement de la fusion) et `service.fused_scores` (score avec fusion).
- Reçoit : un tableau de fenêtres (`recording_id`, `offset_s`, `dur_s`) et le dictionnaire de `load_onsets`. Renvoie : un tableau aligné sur les fenêtres, colonnes `RHYTHM_COLUMNS`.
- Fait : pour chaque fenêtre, garde les débuts de notes de son enregistrement qui tombent dans `[offset, offset + dur[`, les ramène au début de la fenêtre, et appelle `rhythm_features` ; si l'enregistrement n'a pas de débuts de notes calculés, la ligne est `nan`.

#### `persistence_features(window_scores, threshold, offsets_s=None, gap_radius_s=GAP_RADIUS_S)` — [signal_processing.py:114](signal_processing.py#L114)

- Quand : appelée par `recording_persistence` et `signal_features`.
- Reçoit : les scores des fenêtres d'un enregistrement, dans l'ordre du temps, un seuil, et au besoin les décalages des fenêtres. Renvoie : un dictionnaire de six descripteurs.
- Fait : une fenêtre est positive si son score est ≥ `threshold`. Puis : `frac_windows` (part de fenêtres positives), `n_positive`, `longest_run` (plus longue suite de positives), `n_isolated` (suites d'une seule fenêtre : faux positifs suspects), `span` (écart entre la première et la dernière positive, en secondes avec `offsets_s`, en nombre de fenêtres sinon), `n_gaps` (fenêtres négatives encadrées de positives, par `surrounded_by_positives`).
- Détail : pour `n_gaps`, le rayon vaut `gap_radius_s` secondes quand les décalages sont donnés, et 2 fenêtres sinon (docstring).

#### `recording_persistence(scores, threshold=0.0)` — [signal_processing.py:270](signal_processing.py#L270)

- Quand : appelée par [`stacking.build_level1`](../combination/stacking.py#L172) (dans chaque fold, sur toutes les fenêtres des enregistrements testés, seuil `PERSISTENCE_THRESHOLD` = 0,0) et par [`service.fused_scores`](../service.py#L1540) (seuil relu de la fusion enregistrée, `persistence_threshold`).
- Reçoit : un tableau (`recording_id`, `offset_s`, `score`) de toutes les fenêtres des enregistrements. Renvoie : un tableau indexé par enregistrement, colonnes `PERSISTENCE_COLUMNS`.
- Fait : trie par décalage, groupe par enregistrement, appelle `persistence_features` avec les décalages.

#### `surrounded_by_positives(centers, positive_centers, radius=GAP_RADIUS_S)` et `GAP_RADIUS_S` — [signal_processing.py:95](signal_processing.py#L95)

- Quand : `surrounded_by_positives` est appelée par `persistence_features`, par [`dataset.paired_negatives`](../inputs/dataset.py#L515) (fonction interne `draw_same_recording` : ne jamais tirer comme négatif présumé une fenêtre encadrée de positifs), [`dataset.suspect_false_negatives`](../inputs/dataset.py#L640), [`selection.gap_candidates`](../annotation/selection.py#L432) (`blanci select --method gaps`) et [`explore.recording_windows`](../exploration/explore.py#L147).
- Reçoit : les centres (s) des fenêtres à juger, les centres des positifs, un rayon. Renvoie : un tableau de booléens, vrai pour chaque centre qui a un positif strictement avant lui et un strictement après lui, chacun à `radius` secondes au plus.
- Fait : trie les positifs, cherche par dichotomie le dernier positif avant et le premier après chaque centre ; sans positif ou sans centre, tout est faux.
- Pourquoi (docstring, DECISIONS n° 102) : une fenêtre négative ainsi encadrée est un faux négatif suspect, « le miroir » du positif isolé.
- `GAP_RADIUS_S` ([signal_processing.py:92](signal_processing.py#L92)) vaut 6,0 s ; commentaire : « ≈ 4 notes d'A. blanci (intervalle moyen 1,414 s) ». C'est la valeur de repli de `signal_processing.gap_radius_s` (6,0 aussi dans `config/default.yaml`), lue par [`dataset.pairing_options`](../inputs/dataset.py#L75), `selection.gap_candidates` et `explore.recording_windows`, et la valeur par défaut du rayon de cinq fonctions de `inputs/dataset.py` (paramètre `gap_radius_s` de `paired_negatives`, `training_set`, `embedded_training_set`, `benchmark_recordings` ; paramètre `radius` de `suspect_false_negatives`).

#### `RHYTHM_COLUMNS` et `PERSISTENCE_COLUMNS` — [signal_processing.py:239](signal_processing.py#L239)

- Les noms des cinq descripteurs de rythme et des six de persistance (`PERSISTENCE_COLUMNS`, [signal_processing.py:240](signal_processing.py#L240)). Servent à `window_rhythm` et `recording_persistence` pour nommer leurs colonnes, et à [`stacking.seq_columns`](../combination/stacking.py#L133), qui garde les descripteurs de rythme si `signal_processing.position` contient `parallel` et ceux de persistance s'il contient `downstream`.

#### `signal_features(onsets, window_scores, threshold=0.5, duration_s=120.0, offsets_s=None, ioi_range_s=(1.2, 1.906))` — [signal_processing.py:147](signal_processing.py#L147)

- Quand : aucun appelant hors tests (un test de `test_signal_processing.py`). À ne pas confondre avec [`stacking.signal_features`](../combination/stacking.py#L124), une autre fonction du même nom, qui est celle que la fusion utilise.
- Fait : réunit en un dictionnaire `rhythm_features(onsets, duration_s, ioi_range_s)` et `persistence_features(window_scores, threshold, offsets_s)` pour un enregistrement.

#### `note_snr_db(wav, sr, band=(4400, 5500), note_dur_s=(0.07, 0.13), k_mad=4.0, neighbour_s=0.5)` — [signal_processing.py:161](signal_processing.py#L161)

- Quand : aucun appelant hors tests (un test de `test_signal_processing.py`).
- Reçoit : un signal et les réglages de détection. Renvoie : le rapport signal sur bruit du chant en dB, ou `nan` sans note détectée (ou sans voisinage).
- Fait : 1. détecte les notes (`detect_onsets`) ; 2. passe l'enveloppe en puissance ; 3. marque comme « note » les `note_dur_s[1]` secondes qui suivent chaque début (la durée maximale acceptée, pas la durée mesurée), et comme « voisinage » les `neighbour_s` secondes de part et d'autre, notes exclues ; 4. rend `10 × log10(puissance moyenne des notes ÷ puissance moyenne du voisinage)`.
- Pourquoi (docstring) : ventiler le rappel par RSB. Aucun code du projet ne le fait aujourd'hui.

#### `TRANSFORMS`, `GATES`, `ONSET_GATES`, `FEATURES`, `GATE_THRESHOLD_KEY`, `GATE_DEFAULTS`, `GATED_SCORE` — [signal_processing.py:333](signal_processing.py#L333)

- `TRANSFORMS` = `bandpass`, `denoise` : elles changent le son donné à l'encodeur. `GATES` = `band_energy`, `band_contrast`, `notes`, `rhythm` : elles arrêtent une fenêtre. `ONSET_GATES` = `notes`, `rhythm` : les deux portes comptées sur les débuts de notes. `FEATURES` = les six noms, ceux qu'accepte `--upstream`.
- `GATE_THRESHOLD_KEY` ([signal_processing.py:337](signal_processing.py#L337)) dit sous quelle clé de la configuration se lit le seuil (`min_db` pour l'énergie et le contraste, `min_count` pour les notes et le rythme) ; `GATE_DEFAULTS` ([signal_processing.py:343](signal_processing.py#L343)) donne les seuils de repli : 6,0 dB, 3,0 dB, 1 note, 1 intervalle.
- `GATED_SCORE` ([signal_processing.py:344](signal_processing.py#L344)) = −1 000 000, le score d'une fenêtre arrêtée ; commentaire : il est fini parce que « les métriques refusent −inf ». Lu par `apply_gate`, [`head.oof_scores`](head.py#L768) (les fenêtres arrêtées ne sont jamais apprises et reçoivent ce score), [`stacking.design_matrix`](../combination/stacking.py#L304) et [`service.fused_scores`](../service.py#L1540) (un score ≤ `GATED_SCORE` redevient « pas de score de tête »).
- `GATES` est aussi lu par [`service.window_gate_values`](../service.py#L1819), [`explore.gate_report`](../exploration/explore.py#L259), [`explore_plots.plot_indices`](../exploration/explore_plots.py#L139) et le notebook `03_traitement_du_signal.ipynb`.
- Pourquoi (commentaire en tête de section, DECISIONS n° 90 et 103) : le traitement du signal et le seuillage spectral « ne font qu'un » ; une transformation change le son, donc l'encodeur et son stock (`<nom>+<étiquette>`) ; une fenêtre arrêtée n'est pas encodée, n'est jamais apprise, et un positif arrêté compte comme manqué.

#### `upstream_from_cfg(cfg, only=None)` — [signal_processing.py:453](signal_processing.py#L453)

- Quand : appelée par [`cli.upstream_chain`](../cli.py#L665) (donc par `blanci embed`, et par `blanci anuraset campaign` avec `"none"`), par [`cli.upstream_bench_command`](../cli.py#L676) (avec la valeur de `--upstream`), par [`encoders._with_transforms`](../embedding/encoders/__init__.py#L25) (pour compléter les réglages d'une transformation), par `onset_gates` ; aussi par le notebook `01_explorer_une_fenetre.ipynb`.
- Reçoit : la configuration, et au besoin `only`, une liste de noms ou un texte « bandpass,notes » ou « none ». Renvoie : un `Upstream`.
- Fait : 1. lit la section (`_upstream_section`) ; 2. avec `only`, les fonctionnalités actives sont exactement celles nommées (« none » et les noms vides sont ignorés) ; un nom inconnu lève `ValueError` « fonctionnalités inconnues : … (connues : …) » ; sans `only`, ce sont celles dont `enabled` est vrai dans la section ; 3. pour chaque transformation active, part des réglages par défaut (`bandpass` : bande 3 000–7 000 Hz, ordre 4 ; `denoise` : force 1,0, plancher 0,1) et pose par-dessus ceux de la configuration ; 4. pour chaque porte active, lit son seuil (`gate_threshold`) ; 5. lit `gates.combine` (`all` par défaut) et recopie la section `signal`.
- Détails : `only` remplace les interrupteurs `enabled` mais pas les réglages, qui restent ceux de la configuration (docstring). Les transformations sortent toujours dans l'ordre `bandpass` puis `denoise`, quel que soit l'ordre des noms donnés. Dans `config/default.yaml`, tout est à `enabled: false`.
- `_upstream_section(cfg)` ([signal_processing.py:447](signal_processing.py#L447)) : rend `signal_processing.upstream` ; si cette clé est absente, l'ancienne section `prefilter` ; sinon un dictionnaire vide.

#### `gate_threshold(cfg, name)` — [signal_processing.py:489](signal_processing.py#L489)

- Quand : appelée par `upstream_from_cfg`, `onset_gates`, [`explore.gate_report`](../exploration/explore.py#L259), [`explore_plots.plot_indices`](../exploration/explore_plots.py#L139) et le notebook `03_traitement_du_signal.ipynb`.
- Reçoit : la configuration et le nom d'une porte. Renvoie : son seuil, lu dans `signal_processing.upstream.gates.<nom>` sous la clé de `GATE_THRESHOLD_KEY`, sinon la valeur de `GATE_DEFAULTS`.

#### `Upstream` — [signal_processing.py:378](signal_processing.py#L378)

- Quand : c'est l'objet qui transporte les fonctionnalités actives en amont. Fabriqué par `upstream_from_cfg`, par [`cli.upstream_chain`](../cli.py#L665) (un `Upstream()` vide quand `signal_processing.position` ne contient pas `upstream` et que `--upstream` n'est pas donné), par [`encoders._with_transforms`](../embedding/encoders/__init__.py#L25) et par [`explore.upstream_stages`](../exploration/explore.py#L249). Reçu par [`embed.embed_recordings`](../embedding/embed.py#L198) (paramètre `gates`), [`UpstreamEncoder`](../embedding/encoders/upstream.py#L17), [`service.upstream_bench`](../service.py#L1858), `embed.stock_identity` et `embed.register_encoder`.
- Contient : `transforms` (`{nom: réglages}`, dans l'ordre d'application), `gates` (`{nom: seuil}`), `combine` (`"all"` par défaut), `signal_cfg` (la section `signal`, dont les portes ont besoin).
- `active` ([signal_processing.py:390](signal_processing.py#L390)) : vrai s'il y a au moins une transformation ou une porte ; `cli.embed` s'en sert pour afficher la ligne « traitement du signal en amont ».
- `transform_tag()` ([signal_processing.py:393](signal_processing.py#L393)) : l'étiquette ajoutée au nom de l'encodeur par `UpstreamEncoder` : `bp3-7k` pour un passe-bande 3–7 kHz, `dn1` pour un débruitage de force 1, joints par `+`. Un ordre de filtre différent de 4 ou un plancher différent de 0,1 s'ajoutent (`bp3-7k-o6`, `dn1-f0.2`). Pourquoi (docstring, DECISIONS n° 143) : deux réglages, deux stocks.
- `gate_tag()` ([signal_processing.py:411](signal_processing.py#L411)) : l'étiquette des portes, ajoutée au nom du stock par `embed_recordings` (`<stock>+g-notes1-rhythm1-all`) et reprise dans le tableau `comparison` de `service.upstream_bench` ; chaîne vide sans porte.
- `transform(wav, sr)` ([signal_processing.py:418](signal_processing.py#L418)) : applique `bandpass` puis `denoise` selon `transforms` et rend du `float32` ; appelée par `UpstreamEncoder.embed`, `UpstreamEncoder.embed_tokens` et `explore.upstream_stages`. La durée de trame du débruitage n'est pas réglable par là (elle reste à 0,02 s).
- `passes(values)` ([signal_processing.py:428](signal_processing.py#L428)) : `gate_mask(values, self.gates, self.combine)` ; appelée par `embed_recordings` (fenêtres à encoder) et `service.upstream_bench`.

#### `bandpass(wav, sr, band_hz, order=4)` — [signal_processing.py:347](signal_processing.py#L347)

- Quand : appelée par `Upstream.transform`, et par [`workbench.wav_bytes`](../annotation/workbench.py#L552) quand on lui passe `band_hz` (le seul appel du poste d'annotation, dans `annotation/app.py`, ne le passe pas ; un test de `tests/annotation/test_workbench.py` le fait).
- Reçoit : un signal ou un lot de fenêtres, la fréquence, la bande. Renvoie : le signal filtré, en `float32`.
- Fait : filtre Butterworth passe-bande appliqué dans les deux sens (phase nulle) le long du dernier axe ; la borne haute est ramenée sous 0,99 × Nyquist ; une bande vide ou négative lève `ValueError` « bande … impossible à … Hz ».

#### `denoise(wav, sr, strength=1.0, floor=0.1, frame_s=0.02)` — [signal_processing.py:356](signal_processing.py#L356)

- Quand : appelée par `Upstream.transform` seulement.
- Reçoit : un signal ou un lot de fenêtres. Renvoie : le signal débruité, de la même longueur, en `float32`.
- Fait : 1. découpe en trames d'une puissance de 2 proche de `frame_s × sr` échantillons (au moins 16), recouvertes aux trois quarts (`stft`) ; 2. pour chaque fréquence, estime le fond par la médiane de l'amplitude sur la durée du signal reçu ; 3. retranche `strength` × ce fond, sans descendre sous `floor` × l'amplitude d'origine ; 4. garde la phase et revient au signal (`istft`), complété par des zéros ou coupé à la longueur de départ.
- Détails : un signal plus court qu'une trame est rendu tel quel. Le fond est celui de chaque fenêtre prise seule (docstring : « fenêtre par fenêtre »).

#### `gate_values(windows, sr, signal_cfg, offsets_s=None, onsets=None, dur_s=None)` — [signal_processing.py:525](signal_processing.py#L525)

- Quand : appelée par [`embed.embed_recordings`](../embedding/embed.py#L198) (toutes les fenêtres d'un enregistrement, à la fréquence d'origine, avec ses débuts de notes), [`service.window_gate_values`](../service.py#L1819) (une fenêtre à la fois, sans débuts de notes) et [`explore.window_indices`](../exploration/explore.py#L222).
- Reçoit : un lot de fenêtres `(n, échantillons)`, la fréquence, les réglages `signal`, et au besoin les décalages et la durée des fenêtres avec les débuts de notes de l'enregistrement. Renvoie : un tableau à quatre colonnes (`GATES`), une ligne par fenêtre.
- Fait : 1. pour chaque fenêtre, calcule les scores fixes de [`baselines.window_features`](baselines.py#L132) et en tire `band_energy`, `band_contrast`, `notes` et `rhythm` (la partie entière du score `rhythm` des baselines, qui porte un départage en centièmes) ; 2. si `onsets` et `offsets_s` sont donnés, remplace `notes` et `rhythm` par les comptes de `onset_counts` sur les débuts de notes de l'enregistrement.
- Pourquoi (docstring) : notes et rythme sont « les mêmes que le module en parallèle » ; à défaut de débuts de notes de l'enregistrement, ils sont détectés dans la fenêtre seule. `baselines` est importé dans la fonction, pas en tête de fichier (`baselines.py` importe lui-même `detect_onsets` d'ici).

#### `onset_counts(windows, onsets, ioi_range_s)` — [signal_processing.py:504](signal_processing.py#L504)

- Quand : appelée par `gate_values`, [`service.fused_scores`](../service.py#L1540) (porte de rythme d'une fusion placée en amont), [`service.window_gate_values`](../service.py#L1819) et [`stacking.build_level1`](../combination/stacking.py#L172).
- Reçoit : des fenêtres (`recording_id`, `offset_s`, `dur_s`), le dictionnaire des débuts de notes, la plage d'intervalles. Renvoie : un tableau à deux colonnes, `notes` (débuts de notes dans la fenêtre) et `rhythm` (intervalles consécutifs dans la plage d'A. blanci).
- Détail : un enregistrement absent du dictionnaire donne `nan` (une porte laisse alors passer) ; un enregistrement présent mais sans aucune note donne 0, ce qu'une porte `notes ≥ 1` arrête.

#### `gate_mask(values, gates, combine="all")` — [signal_processing.py:433](signal_processing.py#L433)

- Quand : appelée par `Upstream.passes`, `gate_sweep`, [`service.fused_scores`](../service.py#L1540), [`service.upstream_bench`](../service.py#L1858), [`stacking.upstream_pass`](../combination/stacking.py#L111) et le notebook `03_traitement_du_signal.ipynb`.
- Reçoit : le tableau des valeurs, les seuils `{porte: seuil}`, la combinaison. Renvoie : un tableau de booléens, vrai si la fenêtre passe.
- Fait : une porte est passée si la valeur est ≥ au seuil ou si elle manque (`nan`) ; `all` exige toutes les portes, `any` une seule ; sans porte, tout passe ; une autre combinaison lève `ValueError` « combinaison inconnue ».
- Pourquoi (docstring) : une valeur manquante laisse passer, « dans le doute, on encode ».

#### `onset_gates(cfg)` — [signal_processing.py:495](signal_processing.py#L495)

- Quand : appelée par [`service.train_fusion`](../service.py#L1428) (le résultat est rangé avec la fusion, sous la clé `gate`) et par [`stacking.upstream_pass`](../combination/stacking.py#L111) (appelée par `stacking._fused` quand l'emplacement contient `upstream`).
- Renvoie : `(portes, combinaison)` pour la chaîne de décision : parmi les portes activées dans la configuration, celles de `ONSET_GATES` ; s'il n'y en a aucune, la porte `notes` à son seuil.
- Pourquoi (docstring) : un module placé en amont sans porte de rythme activée garde au moins celle-là.

#### `apply_gate(scores, passed)` — [signal_processing.py:624](signal_processing.py#L624)

- Quand : partout où un score est rendu pour des fenêtres dont certaines ont été arrêtées : 8 fonctions, `service.py` (5 : `score_and_decide`, `evaluate_holdout`, `evaluate_frozen`, `fused_scores`, `upstream_bench`) et `combination/stacking.py` (3 : `build_level1`, `_secondary_head_oof`, `_fused`).
- Reçoit : des scores et un tableau de booléens « passe ». Renvoie : une copie des scores où chaque fenêtre arrêtée vaut `GATED_SCORE`.

#### `gate_sweep(values, y, recordings, grids=None)` et `GATE_GRIDS` — [signal_processing.py:573](signal_processing.py#L573)

- Quand : appelée par [`service.upstream_bench`](../service.py#L1858) (`blanci upstream-bench`) et par le notebook `03_traitement_du_signal.ipynb`.
- Reçoit : les valeurs des portes, les labels 0/1, l'enregistrement de chaque fenêtre, et au besoin d'autres grilles de seuils. Renvoie : un tableau, une ligne par (porte, seuil).
- Fait : pour chaque porte et chaque seuil de la grille, prise seule : part des négatifs arrêtés (`neg_stopped`, le calcul économisé), fenêtres positives perdues, enregistrements positifs gardés, et `recall_ceiling` = enregistrements positifs gardés ÷ enregistrements positifs ; puis l'intervalle de Wilson de ce rappel plafond (`evaluation.evaluate.wilson_interval`).
- Pourquoi (docstring) : un enregistrement positif n'est perdu que si toutes ses fenêtres positives sont arrêtées ; il ne peut alors plus être détecté, quel que soit l'encodeur.
- `GATE_GRIDS` ([signal_processing.py:565](signal_processing.py#L565)) : les seuils essayés par défaut : énergie 0 à 15 dB par pas de 3, contraste 0 ; 1,5 ; 3 ; 4,5 ; 6 ; 9 dB, notes 1 à 4, rythme 1 à 3 (19 lignes en tout).
- `_sweep_row(name, threshold, passed, y, recordings, positive_recordings)` ([signal_processing.py:606](signal_processing.py#L606)) : calcule la ligne d'un couple (porte, seuil).

## `baselines.py` : détecteurs sans encodeur

Calcule six scores par fenêtre sans aucun encodeur (énergie en bande, contraste de bande, nombre de notes, rythme, deux corrélations à un gabarit de note) et les évalue comme une tête d'encodeur : fenêtres annotées et négatifs appariés, folds groupés par micro, AP et rappel. Lit : les annotations et les enregistrements de la base (par `inputs/dataset.py`), le jeu gelé, des morceaux des fichiers audio sous `paths.raw` (lecture partielle avec `soundfile`), les sections `signal`, `benchmark`, `head` et `grids.w3` de la configuration. Écrit : un fichier de scores hors-fold par baseline et par canal sous `paths.reports/oof/` (source `baseline/<nom>/c<canal>`), et `baselines.csv`, `baselines_scores.csv`, `baselines.md` sous `paths.reports` ; jamais sur l'audio. Utilisé par `blanci baselines` ; `evaluation_windows` et `read_windows` servent aussi à `blanci upstream-bench` et `blanci detector-bench`, `window_features` aux portes en amont (`signal_processing.gate_values`) et au détecteur `band_contrast` ; notebooks `02_negatifs_apparies.ipynb` et `03_traitement_du_signal.ipynb`. Tests : [`tests/heads/test_baselines.py`](../../tests/heads/test_baselines.py) ; la commande est lancée dans `tests/interfaces/test_cli_pipeline.py`.

#### `run_baselines(con, cfg, raw_root, channels=(0,), progress_every=200)` — [baselines.py:236](baselines.py#L236)

- Quand : c'est le cœur de `blanci baselines`, seul appelant ([`cli.baselines`](../cli.py#L753)). La commande passe `paths.raw` et les canaux de `--channels`, sinon `audio.channel` s'il est un entier, sinon le canal 0.
- Reçoit : la connexion, la configuration, la racine de l'audio, les canaux à comparer. Renvoie : `(tableau des métriques par baseline, canal et niveau ; scores par fenêtre)`.
- Fait : 1. dresse les fenêtres (`evaluation_windows`) ; s'il n'y a qu'une classe, `ValueError` « il faut des positifs et des négatifs pour évaluer les baselines » ; 2. lit chaque fenêtre (`read_windows`) et calcule ses descripteurs sur chaque canal (`window_features`), avec une ligne d'avancement toutes les `progress_every` fenêtres ; 3. fixe la largeur d'un gabarit à deux durées de note : `max(3, round(2 × signal.note_max_s × 24 000 / 256))`, soit 19 trames (≈ 0,2 s) avec `note_max_s` = 0,103 ; 4. relit les folds communs (`dataset.folds_for`) et l'empreinte des labels (`oof.labels_fingerprint`) ; 5. pour chaque canal, prend les quatre scores fixes tels quels et calcule les deux scores de gabarit hors-fold (`oof_template_scores`) ; 6. pour chaque baseline, range ses scores hors-fold (`oof.save_oof`) et calcule les métriques (`evaluation.evaluate.evaluate`) aux niveaux `window` et `recording`, sur les seules fenêtres au score fini, avec le nombre de micros ; 7. trie par niveau puis par AP décroissante.
- Détails qui ne se voient pas : les groupes des folds sont les `point` (site/micro). Un fichier absent ou illisible arrête la commande (pas de `try` autour de la lecture). Les scores des baselines fixes ne dépendent d'aucun apprentissage : le texte du rapport dit qu'ils sont « hors-fold par nature ».
- Pourquoi (docstring du fichier) : répondre au go/no-go de P0 : si aucun encodeur n'atteint AP ≥ 0,5 et rappel ≥ 0,8 à précision ≥ 0,1, le détecteur primaire sera « DSP + template matching ».

#### `evaluation_windows(con, cfg)` — [baselines.py:75](baselines.py#L75)

- Quand : appelée par `run_baselines`, par [`service.upstream_bench`](../service.py#L1858) quand aucun encodeur n'est donné, par [`detectors/base.read_evaluation_audio`](detectors/base.py#L40) (`blanci detector-bench`), et par les notebooks 02 et 03.
- Reçoit : la connexion et la configuration. Renvoie : un tableau de fenêtres : `window_id`, `recording_id`, `path`, `offset_s`, `dur_s`, `label`, `y`, `presumed`, `point`, `site` (docstring).
- Fait : 1. lit les enregistrements du jeu gelé (`frozen.frozen_recordings`) ; 2. prend les enregistrements du benchmark (`dataset.benchmark_recordings` : annotés et candidats aux négatifs appariés) et retire les gelés ; 3. pose sur chacun la grille `grids.w3` (fenêtre de 3,0 s, pas de 1,5 s dans `config/default.yaml`), avec les identifiants de `core.db.window_id_for` ; 4. appelle `dataset.training_set` sur cette grille, avec `benchmark.negatives_per_positive` négatifs appariés par enregistrement positif (20), la graine `head.seed` et les réglages d'appariement (`dataset.pairing_options`) ; 5. ajoute la durée et le chemin du fichier.
- Pourquoi (docstring, DECISIONS n° 182 pour les intervalles) : ce sont les fenêtres des têtes, étiquetées par les labels de fenêtres et par les intervalles, jeu gelé exclu.

#### `read_windows(windows, raw_root, channels)` — [baselines.py:108](baselines.py#L108)

- Quand : appelée par `run_baselines`, [`service.window_gate_values`](../service.py#L1819) et [`detectors/base.read_evaluation_audio`](detectors/base.py#L40).
- Reçoit : un tableau de fenêtres avec `path`, `offset_s`, `dur_s`, la racine de l'audio, les canaux voulus. Renvoie : un itérateur de `(indice de ligne, {canal: forme d'onde}, fréquence d'échantillonnage)`.
- Fait : groupe les fenêtres par fichier, ouvre chaque fichier une seule fois (`soundfile.SoundFile`), se place au début de chaque fenêtre et ne lit que sa durée.
- Détails : un fichier mono donne son unique canal pour chaque canal demandé (docstring). L'« indice de ligne » est l'étiquette d'index du tableau : les trois appelants remettent l'index à 0, 1, 2… avant l'appel et s'en servent comme d'une position. Une fenêtre qui dépasse la fin du fichier rend un signal plus court, sans silence ajouté. Le fichier est cherché à `raw_root/<chemin de la base>`, sans `core.locate`.

#### `window_features(wav, sr, signal_cfg)` et `WindowFeatures` — [baselines.py:132](baselines.py#L132)

- Quand : appelée par `run_baselines`, par [`signal_processing.gate_values`](signal_processing.py#L525) (valeurs des portes en amont), par [`BandContrastDetector.score`](detectors/__init__.py#L29) et par le notebook `02_negatifs_apparies.ipynb`.
- Reçoit : la forme d'onde mono d'une fenêtre, sa fréquence, les réglages `signal`. Renvoie : un `WindowFeatures` ([baselines.py:68](baselines.py#L68)) : `fixed`, les quatre scores fixes, et `spec_db`, le spectrogramme en dB entre 3 et 7 kHz (fréquences × temps), gardé pour les gabarits.
- Fait : 1. rééchantillonne à 24 kHz (`core.audio.resample`) et complète par des zéros un signal de moins de 512 échantillons ; 2. calcule un spectrogramme de puissance (`scipy.signal.spectrogram`, trames de 512 échantillons, pas de 256) ; 3. moyenne la puissance dans la bande de la note (`signal.band_hz`) et dans les deux bandes voisines (`FLANKS_HZ`) ; 4. lisse en dB sur la durée d'une note (`round(signal.note_max_s × 24 000 / 256)` trames, 10 avec 0,103 s) l'énergie en bande et le contraste bande ÷ voisines ; 5. détecte les notes (`detect_onsets` avec les réglages `signal`) et compte les intervalles dans `signal.ioi_range_s` ; 6. rend `band_energy` et `band_contrast` (maximum moins médiane de la courbe lissée, dans la fenêtre), `notes` (nombre de notes) et `rhythm` (nombre d'intervalles dans la plage, plus le nombre de notes divisé par 100, pour départager).
- Pourquoi (docstring du fichier) : le contraste ne compte pas un son large bande (pluie, insecte, cri d'oiseau), qui « monte partout ».
- `_band_rows(freqs, band)` ([baselines.py:128](baselines.py#L128)) : les lignes du spectrogramme dont la fréquence est dans la bande, bornes comprises.

#### `FIXED`, `LEARNED`, `BASELINES`, `SR`, `NPERSEG`, `HOP`, `SPEC_BAND_HZ`, `FLANKS_HZ` — [baselines.py:54](baselines.py#L54)

- `FIXED` = `band_energy`, `band_contrast`, `notes`, `rhythm` (rien n'est appris) ; `LEARNED` = `template_mean`, `template_max` (gabarits appris dans chaque fold) ; `BASELINES` = les six.
- `SR` = 24 000 Hz ([baselines.py:60](baselines.py#L60)), `NPERSEG` = 512 (21 ms, cases de 47 Hz), `HOP` = 256 (10,7 ms), `SPEC_BAND_HZ` = 3 000–7 000 Hz, `FLANKS_HZ` = 3 000–4 200 et 5 700–7 000 Hz.
- Pourquoi (commentaire) : un rééchantillonnage commun pour que les gabarits aient les mêmes cases de fréquence quel que soit l'enregistreur ; 24 kHz couvre la bande de la note et ses voisines.

#### `oof_template_scores(specs, y, presumed, groups, width, n_splits=5, n_exemplars=30, seed=0, assignment=None)` — [baselines.py:205](baselines.py#L205)

- Quand : appelée par `run_baselines`, une fois par canal, avec `head.n_splits`, `head.seed` et les folds communs.
- Reçoit : les spectrogrammes des fenêtres, les labels, le drapeau « négatif présumé », le micro de chaque fenêtre, la largeur du gabarit. Renvoie : `{"template_mean": scores, "template_max": scores}`, alignés sur les fenêtres.
- Fait, dans chaque fold (`evaluation.evaluate.grouped_folds`) : 1. prend les positifs annotés du côté entraînement ; 2. découpe dans chacun un gabarit centré sur sa trame la plus forte (`note_patch`) et le centre-réduit ; 3. `template_mean` : corrélation des fenêtres testées avec la moyenne de ces gabarits ; 4. `template_max` : meilleure corrélation avec `n_exemplars` gabarits tirés au hasard sans remise (tous s'il y en a moins).
- Détails : un fold sans aucun gabarit d'entraînement laisse ses fenêtres testées à `nan`. Le générateur aléatoire est créé une fois avant la boucle des folds.
- Pourquoi (docstring) : les gabarits sont appris sur les seuls positifs annotés du fold d'entraînement, jamais sur le micro tenu à l'écart.

#### `note_patch(spec_db, width)` — [baselines.py:183](baselines.py#L183)

- Quand : appelée par `oof_template_scores`.
- Reçoit : le spectrogramme d'une fenêtre et une largeur en trames. Renvoie : un gabarit aplati (fréquences × `width`), ou `None` si la fenêtre a moins de `width` trames.
- Fait : lisse le profil d'énergie dans la bande de la note, prend la trame la plus forte, et découpe `width` trames centrées dessus (sans sortir de la fenêtre).
- `_band_profile(spec_db)` ([baselines.py:171](baselines.py#L171)) : l'énergie moyenne par trame entre 4 400 et 5 500 Hz ; cette bande est écrite en dur ici, et les fréquences des lignes sont reconstituées par un découpage régulier de 3 000 à 7 000 Hz.

#### `template_scores(specs, templates, width)` — [baselines.py:192](baselines.py#L192)

- Quand : appelée par `oof_template_scores`, deux fois par fold.
- Reçoit : des spectrogrammes, un ou plusieurs gabarits aplatis, la largeur. Renvoie : un score par fenêtre, `nan` si la fenêtre est plus courte que le gabarit.
- Fait : fait glisser le gabarit trame par trame sur la fenêtre ; à chaque position, centre-réduit le morceau et le gabarit et calcule leur produit scalaire (corrélation normalisée, entre −1 et 1) ; garde le maximum sur les positions et sur les gabarits.
- `_znorm(patches)` ([baselines.py:177](baselines.py#L177)) : retire la moyenne de chaque gabarit et le divise par sa norme.

#### `write_baseline_report(table, scores, reports_dir, stem="baselines")` — [baselines.py:327](baselines.py#L327)

- Quand : appelée par [`cli.baselines`](../cli.py#L753) juste après `run_baselines`, avec `paths.reports`.
- Écrit : `<stem>.csv` (tout le tableau), `<stem>_scores.csv` (un score par fenêtre, baseline et canal) et `<stem>.md` (le critère du go/no-go, puis un tableau par niveau avec les colonnes de `REPORT_COLUMNS`, [baselines.py:312](baselines.py#L312)). Crée le dossier s'il manque. Renvoie : les trois chemins.

## `cluster.py` : la voie non supervisée

Regroupe des fenêtres par ressemblance de leurs embeddings, sans label, et dit ce que valent ces groupes. Ne lit et n'écrit rien lui-même : il reçoit des tableaux et rend des tableaux ; c'est `service.py` qui lit le stock et `cli.py` qui écrit les CSV. Utilisé par `blanci cluster` ([`service.run_clustering`](../service.py#L1168)), `blanci pca` ([`service.run_pca_curve`](../service.py#L1253)), `blanci select --method cluster` et le poste d'annotation ([`selection.cluster_candidates`](../annotation/selection.py#L260)), `blanci cluster-label` (`NOISE`). Tests : [`tests/heads/test_cluster.py`](../../tests/heads/test_cluster.py) ; les commandes `cluster` et `pca` sont lancées dans `tests/interfaces/test_cli_pipeline.py`.

#### `cluster_embeddings(X, n_components=50, min_cluster_size=15, min_samples=5, seed=0)` et `NOISE` — [cluster.py:30](cluster.py#L30)

- Quand : appelée par [`service.run_clustering`](../service.py#L1168) (modes `c0` et `c1`) et [`selection.cluster_candidates`](../annotation/selection.py#L260), tous deux avec les réglages de la section `cluster` (`pca_components` 50, `min_cluster_size` 15, `min_samples` 5, `seed` 0 dans `config/default.yaml`).
- Reçoit : les embeddings des fenêtres. Renvoie : le numéro de groupe de chaque fenêtre, −1 pour celles que HDBSCAN laisse hors de tout groupe.
- Fait : 1. ramène chaque embedding à la norme 1 (`embedding.index.l2_normalize`) ; 2. réduit par ACP à `n_components` dimensions (au plus la dimension de l'embedding et le nombre de fenêtres moins une) ; 3. lance HDBSCAN de scikit-learn sur le résultat.
- `NOISE` ([cluster.py:27](cluster.py#L27)) = −1, le « bruit » de HDBSCAN ; lu aussi par `selection.cluster_candidates` (aucun candidat tiré dans le bruit) et [`selection.label_cluster`](../annotation/selection.py#L623) (refus d'étiqueter le bruit en bloc).

#### `cluster_table(labels, mics, y=None, flagged=None)` — [cluster.py:54](cluster.py#L54)

- Quand : appelée par [`service.run_clustering`](../service.py#L1168) (le tableau écrit dans `cluster_<mode>_<encodeur>.csv`) et par `c1_verdict`.
- Reçoit : le groupe et le micro de chaque fenêtre, et au besoin les labels 0/1 et un drapeau « enregistrement signalé ». Renvoie : un tableau, une ligne par groupe (le bruit −1 compris).
- Fait : pour chaque groupe : taille `n`, micro dominant et sa part, nombre de micros ; avec `y` : `n_pos`, `recall` (part des positifs de l'échantillon qui sont dans le groupe), `precision` (part de positifs dans le groupe), `enrichment` (`precision` ÷ part de positifs dans tout l'échantillon) ; avec `flagged` : `flagged_share`. Trié par `n_pos` décroissant s'il y a des labels, par taille sinon.

#### `c0_summary(labels, mics, sites=None)` — [cluster.py:96](cluster.py#L96)

- Quand : appelée par [`service.run_clustering`](../service.py#L1168) en mode `c0` (avec les sites) et par `c1_verdict` (sans).
- Renvoie : `n_windows`, `n_clusters` (hors bruit), `noise_share`, `ami_mic`, et `ami_site` s'il y a plus d'un site. Pas de docstring.
- `ami(labels, groups)` ([cluster.py:45](cluster.py#L45)) : information mutuelle ajustée (scikit-learn) entre les groupes trouvés et une partition connue (micros ou sites), bruit exclu ; `nan` s'il reste moins de deux fenêtres ou moins de deux groupes. Pourquoi (docstring du fichier) : une AMI élevée dit que l'embedding encode surtout le paysage sonore du point.

#### `c1_verdict(labels, y, mics, min_recall=0.5, min_enrichment=20.0, max_ami_mic=0.3)` — [cluster.py:185](cluster.py#L185)

- Quand : appelée par [`service.run_clustering`](../service.py#L1168) en mode `c1`, avec `cluster.c1_min_recall`, `c1_min_enrichment` et `c1_max_ami_mic` (0,5 ; 20 ; 0,3 dans `config/default.yaml`).
- Reçoit : les groupes, les labels, les micros, les trois seuils. Renvoie : le résumé de `c0_summary` plus `best_cluster`, `best_recall`, `best_enrichment`, `passed`.
- Fait : 1. dresse `cluster_table` et retire le bruit ; 2. choisit le groupe au meilleur rappel (puis au meilleur enrichissement) ; sans groupe ou sans positif dans un groupe, le rappel vaut 0 ; 3. C1 est réussi si le rappel et l'enrichissement atteignent leurs seuils et si l'AMI groupe/micro ne dépasse pas le sien (une AMI indéfinie ne fait pas échouer).
- Pourquoi (docstring du fichier) : la note occupe 2–3 % d'une fenêtre, les groupes ressemblent d'abord à des paysages sonores ; les seuils sont ceux du §5 bis de la feuille de route, « (jugement) ».

#### `variance_partition(X, sites, mics)` — [cluster.py:111](cluster.py#L111)

- Quand : appelée par [`service.run_clustering`](../service.py#L1168) en mode `c0` seulement ; `cli.cluster` affiche alors la ligne « variance des embeddings ».
- Reçoit : les embeddings, le site et le point (site/micro) de chaque fenêtre. Renvoie : `share_site`, `share_mic`, `share_within` (les trois font 1), `n_sites`, `n_mics`.
- Fait : ramène les embeddings à la norme 1, puis décompose la somme des carrés des écarts à la moyenne générale : écart des moyennes de site à la moyenne générale (site), écart des moyennes de micro à celle de leur site (micro dans son site), le reste (dans un micro : heure, météo, chant, bruit).
- Détail : si la variance totale est nulle, les trois parts valent `nan` et les deux comptes ne sont pas rendus.
- Pourquoi (docstring) : question de Léonard du 27/09 : les micros d'un même site diffèrent-ils autant que deux sites ?

#### `pca_information_curve(X, mics=None)` — [cluster.py:142](cluster.py#L142)

- Quand : appelée par [`service.run_pca_curve`](../service.py#L1253) (`blanci pca`).
- Reçoit : les embeddings et, au besoin, le micro de chaque fenêtre. Renvoie : un tableau, une ligne par nombre de composantes k de 1 à la dimension : `components`, `lost_raw` (% de variance perdue en ne gardant que les k premières composantes) et, avec `mics`, `lost_centred` (la même chose après avoir retiré à chaque fenêtre la moyenne de son micro) ; `attrs["mic_share"]` donne la part de la variance brute que ce centrage retire.
- Fait : pour chaque version (brute, centrée par micro), centre les colonnes, calcule les valeurs propres de la matrice des produits croisés, et cumule. Chaque courbe est en % de sa propre variance totale.
- Détail : ici les embeddings ne sont pas ramenés à la norme 1, à la différence de `cluster_embeddings` et de `variance_partition`.
- Pourquoi (docstring) : dire ce que jette l'ACP de R18 et du clustering ; la variance n'est pas l'information utile, une direction de faible variance peut porter le chant.
- `lost(A, name)` ([cluster.py:159](cluster.py#L159)), fonction interne : la courbe d'une matrice ; rend des zéros si la variance totale est nulle.

#### `components_for(curve, column, kept)` — [cluster.py:179](cluster.py#L179)

- Quand : appelée par [`service.run_pca_curve`](../service.py#L1253), pour chaque part de variance de `PCA_KEPT` (80, 90, 95, 99 %).
- Renvoie : le plus petit nombre de composantes qui garde au moins `kept` % de la variance ; le nombre total si aucune ligne n'y suffit.

## `finetune.py` : emplacement réservé

Un emplacement, pas du code qui tourne : la docstring décrit ce que devra faire l'adaptation de l'encodeur lui-même (fine-tuning complet ou LoRA) et dit en toutes lettres « EN ATTENTE — emplacement gardé pour plus tard, volontairement non branché ; ne pas signaler comme code mort à l'audit ». Ne lit et n'écrit rien. Aucune commande ne l'appelle. Test : un test de [`tests/heads/test_baselines.py`](../../tests/heads/test_baselines.py) vérifie que l'emplacement répond « pas encore programmé ».

#### `finetune_encoder(con, cfg, encoder, method="lora", fold=None)` — [finetune.py:52](finetune.py#L52)

- Quand : aucun appelant hors tests.
- Fait : refuse une méthode hors de `METHODS` (`ValueError` « méthode inconnue »), puis lève toujours `NotImplementedError` avec le texte de `PLANNED`. Les paramètres `con`, `cfg`, `encoder` et `fold` ne sont pas lus.
- Pourquoi (docstring du fichier, DECISIONS n° 97, 121, 123) : le contrat prévu est de rendre un encodeur exporté en paquet ONNX sous un nouveau nom (`<encodeur>-ft<version>`), qui se brancherait partout comme les autres ; un encodeur adapté sur tous les labels ne peut pas être jugé sur ces mêmes labels, d'où l'adaptation fold par fold (`finetune.per_fold`) ou le jugement sur le jeu gelé et de nouveaux sites. Les réglages sont réservés dans la section `finetune` de la configuration.
- `PLANNED` ([finetune.py:48](finetune.py#L48)) : le message de l'erreur ; `METHODS` ([finetune.py:49](finetune.py#L49)) = `lora`, `full`.

## `__init__.py`

Une docstring qui présente le dossier (« Étape 3 — les têtes de détection : embedding (ou signal) → score ») et liste ses modules en deux groupes, sur embeddings et sans encodeur. Aucun code, aucun import : on écrit toujours `from blanci.heads.cluster import …`. Les treize entrées de la liste correspondent aux fichiers et sous-dossiers présents dans `blanci/heads/`.

## Écarts relevés

Signalés, pas corrigés.

1. **Docstrings de `head.py` en retard sur la liste des têtes.** La docstring du fichier et celle de `oof_scores` énumèrent les têtes sans `dann`, `multiclass`, `proto_probe` ni `loss:<nom>`, que `METHODS` et `fit_and_score` connaissent. La docstring de `select_C` ne cite comme ajustement que `fit_logistic` et `fit_logistic_to_prototype`, alors que les têtes `loss:<nom>` y passent aussi. Relevé en passant : le tableau en tête de `evaluation/head_benchmark.py` omet lui aussi `dann`, `multiclass` et `proto_probe`.
2. **`proto_probe` n'est pas dans `TRAINED_METHODS`.** Le commentaire de `TRAINED_METHODS` décrit les « têtes entraînées sans C mais dont l'échelle dépend de l'entraînement du fold (recalibrées) ». `proto_probe` est entraînée dans chaque fold, sans C, mais n'y figure pas : avec `fold_calibration: platt`, `calibrated_fold_scores` la traite comme une tête par similarité et ne la recalibre pas. Oubli ou choix : le code ne le dit pas.
3. **`calibrated_fold_scores` et la tête `multiclass`.** La docstring dit que « le C du fold est choisi une fois ». Pour `multiclass`, `choose_C` rend `None` (la tête n'est pas dans `FITTERS`) : son C est rechoisi par `_choose_multiclass_C` à chaque appel de `fit_and_score`, donc aussi dans chaque fold interne. La docstring liste les réglages rechoisis dans les folds internes (époques, weight decay, σ) sans citer ce C.
4. **`train_head` n'utilise pas `head.n_splits`.** Elle appelle `select_C` sans `n_splits`, donc avec 5 folds internes, alors que dans la même `train_and_register`, `oof_scores` reçoit `head_cfg["n_splits"]`. Sans effet aujourd'hui (`config/default.yaml` : `n_splits: 5`, et aucun autre YAML présent ne le change) ; différent si la configuration passe à `lomo` ou à une autre valeur.
5. **« Seul type accepté en entrée de la fusion ».** La docstring de `OOFScores` le dit. Le contrôle de type n'existe que dans `fusion._check`, appelée par `fit_fusion` et `fusion_oof` ; `fusion_oof` n'a pas d'appelant dans `blanci/` hors de son fichier. La fusion utilisée par `stacking._fused` (`fusion_model_oof`) reçoit une matrice numpy, sans contrôle de type.
6. **Méthodes sans appelant hors tests.** `Head.predict_proba`, `MulticlassHead.proba` et `GatedHead.gates` ne sont appelées que par les tests. La branche `logistic` de `losses.loss_and_grad` aussi.
7. **Fonction privée importée ailleurs.** `combination/stacking.py` importe `_choose_C` de `head.py`.
8. **Paramètres sans effet.** `patience` de `fit_proto_probe` (aucune validation n'est passée à `optimise`) ; `seed` de `fit_logistic_to_prototype` (jamais lu) ; `glmm_grid` de `fit_loss` (accepté, ignoré, dit par son commentaire).
9. **Valeurs par défaut différentes pour `monitor`.** `fit_gated` : `"loss"` ; `fit_with_options` et `regularization.R42` dans `config/default.yaml` : `"ap"`. Sans effet par le benchmark : `fit_with_options` passe toujours `monitor` à `fit_gated`.
10. **`gated.py` : docstring et commentaires ne donnent pas les mêmes formes.** La docstring du fichier écrit « A (r × d) et B (d × r) » ; les commentaires de `GatedHead` écrivent `down` « A : (d, r) » et `up` « B : (r, d) », ce que fait le code (`x @ down @ up`). Même calcul, convention transposée.
11. **`token_grid` et `notes.md`.** La docstring de `pooling.py` renvoie à `encoders.models.<nom>.token_grid` et à `notes.md`. Aucun des YAML présents (`config/default.yaml`, `config/local.example.yaml`, le `config/local.yaml` de ce poste, `anuraset/anuraset.yaml`) ne contient `token_grid`, et aucun fichier `notes.md` n'existe dans le dossier du projet : l'appel à `as_grid` dans `load_tokens` n'est déclenché par aucune configuration présente. Les encodeurs du projet rendent déjà leurs jetons en grille (`BacpipeEncoder._model_tokens`, les jetons spatiaux de perch_v2, `AvexEncoder._grid`).
12. **`dann` sans régularisateur.** `fit_and_score("dann", …)` sans `regularizer` échoue sur un `AttributeError` de Python, là où `multiclass` lève un message explicite.
13. **Tests de `pooling.py` rangés ailleurs.** Pas de `tests/heads/test_pooling.py` : les tests sont dans `tests/evaluation/test_head_benchmark.py` et `tests/heads/test_regularization.py`. De même, `exemplar_scores`, `medoid_scores` et `cascade_scores` sont testées dans `tests/evaluation/test_head_benchmark.py`, pas dans `test_head.py`.
14. **Têtes torch sans sauvegarde.** `GatedHead`, `ProtoProbeHead` et `MulticlassHead` n'ont ni `save` ni `load` : seule la tête logistique (`Head`) peut être enregistrée par `blanci train`. Conforme au commentaire de `config/default.yaml` (« train, benchmark et fusion gardent la logistique L2 tant que rien n'est retenu »), signalé pour mémoire.
15. **`compute_onsets` ne cherche pas l'audio comme `embed`.** Elle lit `paths.raw/<chemin de la base>` sans `core/locate.variants` ni `locate` (`signal_processing.py`, ligne 303) : elle ne voit ni les dossiers renommés (`audio.renamed`) ni un autre disque, et compte ces fichiers comme « illisibles ». `baselines.read_windows` (`baselines.py`, ligne 117) fait de même, donc aussi `blanci baselines`, `blanci upstream-bench` et `blanci detector-bench` ; là, un fichier absent arrête la commande. Déjà relevé dans la fiche de `core/` (n° 4) pour `onsets` et `tokens`.
16. **Canal des débuts de notes.** `compute_onsets` lit toujours `audio.channel`. `embed.embed_recordings` lit, lui, le canal de `embed.recording_channel` (la piste 1 quand la piste 0 est étouffée). Pour ces enregistrements, les débuts de notes relus de la table par les portes de `embed` viennent d'une autre piste que celle encodée.
17. **Table `onsets` : une ligne par enregistrement, jamais recalculée.** La clé est `recording_id` ; `compute_onsets` saute tout enregistrement qui a une ligne, sans regarder le canal rangé ni les réglages `signal` (qui ne sont pas rangés). Changer `signal.onset_k_mad`, `band_hz`, `note_dur_s` ou `audio.channel` ne recalcule rien, et `cli.onsets` n'a pas d'option pour forcer. À côté, `embed.stock_identity` range bien les réglages `signal` d'un stock à portes (`gate_signal`), mais les débuts de notes relus de la table peuvent dater d'autres réglages.
18. **Docstring de `detect_onsets`.** Elle dit « Sert au traitement du signal, jamais de filtre amont (bande saturée, §3) ». Les portes `notes` et `rhythm` (`GATES`, commentaire de la section « En amont ») arrêtent pourtant des fenêtres avant l'encodeur à partir de ces mêmes débuts de notes.
19. **Fonctions publiques sans appelant hors tests.** `signal_processing.signal_features`, `signal_processing.note_snr_db` (sa docstring dit « Sert à ventiler le rappel par RSB », aucun code ne le fait), `AttentiveHead.save` ; `AttentiveHead.attention` et `AttentiveHead.load` ne servent qu'au notebook 04 ; `finetune.finetune_encoder` (emplacement réservé, annoncé comme tel). `compute_onsets` n'a pas de test dans `tests/heads/` : elle n'est exercée que par la commande dans `tests/interfaces/test_cli_pipeline.py`.
20. **Deux fonctions `signal_features`.** `signal_processing.signal_features` (inutilisée) et `combination/stacking.signal_features` (celle de la fusion) portent le même nom et ne font pas la même chose.
21. **Docstring de `recording_persistence`.** Elle annonce quatre descripteurs (« fraction positive, plus longue série, isolées, étendue ») ; la fonction en rend six (`PERSISTENCE_COLUMNS` : aussi `n_positive` et `n_gaps`).
22. **Aucune tête attentive ni DANN n'est enregistrée.** La docstring d'`attentive.py` parle de « sauvegarde JSON + npz, sans pickle (§7) », mais aucune commande n'appelle `AttentiveHead.save` : `blanci train` enregistre une tête logistique (`head.train_head`). `DannHead` n'a ni `save` ni `load`. Ces deux têtes n'existent que dans les benchmarks.
23. **Régularisations de l'attentive : trois listes.** Le tableau de la docstring d'`attentive.py` va de R40 à R47 ; la docstring de `fit_attentive` ajoute R59 et R64 ; `regularization/names.py` (`TORCH_REGULARIZATIONS`) accepte R40, R41, R42, R43, R45, R46, R47, R59, R64 ; le commentaire de `config/default.yaml` (ligne 260) cite « attentive (R40–R42, R45–R47, R59) », sans R43 ni R64.
24. **Baselines « jugées exactement comme les encodeurs ».** La docstring de `baselines.py` le dit (« mêmes métriques »). `run_baselines` appelle `evaluate` sans `clusters` ni `folds` ([baselines.py:291](baselines.py#L291)), alors que `evaluation/benchmark.probe_table` passe `clusters` (le micro) et `folds` pour les encodeurs : l'intervalle de l'AP des baselines est tiré par enregistrement et non par micro, il n'y a pas d'`ap_fold_mean`, le rappel à précision plancher n'est pas jugé avec un seuil pris sur les autres folds, et les scores ne sont pas recalibrés par fold. La grille des baselines est toujours `w3` (3 s), alors que les têtes sont jugées sur les fenêtres du stock de leur encodeur (5 s pour perch_v2) : « mêmes fenêtres » ne vaut que face à un stock en fenêtres de 3 s.
25. **Bande de la note écrite en dur pour les gabarits.** `_band_profile` ([baselines.py:171](baselines.py#L171)) utilise 4 400–5 500 Hz, alors que `window_features` lit `signal.band_hz` ; les deux coïncident avec `config/default.yaml`, plus si on change la configuration. Les fréquences y sont reconstituées par `np.linspace(3000, 7000, n)`, pas reprises du spectrogramme (écart de quelques dizaines de Hz au plus). `FLANKS_HZ` et `SPEC_BAND_HZ` sont aussi des constantes.
26. **Canal par défaut des baselines.** Avec `audio.channel: mean`, `cli.baselines` prend le canal 0, alors que `service.window_gate_values` et `detectors/base.read_evaluation_audio` moyennent les canaux 0 et 1. Sans effet avec la configuration par défaut (`channel: 0`).
27. **`pca_information_curve` et l'ACP du clustering.** Sa docstring dit que le tableau montre ce que jette « l'ACP de R18 et du clustering ». `cluster_embeddings` normalise les embeddings à la norme 1 avant son ACP, la courbe ne le fait pas : elle ne décrit pas exactement l'ACP du clustering.
28. **`variance_partition` à variance nulle.** Elle rend alors trois `nan` sans `n_sites` ni `n_mics` ; `cli.cluster` lit `summary['n_sites']` dès que `share_site` est présent, et lèverait `KeyError`. Cas limite (tous les embeddings identiques).
29. **Drapeaux « signalés » du clustering.** La docstring de `cluster.py` parle de groupes dominés par un enregistrement signalé « (saturation, son étouffé) », et `service._qc_flagged` cherche les drapeaux `saturation`, `rain`, `muffled`. D'après la fiche de `core/` (écart 3) et un Grep sur `saturation` dans `blanci/`, aucun code n'écrit de drapeau `saturation` : `flagged_share` ne reflète en pratique que le son étouffé.
30. **README, ligne « Traitement du signal ».** `README.md` (ligne 53) dit que l'étape « filtre la bande 4,4–5,5 kHz, repère les fenêtres candidates et mesure le rythme », commande `onsets`. `blanci onsets` ne fait que calculer et ranger les débuts de notes ; rien n'y repère de fenêtre candidate. Les portes qui arrêtent des fenêtres agissent dans `blanci embed` et sont toutes coupées dans `config/default.yaml` (`enabled: false`, `position: [parallel, downstream]`) ; le passe-bande en amont est réglé sur 3–7 kHz, pas 4,4–5,5.
31. **Renvois à des paragraphes « § ».** Les docstrings renvoient à « §3 », « §4 », « §5 », « §5 bis », « §6 », « §7 », « §9 », « §13.7 » : ce ne sont pas des numéros de décision. Le titre « 5 bis. Voie non supervisée : clustering » se trouve dans `documentation/old/feuille-de-route-V4.md` (et V2, V3), un document rangé dans `old/` ; même constat que l'écart 7 de la fiche de `core/`.
32. **Relevé en passant, hors de ce dossier.** Le message d'erreur de `combination/stacking.build_level1` (`combination/stacking.py`, ligne 196) dit que les débuts de notes « se calculent pendant `blanci embed`, ou avec `blanci onsets` » ; la docstring de `cli.onsets` et celle de `embedding/embed.py` disent que `embed` ne les calcule plus (il ne les calcule que pour ses portes, sans les ranger). `service.window_gate_values` garde les valeurs d'énergie et de contraste dans `gate_values.parquet` par `window_id` seul, sans les réglages `signal` ni le canal.
