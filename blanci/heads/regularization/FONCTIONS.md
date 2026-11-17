# `blanci/heads/regularization/` : le code fonction par fonction

Fiche vérifiée le 10/11/2026 sur le commit `9473508` du dépôt public (`git log -1 --format=%h`), arbre de travail sans modification dans ce dossier. Pour savoir si elle est périmée : `git diff --stat 9473508 -- blanci/heads/regularization` ; une sortie vide veut dire que le code de ce dossier n'a pas bougé.

`regularization/` : 11 fichiers, 2 463 lignes. C'est la boîte à outils des régularisations des têtes (numéros R du catalogue `documentation/regularizations/regularizations.md`). Le dossier ne lit ni n'écrit la base ni aucun fichier, à deux exceptions près, en lecture seule : `store_domain_statistics` et `sample_pool` parcourent le stock d'embeddings et la table `recordings`. Une régularisation est un moyen de brider une tête pour qu'elle apprenne le chant et non les particularités de ses micros d'entraînement. Le reste de l'étape 3 est décrit dans la fiche du dossier parent : [`../FONCTIONS.md`](../FONCTIONS.md).

| Fichier | Lignes | Rôle |
|---|---|---|
| [`__init__.py`](__init__.py) | 333 | index (docstring de 125 lignes) et réexport de 91 noms |
| [`names.py`](names.py) | 179 | lire un nom de tête `logistic+R18=16+R13`, le mettre en forme canonique, refuser les combinaisons sans objet |
| [`assembly.py`](assembly.py) | 301 | `Context` (ce qu'on sait des fenêtres), `Regularizer` (application dans l'ordre), `regularizer_for` |
| [`windows.py`](windows.py) | 224 | statistiques du stock par micro (R19, R20), directions du micro (R21), poids des exemples (R13, R15, R36), indicatrices de biais (R37) |
| [`selection.py`](selection.py) | 307 | choix d'un réglage par validation groupée (R26, R40, R50, R75, R76), R74, bagging (R79), pseudo-étiquetage (R81) |
| [`glmm.py`](glmm.py) | 109 | R37 : échelle des biais de micro, σ fixé ou estimé |
| [`torch_training.py`](torch_training.py) | 441 | boucle d'entraînement commune des têtes torch et leurs options (R40 à R47, R59, R61 à R64, R66) |
| [`fusion_logistic.py`](fusion_logistic.py) | 385 | fusion logistique régularisée (R50, R52 à R56) |
| [`decision.py`](decision.py) | 106 | R73 : courbe d'exploitation, seuil à la prévalence du stock, Platt ; recalibration par fold |
| [`classes.py`](classes.py) | 55 | R67 : classe de son de chaque fenêtre |
| [`neighbors.py`](neighbors.py) | 23 | R39 : similarité aux k plus proches références |

## La chaîne

Lecture : `a → b` veut dire « `a` appelle `b` » ; un point-virgule sépare les appels successifs. Les noms sans préfixe sont dans `regularization/`.

- **Toute commande `blanci …`** : `cli.main` → `configure`.
- **`blanci train`** : `cli.train` → `service.train_and_register` → `head.oof_scores` → `head.fit_and_score` → `head._choose_C` → `head.select_C` → `grouped_search` → `usable_folds`, `pick` ; `head.fit_logistic` → `bias_sigmas`, `group_bias_scale`. Puis `threshold_at_prevalence` → `operating_curve`, `precision_at_prevalence` (si `decision.prevalence`) ; `cross_fitted_threshold` → `evaluate.cross_fitted_recall` ; `fold_ids` ; `head.train_head` → `head.select_C` → `grouped_search` ; `platt` ; `service.write_regularization_path` ; `service.write_operating_curve` → `operating_curve`, `precision_at_prevalence`.
- **`blanci retrain`** : `cli.retrain` → `service.retrain` → `train_and_register` (voir `train`).
- **`blanci evaluate`** : `cli.evaluate` → `service.evaluate_holdout` → `head.oof_scores`, `head.train_head` → `grouped_search`.
- **`blanci heads`** : `cli.heads` → `head_benchmark.run_head_benchmark` → `regularization_context` → `window_classes` ; `needs_domain` ; `store_domain_statistics` ; `DomainStats.rows` ; `needs_pool` ; `sample_pool`. Pour chaque tête : `regularizer_for` → `parse_head`, `validate`, `head_name` ; `head.oof_scores` → `head.fit_and_score` → `head._prepared` → `Regularizer.prepare` → `window_transform` ; `fit_projection` → `mean_directions` ou `nuisance_directions` ; `pool_windows` ; `bias_indicators` → `group_indicators`, `site_of` ; `sample_weights` ; `fit_options` → `bias_scales`. Puis `Regularizer.grid` → `fine_grid` ; et selon la tête : `with_pseudo_labels` → `pseudo_positives` ; `bagged` → `bootstrap_weights` ; `head.fit_logistic` → `glmm_scales` → `glmm_evidence` ; `head.fit_multiclass` → `merge_rare_classes` ; `fit_with_options` → `grouped_search`, `usable_folds`, `fit_gated` / `fit_dann` / `fit_attentive` → `import_torch`, `feature_scaling`, `balanced_bce`, `validation_criterion`, `optimise` → `WeightAverage`. Avec `benchmark.fold_calibration: platt` : `head.calibrated_fold_scores` → `head.choose_C`, `usable_folds`, `fold_platt`. À la fin : `canonical` ; `selection_estimate` ; `fold_ids`.
- **`blanci heads-curve`** : `cli.heads_curve` → `head_benchmark.annotation_curve` → `regularization_context` ; `regularizer_for` ; `head.choose_C` → `Regularizer.grid`, `Regularizer.prepare` ; `head.fit_and_score` ; `canonical`.
- **`blanci anuraset heads`** : `cli.anuraset_heads` → `evaluation/anuraset.run_anuraset_heads` → `needs_domain` ; `domain_statistics` ; `canonical` ; `DomainStats.rows` ; `needs_pool` ; `regularizer_for` ; `head.oof_scores` ; `head_benchmark.level_rows` → `fold_ids` ; `selection_estimate` ; `fold_ids`. **`blanci anuraset campaign`** : `cli.anuraset_campaign` → `run_anuraset_campaign` → `run_anuraset_heads`.
- **`blanci benchmark`** : `cli.benchmark` → `benchmark.run_benchmark` → `benchmark_encoder` → `probe_table` → `head.oof_scores` ; `fold_ids`. **`blanci anuraset benchmark`** : `run_anuraset_benchmark` → `probe_table`.
- **`blanci fusion`** : `cli.fusion` → `service.train_fusion` → `stacking.build_level1` → `head._choose_C`, `head.fit_logistic` ; `stacking.fused_oof` → `_fused` → `fusion.fusion_model_oof` → `fusion.fit_fusion_model` → `fusion_settings` ; `choose_fusion_C` → `usable_folds`, `grouped_search` ; `fit_fusion_logistic` → `is_descriptor`, `fit_descriptor_basis`, `DescriptorBasis.transform`, `blocks`, `penalty`, `fit_constrained_logistic` ; `fusion._fold_calibration` → `usable_folds`, `fusion_settings`, `fold_platt`. Puis `stacking.final_model` → `fit_fusion_model` ; `FusionModel.to_dict` → `DescriptorBasis.to_dict` ; `FusionModel.weights` → `DescriptorBasis.amplitudes`.
- **`blanci fusion-bench`** : `cli.fusion_bench` → `stacking.fusion_benchmark` → `_fused` (comme `fusion`) ; `final_model(…).weights()`.
- **`blanci ensemble`** : `cli.ensemble` → `ensemble.run_ensemble` → `fusion_variant` → `fusion_settings` ; `fusion.fusion_model_oof`.
- **`blanci score --fusion`** : `cli.score` → `service.score_and_decide` → `fused_scores` → `FusionModel.from_dict` → `DescriptorBasis.from_dict` ; `FusionModel.decision` → `DescriptorBasis.transform`.
- **Scripts** : `anuraset/calcul/global_bench.py` (au niveau du module) → `needs_domain` ; `_Accumulator.add`, `stats` ; `DomainStats.rows` ; `regularizer_for` ; puis sa fonction `score` → `Context.subset` ; `head.choose_C` ; `head.fit_and_score`. `anuraset/benchmarks/2026-09-29_anuraset_birdmae_base/amorcage.py` → `needs_domain`, `domain_statistics`, `needs_pool`, `regularizer_for`, `head.oof_scores`.

## Ce qui est réellement branché

Vérifié en remontant chaque appel jusqu'à `cli.py`. « Suffixe » = régularisation qu'on active dans le nom de la tête (`logistic+R19`).

| Régularisation | Code de ce dossier | Appelée hors tests par | Commandes | Quand elle tourne |
|---|---|---|---|---|
| R26 (C de la L2) et R75 (règle du choix) | `grouped_search`, `usable_folds`, `pick`, `configure` | `head.select_C`, `head._choose_multiclass_C`, `cli.main` | toutes celles qui ajustent une logistique : `train`, `retrain`, `evaluate`, `heads`, `heads-curve`, `benchmark`, `fusion`, `fusion-bench`, `upstream-bench`, `anuraset …` | toujours |
| R73 (seuil, Platt) | `operating_curve`, `platt` ; `threshold_at_prevalence`, `precision_at_prevalence` | `service.train_and_register`, `service.write_operating_curve` | `train`, `retrain` | courbe et Platt : toujours ; seuil corrigé et précision attendue : seulement si `decision.prevalence` est renseignée (`null` dans `config/default.yaml`) |
| R74 (seuil jugé à part) | `cross_fitted_threshold`, `fold_ids` | `service.train_and_register` | `train`, `retrain` | toujours |
| R74 (variante jugée à part) | `selection_estimate`, `fold_ids` | `head_benchmark.run_head_benchmark`, `evaluation/anuraset.run_anuraset_heads` | `heads`, `anuraset heads`, `anuraset campaign` | dès qu'il y a plus d'une tête |
| R37 sans suffixe (échelles vides) | `bias_sigmas`, `group_bias_scale` | `head.fit_logistic`, `losses.fit_loss` | toutes celles qui ajustent une logistique | à chaque ajustement, sans effet s'il n'y a pas de colonne de biais |
| Suffixes R13, R15, R17, R18, R19, R20, R21, R27, R28, R36, R37, R76, R79, R81 | `regularizer_for`, `Regularizer`, `windows.py`, `selection.py`, `glmm.py` | `head_benchmark.run_head_benchmark` et `annotation_curve`, `evaluation/anuraset.run_anuraset_heads`, `anuraset/calcul/global_bench.py`, `anuraset/benchmarks/2026-09-29_anuraset_birdmae_base/amorcage.py` | `heads`, `heads-curve`, `anuraset heads`, `anuraset campaign` | seulement si le nom de la tête porte le suffixe (`--methods`, `regularization.variants` vide par défaut ; la liste `HEADS` de `evaluation/anuraset.py` en contient). Jamais dans `train`, `score`, `benchmark` ni `fusion` : `train_and_register` appelle `oof_scores` et `train_head` sans `regularizer` |
| Suffixes torch R40, R41, R42, R43, R45, R46, R47, R59, R64 | `Regularizer.torch_options`, `fit_with_options`, `optimise` et ses aides | `head.fit_and_score` (têtes `attentive`, `gated`, `dann`) | `heads`, `heads-curve`, `anuraset heads` | options seulement avec le suffixe ; `fit_with_options` et `optimise` tournent pour toute tête torch, même sans suffixe (dont `attentive` dans `blanci benchmark` s'il y a des jetons) |
| R66 (tête `dann`), R85 (tête `gated`), sonde `proto_probe`, `attentive` | `import_torch`, `feature_scaling`, `balanced_bce`, `optimise`, `validation_criterion`, `dropout`, `grad_reverse`, `dann_strength`, `keep_mask`, `attention_entropy`, `logistic_start`, `l2_sp_penalty` | `dann.fit_dann`, `gated.fit_gated`, `proto_probe.fit_proto_probe`, `attentive.fit_attentive` | `heads` (têtes par défaut si torch est installé), `benchmark` (attentive) | à chaque entraînement de ces têtes |
| R67 (tête `multiclass`) | `window_classes`, `merge_rare_classes` | `head_benchmark.regularization_context`, `head.fit_multiclass` | `heads`, `heads-curve` | `window_classes` à chaque benchmark des têtes ; `merge_rare_classes` quand la tête `multiclass` est ajustée |
| R39 (k voisins) | `nearest_similarity` | `head.exemplar_scores`, `head.knn_scores` | `heads`, `heads-curve`, `benchmark`, `anuraset …` | à chaque tête `knn`, `exemplar` et leurs variantes |
| R50, R52 à R56 (fusion) | `fusion_settings`, `fusion_variant`, `choose_fusion_C`, `fit_fusion_logistic`, `DescriptorBasis` | `fusion.fit_fusion_model`, `fusion._fold_calibration`, `ensemble.run_ensemble`, `fusion.FusionModel` | `fusion-bench` (variantes dans `fusion.benchmark_methods` par défaut), `fusion` et `ensemble` si la méthode porte un suffixe (`fusion.method` vaut `logistic` par défaut), `score --fusion` si la fusion enregistrée a une base | `fusion_settings` à chaque fusion ; le reste seulement avec un suffixe |
| Recalibration par fold | `fold_platt`, `usable_folds` | `head.calibrated_fold_scores`, `fusion._fold_calibration` | `heads`, `benchmark`, `fusion`, `fusion-bench`, `anuraset …` | seulement si `benchmark.fold_calibration: platt` (`none` par défaut) |
| R61, R62 complet, R63 | `layerwise_lr_groups`, `unfreezing_schedule`, `unfreeze_top`, `snapshot`, `l2_sp_model_penalty`, `distillation_loss` | personne (tests seulement) | aucune | jamais ; le code les marque « EN ATTENTE », volontairement non branchées |
| — | `prior_shift`, `DescriptorBasis.curve` | personne (tests seulement) | aucune | jamais |
| — | `DESCRIPTIONS` | personne, ni code ni tests | aucune | jamais |

## `__init__.py` : l'index et les réexports

Une docstring de 125 lignes (l'index des régularisations programmées, par sous-module, par ordre d'application, par tête torch, et celles qui vivent ailleurs), puis l'import de 91 noms des dix sous-modules et la liste `__all__` ([\_\_init\_\_.py:241](__init__.py#L241)) des mêmes 91 noms. Aucun autre code. Tout le reste du projet importe d'ici (`from blanci.heads.regularization import …`), sauf `heads/proto_probe.py`, qui importe de `regularization.torch_training`, et `anuraset/calcul/global_bench.py`, qui prend en plus la classe privée `_Accumulator` dans `regularization.windows`. Tests : [`tests/heads/test_regularization.py`](../../../tests/heads/test_regularization.py), [`tests/heads/test_new_heads.py`](../../../tests/heads/test_new_heads.py), et pour une partie `tests/heads/test_attentive.py`, `tests/heads/test_head.py`, `tests/heads/test_losses.py`, `tests/combination/test_stacking.py`.

- Ce que dit la docstring, en bref : tout est coupé par défaut ; une tête du benchmark active une régularisation dans son nom ; « =v » remplace le réglage principal de la section `regularization` de la configuration ; le nom canonique (régularisations triées par numéro) est celui des rapports et des scores hors-fold.
- Ordre d'application annoncé : R19 ou R20, R17, R18, R21, R37, puis les poids (R13, R15, R36) et la pénalité (R27, R28). C'est bien l'ordre de `Regularizer.prepare`.
- Pourquoi un paquet découpé par thème : docstring, DECISIONS n° 134. Les régularisations elles-mêmes : DECISIONS n° 108 ; R37 : n° 116, 119, 131 ; têtes torch : n° 117, 121, 123 ; sélection : n° 122.

## `names.py` : les noms des têtes régularisées

Des chaînes de caractères seulement : ne lit et n'écrit rien. Sert à `assembly.regularizer_for` et, pour `canonical`, `needs_domain` et `needs_pool`, aux benchmarks des têtes. Tests : `tests/heads/test_regularization.py`.

#### `parse_head(spec)` — [names.py:77](names.py#L77)

- Quand : chaque fois qu'un nom de tête est lu : [`regularizer_for`](assembly.py#L290), `canonical`, `needs_domain`, `needs_pool`. Pas d'appelant hors du dossier, sauf les tests.
- Reçoit : un nom comme `logistic:max+R18=16+R13`. Renvoie : `("logistic:max", {13: None, 18: 16.0})` : la tête de base, et un dictionnaire numéro → valeur (`None` sans « =v »).
- Fait : 1. coupe le nom aux `+` ; le premier morceau est la tête ; 2. chaque morceau suivant doit avoir la forme `R<numéro>` ou `R<numéro>=<valeur>` (`_SUFFIX`, [names.py:71](names.py#L71) : la valeur est faite de chiffres, `.`, `e`, `E`, `+`, `-`, ou bien de lettres minuscules et `_`) ; 3. pour R37, R42 et R64 (`WORD_VALUES`, [names.py:70](names.py#L70)), une valeur doit être un des mots prévus (`glmm` ; `ap`, `loss` ; `ema`, `swa`), sinon `ValueError` « R42=auc : valeurs possibles ap, loss » ; 4. pour les autres, la valeur est convertie en nombre ; un morceau illisible lève `ValueError` « régularisation illisible dans … (ex. R18=16) ».
- Détails : un suffixe répété garde sa dernière valeur. Le nom étant coupé aux `+`, une valeur ne peut pas contenir de `+` (`1e+2`). Voir l'écart n° 1 : pour R37, une valeur numérique est refusée par l'étape 3.

#### `head_name(base, regs)` et `canonical(spec)` — [names.py:102](names.py#L102)

- Quand : `head_name` dans [`regularizer_for`](assembly.py#L290) et `canonical`. `canonical` ([names.py:116](names.py#L116)) pour mettre en forme la tête de référence `head.reference` : [`head_benchmark.run_head_benchmark`](../../evaluation/head_benchmark.py#L228), [`head_benchmark.annotation_curve`](../../evaluation/head_benchmark.py#L451), [`evaluation/anuraset.run_anuraset_heads`](../../evaluation/anuraset.py#L341).
- Renvoie : le nom canonique : la tête, puis les régularisations triées par numéro. `R13` sans valeur, `R42=ap` pour un mot, `R18=16` pour un nombre entier (sans décimale), `R28=0.25` sinon.
- `canonical(spec)` = `head_name(*parse_head(spec))` : `logistic+R18=16+R13` devient `logistic+R13+R18=16`.

#### `validate(base, regs)` — [names.py:120](names.py#L120)

- Quand : dans [`regularizer_for`](assembly.py#L290), seul appelant hors tests.
- Fait, dans l'ordre : 1. un numéro absent de `IMPLEMENTED` lève `ValueError` « R99 n'est pas programmée comme suffixe (programmées : … ; R22 = pooling gem, R30 = tête logistic_to_prototype, R31 = tête lda_shrunk) » ; 2. sans suffixe, rien ; 3. tête `multiclass` : seuls R17 à R21 et R76 sont admis, puis la fonction s'arrête là ; 4. tête `attentive`, `gated` ou `dann` : seuls les numéros de `TORCH_REGULARIZATIONS` pour cette tête, puis la fonction s'arrête là ; 5. pour toute autre tête : un numéro réservé aux têtes torch est refusé ; R19 avec R20 refusé (« R20 contient déjà le centrage de R19 ») ; R27 avec R28 refusé ; R19 ou R20 refusé sur `logistic:<pooling>` (les statistiques du stock n'existent que pour l'embedding par défaut) ; R13, R15, R36 réservés à `WEIGHTED_HEADS` ; R76 à `C_HEADS` ; R79 à `BAGGED_HEADS` ; R81 à `PSEUDO_HEADS` ; R79 avec R81 refusé (« pas programmé ensemble ») ; R37 réservé à `GROUP_BIAS_HEADS` ; `R37=glmm` refusé sur `loss:<nom>` ; R27, R28 réservés à `PENALIZED_HEADS`.
- Détail : la « famille » d'une tête est ce qui précède le premier `:` (`loss:focal` → `loss`, `knn:k=5` → `knn`), et `logistic` pour `logistic:<pooling>`.

#### `needs_domain(specs)` et `needs_pool(specs)` — [names.py:174](names.py#L174)

- Quand : avant de construire le contexte des fenêtres, pour ne lire le stock que si une tête le demande : [`head_benchmark.regularization_context`](../../evaluation/head_benchmark.py#L143), [`evaluation/anuraset.run_anuraset_heads`](../../evaluation/anuraset.py#L341), `anuraset/calcul/global_bench.py`, `anuraset/benchmarks/2026-09-29_anuraset_birdmae_base/amorcage.py`.
- Renvoient : vrai si une des têtes de la liste porte R19 ou R20 (`needs_domain`), R81 (`needs_pool`, [names.py:178](names.py#L178)).

#### `IMPLEMENTED` et les autres constantes — [names.py:8](names.py#L8)

- `IMPLEMENTED` ([names.py:8](names.py#L8)) : les 23 numéros admis comme suffixe : 13, 15, 17, 18, 19, 20, 21, 27, 28, 36, 37 ; 40, 41, 42, 43, 45, 46, 47, 59, 64 ; 76, 79, 81. Lue par `validate` seulement.
- `MAIN_PARAMETER` ([names.py:37](names.py#L37)) : pour chaque numéro, le réglage que « =v » remplace : R15 `hard_weight`, R18 `components`, R21 `iterations`, R28 `l1_ratio`, R36 `power`, R37 `scale`, R41 `weight_decay`, R42 `monitor`, R43 `strength`, R45 et R46 `p`, R47 `strength`, R59 `warmup`, R64 `method`, R76 `points`, R79 `bags`, R81 `weight`. Lue par `Regularizer.param`. R13, R17, R19, R20, R27 et R40 n'y sont pas : une valeur donnée à ces suffixes est acceptée, reste dans le nom canonique, et n'est lue nulle part.
- `WEIGHTED_HEADS`, `PENALIZED_HEADS`, `GROUP_BIAS_HEADS`, `C_HEADS`, `PSEUDO_HEADS`, `BAGGED_HEADS` ([names.py:56](names.py#L56)) : les familles de têtes qui acceptent chaque groupe de suffixes ; lues par `validate`. `CONTEXT_HEADS` ([names.py:60](names.py#L60)) = `multiclass`, `dann` : têtes qui ont besoin du contexte même sans suffixe ; lue par `regularizer_for`.
- `TORCH_REGULARIZATIONS` ([names.py:64](names.py#L64)) : `attentive` accepte 40, 41, 42, 43, 45, 46, 47, 59, 64 ; `gated` et `dann` acceptent 40, 42, 46, 59, 64.
- `DESCRIPTIONS` ([names.py:11](names.py#L11)) : une phrase par numéro. Réexportée, mais lue par aucun code ni aucun test (écart n° 6).

## `assembly.py` : appliquer les régularisations dans l'ordre

Le fichier qui relie un nom de tête aux calculs. Ne lit et n'écrit rien : il reçoit les embeddings et le contexte déjà chargés. Utilisé par les benchmarks des têtes, qui passent le `Regularizer` à [`head.oof_scores`](../head.py#L768), [`head.fit_and_score`](../head.py#L619) et [`head.choose_C`](../head.py#L587). Tests : `tests/heads/test_regularization.py`, `tests/heads/test_new_heads.py`, `tests/heads/test_attentive.py`, `tests/heads/test_losses.py`.

#### `regularizer_for(spec, cfg, context)` — [assembly.py:290](assembly.py#L290)

- Quand : une fois par tête d'un benchmark : [`head_benchmark.run_head_benchmark`](../../evaluation/head_benchmark.py#L228) (`blanci heads`), [`head_benchmark.annotation_curve`](../../evaluation/head_benchmark.py#L451) (`blanci heads-curve`), [`evaluation/anuraset.run_anuraset_heads`](../../evaluation/anuraset.py#L341) (`blanci anuraset heads`, `blanci anuraset campaign`), `anuraset/calcul/global_bench.py`, `anuraset/benchmarks/2026-09-29_anuraset_birdmae_base/amorcage.py`. Aucun appel dans `service.py` : `blanci train` n'y passe pas.
- Reçoit : le nom de la tête, la configuration, le contexte des fenêtres. Renvoie : `(nom canonique, tête de base, Regularizer ou None)`.
- Fait : 1. `parse_head` ; 2. `validate` ; 3. `head_name` ; 4. sans suffixe et pour une tête hors `CONTEXT_HEADS`, renvoie `None` à la place du `Regularizer` ; 5. sinon, sans contexte, `ValueError` « … : contexte des fenêtres manquant » ; 6. sinon construit `Regularizer(regs, cfg["regularization"], context)`.
- Détail : les têtes `multiclass` et `dann` reçoivent toujours un `Regularizer`, même sans suffixe : la première y lit les classes des fenêtres, la seconde les micros.

#### `Context` — [assembly.py:32](assembly.py#L32)

- Quand : construit par [`head_benchmark.regularization_context`](../../evaluation/head_benchmark.py#L143), [`evaluation/anuraset.run_anuraset_heads`](../../evaluation/anuraset.py#L341), `anuraset/calcul/global_bench.py`, `amorcage.py`.
- Contient, une valeur par fenêtre évaluée : `groups` (le groupe `regularization.by`, `point` par défaut : site/micro), `hard` (vrai pour un négatif annoté, R15), `domain` et `domain_rows` (statistiques du stock par groupe et indice du groupe de chaque fenêtre, R19 et R20), `classes` (classe du son, R67) ; et pour R81 le réservoir de fenêtres non annotées : `pool`, `pool_groups`, `pool_domain_rows`.
- `subset(rows)` ([assembly.py:45](assembly.py#L45)) : le même contexte réduit aux lignes `rows` ; les statistiques et le réservoir sont gardés entiers. Appelée seulement par `anuraset/calcul/global_bench.py`, qui traite un site par paquets.

#### `Regularizer(regs, params, context)` — [assembly.py:59](assembly.py#L59)

- Les régularisations d'une tête : `regs` (numéro → valeur du suffixe), `params` (la section `regularization` de la configuration), `context`. Deux champs de travail, remplis par `prepare` : `pool_prepared` et `pool_prepared_groups` ([assembly.py:131](assembly.py#L131)), le réservoir de R81 une fois transformé.
- `param(number, key, default)` ([assembly.py:66](assembly.py#L66)) : la valeur d'un réglage. La valeur du suffixe l'emporte si `key` est le réglage principal de ce numéro (`MAIN_PARAMETER`) ; sinon la section `R<numéro>` de la configuration ; sinon `default`. Appelée aussi de l'extérieur par [`head.fit_and_score`](../head.py#L619) pour `R67.min_count`.

#### `Regularizer.prepare(X, y, train, seed=0)` — [assembly.py:256](assembly.py#L256)

- Quand : au début de chaque ajustement d'une tête régularisée, par `head._prepared` ([head.py:580](../head.py#L580)), elle-même appelée par [`head.fit_and_score`](../head.py#L619) et [`head.choose_C`](../head.py#L587). Dans une validation croisée, une fois par fold.
- Reçoit : les embeddings de toutes les fenêtres, les labels, les indices des fenêtres d'entraînement. Renvoie : `(X transformé pour toutes les fenêtres, poids des fenêtres d'entraînement ou None, options de l'ajustement logistique)`.
- Fait : 1. `window_transform` sur toutes les fenêtres (R19 ou R20, puis R17) ; 2. `fit_projection` ajustée sur les seules fenêtres d'entraînement (R18 puis R21), appliquée à toutes ; 3. avec R81, `pool_windows` prépare le réservoir par les mêmes transformations ; 4. avec R37, `bias_indicators` ajoute à droite de X une colonne par micro de l'entraînement (et par site, si deux σ sont réglés et si tous les groupes d'entraînement contiennent un `/`) ; le réservoir reçoit les mêmes colonnes ; 5. `sample_weights` calcule les poids (R13, R15, R36) avec `R15.hard_weight` (3 par défaut) et `R36.power` (0 par défaut) ; 6. `fit_options` rend les options de pénalité et de biais.
- Détail : `prepare` range le réservoir préparé dans l'objet lui-même (`pool_prepared`), que `fit_and_score` relit juste après ; le même `Regularizer` sert à tous les folds, chaque appel remplace le précédent.
- Pourquoi (docstring de `__init__.py`) : R18 et R21 sont ajustées dans chaque fold sur les seules fenêtres d'entraînement ; le C est ensuite choisi sur ces fenêtres transformées.
- `window_transform(X, domain_rows=None)` ([assembly.py:72](assembly.py#L72)) : fenêtre par fenêtre, sans apprentissage. R19 : retire la moyenne du stock du micro ; R20 : divise en plus par son écart-type (+ `R20.eps`, 1e-6) ; R17 : ramène chaque vecteur à la norme 1 (`embedding/index.l2_normalize`). Sans les statistiques du stock : `ValueError` « R19/R20 demandent les statistiques du stock (Context.domain) ».
- `fit_projection(X, y, groups, seed=0)` ([assembly.py:91](assembly.py#L91)) : renvoie une fonction de projection, ou `None` sans R18 ni R21. R18 : ACP de scikit-learn à `R18.components` composantes (32 par défaut, jamais plus que le nombre de fenêtres ni de dimensions). R21 : sur les seuls négatifs, calcule les directions qui distinguent les micros (`mean_directions` si `R21.method` vaut `means`, le défaut ; `nuisance_directions` si `inlp`) et les retire ; une autre méthode lève `ValueError`.
- `pool_windows(project, train)` ([assembly.py:134](assembly.py#L134)) : R81. Garde du réservoir les fenêtres des micros d'entraînement (`R81.pool_from: train`, le défaut) ou tout (`all`), leur applique `window_transform` et la projection. Sans réservoir : `ValueError` « R81 demande un réservoir de fenêtres non annotées (Context.pool) ».
- `fit_options(bias_columns=0, bias_levels=None)` ([assembly.py:231](assembly.py#L231)) : R27 → `l1_ratio = 1` ; R28 → `l1_ratio = R28.l1_ratio` (0,5) ; avec des colonnes de biais : `bias_columns`, `bias_scale`, et au besoin `bias_levels` et `glmm_grid` (`R37.glmm_grid`, sinon `GLMM_GRID`). Ces options vont telles quelles à [`head.fit_logistic`](../head.py#L251) ou [`losses.fit_loss`](../losses.py#L69).
- `bias_scales()` ([assembly.py:220](assembly.py#L220)) : R37 : `(σ du micro,)` ou `(σ du micro, σ du site)` selon `R37.scale` (3 par défaut) et `R37.site_scale` (`None` par défaut) ; chaque valeur est un nombre ou le mot `glmm`.

#### `grid`, `bags`, `pseudo_options`, `torch_options`, `dann_options` : les autres méthodes de `Regularizer` — [assembly.py:155](assembly.py#L155)

- `grid(C_grid)` ([assembly.py:155](assembly.py#L155)) : avec R76, remplace la grille de C par `fine_grid(C_grid, R76.points)` (13 points) ; sinon la rend telle quelle. Appelée par [`head.fit_and_score`](../head.py#L619) et [`head.choose_C`](../head.py#L587).
- `bags` ([assembly.py:183](assembly.py#L183)) : propriété ; le nombre de tirages de R79 (`R79.bags`, 20), 0 sans R79.
- `pseudo_options()` ([assembly.py:161](assembly.py#L161)) : les réglages de R81 pour `with_pseudo_labels` : `min_score` 3, `max_fraction` 0,01, `weight` 0,3, `rounds` 1.
- `torch_options()` ([assembly.py:187](assembly.py#L187)) : les options des têtes torch selon les suffixes présents. R40 : `weight_decays` (`R40.grid`, 1e-4 à 1e-1) et `n_splits` (3) ; R41 : `optimizer="adamw"`, `weight_decay` (1e-2) ; R42 : `early_stopping`, `monitor` (`ap`), `epochs` (`R42.max_epochs`, 300), `n_splits` ; R43 : `entropy` (0,01) ; R45 : `token_dropout` (0,2) ; R46 : `dim_dropout` (0,2) ; R47 : `shrink` (1e-2) ; R59 : `warmup` (20), `clip_norm` (1,0) ; R64 : `average` (`ema`), `ema_decay` (0,99), `swa_start` (0,75). Sans suffixe : un dictionnaire vide.
- `dann_options()` ([assembly.py:170](assembly.py#L170)) : les réglages de la tête `dann`, lus dans `regularization.R66` : `hidden` 32, `strength` 1,0, `domain_on` `negatives`, `weight_decay` 1e-2, `epochs` 300, `lr` 1e-2.

## `windows.py` : fenêtre par fenêtre, et poids des exemples

Les calculs de R19, R20, R21, R13, R15, R36 et les indicatrices de R37. Lit : seulement dans `store_domain_statistics` (stock d'embeddings et table `recordings`). N'écrit rien. Appelé surtout par `assembly.py`. Tests : `tests/heads/test_regularization.py`.

#### `store_domain_statistics(con, store, filters, by="point")` — [windows.py:68](windows.py#L68)

- Quand : dans [`head_benchmark.regularization_context`](../../evaluation/head_benchmark.py#L143), seulement si une tête demandée porte R19 ou R20 (`blanci heads`, `blanci heads-curve`).
- Reçoit : la connexion, le stock (`EmbeddingStore`), les filtres de partition, la colonne de regroupement. Renvoie : un `DomainStats`.
- Fait : 1. lit dans `recordings` (`inputs/dataset.recordings_table`) le groupe de chaque enregistrement ; 2. parcourt le stock fichier par fichier (`store.fragments`, `store.read`) ; 3. écarte les fenêtres arrêtées par une porte (`gated_mask`) ; 4. cumule par groupe la somme et la somme des carrés des embeddings ; 5. un stock vide lève `ValueError` « stock vide pour … : pas de statistiques (R19/R20) ».
- Pourquoi (docstring de `__init__.py`) : aucune étiquette n'est lue, ce que la chaîne aura aussi sur un nouveau site.
- `_Accumulator` ([windows.py:32](windows.py#L32)) : les sommes par groupe, ajoutées paquet par paquet (`add`), puis `stats(by)` en tire moyenne et écart-type (variance négative ramenée à 0). Pourquoi (docstring) : un stock entier ne tient pas en mémoire. Classe privée, pourtant importée par `anuraset/calcul/global_bench.py`.

#### `domain_statistics(emb, groups, by="point")` — [windows.py:62](windows.py#L62)

- Quand : quand les embeddings sont déjà en mémoire : [`evaluation/anuraset.run_anuraset_heads`](../../evaluation/anuraset.py#L341) et `amorcage.py`, avec `by="site"`.
- Renvoie : le `DomainStats` de ces embeddings, en un seul paquet.

#### `DomainStats` — [windows.py:14](windows.py#L14)

- Moyenne et écart-type des embeddings de chaque groupe : `by`, `names`, `mean`, `std`, `count` (nombre de fenêtres).
- `rows(groups)` ([windows.py:23](windows.py#L23)) : l'indice du groupe de chaque fenêtre ; un groupe sans statistiques lève `ValueError` « pas de statistiques de stock pour … (R19/R20) ». Appelée par les quatre constructeurs de `Context`.

#### `mean_directions(X, groups)` — [windows.py:87](windows.py#L87)

- Quand : dans `Regularizer.fit_projection`, méthode `means` de R21 (le défaut), sur les négatifs d'entraînement.
- Renvoie : une base orthonormée des écarts entre la moyenne de chaque groupe et la moyenne générale, au plus (nombre de groupes − 1) directions ; aucune direction s'il y a moins de deux groupes.
- Fait : décomposition en valeurs singulières des écarts de moyennes ; garde les directions dont la valeur singulière dépasse un millionième de la plus grande.
- Pourquoi (docstring) : une fois ce sous-espace retiré, tous les micros ont la même moyenne ; aucun classifieur linéaire ne les distingue plus par leur fond moyen (principe de LEACE, Belrose et al. 2023).

#### `nuisance_directions(X, groups, iterations=3, max_directions=128, C=1.0, seed=0, tolerance=0.05)` — [windows.py:104](windows.py#L104)

- Quand : dans `Regularizer.fit_projection`, méthode `inlp` de R21.
- Fait : à chaque tour (3 au plus) : 1. retire des embeddings centrés les directions déjà trouvées ; 2. si une régression logistique ne reconnaît plus le micro mieux que le hasard (exactitude en validation croisée ≤ part du groupe majoritaire + `tolerance`), s'arrête ; 3. sinon ajuste cette régression, prend ses directions (les normales des frontières), les rend orthogonales à celles déjà trouvées et les ajoute, sans dépasser `max_directions`.
- Détail : la validation croisée a au plus 3 folds, et moins si le plus petit groupe a moins de 3 fenêtres ; avec un groupe d'une seule fenêtre, le test d'arrêt est sauté.
- Pourquoi `means` par défaut (docstring) : sur données simulées, quand les micros sont très séparés, l'INLP finit par retirer le chant.

#### `sample_weights(y, groups, hard, regs, hard_weight=3.0, class_power=1.0)` — [windows.py:160](windows.py#L160)

- Quand : dans `Regularizer.prepare`, sur les fenêtres d'entraînement.
- Renvoie : un poids par fenêtre, ou `None` si la tête n'a ni R13, ni R15, ni R36.
- Fait : 1. R13 : dans chaque classe, chaque fenêtre pèse 1 / (nombre de fenêtres de son micro dans cette classe) : chaque micro pèse autant ; 2. R15 : les négatifs annotés sont multipliés par `hard_weight` ; 3. dans chaque classe, les poids sont remis à une moyenne de 1 ; 4. R36 : multiplie par (n / 2·n_classe)^(`class_power` − 1).
- Détail : les têtes appliquent déjà le poids de classe « équilibré » n / 2·n_classe ; le produit donne donc (n / 2·n_classe)^`class_power` : 1 = équilibré, 0 = aucun rééquilibrage. La valeur par défaut ici est 1, mais `prepare` passe `R36.power`, qui vaut 0 (écart n° 4).

#### `bias_indicators(groups, train, sites=False)` — [windows.py:211](windows.py#L211)

- Quand : dans `Regularizer.prepare`, avec R37.
- Renvoie : `(indicatrices, niveau de chaque colonne)`. Niveau 0 : une colonne par micro de l'entraînement ; avec `sites`, niveau 1 en plus : une colonne par site de l'entraînement.
- `group_indicators(groups, train)` ([windows.py:197](windows.py#L197)) : une colonne par groupe présent dans `train`, 1 si la fenêtre est de ce groupe. Un micro absent de l'entraînement n'a pas de colonne : ses fenêtres n'ont que des 0, donc le biais commun.
- `site_of(groups)` ([windows.py:206](windows.py#L206)) : le site d'un point `site/micro`, ce qui précède le premier `/`.

## `selection.py` : choisir un réglage, bagging, pseudo-étiquetage

Lit : seulement dans `sample_pool` (stock et table `recordings`). N'écrit rien. C'est le fichier le plus appelé du dossier : `grouped_search` sert à tout ajustement logistique du projet. Tests : `tests/heads/test_regularization.py`, `tests/heads/test_new_heads.py`.

#### `grouped_search(score, grid, y, groups, n_splits=5, seed=0, rule=None, more_regularized="low")` — [selection.py:98](selection.py#L98)

- Quand : pour choisir la force d'une régularisation : [`head.select_C`](../head.py#L459) (le C des têtes logistiques, R26 : donc `blanci train`, `retrain`, `evaluate`, `heads`, `benchmark`, `fusion`…), `head._choose_multiclass_C` ([head.py:558](../head.py#L558)), [`fit_with_options`](torch_training.py#L361) (weight decay, R40), [`choose_fusion_C`](fusion_logistic.py#L15) (R50).
- Reçoit : une fonction `score(valeur, train, test)` qui rend l'AP sur `test` d'un modèle appris sur `train`, la grille de valeurs, les labels, les groupes. Renvoie : `(valeur retenue ou None, {valeur: AP moyenne}, {valeur: erreur type})`.
- Fait : 1. `usable_folds` donne les folds internes ; 2. pour chaque valeur, l'AP de chaque fold, leur moyenne, et l'erreur type (écart-type des AP ÷ √nombre de folds ; 0 avec un seul fold) ; 3. `pick` retient une valeur.
- Détail : sans fold utilisable, toutes les moyennes sont `nan` et la valeur retenue est `None` ; `head.select_C` prend alors la première valeur de la grille.
- Pourquoi (docstring) : les deux derniers dictionnaires forment le chemin de régularisation (R76), que `service.write_regularization_path` écrit en CSV après `blanci train`.

#### `usable_folds(y, groups, n_splits, seed)` — [selection.py:42](selection.py#L42)

- Quand : `grouped_search`, [`fit_with_options`](torch_training.py#L361) (R42), [`choose_fusion_C`](fusion_logistic.py#L15), [`head.calibrated_fold_scores`](../head.py#L842) et `fusion._fold_calibration` ([fusion.py:474](../../combination/fusion.py#L474)) pour les folds internes de la recalibration.
- Renvoie : les folds groupés (micros entiers, `evaluation/evaluate.grouped_folds`) dont l'entraînement et le test ont chacun les deux classes ; une liste vide s'il n'y a qu'un groupe.

#### `pick(means, ses, rule=None, more_regularized="low")` — [selection.py:74](selection.py#L74)

- Quand : dans `grouped_search`, seul appelant hors tests.
- Fait : écarte les valeurs sans AP ; avec la règle `best`, rend celle de meilleure AP moyenne ; avec `one_se`, garde toutes celles dont la moyenne est à moins d'une erreur type de la meilleure, et rend la plus régularisante : la plus petite (`low`, pour C) ou la plus grande (`high`, pour le weight decay). `None` si aucune valeur n'a d'AP.
- Détail : sans `rule`, la règle est celle de `_SELECTION` ([selection.py:63](selection.py#L63)), un réglage global du module, `one_se` au départ.

#### `configure(cfg)` — [selection.py:66](selection.py#L66)

- Quand : au début de chaque commande, par [`cli.main`](../../cli.py#L117). Aucun autre appelant hors tests.
- Fait : lit `head.selection_rule` (`one_se` dans `config/default.yaml`) et le range dans `_SELECTION` ; une valeur hors de `SELECTION_RULES` ([selection.py:62](selection.py#L62) : `one_se`, `best`) lève `ValueError`.
- À savoir : les programmes qui ne passent pas par `cli.main` (ceux de `anuraset/`) n'appellent pas `configure` : ils gardent `one_se`, quel que soit le fichier de configuration.

#### `fine_grid(grid, points=13)` — [selection.py:125](selection.py#L125)

- Quand : dans `Regularizer.grid`, avec R76.
- Renvoie : `points` valeurs régulièrement espacées en échelle logarithmique entre la plus petite et la plus grande de `grid` (13 entre 0,001 et 10 : un facteur d'environ 2,2 entre voisines, docstring).

#### `fold_ids(n, folds)` — [selection.py:136](selection.py#L136)

- Quand : pour passer des folds d'un `OOFScores` au numéro de fold de chaque fenêtre : [`service.train_and_register`](../../service.py#L135), [`head_benchmark.run_head_benchmark`](../../evaluation/head_benchmark.py#L228) et `level_rows`, [`evaluation/benchmark.probe_table`](../../evaluation/benchmark.py#L72), [`evaluation/anuraset.run_anuraset_heads`](../../evaluation/anuraset.py#L341).
- Renvoie : un tableau de `n` entiers, le fold de test de chaque fenêtre, −1 pour une fenêtre jamais testée.

#### `cross_fitted_threshold(y, scores, folds, min_precision)` — [selection.py:144](selection.py#L144)

- Quand : dans [`service.train_and_register`](../../service.py#L135) (`blanci train`, `retrain`), à chaque entraînement.
- Fait : appelle `evaluation/evaluate.cross_fitted_recall` ([evaluate.py:147](../../evaluation/evaluate.py#L147)), sans rien ajouter : pour chaque fold, le seuil est choisi sur les scores hors-fold des autres folds, puis appliqué à ce fold. `train_and_register` en range le rappel et la précision dans les métadonnées de la tête (`crossfit_recall`, `crossfit_precision`).
- Pourquoi (docstring, DECISIONS n° 140) : un seuil choisi et jugé sur les mêmes scores est optimiste.

#### `selection_estimate(scores, y, folds, recordings=None)` — [selection.py:157](selection.py#L157)

- Quand : à la fin d'un benchmark de têtes qui en compare plusieurs : [`head_benchmark.run_head_benchmark`](../../evaluation/head_benchmark.py#L228), [`evaluation/anuraset.run_anuraset_heads`](../../evaluation/anuraset.py#L341).
- Reçoit : les scores hors-fold de chaque tête, les labels, le fold de chaque fenêtre, et au besoin l'enregistrement de chaque fenêtre (l'AP se calcule alors par enregistrement). Renvoie : `winner` (la tête de meilleure AP sur tout), `winner_ap`, `chosen` (la tête choisie pour chaque fold), `fold_ap_winner` et `fold_ap_selection` ; un dictionnaire vide si aucune tête n'a d'AP.
- Fait : pour chaque fold, choisit la meilleure tête sur les autres folds, puis mesure son AP sur ce fold ; compare à l'AP, sur les mêmes folds, de la gagnante du tableau.
- Pourquoi (docstring, R74 et R80) : dit combien la gagnante doit à la chance, puisque le choix ne voit jamais les labels qu'on mesure.

#### `sample_pool(con, store, filters, by, exclude, size, seed=0)` — [selection.py:12](selection.py#L12)

- Quand : dans [`head_benchmark.regularization_context`](../../evaluation/head_benchmark.py#L143), seulement si une tête porte R81.
- Renvoie : `(embeddings, groupe)` d'au plus `size` fenêtres non annotées du stock (`R81.pool`, 20 000).
- Fait : dans chaque fichier du stock, tire au hasard sans remise au plus `size ÷ nombre de fichiers` fenêtres (arrondi au-dessus), hors fenêtres du benchmark (`exclude`) et fenêtres arrêtées par une porte ; coupe le tout à `size`. Un stock sans fenêtre utilisable lève `ValueError` « … : pas de réservoir (R81) ».

#### `with_pseudo_labels(fit_rows, X, y, sample_weight, pool, min_score=3.0, max_fraction=0.01, weight=0.3, rounds=1)` — [selection.py:223](selection.py#L223)

- Quand : dans [`head.fit_and_score`](../head.py#L619), pour les têtes `logistic`, `logistic_to_prototype` et `loss:<nom>` qui portent R81.
- Fait : 1. apprend une tête sur les labels ; 2. note le réservoir ; 3. `pseudo_positives` retient les fenêtres sûres ; 4. réapprend la tête avec ces fenêtres ajoutées comme positifs, au poids `weight` ; 5. recommence `rounds` fois, ou s'arrête dès qu'aucune fenêtre n'est retenue ; 6. range le nombre de pseudo-positifs du dernier tour dans `model.meta["pseudo_positives"]`.
- Pourquoi (docstring) : auto-apprentissage (Lee 2013) ; pas de pseudo-négatifs, les fenêtres notées bas étant aussi incertaines que les négatifs présumés (DECISIONS n° 106) ; risque : le biais de confirmation.
- `pseudo_positives(scores, min_score=3.0, max_fraction=0.01)` ([selection.py:209](selection.py#L209)) : les indices des fenêtres dont le score (logit) atteint `min_score`, au plus la fraction `max_fraction` du réservoir, les mieux notées d'abord.

#### `bagged(fit, y, groups, bags, seed=0, sample_weight=None)` — [selection.py:280](selection.py#L280)

- Quand : dans [`head.fit_and_score`](../head.py#L619), pour les têtes légères qui portent R79 (et pas R81).
- Renvoie : un `Bagged` ([selection.py:271](selection.py#L271)), dont `decision(X)` est la moyenne des scores de ses têtes.
- Fait : jusqu'à avoir `bags` têtes (20), tire les micros au hasard avec remise (`bootstrap_weights`), saute un tirage où il manque une classe, apprend une tête sur les fenêtres tirées, pondérées par le nombre de tirages de leur micro (× `sample_weight` s'il y en a). S'arrête après `10 × bags` essais ; aucune tête : `ValueError` « R79 : aucun tirage bootstrap n'a les deux classes ».
- Détail : si les essais s'épuisent, le résultat peut compter moins de `bags` têtes, sans message.
- `bootstrap_weights(groups, rng)` ([selection.py:261](selection.py#L261)) : tire autant de micros qu'il y en a, avec remise ; le poids d'une fenêtre est le nombre de fois où son micro est tiré. Pourquoi (docstring) : tirer des fenêtres ferait croire à des exemples indépendants.

## `glmm.py` : R37, l'échelle des biais de micro

Des calculs seulement. Appelé par [`head.fit_logistic`](../head.py#L251) et [`losses.fit_loss`](../losses.py#L69). Tests : `tests/heads/test_regularization.py`.

#### `group_bias_scale(sigma, C)` — [glmm.py:10](glmm.py#L10)

- Quand : à chaque ajustement de [`head.fit_logistic`](../head.py#L251) et de [`losses.fit_loss`](../losses.py#L69), juste avant `head.standardize`, même sans R37 (le tableau des σ est alors vide).
- Renvoie : σ ÷ √C, pour un σ ou un tableau de σ.
- Pourquoi (docstring) : avec cette échelle, le biais b d'un micro coûte b² / 2σ² quel que soit C : c'est un a priori N(0, σ²).

#### `bias_sigmas(bias_scale, bias_columns, bias_levels=None)` — [glmm.py:18](glmm.py#L18)

- Quand : mêmes appelants que `group_bias_scale`, et `glmm_scales`.
- Renvoie : le σ de chaque colonne de biais : `bias_scale` est un σ, ou un σ par niveau (micro, site) que `bias_levels` répartit ; tout au niveau 0 par défaut. Un σ encore égal au mot `glmm` lève `ValueError` « R37 : σ « glmm » à estimer d'abord ».

#### `glmm_scales(fit, X, y, C, bias_columns, bias_scale, bias_levels=None, sample_weight=None, grid=GLMM_GRID)` — [glmm.py:63](glmm.py#L63)

- Quand : dans [`head.fit_logistic`](../head.py#L251), quand il y a des colonnes de biais et qu'un σ vaut `glmm` (`logistic+R37=glmm`, ou `R37.scale: glmm`). `fit_logistic` réajuste ensuite la tête avec les σ trouvés et range le tableau des essais dans `meta["glmm_evidence"]`.
- Reçoit : `fit(σ par niveau)`, qui rend la tête ajustée avec ces σ, les données, C, le nombre de colonnes de biais. Renvoie : `(σ retenu par niveau, {essai: log-vraisemblance approchée})`.
- Fait : 1. part de σ = 1 pour chaque niveau à estimer ; 2. pour chaque niveau à estimer, essaie chaque valeur de `grid` en gardant les autres niveaux, et retient celle de plus grande `glmm_evidence` ; 3. avec deux niveaux à estimer, refait une seconde passe ; 4. un essai déjà fait n'est pas recalculé.
- `GLMM_GRID` ([glmm.py:30](glmm.py#L30)) : les σ essayés faute de `R37.glmm_grid` : 0,1, 0,3, 1, 3, 10.
- `glmm_evidence(decision, y, sample_weight, indicators, biases, sigmas, weight_penalty)` ([glmm.py:33](glmm.py#L33)) : la log-vraisemblance marginale approchée (Laplace) : − perte pondérée − pénalité des poids − Σ b² / 2σ² − ½ log det(I + Σ^½ H Σ^½), H étant la courbure de la perte le long des biais. Pourquoi (docstring) : un σ grand laisse chaque micro coller à ses données mais se paie dans le dernier terme ; un σ petit coûte en ajustement si les micros diffèrent vraiment.

## `torch_training.py` : la mécanique des têtes torch

Ne lit et n'écrit rien. `torch` n'est importé qu'à l'intérieur des fonctions. Appelé par [`attentive.fit_attentive`](../attentive.py#L102), [`gated.fit_gated`](../gated.py#L49), [`dann.fit_dann`](../dann.py#L53), [`proto_probe.fit_proto_probe`](../proto_probe.py#L49) et [`head.fit_and_score`](../head.py#L619). Tests : `tests/heads/test_attentive.py`, `tests/heads/test_regularization.py`, `tests/heads/test_new_heads.py`.

#### `fit_with_options(fit, X, y, groups, seed=0, *, weight_decays=None, early_stopping=False, monitor="ap", n_splits=3, **options)` — [torch_training.py:361](torch_training.py#L361)

- Quand : dans [`head.fit_and_score`](../head.py#L619), pour les têtes `gated`, `dann` et `attentive`, avec les options de `Regularizer.torch_options` (aucune sans suffixe). `proto_probe` et la cascade n'y passent pas.
- Reçoit : la fonction d'entraînement (`fit_gated`, `fit_dann`, `fit_attentive`), les données, les groupes. Renvoie : la tête entraînée, avec dans `meta` ce qui a été choisi.
- Fait : 1. R40, s'il y a au moins deux valeurs dans `weight_decays` : chaque valeur est jugée par `grouped_search` (AP moyenne sur des folds internes par micro ; avec la règle `one_se`, la plus grande valeur parmi celles à moins d'une erreur type de la meilleure) ; la valeur retenue remplace `weight_decay`, le tableau va dans `meta["weight_decay_cv"]` ; 2. R42, si `early_stopping` : dans chaque fold interne, la tête est entraînée sans arrêt (`patience=None`) et sa courbe de validation relevée ; l'époque qui minimise la courbe moyenne est retenue ; la tête est réentraînée sur tout pour ce nombre d'époques ; `meta["early_stopping"]` garde l'époque, le critère et le nombre de folds ; 3. sinon, un entraînement simple.
- Détails : faute de deux micros ou d'un fold à deux classes, R40 et R42 sont sautées (docstring). Une fonction `fit` qui porte l'attribut `needs_groups` (c'est le cas de `fit_dann`) reçoit en plus les groupes des lignes.
- Pourquoi (docstring, DECISIONS n° 117) : un fold unique de 1 à 3 micros donnait une époque au hasard.

#### `optimise(parameters, decayed, loss_of, *, epochs, lr, weight_decay, optimizer="adam", validation_loss=None, patience=20, curve=None, warmup=0, clip_norm=None, average=None, ema_decay=0.99, swa_start=0.75)` — [torch_training.py:70](torch_training.py#L70)

- Quand : la boucle d'entraînement des quatre têtes torch : `fit_attentive`, `fit_gated`, `fit_dann`, `fit_proto_probe`.
- Reçoit : les paramètres à apprendre, lesquels subissent le weight decay, la fonction `loss_of()` qui rend la perte. Renvoie : l'époque retenue (la dernière sans validation).
- Fait, à chaque époque (un pas sur le lot entier) : 1. R59 : pendant `warmup` époques, le pas d'apprentissage monte linéairement ; 2. calcule la perte et son gradient ; 3. R59 : ramène la norme du gradient à `clip_norm` au plus ; 4. fait le pas (Adam, ou AdamW avec R41) ; 5. R64 : met à jour la moyenne des poids ; 6. avec `validation_loss` (R42) : mesure le critère (sur les poids moyens si R64 est active), garde une copie des paramètres s'il s'améliore, et s'arrête `patience` époques après la meilleure. À la fin : restaure les paramètres de la meilleure époque, ou, sans validation, remplace les poids par leur moyenne.
- Détails : un optimiseur autre que `adam` ou `adamw` lève `ValueError` ; de même une moyenne autre que `ema` ou `swa`. Le critère de chaque époque est ajouté à la liste `curve`.
- `WeightAverage` ([torch_training.py:153](torch_training.py#L153)) : R64. `update(epoch)` : `ema` = moyenne glissante (poids × `decay` à chaque pas) ; `swa` = moyenne simple à partir de l'époque `ceil(start × epochs)`. `apply()` remplace les poids par la moyenne ; `swapped()` met les poids moyens le temps d'une évaluation puis rend les poids courants. Sans méthode, rien ne change. Pourquoi (docstring) : en fin d'entraînement les poids oscillent autour d'un minimum, la moyenne tombe plus près du centre (Izmailov et al. 2018 pour SWA).

#### `import_torch` et le socle commun — [torch_training.py:15](torch_training.py#L15)

- `import_torch(what)` ([torch_training.py:15](torch_training.py#L15)) : rend `torch`, ou lève `RuntimeError` « … s'entraîne avec torch (uv sync --group research) ». Appelée au début des quatre fonctions d'entraînement.
- `feature_scaling(X)` ([torch_training.py:24](torch_training.py#L24)) : moyenne et écart-type de chaque dimension, un écart nul devenant 1 ; rangés avec la tête pour l'inférence en numpy. Quatre appelants, les mêmes.
- `balanced_bce(torch, y)` ([torch_training.py:32](torch_training.py#L32)) : l'entropie croisée binaire où les positifs pèsent n_négatifs ÷ n_positifs. Quatre appelants, les mêmes.
- `validation_criterion(torch, loss_fn, scores_of, y_val, monitor="loss")` ([torch_training.py:284](torch_training.py#L284)) : le critère de R42, à minimiser : la perte sur les fenêtres de validation (`loss`) ou 1 − AP (`ap`) ; autre mot : `ValueError`. Appelée par `fit_attentive`, `fit_gated`, `fit_dann`. Pourquoi deux critères (docstring, DECISIONS n° 117 et 119) : la perte monte dès que la tête devient trop sûre d'elle, même quand son classement s'améliore encore.
- `dropout(torch, x, p, train)` ([torch_training.py:318](torch_training.py#L318)) : R46 ; éteint des sorties avec la probabilité `p`, à l'entraînement seulement. Appelée par `fit_attentive`, `fit_gated`, `fit_dann`.

#### `attention_entropy` et les options de la tête attentive — [torch_training.py:300](torch_training.py#L300)

- `attention_entropy(torch, weights)` ([torch_training.py:300](torch_training.py#L300)) : R43 ; l'entropie moyenne des poids d'attention. `fit_attentive` l'ajoute à la perte avec un coefficient : positif, il pousse vers une attention piquée ; négatif, vers une attention diffuse (docstring).
- `keep_mask(torch, n, t, p)` ([torch_training.py:310](torch_training.py#L310)) : R45 ; quels jetons garder, chacun masqué avec la probabilité `p`, au moins un par fenêtre.
- `logistic_start(Z, y, C, seed=0)` ([torch_training.py:326](torch_training.py#L326)) : R47 ; ajuste `head.fit_logistic` sur `Z` et rend ses poids et son biais réécrits sans la standardisation : le point de départ de l'attentive, et la cible de son rétrécissement.
- `l2_sp_penalty(torch, parameters, references, strength)` ([torch_training.py:338](torch_training.py#L338)) : ½ λ Σ ‖θ − θ_réf‖² : l'écart aux poids de référence plutôt qu'à zéro. Appelée par `fit_attentive` pour R47.

#### `grad_reverse` et `dann_strength` : la tête `dann` — [torch_training.py:42](torch_training.py#L42)

- `grad_reverse(torch, x, strength)` ([torch_training.py:42](torch_training.py#L42)) : R66 ; à l'aller l'identité, au retour le gradient multiplié par −`strength` (Ganin et al. 2016). Appelée par `fit_dann`.
- `dann_strength(progress, strength=1.0)` ([torch_training.py:60](torch_training.py#L60)) : la force de l'inversion selon l'avancement p de l'entraînement : `strength × (2 / (1 + e^(−10p)) − 1)`, nulle au départ, pleine ensuite. Appelée par `fit_dann`.

#### `layerwise_lr_groups` et les autres aides en attente (R61, R62 complet, R63) — [torch_training.py:221](torch_training.py#L221)

- Six fonctions que le code marque « EN ATTENTE — emplacement gardé pour plus tard, volontairement non branché ; ne pas signaler comme code mort à l'audit ». Aucun appelant hors tests (`tests/heads/test_attentive.py`, `tests/heads/test_regularization.py`).
- `layerwise_lr_groups(layers, lr, decay=0.8, weight_decay=0.0)` ([torch_training.py:221](torch_training.py#L221)) : R61 ; un pas d'apprentissage par couche, `lr` pour la dernière, × `decay` à chaque couche en descendant.
- `unfreezing_schedule(n_layers, epochs, every)` ([torch_training.py:241](torch_training.py#L241)) : R61 ; le nombre de couches du haut qui apprennent à chaque époque : 1, puis une de plus toutes les `every` époques.
- `unfreeze_top(layers, n_top)` ([torch_training.py:249](torch_training.py#L249)) : R61 ; gèle tout sauf les `n_top` dernières couches.
- `snapshot(model)` ([torch_training.py:259](torch_training.py#L259)) : R62 ; une copie des poids avant l'adaptation.
- `l2_sp_model_penalty(torch, model, reference, alpha, beta=0.0)` ([torch_training.py:266](torch_training.py#L266)) : R62 ; α/2 Σ ‖θ − θ⁰‖² sur les poids qui existaient, β/2 Σ ‖θ‖² sur les nouveaux ; les poids gelés ne comptent pas.
- `distillation_loss(torch, student_logits, teacher_logits, temperature=2.0)` ([torch_training.py:348](torch_training.py#L348)) : R63 ; entropie croisée entre l'élève et l'enseignant, adoucis par la température, × T².

## `fusion_logistic.py` : la fusion logistique régularisée

Des calculs seulement. Appelé par `combination/fusion.py` et `combination/ensemble.py`. Les tests sont rangés dans `tests/combination/test_stacking.py`.

#### `fusion_settings(method)` et `fusion_variant(method)` — [fusion_logistic.py:59](fusion_logistic.py#L59)

- Quand : `fusion_settings` à chaque appel de [`fusion.fit_fusion_model`](../../combination/fusion.py#L313) et dans `fusion._fold_calibration` ; `fusion_variant` ([fusion_logistic.py:86](fusion_logistic.py#L86)) dans [`ensemble.run_ensemble`](../../combination/ensemble.py#L68), pour reconnaître une méthode valide.
- Reçoit : le nom de la méthode de fusion. Renvoie : `logistic+R50+R55=10` → `{50: None, 55: 10.0}` ; `logistic` → `{}` ; `None` si la méthode n'est pas la fusion logistique. `fusion_variant` rend seulement l'ensemble des numéros.
- Fait : chaque suffixe doit être dans `FUSION_SUFFIXES` ([fusion_logistic.py:47](fusion_logistic.py#L47) : 50, 52, 53, 54, 55, 56), sinon `ValueError` « suffixe … inconnu » ; seuls R54, R55 et R56 prennent une valeur (`FUSION_PARAMETER`, [fusion_logistic.py:56](fusion_logistic.py#L56) : `bins`, `smoothness`, `cap`) ; R54 avec R55 refusé, R53 avec R55 refusé.

#### `choose_fusion_C(X, y, groups, C_grid, n_splits=5, seed=0, method="logistic", columns=None, **options)` — [fusion_logistic.py:15](fusion_logistic.py#L15)

- Quand : dans [`fusion.fit_fusion_model`](../../combination/fusion.py#L313), quand la méthode porte R50 et que les groupes sont fournis.
- Renvoie : `(C retenu, {C: AP moyenne})`, ou `(None, {})` faute de deux valeurs de C, de deux micros ou d'un fold à deux classes ; `fit_fusion_model` garde alors le C fixé.
- Fait : `grouped_search` sur la grille ; pour chaque C et chaque fold interne, ajuste la fusion `method` (la même, sans R50) par `fit_fusion_model` et mesure son AP.

#### `fit_fusion_logistic(z, y, sign, columns, variant, C, cap=2.0, bins=5, segments=8, smoothness=1.0)` — [fusion_logistic.py:342](fusion_logistic.py#L342)

- Quand : dans [`fusion.fit_fusion_model`](../../combination/fusion.py#L313), dès que la méthode porte un des suffixes R52 à R56.
- Reçoit : les entrées standardisées de la fusion, les labels, l'orientation de chaque entrée (+1 si elle monte avec le label), les noms des entrées, les numéros actifs. Renvoie : `(coefficients, biais, base des descripteurs ou None, colonne de la tête ou None)`.
- Fait : 1. avec R54 ou R55, développe chaque descripteur du signal en un bloc de colonnes (`fit_descriptor_basis`) ; 2. avec R52, retourne les entrées restées linéaires selon leur orientation ; 3. impose des coefficients ≥ 0 aux entrées linéaires (R52) et aux paliers (R54) ; 4. repère la colonne `head` ; 5. appelle `fit_constrained_logistic` avec la pénalité L1 (R53), le plafond (R56) ou la pénalité de lissage (R55) ; 6. remet les coefficients dans l'orientation d'origine.
- `is_descriptor(column)` ([fusion_logistic.py:181](fusion_logistic.py#L181)) : vrai pour une entrée qui n'est ni `head` ni `head:<encodeur>` : seuls les descripteurs du traitement du signal sont courbés.

#### `fit_constrained_logistic(z, y, C, nonneg=False, l1=False, cap=None, head_index=None, penalty=None)` — [fusion_logistic.py:93](fusion_logistic.py#L93)

- Quand : dans `fit_fusion_logistic`, seul appelant.
- Renvoie : `(coefficients, biais)` d'une logistique à classes équilibrées, sous contraintes.
- Fait : minimise C × Σ poids × perte logistique + pénalité, par L-BFGS-B (5 000 itérations au plus), en partant de coefficients à 0,01. R52 : bornes ≥ 0 sur les colonnes désignées ; R53 : pénalité Σ |w| (valeur absolue lissée sur les colonnes sans borne) ; R55 : pénalité ½ wᵀ P w ; R56 : le score est w_tête·z_tête + `cap` × tanh(reste ÷ `cap`) + b.
- Détails : R53 avec R55 lève `ValueError` ; R56 sans entrée `head` lève `ValueError` « la fusion n'a pas d'entrée « head » à protéger ». Le calcul est limité à un fil BLAS (commentaire : sur de petits produits, plusieurs fils perdent jusqu'à 100 fois leur temps).
- Pourquoi R56 (docstring) : les descripteurs déplacent le score d'au plus `cap` en logit ; une tête assez sûre d'elle passe toujours, aucun descripteur n'a de droit de veto.

#### `DescriptorBasis(kind, n_inputs, expanded)` — [fusion_logistic.py:189](fusion_logistic.py#L189)

- Quand : créée par `fit_descriptor_basis` ; rangée dans [`fusion.FusionModel`](../../combination/fusion.py#L193) (champ `basis`), enregistrée avec la fusion et relue à `blanci score --fusion`.
- Chaque descripteur devient un bloc de colonnes. `steps` (R54) : une colonne par seuil, 1 si la valeur orientée dépasse le seuil ; `spline` (R55) : B-splines cubiques entre les quantiles 1 % et 99 % de l'entraînement, plates au-delà. Les colonnes sont centrées sur l'entraînement.
- `transform(z)` ([fusion_logistic.py:230](fusion_logistic.py#L230)) : la matrice développée ; appelée par `fit_fusion_logistic` et par `FusionModel.decision`.
- `blocks()` ([fusion_logistic.py:220](fusion_logistic.py#L220)) : les colonnes (début, fin) de chaque entrée.
- `penalty(smoothness=1.0, order=2)` ([fusion_logistic.py:241](fusion_logistic.py#L241)) : la matrice P de R55 : 1 pour une entrée linéaire ; pour un bloc de spline, sa taille plus `smoothness` × la courbure (différences d'ordre 2 des coefficients voisins).
- `amplitudes(coef)` ([fusion_logistic.py:261](fusion_logistic.py#L261)) : l'amplitude de la contribution de chaque entrée entre −2 et +2 écarts-types ; appelée par `FusionModel.weights` pour donner la part de chaque entrée.
- `to_dict()` et `from_dict(d)` ([fusion_logistic.py:289](fusion_logistic.py#L289)) : aller-retour JSON ; appelées par `FusionModel.to_dict` et `FusionModel.from_dict`.
- `curve(j, coef, x)` ([fusion_logistic.py:279](fusion_logistic.py#L279)) : la contribution de l'entrée `j` ; appelée par les tests seulement.
- `_raw(spec, x)` ([fusion_logistic.py:210](fusion_logistic.py#L210)) : les colonnes d'une entrée avant centrage.
- `fit_descriptor_basis(kind, z, sign, expand, bins=5, segments=8, degree=3)` ([fusion_logistic.py:301](fusion_logistic.py#L301)) : apprend la base sur l'entraînement : seuils aux quantiles (`bins` classes de même effectif) ou nœuds de `segments` intervalles égaux ; une entrée constante n'est pas développée ; un `kind` inconnu lève `ValueError`. Appelée par `fit_fusion_logistic` seulement.

## `decision.py` : R73 et la recalibration par fold

Des calculs seulement. Appelé par `service.py`, `head.py` et `combination/fusion.py`. Tests : `tests/heads/test_regularization.py`, `tests/heads/test_head.py`.

#### `operating_curve(y, scores)` — [decision.py:11](decision.py#L11)

- Quand : dans `threshold_at_prevalence`, et dans [`service.write_operating_curve`](../../service.py#L263), qui écrit `seuils_<encodeur>_<version>.csv` après chaque `blanci train`.
- Renvoie : pour chaque seuil possible (les scores distincts, du plus haut au plus bas) : `threshold`, `tpr` (rappel), `fpr` (taux de fausses alertes), `precision` dans cet échantillon.

#### `precision_at_prevalence(tpr, fpr, prevalence)` — [decision.py:26](decision.py#L26)

- Quand : dans `threshold_at_prevalence`, et dans `service.write_operating_curve` si `decision.prevalence` est renseignée.
- Renvoie : la précision attendue là où la part de positifs vaut `prevalence` : π·TPR / (π·TPR + (1 − π)·FPR).
- Limite (docstring) : suppose que les négatifs jugés ressemblent à ceux du stock, alors que le benchmark tire ses négatifs près des positifs.

#### `threshold_at_prevalence(y, scores, min_precision, prevalence)` — [decision.py:36](decision.py#L36)

- Quand : dans [`service.train_and_register`](../../service.py#L135), seulement si `decision.prevalence` est renseignée ; le seuil et le rappel rendus remplacent alors ceux du benchmark.
- Renvoie : `recall`, `threshold`, `precision` : le seuil de plus grand rappel dont la précision à la prévalence du stock atteint `min_precision` (à rappel égal, le seuil le plus haut) ; rappel 0 et seuil infini si aucun seuil n'y arrive.

#### `platt(scores, y)` — [decision.py:56](decision.py#L56)

- Quand : dans [`service.train_and_register`](../../service.py#L135), à chaque entraînement, sur les scores hors-fold finis ; le couple est rangé dans les métadonnées de la tête (clé `platt`).
- Renvoie : `(a, b)` tels que σ(a·score + b) soit la probabilité d'être un positif, à la proportion de positifs de l'échantillon. Une logistique à une variable, presque sans pénalité (C = 1e6).
- À savoir : aucune lecture de la clé `platt` n'a été trouvée dans `blanci/` (écart n° 9).

#### `prior_shift(p, source_prevalence, target_prevalence)` — [decision.py:68](decision.py#L68)

- Quand : dans les tests seulement.
- Renvoie : une probabilité ramenée d'une proportion de positifs à une autre (règle de Bayes sur les cotes).

#### `fold_platt(scores, y)` — [decision.py:82](decision.py#L82)

- Quand : dans [`head.calibrated_fold_scores`](../head.py#L842) et `fusion._fold_calibration` ([fusion.py:474](../../combination/fusion.py#L474)), donc seulement si `benchmark.fold_calibration: platt`.
- Renvoie : `(a, b)` : a·score + b est la cote « A. blanci contre fond » à classes équilibrées, apprise sur des scores hors-fold.
- Fait : centre et réduit les scores, puis ajuste une logistique à une variable faiblement pénalisée (C = 100, classes équilibrées). Sans les deux classes, ou si la pente n'est pas un nombre strictement positif, se replie sur le seul centrage-réduction.
- Pourquoi (docstring, DECISIONS n° 133 et 135) : mettre les folds sur une même échelle avant de les mettre bout à bout ; le classement d'un fold n'est jamais retourné.

## `classes.py` : R67, les classes des sons

Des calculs seulement. Tests : `tests/heads/test_new_heads.py`.

#### `window_classes(labels, y, presumed=None)` — [classes.py:26](classes.py#L26)

- Quand : dans [`head_benchmark.regularization_context`](../../evaluation/head_benchmark.py#L143), à chaque `blanci heads` et `blanci heads-curve`, que la tête `multiclass` soit demandée ou non.
- Renvoie : la classe de chaque fenêtre : `blanci` pour un positif ; `fond` pour un négatif présumé ou sans label ; sinon la classe de son label d'après `CLASS_OF_LABEL` ; `autre` pour un label inconnu.
- `CLASS_OF_LABEL` ([classes.py:11](classes.py#L11)) : `bird` → oiseau ; `amphibian`, `amphibian_contact_call` → amphibien ; `false_friend` → faux ami ; `orthoptera` → orthoptère ; `rain` → pluie ; `background` → fond ; `artefact_muffled`, `artefact_in_bag` → artefact ; `other` → autre. `POSITIVE_CLASS` ([classes.py:23](classes.py#L23)) = `blanci`.
- Pourquoi (commentaire) : les négatifs présumés rejoignent le fond, aucun événement n'y ayant été noté (n° 124) ; `artefact_in_bag` est l'ancien nom, d'avant le n° 203.

#### `merge_rare_classes(classes, min_count=10)` — [classes.py:44](classes.py#L44)

- Quand : dans [`head.fit_multiclass`](../head.py#L428), à chaque ajustement de la tête `multiclass`, avec `R67.min_count`.
- Fait : une classe de moins de `min_count` fenêtres (hors `blanci`, `fond`, `autre`) rejoint `autre` ; si `autre` reste sous le seuil, elle rejoint `fond`.
- Pourquoi (docstring) : une classe de 3 exemples ne s'apprend pas, elle ajoute du bruit.

## `neighbors.py` : R39, les k plus proches références

Une fonction. Tests : `tests/heads/test_head.py`.

#### `nearest_similarity(sims, k=1, weighted=False)` — [neighbors.py:10](neighbors.py#L10)

- Quand : dans [`head.exemplar_scores`](../head.py#L71) et [`head.knn_scores`](../head.py#L133), donc pour les têtes `exemplar`, `knn` et leurs variantes `knn:k=5`, `exemplar:k=3`, `knn:k=5:w`.
- Reçoit : la matrice des similarités cosinus (fenêtres × références). Renvoie : pour chaque fenêtre, sa similarité au plus proche (k = 1) ou la moyenne de ses `k` plus proches ; avec `weighted`, moyenne pondérée par l'inverse de la distance euclidienne.
- Détail : `k` est ramené entre 1 et le nombre de références.
- Pourquoi (docstring, DECISIONS n° 115) : la moyenne de k voisins est plus lisse et moins sensible à une référence bizarre.

## Écarts relevés

Signalés, pas corrigés.

1. **`logistic+R37=<nombre>` est refusé par `parse_head`.** Pour un numéro de `WORD_VALUES` (37, 42, 64), toute valeur doit être un des mots prévus ([names.py:86](names.py#L86)) : `logistic+R37=0.3` lève donc « R37=0.3 : valeurs possibles glmm ». La docstring de `__init__.py` (« `logistic+R37=0.3`, `=1`, `=3`, `=10` »), le commentaire de `config/default.yaml`, le catalogue (« `+R37=σ` ») et un commentaire de `tests/heads/test_regularization.py` (« R37 : un nombre ou « glmm » ») disent l'inverse. Aucun test ne lit un nom `R37=<nombre>` : ceux qui règlent σ construisent `Regularizer({37: 0.01}, …)` directement. σ reste réglable par `regularization.R37.scale`. Établi à la lecture de `parse_head` (lignes 86 à 91) ; rien n'a été lancé.
2. **`__init__.py` annonce « R1–R84 ».** Le catalogue est titré « R1–R85 », et R85 (tête `gated`) figure plus bas dans la même docstring.
3. **Le message de `validate` oublie `dann`.** « R… : têtes attentive et gated seulement (entraînées avec torch) » (`names.py`, ligne 147), alors que `TORCH_REGULARIZATIONS` accepte R40, R42, R46, R59, R64 pour `dann`. La docstring de `__init__.py` ne cite pas non plus `proto_probe.py` parmi les têtes qui appellent la mécanique torch.
4. **R36 : trois valeurs par défaut.** `sample_weights` a `class_power=1.0` ; `Regularizer.prepare` passe `R36.power` avec 0,0 par défaut, et `config/default.yaml` écrit `R36: {power: 0.0}`. `logistic+R36` sans valeur retire donc tout rééquilibrage des classes. La docstring de `__init__.py` (« 1 équilibré (défaut), 0 aucun ») et le catalogue (« équilibré par défaut ») parlent de la tête sans R36 ; lus vite, ils laissent croire que `+R36` seul ne change rien.
5. **R21 est décrite comme « INLP ».** `DESCRIPTIONS[21]` et le catalogue disent « INLP » ; la méthode par défaut est `means` (`R21.method`), l'INLP est l'option. De plus le réglage principal de R21 est `iterations` : `logistic+R21=5` ne change rien avec la méthode par défaut.
6. **`DESCRIPTIONS` n'est lue par personne.** Réexportée par `__init__.py`, absente de tout autre fichier de `blanci/`, `anuraset/`, `scripts/`, `documentation/tableau-de-bord/` et `tests/`.
7. **Fonctions sans appelant hors tests.** `prior_shift`, `DescriptorBasis.curve`, et les six aides « EN ATTENTE » (`layerwise_lr_groups`, `unfreezing_schedule`, `unfreeze_top`, `snapshot`, `l2_sp_model_penalty`, `distillation_loss`), que le code demande de ne pas signaler comme code mort. `Context.subset` n'est appelée que par `anuraset/calcul/global_bench.py`.
8. **Une valeur donnée à un suffixe sans réglage principal est acceptée et ignorée.** `logistic+R17=3` ou `attentive+R40=0.01` passent `validate`, changent le nom canonique (donc le nom du fichier de scores hors-fold), et ne changent aucun calcul : R13, R17, R19, R20, R27 et R40 ne sont pas dans `MAIN_PARAMETER`.
9. **Le Platt de R73 est rangé mais pas relu.** `service.train_and_register` range `platt(…)` dans les métadonnées de la tête ; aucune lecture de cette clé dans `blanci/` (seul `tests/interfaces/test_service.py` vérifie sa présence). `prior_shift`, que la docstring de `platt` donne comme correction, n'est pas branchée. Relevé en passant, à reprendre dans la fiche de `service.py` : ce Platt est ajusté sur les scores « finis », ce qui garde les fenêtres arrêtées par une porte (score `GATED_SCORE`, −1e6).
10. **`multiclass+R19+R20` n'est pas refusé.** `validate` s'arrête après le contrôle propre à `multiclass` ([names.py:133](names.py#L133)), avant le refus « R20 contient déjà le centrage de R19 ». Sans effet sur le calcul : `window_transform` ne centre qu'une fois.
11. **Le catalogue range `multiclass` parmi les « têtes entraînées avec torch ».** Dans le code, `head.fit_multiclass` est une régression logistique de scikit-learn, et `validate` ne lui accepte aucun suffixe torch.
12. **`_Accumulator` est privée mais importée de l'extérieur.** `anuraset/calcul/global_bench.py` fait `from blanci.heads.regularization.windows import _Accumulator` pour cumuler les statistiques par paquets ; `domain_statistics`, la fonction publique, ne le permet pas.
13. **La règle R75 ne s'applique qu'en passant par `cli.main`.** `configure` n'a pas d'autre appelant : les programmes de `anuraset/` gardent `one_se` même si la configuration demande `best`.
14. **Commentaire de `config/default.yaml` en retard.** Il liste pour les têtes torch « attentive (R40–R42, R45–R47, R59), gated (R40, R42, R46, R59) » : R43 et R64 manquent, et `dann` aussi, par rapport à `TORCH_REGULARIZATIONS`.
