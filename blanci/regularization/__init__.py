"""Régularisations des têtes (DECISIONS n° 108), numérotées comme la liste du 25/09/2026
(R1–R84, `documentation/regularisation.md`).

Toutes coupées par défaut. Une tête du benchmark les active dans son nom :
`logistic+R18=16+R19` = régression logistique, ACP à 16 composantes (R18), centrage par micro
(R19). « =v » remplace le réglage principal de la section `regularization` de la config. Le nom
canonique (régularisations triées par numéro) est celui des rapports et des scores hors-pli :
on sait toujours lesquelles ont servi.

Un paquet découpé par thème (DECISIONS n° 134) ; tout s'importe d'ici
(`from blanci.regularization import …`), « ici » désigne le paquet :

| sous-module | contenu |
|---|---|
| `names` | noms des têtes (`parse_head`, `canonical`, `validate`), réglages principaux |
| `windows` | R19, R20 (stock par micro), R21, poids R13/R15/R36, indicatrices R37 |
| `assembly` | `Context`, `Regularizer` (application dans l'ordre), `regularizer_for` |
| `selection` | validation groupée (R26, R40, R75, R76), R74, R79, R81 |
| `torch_training` | têtes torch : `optimise`, R41–R47, R59, R61–R64, R66, `fit_with_options` |
| `glmm` | R37 : échelle des biais, σ estimé (GLMM) |
| `fusion_logistic` | fusion : R50, R52–R56 |
| `decision` | R73 : seuil, Platt, prévalence |
| `classes` | R67 : classes des sons |
| `neighbors` | R39 : k plus proches voisins |

Programmées ici, appliquées dans cet ordre :

| R | où | quoi |
|---|---|---|
| R19 | fenêtre | centrage par micro : − moyenne de tout le stock du micro (sans labels) |
| R20 | fenêtre | AdaBN : (x − moyenne du micro) / écart-type du micro |
| R17 | fenêtre | embedding ramené à la norme 1 |
| R18 | pli | ACP ajustée sur l'entraînement du pli (`components`) |
| R21 | pli | retrait des directions qui trahissent le micro (négatifs seulement) |
| R37 | fenêtre | indicatrices du micro (et du site) ajoutées à l'entrée : biais pénalisés |
| R13 | poids | chaque micro pèse autant dans sa classe |
| R15 | poids | négatifs annotés (faux amis, espèces) × `hard_weight` face aux présumés |
| R36 | poids | poids des classes (n / 2·n_classe)^`power` : 1 équilibré (défaut), 0 aucun |
| R27 | pénalité | L1 (lasso) au lieu de L2 |
| R28 | pénalité | Elastic Net (`l1_ratio`) |

Sélection des réglages et entraînements légers (DECISIONS n° 122) :

| R | quoi | fonction, activation |
|---|---|---|
| R74 | seuil et variante jugés à part | `cross_fitted_threshold`, `selection_estimate` |
| R75 | règle du « 1 écart-type » (`head.selection_rule`) | `pick`, `grouped_search` |
| R76 | grille de C plus fine, chemin de régularisation (`train`) | `+R76`, `fine_grid` |
| R77 | un pli par micro | `head.n_splits: lomo` (`evaluate.lomo_assignment`) |
| R79 | bagging par tirage bootstrap des micros | `+R79`, `bagged` |
| R81 | pseudo-étiquetage sur un réservoir non annoté | `+R81`, `with_pseudo_labels` |

Réglage choisi par validation groupée : `grouped_search`, commun au C des têtes (R26,
`head.select_C`), au weight decay (R40) et au C de la fusion (R50, `choose_fusion_C`).

Têtes et réseaux entraînés avec torch (DECISIONS n° 117, 121, 123) : toute la mécanique est
ici, les têtes (`attentive.py`, `gated.py`, `dann.py`) l'appellent dans leur entraînement.

| R | têtes | quoi | fonction |
|---|---|---|---|
| R40 | attentive, gated, dann | weight decay par validation groupée | `fit_with_options` |
| R41 | attentive | AdamW (weight decay découplé, `weight_decay`) | `optimise` |
| R42 | attentive, gated, dann | époques par validation groupée (`=ap`, `=loss`) | idem |
| R43 | attentive | entropie de l'attention (`strength` β, signé) | `attention_entropy` |
| R45 | attentive | dropout des jetons (`p`) | `keep_mask` |
| R46 | attentive, gated, dann | dropout des dimensions (`p`) | `dropout` |
| R47 | attentive | départ et rétrécissement vers la logistique (`strength`) | `logistic_start` |
| R59 | attentive, gated, dann | warm-up du pas, écrêtage du gradient | `optimise` |
| R64 | attentive, gated, dann | moyenne des poids (`=ema`, `=swa`) | `WeightAverage` |
| R66 | dann | inversion du gradient contre le micro | `grad_reverse`, `dann_strength` |

Prêtes pour les réseaux à venir (fine-tuning, distillation, modèle maison, n° 121, 123) :
R61 `layerwise_lr_groups`, `unfreezing_schedule`, `unfreeze_top` ; R62 `snapshot`,
`l2_sp_model_penalty` (et `l2_sp_penalty`, déjà utilisée par R47) ; R63 `distillation_loss`.

Ce paquet est l'index de toutes les régularisations programmées. Celles qui sont une tête ou
un réglage d'un autre étage vivent là où elles s'appliquent, et appellent ce paquet quand elles
ont une mécanique propre :

| R | où | comment l'activer |
|---|---|---|
| R22 | `pooling.gem` | tête `logistic:gem` (gem2, gem5…) |
| R26 | `head.fit_logistic` | la L2, toujours là ; C par `grouped_search` |
| R30, R31 | `head.py` | têtes `logistic_to_prototype`, `lda_shrunk` |
| R34, R35 | `losses.py` | têtes `loss:<nom>`, `--methods losses` |
| R37 | `head.standardize` | échelle des colonnes : `group_bias_scale` (ici) |
| R37=glmm | `head.fit_logistic` | σ estimé sur les données : `glmm_scales` (ici) |
| R39 | `head.py` | têtes `knn:k=…`, `exemplar:k=…` (`:w`) ; calcul : `nearest_similarity` (ici) |
| R50 | `fusion.fit_fusion_model` | méthode `logistic+R50` ; C : `choose_fusion_C` (ici) |
| R52, R53, R56 | `fusion.py` | `logistic+R52`, `+R53`, `+R56` ; `fit_constrained_logistic` |
| R54, R55 | `fusion.py` | `logistic+R54`, `+R55` ; `DescriptorBasis`, `fit_fusion_logistic` |
| R57 | `stacking.py` | toujours là : la fusion n'apprend que sur des scores hors-pli |
| R60 | `finetune.py` | LoRA sur les couches hautes (`finetune.lora.layers`), à écrire |
| R66 | `dann.py` | tête `dann` ; mécanique ici |
| R67 | `head.fit_multiclass` | tête `multiclass` ; classes : `window_classes` (ici) |
| R70 | `evaluate.to_recordings` | le maximum, déjà le défaut |
| R73 | `service.train_and_register` | `decision.prevalence` ; `threshold_at_prevalence`, `platt` |
| R78 | `anuraset.run_anuraset_heads` | `blanci anuraset-heads`, un pli par site |
| R85 | `gated.py` | tête `gated` |

Écartées au tri des 26 et 27/09 : R33, R48, R51 (remplacée par R52), R58, R68, R69, R71, R72,
R82 (DECISIONS n° 116, 125–127). Plus tard : R83.

R19 et R20 lisent le stock d'embeddings entier du micro (`store_domain_statistics`) : aucune
étiquette, ce que la chaîne aura aussi sur un nouveau site. Elles ne valent que pour
l'embedding par défaut (pas pour les jetons résumés `logistic:<pooling>`). R18 et R21 sont
ajustées dans chaque pli sur les seules fenêtres d'entraînement ; le C des têtes logistiques est
ensuite choisi sur ces fenêtres transformées.

R37 (DECISIONS n° 116) : une colonne par micro de l'entraînement, non standardisée ; son poids
est le biais du micro, d'a priori N(0, σ²) (σ = `scale`, en logit, indépendant de C : voir
`head.standardize`). Un micro absent de l'entraînement (le micro du pli de test, un nouveau
site) a toutes ses colonnes à 0 : biais commun. Le biais absorbe le niveau de chaque micro
pendant l'apprentissage, w n'a plus à le coder. Sur données simulées, un σ petit (biais « très
pénalisés ») laissait le raccourci dans w : le mécanisme dépend de σ, à choisir sur la base
complète (`logistic+R37=0.3`, `=1`, `=3`, `=10`) ; σ = 3 est un défaut provisoire (n° 119).

R37 en GLMM (DECISIONS n° 131) : `logistic+R37=glmm` estime σ sur les données au lieu de le
fixer — la variance de l'effet aléatoire « micro » d'un modèle linéaire généralisé mixte,
choisie par vraisemblance marginale approchée (Laplace, `glmm_evidence`) parmi `glmm_grid`.
`R37.site_scale` ajoute un niveau site (biais de site + biais de micro dans son site, emboîtés) :
un micro nouveau d'un site connu hérite du biais de son site. Ce qui est lissé vers la moyenne,
ce sont les **biais de la tête** (le niveau de score propre à chaque micro), pas une probabilité
de présence : la tête reste un détecteur, fenêtre par fenêtre.
"""

from __future__ import annotations

from blanci.regularization.assembly import (
    Context,
    Regularizer,
    regularizer_for,
)
from blanci.regularization.classes import (
    CLASS_OF_LABEL,
    POSITIVE_CLASS,
    merge_rare_classes,
    window_classes,
)
from blanci.regularization.decision import (
    fold_platt,
    operating_curve,
    platt,
    precision_at_prevalence,
    prior_shift,
    threshold_at_prevalence,
)
from blanci.regularization.fusion_logistic import (
    FUSION_PARAMETER,
    FUSION_SUFFIXES,
    DescriptorBasis,
    choose_fusion_C,
    fit_constrained_logistic,
    fit_descriptor_basis,
    fit_fusion_logistic,
    fusion_settings,
    fusion_variant,
    is_descriptor,
)
from blanci.regularization.glmm import (
    GLMM_GRID,
    bias_sigmas,
    glmm_evidence,
    glmm_scales,
    group_bias_scale,
)
from blanci.regularization.names import (
    BAGGED_HEADS,
    C_HEADS,
    CONTEXT_HEADS,
    DESCRIPTIONS,
    GROUP_BIAS_HEADS,
    IMPLEMENTED,
    MAIN_PARAMETER,
    PENALIZED_HEADS,
    PSEUDO_HEADS,
    TORCH_REGULARIZATIONS,
    WEIGHTED_HEADS,
    WORD_VALUES,
    canonical,
    head_name,
    needs_domain,
    needs_pool,
    parse_head,
    validate,
)
from blanci.regularization.neighbors import (
    nearest_similarity,
)
from blanci.regularization.selection import (
    SELECTION_RULES,
    Bagged,
    bagged,
    bootstrap_weights,
    configure,
    cross_fitted_threshold,
    fine_grid,
    fold_ids,
    grouped_search,
    pick,
    pseudo_positives,
    sample_pool,
    selection_estimate,
    usable_folds,
    with_pseudo_labels,
)
from blanci.regularization.torch_training import (
    WeightAverage,
    attention_entropy,
    dann_strength,
    distillation_loss,
    dropout,
    fit_with_options,
    grad_reverse,
    keep_mask,
    l2_sp_model_penalty,
    l2_sp_penalty,
    layerwise_lr_groups,
    logistic_start,
    optimise,
    snapshot,
    unfreeze_top,
    unfreezing_schedule,
    validation_criterion,
)
from blanci.regularization.windows import (
    DomainStats,
    bias_indicators,
    domain_statistics,
    group_indicators,
    mean_directions,
    nuisance_directions,
    sample_weights,
    site_of,
    store_domain_statistics,
)

__all__ = [
    "attention_entropy",
    "Bagged",
    "bagged",
    "BAGGED_HEADS",
    "bias_indicators",
    "bias_sigmas",
    "bootstrap_weights",
    "C_HEADS",
    "canonical",
    "choose_fusion_C",
    "CLASS_OF_LABEL",
    "configure",
    "Context",
    "CONTEXT_HEADS",
    "cross_fitted_threshold",
    "dann_strength",
    "DESCRIPTIONS",
    "DescriptorBasis",
    "distillation_loss",
    "domain_statistics",
    "DomainStats",
    "dropout",
    "fine_grid",
    "fit_constrained_logistic",
    "fit_descriptor_basis",
    "fit_fusion_logistic",
    "fit_with_options",
    "fold_ids",
    "fold_platt",
    "FUSION_PARAMETER",
    "fusion_settings",
    "FUSION_SUFFIXES",
    "fusion_variant",
    "glmm_evidence",
    "GLMM_GRID",
    "glmm_scales",
    "grad_reverse",
    "GROUP_BIAS_HEADS",
    "group_bias_scale",
    "group_indicators",
    "grouped_search",
    "head_name",
    "IMPLEMENTED",
    "is_descriptor",
    "keep_mask",
    "l2_sp_model_penalty",
    "l2_sp_penalty",
    "layerwise_lr_groups",
    "logistic_start",
    "MAIN_PARAMETER",
    "mean_directions",
    "merge_rare_classes",
    "nearest_similarity",
    "needs_domain",
    "needs_pool",
    "nuisance_directions",
    "operating_curve",
    "optimise",
    "parse_head",
    "PENALIZED_HEADS",
    "pick",
    "platt",
    "POSITIVE_CLASS",
    "precision_at_prevalence",
    "prior_shift",
    "PSEUDO_HEADS",
    "pseudo_positives",
    "Regularizer",
    "regularizer_for",
    "sample_pool",
    "sample_weights",
    "selection_estimate",
    "SELECTION_RULES",
    "site_of",
    "snapshot",
    "store_domain_statistics",
    "threshold_at_prevalence",
    "TORCH_REGULARIZATIONS",
    "unfreeze_top",
    "unfreezing_schedule",
    "usable_folds",
    "validate",
    "validation_criterion",
    "WeightAverage",
    "WEIGHTED_HEADS",
    "window_classes",
    "with_pseudo_labels",
    "WORD_VALUES",
]
