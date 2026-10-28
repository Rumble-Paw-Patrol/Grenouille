# Régularisations (R1–R85)

Liste des régularisations envisagées, rangées par type, avec leur sort. Numéros stables : ce sont ceux du code, de la config et de `DECISIONS.md`. Mécanique et index côté code : `blanci/heads/regularization/` (`__init__.py`). État en image : `documentation/tableaux/regularisations.png`. Aucune conclusion sur leur efficacité tant que la base complète n'est pas accessible (DECISIONS n° 119).

Conservée : **Oui** = programmée ou en place · **Écartée** = refusée · **En attente** = outil prêt ou accordée, pas encore utilisée · **Pas encore** = ni programmée ni écartée.

## Augmentation des données (avant l'encodeur)

| R | Description | Conservée | Où |
|---|---|---|---|
| R1 | Mélange de fond : un positif collé sur le fond d'autres micros ou sites, SNR aléatoire | Pas encore (prévue pour le modèle maison) | |
| R2 | Mixup / CutMix audio, labels souples | Pas encore | |
| R3 | Variation de gain aléatoire (simule l'éloignement) | Pas encore | |
| R4 | Décalage temporel de la note dans la fenêtre | Pas encore | |
| R5 | Décalage de hauteur et étirement temporel légers (±2–3 %) | Pas encore | |
| R6 | SpecAugment (masques temps et fréquence), utile seulement sur un modèle entraîné | Pas encore | |
| R7 | Réverbération, filtre passe-bas (distance, végétation) | Pas encore | |
| R8 | Ajout de pluie ou de bruit réel | Pas encore | |
| R9 | Le micro 18 dB comme deuxième vue du même événement | Pas encore | |
| R10 | Test-time augmentation : moyenne des scores sur plusieurs décalages ou canaux | Pas encore | |

## Labels et pondération des exemples

| R | Description | Conservée | Où |
|---|---|---|---|
| R11 | Lissage des labels (0,05 / 0,95) | Pas encore | |
| R12 | Labels souples selon la confiance (« second plan », « lointain ») | Pas encore | |
| R13 | Chaque micro pèse autant dans sa classe | Oui | `regularization/windows.py`, suffixe `+R13` |
| R14 | Plafond de k fenêtres positives par enregistrement | Pas encore | |
| R15 | Négatifs annotés (faux amis, espèces) surpondérés face aux présumés | Oui | `regularization/windows.py`, suffixe `+R15` |
| R16 | Ratio de négatifs (1:20 à 1:50) | Pas encore | |
| R36 | Poids des classes réglable (équilibré par défaut, 0 = aucun) | Oui | `regularization/windows.py`, suffixe `+R36` |

## Représentation (embeddings)

| R | Description | Conservée | Où |
|---|---|---|---|
| R17 | Normalisation L2 des embeddings | Oui | `regularization/`, suffixe `+R17` |
| R18 | ACP avant la tête (16, 32, 64 composantes) | Oui | `regularization/`, suffixe `+R18=k` ; graphe : `blanci pca` |
| R19 | Centrage par micro (− moyenne du stock du micro, sans labels) | Oui | `regularization/windows.py`, suffixe `+R19` |
| R20 | AdaBN : centrage et réduction par micro | Oui | `regularization/windows.py`, suffixe `+R20` |
| R21 | Retrait des directions qui trahissent le micro (INLP) | Oui | `regularization/`, suffixe `+R21` |
| R22 | Pooling des jetons : moyenne, max, GeM… | Oui | `blanci/heads/pooling.py`, tête `logistic:gem` |
| R23 | Choix de la couche de l'encodeur (couches intermédiaires) | Pas encore (coût de stockage, `documentation/encodeurs-bacpipe.md`) | |
| R24 | Concaténation de plusieurs encodeurs | Pas de régularisation propre ; benchmark des ensembles à part | |
| R25 | Quantification ou binarisation des embeddings | Pas encore | |

## Tête linéaire : pénalités et pertes

| R | Description | Conservée | Où |
|---|---|---|---|
| R26 | L2 (ridge), C choisi par validation groupée | Oui | `blanci/heads/head.py` (`fit_logistic`) |
| R27 | L1 (lasso) | Oui | suffixe `+R27` |
| R28 | Elastic Net | Oui | suffixe `+R28` |
| R29 | Group lasso (un bloc par encodeur ou par groupe de descripteurs) | Pas encore | |
| R30 | Rétrécissement des poids vers le prototype différentiel | Oui | `head.py`, tête `logistic_to_prototype` |
| R31 | LDA à covariance rétrécie (Ledoit-Wolf) | Oui | `head.py`, tête `lda_shrunk` |
| R32 | Logistique bayésienne (incertitude par fenêtre) | Pas encore | |
| R33 | Contrainte de norme maximale | Écartée (équivalente à la L2) | |
| R34 | SVM linéaire à marge large | Oui | `blanci/heads/losses.py`, `--methods losses` |
| R35 | Pertes robustes (focale, tronquée, symétrique) | Oui | `losses.py`, `--methods losses` |
| R39 | kNN : choix de k, recherche par l'exemple | Oui | `head.py`, `knn:k=…`, `--methods neighbors` |

## Effets de site et de micro

| R | Description | Conservée | Où |
|---|---|---|---|
| R37 | Biais par micro très pénalisé (σ à choisir) ; variante GLMM : σ estimé sur les données | Oui | `regularization/glmm.py`, `+R37=σ`, `+R37=glmm` |
| R38 | Modèle mixte complet : écarts de w par micro | En attente (après une ACP) | |

## Têtes entraînées avec torch (attentive, gated, dann, multiclass)

| R | Description | Conservée | Où |
|---|---|---|---|
| R40 | Weight decay choisi par validation groupée | Oui | `regularization/torch_training.py` |
| R41 | AdamW (weight decay découplé) | Oui | idem |
| R42 | Arrêt précoce sur validation groupée (`=ap`, `=loss`) | Oui | idem |
| R43 | Entropie de l'attention | Oui | idem, `attention_entropy` |
| R44 | Température de l'attention | Pas encore | |
| R45 | Dropout des jetons | Oui | idem |
| R46 | Dropout des dimensions | Oui | idem |
| R47 | Rétrécissement vers la logistique (L2-SP) | Oui | idem |
| R48 | Rang faible de l'attention | Écartée (R85 porte déjà le rang faible) | |
| R49 | Attention multi-têtes à paramètres partagés | Pas encore | |
| R59 | Warm-up et écrêtage du gradient | Oui | idem |
| R64 | Moyenne des poids (`=ema`, `=swa`) | Oui | idem |
| R66 | Inversion du gradient contre le micro (DANN) | Oui | `blanci/heads/dann.py` |
| R67 | Tête multi-classes (blanci, congénères, faux amis, bruit) | Oui | `head.fit_multiclass`, tête `multiclass` |
| R67 bis | Sous-classes positives (solo / chœur), qualité A/B/C en tâches auxiliaires | Pas encore (il faut ≥ 30 enregistrements par sous-classe) | |
| R85 | Sonde à portes : poids par dimension dépendant de la fenêtre, rang faible | Oui | `blanci/heads/gated.py`, tête `gated` |

## Fine-tuning, distillation, modèle maison

| R | Description | Conservée | Où |
|---|---|---|---|
| R60 | Rang du LoRA, couches adaptées | En attente (rang réglable, fine-tuning à écrire) | `finetune.py`, `finetune.lora.rank` |
| R61 | Gel progressif, taux d'apprentissage par couche | En attente (outil prêt) | `regularization/torch_training.py` |
| R62 | L2-SP : pénalité vers les poids pré-entraînés | En attente (prête ; déjà utilisée par R47) | idem |
| R63 | Distillation (labels souples, température) | En attente (perte prête) | idem, `distillation_loss` |
| R65 | Pré-entraînement sur AnuraSet puis ajustement | En attente (accordée avec le modèle maison et le LoRA) | |

## Fusion et empilement

| R | Description | Conservée | Où |
|---|---|---|---|
| R50 | C de la fusion choisi par validation groupée | Oui | `regularization/fusion_logistic.py`, `logistic+R50` |
| R51 | Contraintes de signe sur les coefficients | Remplacée par R52 | |
| R52 | Coefficients ≥ 0 sur les entrées orientées (combinaison convexe pour `weighted`, `weight_grid`) | Oui | `fusion_logistic.py`, `logistic+R52` |
| R53 | Sélection L1 parmi les descripteurs | Oui | `fusion_logistic.py`, `logistic+R53` |
| R54 | Descripteurs en marches monotones (5 classes) | Oui | `fusion_logistic.py`, `logistic+R54` |
| R55 | GAM par P-splines (courbe lisse) | Oui | `fusion_logistic.py`, `logistic+R55` |
| R56 | Pas de veto : plafond sur le poids des descripteurs séquentiels | Oui | `fusion_logistic.py`, `logistic+R56` |
| R57 | Score de niveau 1 hors-fold pour l'empilement | Oui (toujours actif) | `blanci/combination/stacking.py` |
| R58 | Phénologie comme a priori bayésien | Écartée | |

## Post-traitement (enregistrement, point, temps)

| R | Description | Conservée | Où |
|---|---|---|---|
| R68 | Lissage temporel des scores | Écartée (efface les faux négatifs suspects, utiles au « positive mining ») | |
| R69 | HMM / CRF sur la séquence de fenêtres | Écartée (même raison) | |
| R70 | Agrégation par enregistrement (le maximum) | Oui (défaut de l'évaluation) | `blanci/evaluation/`, `to_recordings` |
| R71 | Lissage entre créneaux voisins du même micro | Écartée | |
| R72 | Modèle d'occupation hiérarchique par point | Écartée | |
| R73 | Calibration (Platt) et seuil selon la prévalence | Oui | `regularization/decision.py`, `decision.prevalence`, `blanci prevalence` |

## Réglage et sélection des hyperparamètres

| R | Description | Conservée | Où |
|---|---|---|---|
| R74 | Validation imbriquée : seuil et variante jugés à part | Oui | `regularization/selection.py` |
| R75 | Règle du « 1 écart-type » (défaut) | Oui | `selection.py`, `head.selection_rule` |
| R76 | Grille de C plus fine, chemin de régularisation | Oui | `selection.py`, suffixe `+R76` |
| R77 | Leave-one-micro-out | Oui | `head.n_splits: lomo` |
| R78 | Réglage visant la généralisation entre sites (un fold par site sur AnuraSet) | Oui | `blanci anuraset heads` |
| R79 | Bagging par tirage bootstrap des micros | Oui | `selection.py`, suffixe `+R79` |
| R80 | Limiter les comparaisons (présélection des variantes, jeu gelé) | Oui (règle de travail) | |

## Semi-supervisé et apprentissage actif

| R | Description | Conservée | Où |
|---|---|---|---|
| R81 | Pseudo-étiquetage (seuil haut, poids 0,3) | Oui | `selection.py`, suffixe `+R81` |
| R82 | Cohérence entre augmentations (Mean Teacher, FixMatch) | Écartée | |
| R83 | Minimisation d'entropie sur les nouveaux sites | En attente (« plus tard ») | |
| R84 | Part d'aléatoire dans la file d'annotation 60-20-20 | Oui | `blanci/annotation/selection.py` |
