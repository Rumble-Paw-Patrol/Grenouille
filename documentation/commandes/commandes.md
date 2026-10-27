# Commandes

`uv run blanci [--config config/local.yaml] <commande>` ; `--help` donne les options de chacune.

## Données

- `ingest` : inventaire des enregistrements d'un dossier
- `qc` : contrôle audio de tout le corpus avant l'encodage
- `qc-calibrate` : mesure les seuils du contrôle qualité
- `flag` : recalcule les drapeaux d'enregistrement
- `status` : résumé de la base (enregistrements, drapeaux, labels)
- `check-grid` : vérifie que les grilles de fenêtres gardent les notes entières
- `onsets` : débuts de notes par traitement du signal

## Annotation

- `candidates` : file d'écoute (plan, hasard, enregistrements entiers, drapeau)
- `annotate` : poste d'annotation dans le navigateur
- `label` : ajoute un label à une fenêtre
- `export-labels` : exporte ses annotations en CSV
- `import-labels` : recharge des annotations exportées ailleurs
- `paquet` : paquet d'écoute pour un encadrant sans GitHub ni disques
- `agreement` : accord entre deux annotateurs
- `select` : sélection de candidats (actif, similarité, clusters, trous, négatifs difficiles)
- `cluster` : clustering HDBSCAN des embeddings
- `cluster-status` : avancement des groupes
- `cluster-label` : étiquette un groupe en bloc
- `clips-export` : extraits WAV d'une file pour un outil externe
- `clips-import` : relit les réponses de cet outil

## Encodage

- `throughput` : débit, mémoire et dimension de chaque encodeur
- `embed` : extrait les embeddings (reprenable)
- `tokens` : jetons pour la sonde attentive
- `pca` : variance perdue selon le nombre de dimensions gardées

## Têtes, détection

- `train` : entraîne la tête logistique et calibre son seuil
- `score` : score toutes les fenêtres, décide par enregistrement
- `retrain` : réentraîne la tête, l'adopte si elle n'est pas moins bonne sur le jeu gelé
- `fusion` : fusion tête + rythme + persistance
- `prevalence` : part de positifs dans le stock
- `freeze` : gèle des enregistrements en jeu de test

## Benchmarks

- `baselines` : références sans encodeur
- `benchmark` : compare les encodeurs
- `heads` : compare les têtes
- `heads-curve` : performance selon le nombre d'annotations du site
- `fusion-bench` : compare les fusions
- `ensemble` : combine plusieurs modèles
- `sources` : contenu du stock de scores hors-pli
- `upstream-bench` : portes du traitement du signal en amont
- `detector-bench` : détecteurs audio
- `benchmark-all` : tout, sur les mêmes enregistrements
- `evaluate` : évaluation par sites tenus à l'écart ou plis par micro
- `activity` : courbes d'activité journalières et saisonnières

## AnuraSet (archivé)

- `anuraset prepare`, `anuraset profile`, `anuraset benchmark`, `anuraset heads`, `anuraset campaign`
