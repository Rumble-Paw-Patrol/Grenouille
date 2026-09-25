# blanci — détection acoustique d'*Anomaloglossus blanci*

Stage ONF Guyane, 15/09/2026 → 14/03/2027. Feuille de route :
`documentation/feuille-de-route-V4.md` (§13 = spécification d'implémentation).
Écarts et précisions : `DECISIONS.md`.

## Installation

```sh
uv sync          # Python 3.11, dépendances de base + outils de dev
uv run pytest
uv run blanci --help
```

## Données

L'audio reste sur le disque externe, jamais modifié. Sa racine se déclare dans
`config/local.yaml` (copie de `config/local.example.yaml`, ignorée par git) ; les chemins sont
stockés relatifs à cette racine, ils survivent à un changement de lettre ou de machine.

```
<disque>/Projet blanci 2025/RELEVE 3 Mataroni - 06-13 janv 2026/2LA04530_MGM06/Data/
    2LA04530_20260106_103000.wav      # micro 2LA04530, 6 janvier 2026, 10 h 30 locales
```

Enregistrements de 2 min, 48 kHz stéréo. Le micro est le numéro de série (GUANO, sinon
préfixe du nom), l'heure vient du GUANO avec le fuseau de l'enregistreur. WAV et FLAC sont
inventoriés ; l'appariement avec les annotations se fait sur le nom sans extension.

Annotations : l'export Excel de Blancinet v0.1.0 (une ligne par détection de 3 s, clé S3,
score, vérification `True` / `False`, commentaires). Seules les lignes vérifiées deviennent des
labels.

Hors git, sous `data/` : `db/blanci.sqlite` (inventaire, fenêtres, labels en ajout seul),
`embeddings/<encodeur>/<jeu>/<site>/<aaaamm>.parquet`, `models/`, `reports/`.

## Commandes

Toutes passent par `blanci/service.py`, que la future GUI appellera de la même façon (§4).

```sh
B="uv run blanci --config config/local.yaml"

# --- Données -------------------------------------------------------------
# Inventaire, relevé par relevé, DANS L'ORDRE CHRONOLOGIQUE : les cartes SD d'un relevé
# contiennent encore les fichiers du précédent ; le premier inventorié garde le fichier.
# --no-qc --no-hash : en-têtes seuls, le disque entier en quelques minutes.
$B ingest --dataset 2026 --site CDR --no-qc --no-hash "D:/Projet blanci 2025/RELEVE 1 CDR - 21-27 décembre 2025"
$B ingest --dataset 2026 --site Mataroni --no-qc --no-hash "D:/Projet blanci 2025/RELEVE 3 Mataroni - 06-13 janv 2026"

# Import des annotations ; --dry-run d'abord pour relire verdicts, espèces et lignes signalées
$B import-labels documentation/All_detections_blancinet_v0.1.0_dataset1BV.xlsx --dry-run
$B import-labels documentation/All_detections_blancinet_v0.1.0_dataset1BV.xlsx
# Toutes les détections Blancinet, comme scores (pas comme labels) : comparables aux nôtres
$B import-detections documentation/All_detections_blancinet_v0.1.0_dataset1BV.xlsx
$B export-labels     # fenêtres annotées : label, qualité, espèce, commentaire

# Les notes annotées tiennent-elles entières dans les fenêtres des grilles 3 s et 5 s ?
$B check-grid
# Drapeaux (remarques sur un enregistrement, DECISIONS n° 79) : écartent du corpus
# silencieux, micro dans sac, durée anormale, hors relevé ; pluie et saturation restent.
# Recalcule inventaire, audio (seuils actuels, sans relire l'audio) et drapeaux d'écoute.
$B flag
$B status

# --- Baselines sans encodeur (§3) : lit l'audio, n'encode rien -------------
$B baselines --channels 0,1                          # → data/reports/baselines.md

# --- Annotation (§5) : uv sync --group app -------------------------------
$B candidates --from documentation/All_detections_blancinet_v0.1.0_dataset1BV.xlsx    --per-site 30 --random 12 --name lot1             # → data/reports/candidats_lot1.csv
$B annotate                                          # poste d'écoute dans le navigateur

# Enregistrements entiers (audit aléatoire, jeu gelé) ; accord entre deux annotateurs
$B candidates --entiers 300 --sites Mataroni --reason audit_aleatoire --random 0 --name audit
$B agreement --annotators léonard,tuteur

# --- Relevé des encodeurs (§2, §7) : bruit synthétique, rien n'est lu -----
$B throughput --encoders birdnet,beats,perch_v2,birdmae_base   # → debit.md

# --- Embeddings et choix d'encodeur (§2) : uv sync --group research -------
uv run blanci embed --encoder birdmae --subset benchmark   # annotés + négatifs appariés
# (contrôle audio au passage : silencieux et micro dans sac écartés ; --no-qc pour s'en passer)
uv run blanci embed --encoder birdmae --peak-hours    # reprenable
$B cluster --encoder birdmae-bacpipe1.3.5 --mode c1   # clustering C0/C1 (§5 bis)
$B candidates --congeners perch_v2-bacpipe1.3.5       # logits des congénères de Perch
uv run blanci benchmark --encoders birdmae-1,beats-1  # → data/reports/benchmark.md

# --- Pré-benchmark AnuraSet (§2) : base et stocks à part -----------------
A="uv run blanci --config config/anuraset.yaml"
$A anuraset-prepare && $A anuraset-profile            # extraction, inventaire, espèces
$A embed --encoder perch_v2 && $A anuraset-benchmark --encoders perch_v2-bacpipe1.3.5

# --- Détection (§1, §5) --------------------------------------------------
uv run blanci train --encoder birdmae-1               # tête + seuil à précision ≥ 0,1
uv run blanci score --encoder birdmae-1               # tête adoptée ; décisions, points
uv run blanci queue --encoder birdmae-1 --n 40        # file de vérification 60/20/20
uv run blanci search --encoder birdmae-1 --site tresor --k 300   # récolte de positifs
uv run blanci label <window_id> --label blanci_solo --source active
$B fusion --encoder birdmae_base-bacpipe1.3.5        # tête + rythme + persistance (§3)
$B score --encoder birdmae_base-bacpipe1.3.5 --fusion
$B tokens --encoder perch_v2                         # jetons pour la sonde attentive
$B qc-calibrate                                      # seuils QC mesurés, config inchangée
# Réentraînement (ONF, M5) : nouvelle tête jugée sur le jeu gelé, adoptée si pas moins bonne
$B retrain --encoder birdmae_base-bacpipe1.3.5

# --- Variantes de la chaîne (DECISIONS n° 88–90, 101–103) -----------------
# Négatifs appariés : benchmark.pairing = nearest (défaut) | other_day | same_day | mixed
$B embed --encoder birdmae --subset benchmark --overlap 0.75   # stock birdmae-…@o75
$B embed --encoder birdmae --subset benchmark --upstream bandpass      # stock birdmae+bp3-7k-…
$B upstream-bench                                    # portes : arrêtées / positifs perdus
$B upstream-bench --encoder birdmae-bacpipe1.3.5 --upstream notes,rhythm

# --- Benchmarks (DECISIONS n° 91–98) : plis communs, scores hors-pli rangés -
$B heads --encoder perch_v2-bacpipe1.3.5             # toutes les têtes, poolings, cascade
$B heads-curve --encoder perch_v2-bacpipe1.3.5       # différentiel ou linear probe, selon k
$B heads --encoder perch_v2-bacpipe1.3.5 --methods logistic,logistic+R19,logistic+R18=32  # régularisations (n° 108)
$B fusion-bench --encoder birdmae-bacpipe1.3.5 --sources head:perch_v2-bacpipe1.3.5
$B ensemble --sources birdmae-bacpipe1.3.5/logistic,perch_v2-bacpipe1.3.5/logistic
$B detector-bench --detector band_contrast           # distilled, homemade : réservés
$B sources                                           # ce que le stock hors-pli contient
$B benchmark-all --external blancinet                # tout, mêmes enregistrements

# --- Sélection des candidats (DECISIONS n° 99–100) -------------------------
$B select --method active --encoder birdmae-bacpipe1.3.5 --n 40 --mix 0.4,0.2,0.4
$B select --method negative_mining --encoder birdmae-bacpipe1.3.5 --mode false_friends
$B select --method cluster --encoder birdmae-bacpipe1.3.5 --n 10   # puis cluster-status
$B select --method gaps --encoder birdmae-bacpipe1.3.5   # trous : faux négatifs suspects
$B select --method gaps --mode labels                     # négatifs annotés à réécouter
$B cluster-label --encoder birdmae-bacpipe1.3.5 --cluster 7        # groupe homogène
$B yapat-export data/reports/candidats_active.csv    # extraits + manifeste pour YAPAT
$B annotate                                          # mode de sélection, carte, « Envoyer »

# --- Évaluation (§6) -----------------------------------------------------
uv run blanci evaluate --encoder birdmae-1                       # plis par micro
uv run blanci evaluate --encoder birdmae-1 --holdout tresor,kaw  # sites tenus à l'écart
$B freeze data/reports/candidats_gele.csv --version v1   # jeu gelé : jamais entraîné
$B evaluate --encoder birdmae-1 --frozen last            # la tête jugée sur le jeu gelé
# Courbes d'activité (M4) contre les patrons de Courtois et al. 2025 ; courbes numérisées
# en option (CSV [site,] hour, value / [site,] month, value)
$B activity --encoder birdmae-1 --dataset 2023 --reference-hours ref_heures.csv
```

Les colonnes du fichier d'annotations sont reconnues automatiquement, y compris sous forme
de libellé composé (« Nom de l'enregistrement ») : fichier, timecode, vérification manuelle,
score, commentaire, qualité, site, micro. Si une colonne n'est pas trouvée, ajouter son nom
dans `labels.import.columns` de la config.

La colonne de vérification tranche positif/négatif : `True` / `False`, oui/non et leurs
variantes, réponses rédigées mentionnant *blanci*, ou nom d'un faux ami connu (qui donne
aussi l'espèce). Une cellule vide est une détection jamais écoutée : ignorée, ce n'est pas un
label. « à vérif » / « à conf » sont mises de côté et listées. Toute autre valeur non reconnue
**bloque l'import** au lieu d'être rangée en négatif ; `--dry-run` les liste toutes. Chaque
ligne doit désigner un enregistrement déjà inventorié.

## Notebooks d'exploration

`uv sync --inexact --group notebook` (`--inexact` garde les groupes research et app déjà
installés), puis ouvrir dans VS Code avec le noyau du dépôt (`.venv`). La base est ouverte en
lecture seule, l'audio des disques est lu, jamais écrit ; tout se règle dans la cellule
« Réglages » (`config/local.yaml` pour le disque externe).

| Notebook | Pour |
|---|---|
| `notebooks/01_explorer_une_fenetre.ipynb` | un enregistrement et une fenêtre à travers la chaîne : écoute, grille, portes, module séquentiel (amont, parallèle, aval), négatif apparié, embeddings, prototype différentiel |
| `notebooks/02_negatifs_apparies.ipynb` | ce que contiennent les négatifs appariés de chaque stratégie ; feuille d'écoute et taux de contamination |
| `notebooks/03_module_sequentiel.ipynb` | régler la détection des notes et les seuils des portes (banc d'essai amont sans encodeur) |

**Ne jamais committer les sorties** : les lecteurs audio embarquent le son des
enregistrements (*Clear All Outputs* avant un commit ; `tests/test_notebooks.py` le vérifie).

## Avancement

| Jalon | État |
|---|---|
| M0 dépôt, config, `ingest`, `import-labels` | **accepté sur données réelles** le 22/09 (DECISIONS n° 49) |
| M1 `embed`, `benchmark` en plis par micro | écrits, testés, branchés sur la CLI ; baselines sans encodeur mesurées (DECISIONS n° 62) ; **aucun encodage lancé** |
| M2 `head`, `search`, `queue`, prototype Streamlit | `train`, `score`, `queue`, `search` en service et en CLI ; poste d'annotation Streamlit (`annotate`, `candidates`) |
| M3 `sequential`, `fusion`, `aggregate`, audit aléatoire | branchés : `onsets`, `fusion`, `score --fusion` ; audit par `candidates --entiers` |
| M4 jeu gelé, `evaluate --holdout`, patrons 2023 | `freeze`, `evaluate --frozen`, `activity` écrits ; en attente du jeu gelé et de l'inventaire 2023 |
| M5 export ONNX, réentraînement sans intervention | `retrain` (adoption jugée sur le jeu gelé) écrit ; export ONNX après le choix d'encodeur |
| Ajouts du 24/09 (DECISIONS n° 88–100) | négatifs appariés à trois stratégies, chevauchement réglable, seuillage en amont activable, plis communs et scores hors-pli rangés, toutes les têtes et courbe selon les annotations, fusion à N entrées, emplacement du module séquentiel, ensembles, benchmark complet, outil de sélection, poste d'annotation refondu ; emplacements réservés : distillation, modèle maison, LoRA |

Acceptation M0 : 29 513 enregistrements (980 h, 5 relevés) inventoriés, 345 positifs et
150 négatifs importés, aucune annotation coupée par les grilles 3 s et 5 s. Les 345 positifs
viennent de 51 enregistrements et 13 micros, tous à Mataroni (DECISIONS n° 35).

643 tests passent sur Python 3.11 (`uv run pytest`). Tous les modules sont couverts sauf
`encoders/onnx_encoder.py` et `encoders/export.py`, qui demandent un modèle exporté ;
les neuf encodeurs bacpipe du §2 sont installés et mesurés (DECISIONS n° 64, 72).

Enregistrements de test et hors relevé signalés (149, jamais encodés, DECISIONS n° 51) ;
drapeaux posés à l'écoute sur les 131 enregistrements annotés (8 micro dans sac, 5 pluie,
DECISIONS n° 79) ;
encodage sur le premier micro (gain 6 dB), les deux micros à l'écoute (DECISIONS n° 50, 58).

Négatifs appariés changés (DECISIONS n° 88, 101 : les plus proches, même enregistrement
compris) : **relancer les baselines** avant de comparer quoi que ce soit.

**Reste à faire avant M1** : regrouper les enregistrements sur le disque du stage puis
réinventorier (les labels suivent, DECISIONS n° 46) ; inventorier la phénologie 2023-2024 ;
encoder (ce week-end) les données ONF et AnuraSet.
