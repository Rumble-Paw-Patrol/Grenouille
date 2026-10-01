# Commandes et données

Référence détaillée de la ligne de commande `blanci` (une cinquantaine de commandes) et du
format des données. Le [README](../README.md) donne la vue d'ensemble et le démarrage rapide.
Les renvois « n° » désignent les décisions numérotées de [`DECISIONS.md`](../DECISIONS.md),
les « § » les sections de son cadre. Les chemins `data/` et `config/local.yaml` sont locaux,
ignorés par git.

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

### Inventaire (base au 29/09/2026)

| Jeu | Site | Enregistrements | dont 120 s | Heures (120 s) | Go |
|---|---|---:|---:|---:|---:|
| 2023 (phénologie, déc. 2023 → nov. 2024, 6 relevés) | Kaw | 22 659 | 22 650 | 755,0 | 521,9 |
| | Molokoi | 21 007 | 20 999 | 700,0 | 483,9 |
| | Trésor | 23 113 | 23 105 | 770,2 | 532,7 |
| | **total 2023** | **66 779** | **66 754** | **2 225,2** | **1 538,5** |
| 2026 (Projet blanci 2025) | CDR | 5 438 | 5 391 | 179,7 | 124,3 |
| | Mataroni | 13 059 | 12 974 | 432,5 | 299,4 |
| | Patawa Est | 2 624 | 2 621 | 87,4 | 60,4 |
| | Patawa Ouest | 1 107 | 1 099 | 36,6 | 25,4 |
| | RNRT | 7 285 | 7 280 | 242,7 | 167,8 |
| | **total 2026** | **29 513** | **29 365** | **978,9** | **677,3** |
| **Total** | | **96 292** | **96 119** | **3 204** | **2 216** |

Go : taille réelle des fichiers sur les disques (10⁹ octets), aucun fichier inventorié
manquant. 96 176 enregistrements sont en 48 kHz stéréo, les autres (116) en 24 ou 32 kHz.
Fenêtres potentielles, chevauchement de 50 % (`encoders.overlap`), sur les 96 119
enregistrements de 120 s :

| Fenêtre | Pas | Par enregistrement | Au total |
|---|---|---:|---:|
| 3 s | 1,5 s | 79 | 7 593 401 |
| 5 s | 2,5 s | 47 | 4 517 593 |
| 6 s | 3 s | 39 | 3 748 641 |

Écartés par les drapeaux (jamais encodés) : 173 de durée anormale, 88 hors relevé, 8 micros
dans le sac, 1 521 à l'horloge douteuse (Molokoi SMA14636, avril 2024, en file d'écoute) ;
94 588 enregistrements de 120 s restent encodables. Détail et anomalies : DECISIONS n° 153
et 154.

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
# Écartés par un drapeau, à écouter en entier pour juger ce qu'ils valent (source « flag »)
$B candidates --drapeau clock_off --random 0 --name horloge_SMA14636
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
$A anuraset-campaign --encoders perch_v2             # tout d'un coup : espèces, encodage, têtes (n° 136)

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
$B heads --encoder perch_v2-bacpipe1.3.5 --methods losses   # benchmark des pertes (n° 112)
$B heads --encoder perch_v2-bacpipe1.3.5 --methods neighbors  # k voisins, par similarité (R39, n° 115)
$B heads --encoder perch_v2-bacpipe1.3.5 --methods logistic,logistic+R36,logistic+R37  # n° 116
$B heads --encoder perch_v2-bacpipe1.3.5 --methods attentive,attentive+R41+R42  # n° 117
$B heads --encoder perch_v2-bacpipe1.3.5 --methods logistic,logistic+R79,logistic+R81,multiclass,dann  # n° 122–124
$A anuraset-heads --encoder perch_v2-bacpipe1.3.5     # têtes jugées un site à la fois (R78, n° 125)
$B prevalence                                        # part de positifs au hasard → decision.prevalence (n° 128)
$B heads --encoder perch_v2-bacpipe1.3.5 --methods logistic+R37,logistic+R37=glmm  # σ du biais estimé (GLMM, n° 131)
$B pca --encoder perch_v2-bacpipe1.3.5               # variance perdue selon les dimensions gardées (n° 132)
# Rapports : AP poolée et AP moyenne par pli ; benchmark.fold_calibration: platt recalibre chaque pli (n° 135)
$B fusion-bench --encoder birdmae-bacpipe1.3.5 --sources head:perch_v2-bacpipe1.3.5
$B fusion-bench --encoder birdmae-bacpipe1.3.5         # dont logistic+R50+R54, +R50+R55 (n° 130)
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
