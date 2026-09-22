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

# Les notes annotées tiennent-elles entières dans les fenêtres des grilles 3 s et 5 s ?
$B check-grid
$B status

# --- Embeddings et choix d'encodeur (§2) ---------------------------------
uv run blanci embed --encoder birdmae --peak-hours    # reprenable
uv run blanci benchmark --encoders birdmae-1,beats-1  # → data/reports/benchmark.md

# --- Détection (§1, §5) --------------------------------------------------
uv run blanci train --encoder birdmae-1               # tête + seuil à précision ≥ 0,1
uv run blanci score --encoder birdmae-1               # décisions + points classés
uv run blanci queue --encoder birdmae-1 --n 40        # file de vérification 60/20/20
uv run blanci search --encoder birdmae-1 --site tresor --k 300   # récolte de positifs
uv run blanci label <window_id> --label blanci_solo --source active

# --- Évaluation (§6) -----------------------------------------------------
uv run blanci evaluate --encoder birdmae-1                       # plis par micro
uv run blanci evaluate --encoder birdmae-1 --holdout tresor,kaw  # sites tenus à l'écart
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

## Avancement

| Jalon | État |
|---|---|
| M0 dépôt, config, `ingest`, `import-labels` | **accepté sur données réelles** le 22/09 (DECISIONS n° 49) |
| M1 `embed`, `benchmark` en plis par micro | écrits, testés, branchés sur la CLI ; **adaptateur bacpipe non validé** |
| M2 `head`, `search`, `queue`, prototype Streamlit | `train`, `score`, `queue`, `search` en service et en CLI ; GUI Streamlit à faire |
| M3 `sequential`, `fusion`, `aggregate`, audit aléatoire | modules écrits et testés ; `aggregate` branché, `sequential`/`fusion` pas encore |

Acceptation M0 : 29 513 enregistrements (980 h, 5 relevés) inventoriés, 345 positifs et
150 négatifs importés, aucune annotation coupée par les grilles 3 s et 5 s. Les 345 positifs
viennent de 51 enregistrements et 13 micros, tous à Mataroni (DECISIONS n° 35).

410 tests passent sur Python 3.11 (`uv run pytest`). Tous les modules sont couverts sauf
`encoders/bacpipe_encoder.py`, `encoders/onnx_encoder.py` et `encoders/export.py`, qui
demandent respectivement bacpipe (groupe `research`) et un modèle exporté.

**Reste à faire avant M1** : écarter les enregistrements hors campagne (durée ≠ 2 min, dates
hors relevé) ; décider du canal audio (stéréo, gains 6 et 18 dB, DECISIONS n° 44) ; valider
l'adaptateur bacpipe contre la bibliothèque installée.
