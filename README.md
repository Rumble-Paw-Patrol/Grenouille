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

Enregistrements de 2 min, nommés `<micro>_<AAAAMMJJ>_<HHMMSS>` — par exemple
`2la04530_20260106_103000.wav` : micro `2LA04530`, 6 janvier 2026 à 10 h 30 locales
(UTC−3). WAV et FLAC sont inventoriés tous les deux ; l'appariement avec le fichier
d'annotations se fait sur le nom **sans extension**.

Annotations : un tableau Excel, une ligne par fenêtre de 3 s, avec le nom de
l'enregistrement, le timecode, le score de l'ancien prestataire et la vérification manuelle.

Hors git, sous `data/` (chemins dans `config/default.yaml`) :

```
data/raw/{2023,2026}/<site>/<micro>/*.{wav,flac}   # audio brut, jamais modifié
data/labels/imports/                        # fichiers d'annotation reçus, jamais modifiés
data/db/blanci.sqlite                       # inventaire, fenêtres, labels (ajout seul)
data/embeddings/<encodeur>/<jeu>/<site>/<aaaamm>.parquet
data/reports/                               # rapports d'erreurs d'inventaire
```

## Commandes

Toutes passent par `blanci/service.py`, que la future GUI appellera de la même façon (§4).

```sh
# --- Données -------------------------------------------------------------
# Inventaire + contrôle qualité (reprenable : les fichiers déjà connus sont sautés)
uv run blanci ingest --dataset 2026

# Import des annotations ; --dry-run d'abord pour relire l'analyse des commentaires
uv run blanci import-labels data/labels/imports/positifs.csv --kind positive --dry-run
uv run blanci import-labels data/labels/imports/positifs.csv --kind positive
uv run blanci import-labels data/labels/imports/faux_amis.csv --kind negative

# Les notes annotées tiennent-elles entières dans les fenêtres des grilles 3 s et 5 s ?
uv run blanci check-grid
uv run blanci status

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

La colonne de vérification tranche positif/négatif : oui/non et leurs variantes, réponses
rédigées mentionnant *blanci*, ou nom d'un faux ami connu (qui donne aussi l'espèce). Une
valeur non reconnue (« à revoir ») **bloque la ligne** au lieu d'être rangée en négatif ;
`--dry-run` les liste toutes. Chaque ligne doit désigner un enregistrement déjà inventorié.

## Avancement

| Jalon | État |
|---|---|
| M0 dépôt, config, `ingest`, `import-labels` | code et tests écrits ; **acceptation à faire sur les données réelles** |
| M1 `embed`, `benchmark` en plis par micro | écrits, testés, branchés sur la CLI ; **adaptateur bacpipe non validé** |
| M2 `head`, `search`, `queue`, prototype Streamlit | `train`, `score`, `queue`, `search` en service et en CLI ; GUI Streamlit à faire |
| M3 `sequential`, `fusion`, `aggregate`, audit aléatoire | modules écrits et testés ; `aggregate` branché, `sequential`/`fusion` pas encore |

386 tests passent sur Python 3.11 (`uv run pytest`), dont la chaîne complète en ligne de
commande sur un corpus synthétique. Tous les modules sont couverts sauf
`encoders/bacpipe_encoder.py`, `encoders/onnx_encoder.py` et `encoders/export.py`, qui
demandent respectivement bacpipe (groupe `research`) et un modèle exporté.

**Non validé sur données réelles** : l'adaptateur bacpipe devine l'API de la bibliothèque
(noms de modules, `SAMPLE_RATE`, `preprocess()`) ; il sera confronté à bacpipe installé en M1.

Le format d'entrée est écrit d'après la description de Léonard et couvert par
`tests/test_real_format.py`, mais aucun fichier réel n'a encore été lu. L'acceptation M0
reste à faire : `ingest` sur une centaine de fichiers, puis `import-labels --dry-run` sur le
vrai tableau, pour relire les verdicts et les lignes signalées avant d'écrire quoi que ce
soit en base.
