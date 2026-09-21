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

Hors git, sous `data/` (chemins dans `config/default.yaml`) :

```
data/raw/{2023,2026}/<site>/<micro>/*.wav   # audio brut, jamais modifié
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

Les colonnes des fichiers d'annotation sont reconnues automatiquement (fichier, début,
commentaire, qualité, site, micro) ; si une colonne n'est pas trouvée, ajouter son nom dans
`labels.import.columns` de la config. Chaque ligne doit désigner un enregistrement déjà
inventorié.

## Avancement

| Jalon | État |
|---|---|
| M0 dépôt, config, `ingest`, `import-labels` | code et tests écrits ; **acceptation à faire sur les données réelles** |
| M1 `embed`, `benchmark` en plis par micro | écrits, testés, branchés sur la CLI ; **adaptateur bacpipe non validé** |
| M2 `head`, `search`, `queue`, prototype Streamlit | `train`, `score`, `queue`, `search` en service et en CLI ; GUI Streamlit à faire |
| M3 `sequential`, `fusion`, `aggregate`, audit aléatoire | modules écrits et testés ; `aggregate` branché, `sequential`/`fusion` pas encore |

320 tests passent sur Python 3.11 (`uv run pytest`), dont la chaîne complète en ligne de
commande sur un corpus synthétique. Tous les modules sont couverts sauf
`encoders/bacpipe_encoder.py`, `encoders/onnx_encoder.py` et `encoders/export.py`, qui
demandent respectivement bacpipe (groupe `research`) et un modèle exporté.

**Non validé sur données réelles** : l'adaptateur bacpipe devine l'API de la bibliothèque
(noms de modules, `SAMPLE_RATE`, `preprocess()`). Les annotations sont des fenêtres de 3 s ;
reste à savoir si elles arrivent en tableau (fichier + début) ou en extraits WAV découpés —
dans ce dernier cas, il faudra un importeur qui retrouve l'enregistrement et le décalage.
