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

## Commandes (jalon M0)

```sh
# Inventaire + contrôle qualité (reprenable : les fichiers déjà connus sont sautés)
uv run blanci ingest --dataset 2026

# Import des annotations ; --dry-run d'abord pour relire l'analyse des commentaires
uv run blanci import-labels data/labels/imports/positifs.csv --kind positive --dry-run
uv run blanci import-labels data/labels/imports/positifs.csv --kind positive
uv run blanci import-labels data/labels/imports/faux_amis.csv --kind negative

# Les notes annotées tiennent-elles entières dans les fenêtres des grilles 3 s et 5 s ?
uv run blanci check-grid

uv run blanci status
```

Les colonnes des fichiers d'annotation sont reconnues automatiquement (fichier, début,
commentaire, qualité, site, micro) ; si une colonne n'est pas trouvée, ajouter son nom dans
`labels.import.columns` de la config. Chaque ligne doit désigner un enregistrement déjà
inventorié.

## Avancement

| Jalon | État |
|---|---|
| M0 dépôt, config, `ingest`, `import-labels`, tests `grid` / `resample` / `search` | code et tests écrits ; **acceptation à faire sur les données réelles** |
| M1 `embed` via bacpipe, `benchmark` en plis par micro | modules écrits et testés ; adaptateur bacpipe non validé, `benchmark.py` à écrire |
| M2 `head`, `search`, `queue`, prototype Streamlit | `head`, `active`, `dataset` écrits et testés ; `service.py` et la GUI à faire |
| M3 `sequential`, `fusion`, `aggregate`, audit aléatoire | modules écrits et testés ; non branchés sur la CLI |

244 tests passent sur Python 3.11 (`uv run pytest`). Tous les modules sont couverts sauf
`encoders/bacpipe_encoder.py`, `encoders/onnx_encoder.py` et `encoders/export.py`, qui
demandent respectivement bacpipe (groupe `research`) et un modèle exporté.

**Non validé sur données réelles** : l'adaptateur bacpipe devine l'API de la bibliothèque
(noms de modules, `SAMPLE_RATE`, `preprocess()`) ; le format des 345 + 158 annotations est
inconnu, l'importeur suppose un tableau CSV ou Excel.
