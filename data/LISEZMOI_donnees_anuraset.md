# Embeddings birdnet d'AnuraSet (branche `donnees-anuraset-birdnet`)

Calculés le 29/09/2026 dans le cloud (même sélection que le benchmark 01, DECISIONS n° 141) :
1 599 enregistrements, 31 938 fenêtres de 3 s jointives (fenêtre native de BirdNET), 1024 dimensions, encodage 23 min.

- `data/embeddings_anuraset/birdnet-bacpipe1.3.5@o0/` : le stock (Parquet), 56M ;
- `data/db/anuraset.sqlite` : l'inventaire (enregistrement → fichier, site).

Pour les utiliser : placer ces deux dossiers dans `data/` du dépôt (la base peut remplacer celle
de `donnees-anuraset` : mêmes enregistrements). Les têtes se relancent sans audio :
`uv run blanci --config config/anuraset.yaml anuraset-heads --encoder birdnet-bacpipe1.3.5@o0`.
