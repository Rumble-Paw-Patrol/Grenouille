# Embeddings perch_bird d'AnuraSet (branche `donnees-anuraset-perch_bird`)

Calculés le 29/09/2026 dans le cloud (même sélection que le benchmark 01, DECISIONS n° 141) :
1 599 enregistrements, 19 166 fenêtres de 5 s jointives, 1280 dimensions, encodage 88 min.

- `data/embeddings_anuraset/perch_bird-bacpipe1.3.5@o0/` : le stock (Parquet), 46M ;
- `data/db/anuraset.sqlite` : l'inventaire (enregistrement → fichier, site).

Pour les utiliser : placer ces deux dossiers dans `data/` du dépôt (la base peut remplacer celle
de `donnees-anuraset` : mêmes enregistrements). Les têtes se relancent sans audio :
`uv run blanci --config config/anuraset.yaml anuraset-heads --encoder perch_bird-bacpipe1.3.5@o0`.
