# Embeddings protoclr d'AnuraSet (branche `donnees-anuraset-protoclr`)

Calculés le 29/09/2026 dans le cloud (même sélection que le benchmark 01, DECISIONS n° 141) :
1 599 enregistrements, 15 970 fenêtres de 6 s jointives (fenêtre native de protoclr), 384 dimensions, encodage 25 min.

- `data/embeddings_anuraset/protoclr-bacpipe1.3.5@o0/` : le stock (Parquet), 13M ;
- `data/db/anuraset.sqlite` : l'inventaire (enregistrement → fichier, site).

Pour les utiliser : placer ces deux dossiers dans `data/` du dépôt (la base peut remplacer celle
de `donnees-anuraset` : mêmes enregistrements). Les têtes se relancent sans audio :
`uv run blanci --config config/anuraset.yaml anuraset-heads --encoder protoclr-bacpipe1.3.5@o0`.
