# Embeddings birdmae_base d'AnuraSet (branche `donnees-anuraset-birdmae_base`)

Calculés le 29/09/2026 dans le cloud, CPU seul (même sélection que le benchmark 01, DECISIONS n° 141) :
1 599 enregistrements, 19 166 fenêtres de 5 s jointives, 768 dimensions (Bird-MAE-Base,
`DBD-research-group/Bird-MAE-Base` via bacpipe 1.3.5), ~7,5 fenêtres/s sur 4 cœurs (≈ 45 min
d'encodage, en trois reprises : le conteneur a redémarré deux fois, l'encodage a repris où il en était).

- `data/embeddings_anuraset/birdmae_base-bacpipe1.3.5@o0/` : le stock (Parquet), 28M ;
- `data/db/anuraset.sqlite` : l'inventaire (enregistrement → fichier, site).

Pour les utiliser : placer ces deux dossiers dans `data/` du dépôt (la base peut remplacer celle
de `donnees-anuraset` : mêmes enregistrements). Les têtes se relancent sans audio :
`uv run blanci --config config/anuraset.yaml anuraset-heads --encoder birdmae_base-bacpipe1.3.5@o0`.
