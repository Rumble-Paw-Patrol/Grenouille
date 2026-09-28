# Embeddings perch_v2 d'AnuraSet (branche `donnees-anuraset`)

Calculés le 28/09/2026 dans le cloud (benchmark 01, DECISIONS n° 141) : 1 599 enregistrements,
19 166 fenêtres de 5 s jointives, 1 536 dimensions.

- `data/embeddings_anuraset/perch_v2-bacpipe1.3.5@o0/` : le stock (Parquet), ~55 Mo ;
- `data/db/anuraset.sqlite` : l'inventaire (enregistrement → fichier, site), ~16 Mo.

Pour les utiliser : placer ces deux dossiers dans `data/` du dépôt, et l'audio d'AnuraSet
(Zenodo 8342596, `raw_data.zip`) sous `data/external/anuraset/` seulement si l'on veut
réencoder ou mesurer la fréquence dominante. Les têtes se relancent sans audio :
`uv run blanci --config config/anuraset.yaml anuraset-heads --encoder perch_v2-bacpipe1.3.5@o0`.
