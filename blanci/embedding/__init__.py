"""Étape 2 — des enregistrements aux embeddings.

- `grid` : grille de fenêtres, indépendante de l'encodeur.
- `encoders/` : encodeurs (bacpipe, AVEX, ONNX du livrable), passe-bas, export ONNX.
- `embed` : extraction (enregistrements → grille → encodeur → stock).
- `channels` : piste lue par enregistrement (règles par micro et période, son étouffé),
  et retrait d'un stock de ce qui est encodé sur une autre piste.
- `store` : stockage Parquet partitionné des embeddings.
- `index` : recherche par similarité cosinus.
"""
