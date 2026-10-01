"""Étape 2 — des enregistrements aux embeddings.

- `grid` : grille de fenêtres, indépendante de l'encodeur.
- `encoders/` : encodeurs (bacpipe, AVEX, ONNX du livrable), passe-bas, export ONNX.
- `embed` : extraction (enregistrements → grille → encodeur → stock).
- `store` : stockage Parquet partitionné des embeddings.
- `index` : recherche par similarité cosinus.
"""
