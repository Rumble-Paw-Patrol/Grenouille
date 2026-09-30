# Embeddings birdnet_v3 d'AnuraSet (branche `donnees-anuraset-birdnet_v3`)

Calculés le 30/09/2026 dans le cloud (vague d'encodage 2, session 5 ; sélection du n° 141) :
1 599 enregistrements, fenêtres de 3 s jointives (fenêtre native, 32 kHz), 1 280 dimensions,
encodage 38 min (22 fenêtres/s, CPU, 4 threads). BirdNET+ V3.0 préversion via bacpipe 1.3.5.
Les probabilités de son classifieur (DENMIN, LEPLAT, PHYCUV, BOAFAB, A. baeobatrachus) sont
rangées dans la base (table `scores`), pour `global_bench.py --native`.

- `data/embeddings_anuraset/birdnet_v3-bacpipe1.3.5@o0/` : le stock (Parquet), 70M ;
- `data/db/anuraset.sqlite` : l'inventaire et les scores du classifieur.

Pour les utiliser : `uv run python documentation/benchmarks/outils_anuraset/importer_stock.py birdnet_v3`.
