# Tableaux des benchmarks

Un tableau par benchmark du projet, en image PNG : les tableaux Markdown s'affichent mal dans
Xcode (DECISIONS n° 118). Cliquer sur l'image dans le navigateur de fichiers d'Xcode l'ouvre.

- `protocole.png` : comment se lit un benchmark (plis, niveaux, métriques, comparaisons) et
  la liste de tous les benchmarks avec leur commande.
- `encodeurs.png` : les 9 encodeurs du projet (f_e, fenêtre, dimension, débit mesuré, accès
  aux jetons et aux couches) et les autres modèles de bacpipe.
- `tetes.png` : toutes les têtes de `blanci heads`.
- `poolings.png` : les résumés des jetons (`logistic:<pooling>`).
- `pertes.png` : le benchmark des pertes (R34, R35), `--methods losses`.
- `voisins.png` : les têtes par similarité et leurs k (R39), `--methods neighbors`.
- `regularisations.png` : toutes les régularisations numérotées, leur état et leur réglage.
- `fusion.png` : méthodes et emplacements de `blanci fusion-bench`.
- `seuillage-amont.png` : transformations et portes de `blanci upstream-bench`.
- `negatifs-apparies.png` : les stratégies de négatifs présumés et leur contamination.
- `baselines.png` : les baselines sans encodeur.
- `ensembles.png` : les trois façons de combiner des modèles.

Une seule source : `generer.py` (données et rendu). Pour ajouter un benchmark ou corriger un
tableau, modifier la liste `TABLEAUX` puis :

    uv run --group notebook python documentation/tableaux/generer.py

Rendu avec Pillow et les polices fournies par matplotlib : rien à télécharger. Images en
palette réduite, 20 à 130 Ko chacune (~600 Ko en tout).
