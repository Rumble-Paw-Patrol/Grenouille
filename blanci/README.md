# blanci

Le paquet Python du projet : un sous-dossier par étape de la chaîne (inventaire, encodage, têtes, fusion, évaluation, annotation, résultats). La commande `blanci` (`cli.py`) donne accès à toutes les étapes.

- `cli.py` : l'interface en ligne de commande `blanci`.
- `service.py` : la couche de service, seul point d'entrée de la CLI et de la future interface graphique.
- `core/` : le socle commun (configuration, base SQLite, lecture de l'audio, recherche des enregistrements sur les disques).
- `inputs/` : étape 1, les enregistrements et les annotations (inventaire, contrôle qualité, labels, jeu gelé).
- `embedding/` : étape 2, des enregistrements aux embeddings (grille de fenêtres, encodeurs, stock Parquet, recherche par similarité).
- `heads/` : étape 3, les têtes de détection (embedding ou signal vers score), leurs régularisations et les baselines.
- `combination/` : étape 4, la combinaison de plusieurs modèles (fusion, stacking, ensemble).
- `evaluation/` : étape 5, la mesure (plis groupés, AP, rappel à précision fixée, benchmarks).
- `annotation/` : étape 6, la boucle d'écoute (sélection des candidats, poste d'annotation Streamlit).
- `results/` : étape 7, les sorties écologiques (agrégation fenêtre vers point d'écoute, courbes d'activité).
- `exploration/` : les fonctions et graphiques des notebooks de `notebooks/`.
