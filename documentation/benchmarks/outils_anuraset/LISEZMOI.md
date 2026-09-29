# Outils des benchmarks AnuraSet par encodeur (02 et suivants)

À lancer depuis la racine du dépôt, `config/anuraset.yaml` et bacpipe installés.

- `encoder.py <encodeurs…>` : même sélection que `anuraset-campaign` (n° 141), encodage seul.
- `bench.py <encoder_id> <ESPÈCE> <sorties>` : les 14 têtes de `anuraset.CAMPAIGN_HEADS` sur
  une espèce (un processus par espèce, 1 thread BLAS) ; garde aussi les scores hors-pli.
- `rassembler.py <sorties> <dossier du rapport>` : `donnees/` (Holm sur toutes les espèces,
  `amorcage.csv` : seuil de précision 0,5 choisi sur les autres sites, appliqué au site).
- `generer_modele.py` : modèle du `generer.py` d'un rapport (remplacer `__DOSSIER__`,
  `__ENCODEUR__`, `__FENETRE__`).
