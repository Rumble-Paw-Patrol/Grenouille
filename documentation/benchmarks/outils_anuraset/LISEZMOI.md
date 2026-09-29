# Outils des benchmarks AnuraSet par encodeur (02 et suivants)

À lancer depuis la racine du dépôt, `config/anuraset.yaml` et bacpipe installés.

- `encoder.py <encodeurs…>` : même sélection que `anuraset-campaign` (n° 141), encodage seul.
- `bench.py <encoder_id> <ESPÈCE> <sorties>` : les 14 têtes de `anuraset.CAMPAIGN_HEADS` sur
  une espèce (un processus par espèce, 1 thread BLAS) ; garde aussi les scores hors-pli.
- `rassembler.py <sorties> <dossier du rapport>` : `donnees/` (Holm sur toutes les espèces,
  `amorcage.csv` : seuil de précision 0,5 choisi sur les autres sites, appliqué au site).
- `generer_modele.py` : modèle du `generer.py` d'un rapport (remplacer `__DOSSIER__`,
  `__ENCODEUR__`, `__FENETRE__`).

Benchmark global (07), sur les stocks des six encodeurs réunis dans une même base :

- `global_bench.py <encodeur> <ESPÈCE> <sorties> [--curve]` : transfert vers un site neuf (un pli
  par site, toutes les fenêtres du site jugées, scores gardés) ; `--curve` : courbe d'amorçage
  (k enregistrements positifs du site cible ajoutés à l'entraînement).
- `rassembler_global.py <sorties> <dossier du rapport>` : `donnees/` (niveaux fenêtre et minute,
  comparaisons appariées à perch_v2 + logistique, Holm ; seuil choisi ailleurs ; courbe).
