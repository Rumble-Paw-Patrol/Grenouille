# Outils des benchmarks AnuraSet par encodeur (02 et suivants)

**Pré-benchmark terminé** : plus utilisé pour la suite du projet, gardé pour reproduire les
rapports `documentation/benchmarks/*_anuraset_*` (DECISIONS n° 166). Le module est
`blanci/evaluation/anuraset.py`, les commandes `blanci anuraset …`.

À lancer depuis la racine du dépôt, `anuraset/anuraset.yaml` et bacpipe installés.

- `encoder.py <encodeurs…>` : même sélection que `anuraset campaign` (n° 141), encodage seul.
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

Vague d'encodage 2 (n° 151, 152 ; plan des sessions : `VAGUE_ENCODAGE_2.md`) :

- `telecharger_anuraset.sh` : `raw_data.zip` d'AnuraSet par 12 plages d'octets en parallèle.
- `jetons.py <encodeur>` : jetons de toutes les fenêtres d'un stock (moyennés sur la fréquence,
  float16, par paquets de 50 enregistrements), pour `global_bench.py --tokens`.
- `importer_stock.py <encodeur>` : stock et modèle d'une branche `donnees-anuraset-*` ajoutés
  à la base locale, sans l'écraser.
- `global_bench.py` : `--tokens` (attentive, logistic:max, proto_probe), `--native`
  (classifieur de l'encodeur, sans entraînement) ; stock trouvé sur le disque (bacpipe, avex).
