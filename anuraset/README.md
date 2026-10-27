# Outils des benchmarks AnuraSet par encodeur (02 et suivants)

**Pré-benchmark terminé** : plus utilisé pour la suite du projet, gardé pour reproduire les rapports `anuraset/benchmarks/*_anuraset_*` (DECISIONS n° 166). Le module est `blanci/evaluation/anuraset.py`, les commandes `blanci anuraset …`.

À lancer depuis la racine du dépôt, `anuraset/anuraset.yaml` et bacpipe installés.

Rangement du dossier :

- `anuraset.yaml` : réglages et base séparés des données ONF.
- `benchmarks/` : les 8 rapports (données, figures, `generer.py`), l'index `LISEZMOI.md`, `MODELE_RAPPORT.md`, `generer_modele.py` et les fiches par encodeur (`fiches/`).
- `encodage/` : téléchargement des données, encodage, jetons, import d'un stock.
- `calcul/` : benchmarks (`bench.py`, `global_bench.py`) et rassemblement des sorties en `donnees/` (`rassembler*.py`, `apparie.py`).
- `notes/` : plans de session (`VAGUE_ENCODAGE_2.md`).

- `encodage/encoder.py <encodeurs…>` : même sélection que `anuraset campaign` (n° 141), encodage seul.
- `calcul/bench.py <encoder_id> <ESPÈCE> <sorties>` : les 14 têtes de `anuraset.CAMPAIGN_HEADS` sur une espèce (un processus par espèce, 1 thread BLAS) ; garde aussi les scores hors-pli.
- `calcul/rassembler.py <sorties> <dossier du rapport>` : `donnees/` (Holm sur toutes les espèces, `amorcage.csv` : seuil de précision 0,5 choisi sur les autres sites, appliqué au site).
- Le modèle du `generer.py` d'un rapport est dans `anuraset/benchmarks/generer_modele.py` (remplacer `__DOSSIER__`, `__ENCODEUR__`, `__FENETRE__`).

Benchmark global (07), sur les stocks des six encodeurs réunis dans une même base :

- `calcul/global_bench.py <encodeur> <ESPÈCE> <sorties> [--curve]` : transfert vers un site neuf (un pli par site, toutes les fenêtres du site jugées, scores gardés) ; `--curve` : courbe d'amorçage (k enregistrements positifs du site cible ajoutés à l'entraînement).
- `calcul/rassembler_global.py <sorties> <dossier du rapport>` : `donnees/` (niveaux fenêtre et minute, comparaisons appariées à perch_v2 + logistique, Holm ; seuil choisi ailleurs ; courbe).

Vague d'encodage 2 (n° 151, 152 ; plan des sessions : `notes/VAGUE_ENCODAGE_2.md`) :

- `encodage/telecharger_anuraset.sh` : `raw_data.zip` d'AnuraSet par 12 plages d'octets en parallèle.
- `encodage/jetons.py <encodeur>` : jetons de toutes les fenêtres d'un stock (moyennés sur la fréquence, float16, par paquets de 50 enregistrements), pour `calcul/global_bench.py --tokens`.
- `encodage/importer_stock.py <encodeur>` : stock et modèle d'une branche `donnees-anuraset-*` ajoutés à la base locale, sans l'écraser.
- `calcul/global_bench.py` : `--tokens` (attentive, logistic:max, proto_probe), `--native` (classifieur de l'encodeur, sans entraînement) ; stock trouvé sur le disque (bacpipe, avex).
