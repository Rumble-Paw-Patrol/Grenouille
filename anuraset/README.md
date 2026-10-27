# Outils des benchmarks AnuraSet par encodeur (02 et suivants)

Le module est `blanci/evaluation/anuraset.py`, les commandes `blanci anuraset …`.

À lancer depuis la racine du dépôt, `anuraset/anuraset.yaml` et bacpipe installés.

## Racine

- `README.md` : ce fichier.
- `anuraset.yaml` : réglages et base séparés des données ONF.

## `benchmarks/`

- `LISEZMOI.md` : l'index des rapports.
- `MODELE_RAPPORT.md` : le modèle de rapport.
- `generer_modele.py` : le modèle du `generer.py` d'un rapport (remplacer `__DOSSIER__`, `__ENCODEUR__`, `__FENETRE__`).
- `fiches/` : une fiche par encodeur.
- `<date>_anuraset_<nom>/` : les 8 rapports, chacun avec `RAPPORT.md`, `donnees/`, `figures/` et `generer.py`.

## `encodage/`

- `encoder.py <encodeurs…>` : même sélection que `anuraset campaign` (n° 141), encodage seul.
- `telecharger_anuraset.sh` : `raw_data.zip` d'AnuraSet par 12 plages d'octets en parallèle (vague d'encodage 2).
- `jetons.py <encodeur>` : jetons de toutes les fenêtres d'un stock (moyennés sur la fréquence, float16, par paquets de 50 enregistrements), pour `calcul/global_bench.py --tokens` (vague d'encodage 2).
- `importer_stock.py <encodeur>` : stock et modèle d'une branche `donnees-anuraset-*` ajoutés à la base locale, sans l'écraser (vague d'encodage 2).

## `calcul/`

- `bench.py <encoder_id> <ESPÈCE> <sorties>` : les 14 têtes de `anuraset.CAMPAIGN_HEADS` sur une espèce (un processus par espèce, 1 thread BLAS) ; garde aussi les scores hors-fold.
- `rassembler.py <sorties> <dossier du rapport>` : `donnees/` (Holm sur toutes les espèces, `amorcage.csv` : seuil de précision 0,5 choisi sur les autres sites, appliqué au site).
- `global_bench.py <encodeur> <ESPÈCE> <sorties> [--curve]` : benchmark global (07), sur les stocks des six encodeurs réunis dans une même base ; transfert vers un site neuf (un fold par site, toutes les fenêtres du site jugées, scores gardés). `--curve` : courbe d'amorçage (k enregistrements positifs du site cible ajoutés à l'entraînement). Vague d'encodage 2 : `--tokens` (attentive, logistic:max, proto_probe), `--native` (classifieur de l'encodeur, sans entraînement) ; stock trouvé sur le disque (bacpipe, avex).
- `rassembler_global.py <sorties> <dossier du rapport>` : `donnees/` du benchmark global (niveaux fenêtre et minute, comparaisons appariées à perch_v2 + logistique, Holm ; seuil choisi ailleurs ; courbe).
- `rassembler_08.py <sorties> <dossier du rapport>` : `donnees/` du rapport 08, sur les sorties de `global_bench.py` de tous les encodeurs réunis.
- `apparie.py` : briques communes aux rassemblements du benchmark global (07, 08) : scores par minute, AP moyenne par site, bootstrap apparié.

## `notes/`

- `VAGUE_ENCODAGE_2.md` : plan des sessions de la vague d'encodage 2 (n° 151, 152).
