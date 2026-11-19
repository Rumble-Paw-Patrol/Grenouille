# Contrôle qualité : vérifications sur échantillon

Scripts qui vérifient ce que l'inventaire et le contrôle qualité (`blanci ingest`, `blanci qc`, `blanci flag`) décident, sans rien écrire dans la base ni sur les disques. Leurs sorties vont dans `paths.reports/` et sont lues par les notebooks `05_inventaire` et `06_controle_qualite`.

- `echantillon_pistes.py` : un enregistrement par micro et par jour, plus ceux qui encadrent une bascule de piste étouffée. Mesure les deux pistes : gains réglés (en-tête GUANO), écart de niveau entre pistes, `hf_ratio` par tranche de 10 s, indices de défaut (clics, trous numériques, facteur de crête). Reprenable. Environ 45 min pour les deux disques.
- `spectre_pistes.py` : sur les mêmes enregistrements (ids de `echantillon_pistes.csv`), le spectre moyen de chaque piste (Welch de `blanci qc`, 2 min) en 24 tiers d'octave de 100 Hz à 20 kHz (`inventaire.track_band_levels`), écrit dans `pistes/spectre_pistes.csv` pour ne plus relire le son. L'écart bande par bande entre pistes, corrigé du gain réglé, se lit avec `inventaire.spectral_gaps` et `spectral_gap_by_point` ; figure : `documentation/prez/scripts/figure_spectre_pistes.py`. Reprenable. Environ 1 h pour les deux disques.
- `verifier_doublons.py` : relit les doublons écartés à l'inventaire (`ingest_duplicates_*.csv`) et compare le contenu des deux copies (`ingest.same_recording`), ou cherche les noms répétés sous un dossier (`--disque`). Les copies écartées ne sont que sur le disque d'origine.
- `files_apres_prez4.py` : trois files d'écoute du récap de la présentation n° 4 (`documentation/prez/recap_qc_presentation_4.md`) : `kaw_d_micro_sourd` (la panne de la piste 0 de Kaw D en 2024, étape par étape), `blancinet_ecartes` (écartés que BlanciNet note à 0,5 ou plus, jamais écoutés) et `coupes_programme` (fichiers coupés sur un créneau du programme, écartés, jamais écoutés).

```
uv run python scripts/controle_qualite/echantillon_pistes.py
uv run python scripts/controle_qualite/spectre_pistes.py
uv run python scripts/controle_qualite/verifier_doublons.py --disque F:/audio
uv run python scripts/controle_qualite/files_apres_prez4.py
```
