# Contrôle qualité : vérifications sur échantillon

Scripts qui vérifient ce que l'inventaire et le contrôle qualité (`blanci ingest`, `blanci qc`, `blanci flag`) décident, sans rien écrire dans la base ni sur les disques. Leurs sorties vont dans `paths.reports/` et sont lues par les notebooks `05_inventaire` et `06_controle_qualite`.

- `echantillon_pistes.py` : un enregistrement par micro et par jour, plus ceux qui encadrent une bascule de piste étouffée. Mesure les deux pistes : gains réglés (en-tête GUANO), écart de niveau entre pistes, `hf_ratio` par tranche de 10 s, indices de défaut (clics, trous numériques, facteur de crête). Reprenable. Environ 45 min pour les deux disques.
- `verifier_doublons.py` : relit les doublons écartés à l'inventaire (`ingest_duplicates_*.csv`) et compare le contenu des deux copies (`ingest.same_recording`), ou cherche les noms répétés sous un dossier (`--disque`). Les copies écartées ne sont que sur le disque d'origine.

```
uv run python scripts/controle_qualite/echantillon_pistes.py
uv run python scripts/controle_qualite/verifier_doublons.py --disque F:/audio
```
