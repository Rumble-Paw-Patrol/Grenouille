# Son étouffé

Les scripts qui repèrent les enregistrements au son étouffé, piste par piste. Un enregistrement n'est écarté que si les deux pistes sont étouffées ; s'il en reste une valable, il est gardé avec son drapeau.

- `qc_deux_pistes.py` : mesure `hf_ratio` et `rms_dbfs` sur les deux canaux des micros signalés, puis applique la règle des suites piste par piste. Lecture seule, sorties dans `paths.reports/son_etouffe/`.
- `files_son_etouffe.py` : écrit les files d'écoute (`muffled_perdus`, `muffled_douteux`, `autres_drapeaux`) d'après les mesures de `qc_deux_pistes.py`. Elles arrivent dans `annotations/files_ecoute/`.
