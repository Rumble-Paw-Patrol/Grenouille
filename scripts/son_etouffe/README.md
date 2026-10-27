# Son étouffé

Chaque enregistreur a deux micros, ce qui fait que chaque enregistrement a deux pistes quasi identiques,
seul le gain change (de 6 à 18 dB) ainsi qu'un décalage temporel de l'ordre du centième de seconde.
J'ai initialement encodé mes données sur la piste 0, le module de contrôle qualité qc.py écartant les
pistes audio étouffées : un flag "muffled" leur est attribué (micro dans le sac, micro HS, tombé par terre...).
Je me suis rendu compte récemment que certains enregistrements dont la piste 0 était étouffée avaient une
piste 1 valable. 497 enregistrements dont la piste 1 était valable ont donc été réintégrés au corpus,
104 douteux vont être réécoutés pour être sûr et 399 enregistrements sont considérés perdus car les deux pistes sont étouffées.

Les scripts qui repèrent les enregistrements au son étouffé, piste par piste. Un enregistrement n'est écarté que si les deux pistes sont étouffées ; s'il en reste une valable, il est gardé avec son drapeau.

- `qc_deux_pistes.py` : mesure `hf_ratio` et `rms_dbfs` sur les deux canaux des micros signalés. Un enregistrement n'est écarté que si les deux pistes sont étouffées ; s'il en reste une valable, il est gardé avec son drapeau. Sorties dans `paths.reports/son_etouffe/`.
  
- `files_son_etouffe.py` : écrit les files d'écoute (`muffled_perdus`, `muffled_douteux`, `autres_drapeaux`) d'après les mesures de `qc_deux_pistes.py`. Elles arrivent dans `annotations/files_ecoute/`.

- `muffled_perdus`   : deux pistes étouffées, un extrait de 5 s par enregistrement ;
- `muffled_douteux`  : une piste étouffée et l'autre pas franchement claire, un extrait de 5 s par enregistrement ;
- `autres_drapeaux`  : durée anormale / date hors relevé, enregistrements entiers ;
