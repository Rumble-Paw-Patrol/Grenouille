# esp-aves2 EfficientNet-B0 (bio, all, audioset) sur AnuraSet : benchmark 07, vague 2 (session 12)
**Réglages lus** : adaptateur `avex` (n° 152), f_e 16 kHz, fenêtre 5 s jointive, embedding 1280
(jetons 16 × 4 × 1280 non utilisés : ligne 12 = `--curve` seul). avex 1.3.0, torch 2.14.0+cpu,
transformers 4.57.6, onnxruntime 1.30.0 ; tensorflow et librosa non installés (inutiles).
1 599 enregistrements encodés, 0 sauté. **Débit** (4 cœurs) : 44,6 (bio), 43,4 (all), 42,6
(audioset) fenêtres/s. Benchmark : 15 tâches `--curve`, 4 en parallèle, 80 à 790 s chacune.

**Témoin BOAFAB** (n° 151, logistique, AP moyenne par site, minute) : bio 0,96, all 0,97,
audioset 0,92 : au-dessus de 0,85, tuyau sain.

**Logistique, minute, AP moyenne par site / AP poolée** (sites où l'espèce chante : 3, 2, 3, 2, 2)

| | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB | moyenne |
|---|---|---|---|---|---|---|
| bio | 0,42 / 0,83 | 0,48 / 0,16 | 0,69 / 0,69 | 0,37 / 0,11 | 0,96 / 0,86 | 0,59 / 0,53 |
| all | 0,36 / 0,63 | 0,38 / 0,15 | 0,65 / 0,43 | 0,58 / 0,23 | 0,97 / 0,88 | 0,59 / 0,46 |
| audioset | 0,32 / 0,14 | 0,26 / 0,10 | 0,52 / 0,50 | 0,49 / 0,11 | 0,92 / 0,63 | 0,50 / 0,29 |

**Autres têtes** (moyenne des 5 espèces, minute, par site / poolée ; bio, all, audioset) :
logistic+R37=glmm 0,59/0,50, 0,59/0,42, 0,50/0,23 ; lda_shrunk 0,51/0,53, 0,51/0,50, 0,50/0,34 ;
simple_prototype 0,49/0,31, 0,42/0,26, 0,27/0,13. Niveau fenêtre et courbe d'amorçage :
`resultats/global/esp_aves2_effnetb0_*` (branche `resultats-anuraset-07`). Pas de tête sur jetons.

**Lecture** : bio et all à égalité en AP par site (0,59), audioset en dessous (0,50), surtout en
poolée (0,29). 2 à 3 sites par espèce, pas de test apparié (rapport global). CNN : « en retrait en
sondage linéaire » sans objet (règle n° 151 : transformers).

**Anomalies** : l'encodage de `bio` a duré 1 296 s contre 467 et 476 s pour `all` et `audioset` à
débit affiché voisin, cause non élucidée (stock complet). PITAZU : deux sites, AP très variable.
Stocks : branches `donnees-anuraset-esp_aves2_effnetb0_{bio,all,audioset}`.
