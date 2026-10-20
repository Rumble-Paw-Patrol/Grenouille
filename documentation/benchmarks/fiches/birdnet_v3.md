# birdnet_v3 sur AnuraSet : benchmark 07, vague 2 (session 5, 30/09)
**Réglages lus** (bacpipe 1.3.5) : BirdNET+ V3.0 préversion, ONNX (CPUExecutionProvider), f_e
32 kHz (AnuraSet à 22,05 kHz rééchantillonné), fenêtre 3 s jointive, embedding 1 280 ; logits
de son classifieur rangés (DENMIN, LEPLAT, PHYCUV, BOAFAB = « Hypsiboas faber »,
*A. baeobatrachus*). torch 2.14.0+cpu, transformers 4.57.6, onnxruntime 1.30.0, librosa 0.11.0,
sans TensorFlow. 1 599 enregistrements, 31 938 fenêtres.
**Débit** (4 cœurs) : encodage 22 f/s (38 min) ; benchmark `--curve --native`, 4 processus :
147 (PITAZU) à 1 239 s (PHYCUV) par espèce.
**Témoin BOAFAB** (logistique, AP moyenne par site, minute) : **0,984**, tuyau sain.
**Transfert vers un site neuf, minute, AP moyenne par site / poolée**

| Tête | DENMIN | PITAZU | PHYCUV | LEPLAT | BOAFAB |
|---|---|---|---|---|---|
| logistique | 0,407 / 0,886 | 0,531 / 0,172 | 0,918 / 0,831 | 0,829 / 0,390 | 0,984 / 0,899 |
| logistic+R37=glmm | 0,402 / 0,885 | 0,552 / 0,161 | 0,915 / 0,837 | 0,827 / 0,400 | 0,984 / 0,877 |
| lda_shrunk | 0,401 / 0,870 | 0,429 / 0,593 | 0,901 / 0,829 | 0,657 / 0,527 | 0,963 / 0,932 |
| simple_prototype | 0,357 / 0,340 | 0,413 / 0,125 | 0,526 / 0,321 | 0,441 / 0,113 | 0,922 / 0,694 |
| **classifieur d'origine** (sans entraînement) | **0,614** / 0,890 | — | 0,890 / 0,780 | 0,698 / 0,259 | **0,987 / 0,959** |

Moyenne des 5 espèces (logistique, par site) : **0,73** (benchmark 07 : perch_v2 0,79, birdnet
v2.4 0,68). Par espèce contre perch_v2 : LEPLAT 0,83 / 0,82, PITAZU 0,53 / 0,51, BOAFAB égal,
PHYCUV 0,92 / 0,96, DENMIN 0,41 / 0,67. Pas de test apparié ici (au rapport global).
**Courbe d'amorçage** (logistique, AP minute, moyenne des sites, k = 0 → 20) : DENMIN 0,59→0,79,
PITAZU 0,80→0,97, PHYCUV 0,94→0,96, LEPLAT 0,79→0,87, BOAFAB 0,98→0,98.
**Lecture** : CNN supervisé, tête linéaire = tête des auteurs (n° 151 sans objet). Nette avance
sur la v2.4 en LEPLAT (+0,35) et PHYCUV (+0,11), pas sur perch_v2 en moyenne. Son classifieur
sans entraînement égale ou bat la logistique sur BOAFAB et DENMIN, reste sous elle sur PHYCUV et
LEPLAT ; rien de spectaculaire, donc pas de signe net qu'AnuraSet soit dans ses données
d'entraînement (non exclu).
**Anomalies** : DENMIN par site tiré par INCT4 (2 min positives) ; poolé 0,89 (perch_v2 0,91). Classifieur d'origine : AP poolée LEPLAT 0,26 (calibrage entre sites). Stock : `donnees-anuraset-birdnet_v3` ; scores : `resultats/global/birdnet_v3_*` (historique : `git show 6f2ff13^:resultats/global/<fichier>`).
