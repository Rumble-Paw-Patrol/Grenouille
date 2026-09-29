# Fiche AnuraSet : insect66 et insect459 (session « encodage 10 », 29/09)

**Réglages lus** (bacpipe 1.3.5, `embeddings`) : f_e 44 100 Hz, fenêtre 5,5 s jointive, dimension
1280 (les deux). AnuraSet est à 22,05 kHz : rééchantillonné à 44,1 kHz par l'adaptateur (bande
utile ≤ 11 kHz). Pas de jetons. Versions : torch 2.14.0+cpu, transformers 4.57.6, onnxruntime
1.30.0, librosa 0.11.0, timm 1.0.30, pas de TensorFlow. Architecture non lue.

**Débit** (CPU 4 cœurs) : 24,1 et 26,7 f/s, 1599 enregistrements chacun. Benchmark : 10 tâches
`--curve`, 111 à 686 s chacune.

**Témoin BOAFAB** (logistique, AP moyenne par site, minute) : insect66 **0,916**, insect459
**0,863** : au-dessus de 0,85, tuyau sain (insect459 tout juste).

**Transfert vers un site neuf, niveau minute** (AP moyenne par site / poolée) :

| Espèce | insect66 logistique | insect66 prototype simple | insect459 logistique | insect459 prototype simple |
|---|---|---|---|---|
| DENMIN | 0,314 / 0,158 | 0,289 / 0,241 | 0,297 / 0,214 | 0,288 / 0,318 |
| PITAZU | 0,262 / 0,099 | 0,285 / 0,091 | 0,300 / 0,107 | 0,276 / 0,121 |
| PHYCUV | 0,391 / 0,243 | 0,203 / 0,115 | 0,366 / 0,260 | 0,195 / 0,100 |
| LEPLAT | 0,168 / 0,074 | 0,186 / 0,101 | 0,154 / 0,077 | 0,167 / 0,090 |
| BOAFAB | 0,916 / 0,793 | 0,770 / 0,365 | 0,863 / 0,785 | 0,472 / 0,164 |

Autres têtes : `resultats/global/`. Seule `lda_shrunk` sort du lot, sur PHYCUV (0,407 et 0,472).

**Courbe d'amorçage** (logistique, AP minute, moyenne des sites, k = 0 puis 20 enregistrements
positifs du site) : insect66 DENMIN 0,46→0,58, PITAZU 0,51→0,85, PHYCUV 0,42→0,73, LEPLAT
0,29→0,69, BOAFAB 0,92→0,93 ; insect459 : 0,44→0,58, 0,53→0,90, 0,38→0,73, 0,27→0,64, 0,87→0,95.

**Lecture** : hors BOAFAB, transfert sans exemple faible (AP par site 0,15 à 0,40), la courbe
monte avec les exemples locaux. Pas de comparaison à perch_v2 ici (rapport global). Ni l'un ni
l'autre n'est déclaré transformer : « en retrait en sondage linéaire » sans objet (non vérifié).

**Anomalies** : encodage d'insect66 (2219 s) bien plus long que celui d'insect459 (687 s),
non expliqué. Stocks : branches `donnees-anuraset-insect66` et `-insect459` ; résultats :
`resultats-anuraset-07`, `resultats/global/insect*`.
