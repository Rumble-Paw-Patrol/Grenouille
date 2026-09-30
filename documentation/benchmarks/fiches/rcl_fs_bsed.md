# Fiche AnuraSet : rcl_fs_bsed (session « encodage 11 », 29-30/09)

**Réglages lus** (bacpipe 1.3.5) : f_e 22 050 Hz (celle d'AnuraSet), fenêtre 0,2 s jointive,
dimension 2048, pas de jetons. Essai de 5 min sain. Versions : torch 2.6.0+cpu, transformers
4.57.6, onnxruntime 1.30.0, tensorflow 2.20.0, librosa 0.11.0.
**Débit** (CPU 4 cœurs) : 80,5 trames/s, 478 995 trames, 7197 s. Stock : branche
`donnees-anuraset-rcl_fs_bsed`.
**Benchmark** : transfert seul. **Écart de protocole** (accord de Léonard) : le pli le plus
lourd (262 125 fenêtres × 2048) ne tient pas en 15 Go ; **apprentissage plafonné à 60 000
fenêtres par pli** (tous les positifs, négatifs tirés, graine fixe), test par paquets. Non
strictement comparable aux autres encodeurs. Durées (s) : BOAFAB 2674 (reprise), DENMIN 7854,
PITAZU 4858, PHYCUV 9241, LEPLAT 3785.
**Témoin BOAFAB** (logistique, AP moyenne par site, minute) : **0,718, sous 0,85** ; perch_v2
0,985, birdnet 0,973, birdmae_base 0,934 (mêmes deux sites : INCT20955 0,903, INCT4 0,534).
**Non passé, cause non tranchée** : f_e, fenêtre, dimension vérifiées. Pistes : plafond ;
trame de 0,2 s plus courte que le chant (le max sur 300 trames par minute crée des faux positifs).
**Transfert vers un site neuf, niveau minute** (AP moyenne par site / poolée) :

| Espèce | logistique | prototype simple (tête d'origine) | lda_shrunk |
|---|---|---|---|
| DENMIN | 0,352 / 0,832 | 0,270 / 0,108 | 0,345 / 0,775 |
| PITAZU | 0,426 / 0,255 | 0,323 / 0,377 | 0,448 / 0,718 |
| PHYCUV | 0,418 / 0,253 | 0,165 / 0,081 | 0,482 / 0,250 |
| LEPLAT | 0,431 / 0,112 | 0,175 / 0,075 | 0,380 / 0,129 |
| BOAFAB | 0,718 / 0,474 | 0,270 / 0,197 | 0,624 / 0,404 |

`logistic+R37=glmm` à ±0,02 de la logistique. Moyennes sur les sites avec positifs (DENMIN 3,
PITAZU 2, PHYCUV 3, LEPLAT 2, BOAFAB 2). Sorties : `resultats-anuraset-07`,
`resultats/global/rcl_fs_bsed_*`. CNN : « en retrait en sondage linéaire » sans objet.
**Anomalies** : témoin non passé, plafond ; en AP par site, le prototype simple (tête d'origine)
est le plus faible sur les cinq espèces ; grille de 0,2 s et prototype sur candidats non faits.
